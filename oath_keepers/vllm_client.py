from typing import Callable, List, Tuple, Type, Union, cast

from mcp.types import (
    CallToolRequest,
    CallToolRequestParams,
    EmbeddedResource,
    ImageContent,
    PromptMessage,
    TextContent,
)
from mcp_agent import AgentConfig
from mcp_agent.agents.base_agent import BaseAgent
from mcp_agent.core.prompt import Prompt
from mcp_agent.llm.augmented_llm import (
    AugmentedLLM,
    AugmentedLLMProtocol,
    ModelT,
    RequestParams,
)
from mcp_agent.llm.provider_types import Provider
from mcp_agent.llm.providers.multipart_converter_openai import OpenAIConverter
from mcp_agent.llm.usage_tracking import TurnUsage
from mcp_agent.logging.logger import get_logger
from mcp_agent.mcp.helpers.content_helpers import get_text
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart
from openai._client import AsyncAPIClient
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from pydantic_core import from_json

from oath_keepers.utils.typing import GenerateRequest, SamplingParams, StructuredOutputsParams


class vLLM(AugmentedLLM):
    """
    vLLM interface for the fast-agent library.
    """

    base_url = "http://localhost:8000"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, provider=Provider.GENERIC, **kwargs)

    def _vllm_client(self) -> AsyncAPIClient:
        return AsyncAPIClient(
            version="v1",
            base_url=self.base_url,
            _strict_response_validation=False,
        )

    def _initialize_default_params(self, kwargs: dict) -> RequestParams:
        """Initialize vLLM-specific default parameters"""
        base_params = super()._initialize_default_params(kwargs)

        # Override with our specific settings
        base_params.maxTokens = 1024
        base_params.max_iterations = 10
        base_params.model = "vLLM"

        return base_params

    @staticmethod
    def _convert_messages_to_prompt(messages: List[ChatCompletionMessageParam]) -> str:
        """Convert OpenAI-style messages to a single prompt string for vLLM"""
        prompt_parts = []

        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")

            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")

        return "\n\n".join(prompt_parts)

    async def _vllm_completion(
        self,
        message: ChatCompletionMessageParam,
        request_params: RequestParams | None = None,
    ) -> List[TextContent | ImageContent | EmbeddedResource]:
        """
        Process a query using vLLM and available tools.
        """

        # Store responses of this function
        responses: List[TextContent | ImageContent | EmbeddedResource] = []

        # preparing vllm sampling params
        request_params = self.get_request_params(request_params=request_params)
        sampling_params = SamplingParams(max_tokens=request_params.maxTokens)
        if request_params.response_format is not None:
            sampling_params.structured_outputs = StructuredOutputsParams.from_optional(
                json=request_params.response_format
            )

        # Build messages list
        messages: List[ChatCompletionMessageParam] = []
        system_prompt = self.instruction or request_params.systemPrompt
        if system_prompt:
            messages.append(ChatCompletionSystemMessageParam(role="system", content=system_prompt))

        messages.extend(self.history.get(include_completion_history=request_params.use_history))
        messages.append(message)

        # Convert messages list to prompt format for vLLM
        prompt_text = self._convert_messages_to_prompt(messages)

        # Check for tool availability
        aggregator_response = await self.aggregator.list_tools()
        has_tools = bool(aggregator_response.tools)

        if has_tools:
            # Add tool information to prompt
            tool_descriptions = []
            for tool in aggregator_response.tools:
                tool_desc = f"- {tool.name}: {tool.description or 'No description'}"
                tool_descriptions.append(tool_desc)

            prompt_text += "\n\nAvailable tools:\n" + "\n".join(tool_descriptions)
            prompt_text += (
                "\n\nIf you need to use a tool, respond with: TOOL_CALL: tool_name {arguments}"
            )

        # Store the final assistant response for history
        final_assistant_content = ""

        for i in range(request_params.max_iterations):
            self._log_chat_progress(self.chat_turn(), model=self.default_request_params.model)

            try:
                # Make request to vLLM server
                response = await self._vllm_client().post(
                    "/generate",
                    cast_to=ChatCompletion,
                    body=GenerateRequest(
                        prompts=prompt_text,
                        sampling_params=sampling_params,
                    ).model_dump(),
                )

                # Track usage if available
                if hasattr(response, "usage") and response.usage:
                    try:
                        model_name = self.default_request_params.model
                        turn_usage = TurnUsage.from_openai(response.usage, model_name)
                        self._finalize_turn_usage(turn_usage)
                    except Exception as e:
                        self.logger.warning(f"Failed to track usage: {e}")

                self.logger.debug("vLLM completion response:", data=response)

                if isinstance(response, BaseException):
                    self.logger.error(f"Error: {response}")
                    break

                if not response.choices or len(response.choices) == 0:
                    break

                choice = response.choices[0]
                message_content = choice.message.content

                if not message_content:
                    break

                # Check for tool calls in response
                if has_tools and "TOOL_CALL:" in message_content:
                    # Parse tool call from response
                    lines = message_content.split("\n")
                    tool_call_line = None
                    regular_content = []

                    for line in lines:
                        if line.strip().startswith("TOOL_CALL:"):
                            tool_call_line = line
                        else:
                            regular_content.append(line)

                    # Show regular content if any
                    if regular_content:
                        regular_text = "\n".join(regular_content).strip()
                        if regular_text:
                            await self.show_assistant_message(regular_text)
                            responses.append(TextContent(type="text", text=regular_text))
                            final_assistant_content += regular_text

                    # If there is a line starting with "TOOL_CALL:", call the tool
                    if tool_call_line:
                        # Parse tool call
                        tool_call_content = tool_call_line.replace("TOOL_CALL:", "").strip()
                        parts = tool_call_content.split(" ", 1)
                        tool_name = parts[0]
                        tool_args = parts[1] if len(parts) > 1 else "{}"

                        # Execute tool call
                        try:
                            tool_call_request = CallToolRequest(
                                method="tools/call",
                                params=CallToolRequestParams(
                                    name=tool_name,
                                    arguments=from_json(tool_args, allow_partial=True)
                                    if tool_args.strip() != "{}"
                                    else {},
                                ),
                            )

                            result = await self.call_tool(tool_call_request, None)
                            self.show_tool_result(result)
                            responses.extend(result.content)

                            # Update prompt with tool result for next iteration
                            prompt_text += f"\n\nTool result: {str(result)}"

                        except Exception as e:
                            self.logger.error(f"Tool call failed: {e}")
                            await self.show_assistant_message(f"Tool call failed: {e}")
                            final_assistant_content += f"Tool call failed: {e}"
                            break
                else:
                    # No tool calls, regular response
                    responses.append(TextContent(type="text", text=message_content))
                    await self.show_assistant_message(message_content)
                    final_assistant_content = message_content
                    break

            except Exception as e:
                self.logger.error(f"vLLM completion failed: {e}")
                break

        # Add both user and assistant messages to history if using history
        if request_params.use_history:
            # Get current prompt messages to calculate what's new
            prompt_messages = self.history.get(include_completion_history=False)

            # Calculate new conversation messages (excluding existing prompts and system message)
            new_messages = messages[len(prompt_messages) :]

            # Remove system message from new_messages if it was added
            if system_prompt and new_messages and new_messages[0].get("role") == "system":
                new_messages = new_messages[1:]

            # Add the assistant response to complete the conversation
            if final_assistant_content:
                new_messages.append(
                    ChatCompletionAssistantMessageParam(
                        role="assistant", content=final_assistant_content
                    )
                )

            # Extend history with all new messages (both user and assistant)
            self.history.extend(new_messages)

        self._log_chat_finished(model=self.default_request_params.model)
        return responses

    async def _apply_prompt_provider_specific(
        self,
        multipart_messages: List[PromptMessageMultipart],
        request_params: RequestParams | None = None,
        is_template: bool = False,
    ) -> PromptMessageMultipart:
        last_message = multipart_messages[-1]

        # Add all previous messages to history (or all messages if last is from assistant)
        # if the last message is a "user" inference is required
        messages_to_add = (
            multipart_messages[:-1] if last_message.role == "user" else multipart_messages
        )
        converted = []
        for msg in messages_to_add:
            converted.append(OpenAIConverter.convert_to_openai(msg))

        # Add messages to history
        self.history.extend(converted, is_prompt=is_template)

        # For assistant messages: Return the last message (no completion needed)
        if "assistant" == last_message.role:
            return last_message

        message_param = ChatCompletionUserMessageParam(
            role="user", content=get_text(last_message.content[0])
        )
        responses: List[
            TextContent | ImageContent | EmbeddedResource
        ] = await self._vllm_completion(message_param, request_params)
        return Prompt.assistant(*responses)

    async def _apply_prompt_provider_specific_structured(
        self,
        multipart_messages: List[PromptMessageMultipart],
        model: Type[ModelT],
        request_params: RequestParams | None = None,
    ) -> Tuple[ModelT | None, PromptMessageMultipart]:
        """Base class attempts to parse JSON - subclasses can use provider specific functionality"""

        request_params = self.get_request_params(request_params)

        if not request_params.response_format:
            request_params.response_format = model.model_json_schema()

        result: PromptMessageMultipart = await self._apply_prompt_provider_specific(
            multipart_messages, request_params
        )
        return self._structured_from_multipart(result, model)

    def _structured_from_multipart(
        self, message: PromptMessageMultipart, model: Type[ModelT]
    ) -> Tuple[ModelT | None, PromptMessageMultipart]:
        """Parse the content of a PromptMessage and return the structured model and message itself"""
        try:
            text = get_text(message.content[-1]) or ""
            json_data = from_json(text, allow_partial=True)
            validated_model = model.model_validate(json_data)
            return cast("ModelT", validated_model), message
        except ValueError as e:
            logger = get_logger(__name__)
            logger.warning(f"Failed to parse structured response: {str(e)}")
            return None, message

    async def structured(
        self,
        multipart_messages: List[Union[PromptMessageMultipart, PromptMessage]],
        model: Type[ModelT],
        request_params: RequestParams | None = None,
    ) -> Tuple[ModelT | None, PromptMessageMultipart]:
        """Return a structured response from the LLM using the provided messages."""

        # Convert PromptMessage to PromptMessageMultipart if needed
        if multipart_messages and isinstance(multipart_messages[0], PromptMessage):
            multipart_messages = PromptMessageMultipart.to_multipart(multipart_messages)

        self._precall(multipart_messages)

        result, assistant_response = await self._apply_prompt_provider_specific_structured(
            multipart_messages, model, request_params
        )

        self._message_history.append(assistant_response)
        return result, assistant_response


class LocalAgent(BaseAgent):
    def __init__(self, config: AgentConfig, **kwargs) -> None:
        super().__init__(config, **kwargs)
        self._llm = vLLM

    async def attach_llm(
        self,
        llm_factory: Union[Type[AugmentedLLMProtocol], Callable[..., AugmentedLLMProtocol]],
        **kwargs,
    ) -> AugmentedLLMProtocol:
        return await super().attach_llm(
            llm_factory=self._llm,  # override factory with vLLM
            **kwargs,
        )

    async def structured(
        self,
        multipart_messages: List[PromptMessageMultipart],
        model: Type[ModelT],
        request_params: RequestParams | None = None,
    ) -> Tuple[ModelT | None, PromptMessageMultipart]:
        # TODO: implement retry feature if the response is not structured
        return await self._llm.structured(
            multipart_messages=multipart_messages, model=model, request_params=request_params
        )
