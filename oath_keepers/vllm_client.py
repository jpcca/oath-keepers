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
from mcp_agent.llm.providers.multipart_converter_openai import OpenAIConverter, OpenAIMessage
from mcp_agent.llm.usage_tracking import TurnUsage
from mcp_agent.logging.logger import get_logger
from mcp_agent.mcp.helpers.content_helpers import get_text
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart
from openai._client import AsyncAPIClient
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolParam,
)
from pydantic_core import from_json
from rich.text import Text

from oath_keepers.utils.typing import GenerateRequest, GuidedDecodingParams, SamplingParams


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
        """Initialize Anthropic-specific default parameters"""
        request_params = super()._initialize_default_params(kwargs)
        request_params.sampling_params = SamplingParams(max_tokens=128)
        request_params.model = "vLLM"

        return request_params

    async def _vllm_completion(
        self,
        message: OpenAIMessage,
        request_params: RequestParams | None = None,
    ) -> List[TextContent | ImageContent | EmbeddedResource]:
        """
        Process a query using an LLM provided by the vLLM library and available tools.
        """

        request_params = self.get_request_params(request_params=request_params)
        responses: List[TextContent | ImageContent | EmbeddedResource] = []

        response_format = request_params.response_format
        if response_format is not None:
            request_params.sampling_params = SamplingParams(
                max_tokens=128,
                guided_decoding=GuidedDecodingParams.from_optional(json=response_format),
            )

        # TODO -- move this in to agent context management / agent group handling
        messages: List[ChatCompletionMessageParam] = []
        system_prompt = self.instruction or request_params.systemPrompt
        if system_prompt:
            messages.append(ChatCompletionSystemMessageParam(role="system", content=system_prompt))

        messages.extend(self.history.get(include_completion_history=request_params.use_history))
        messages.append(message)

        response = await self.aggregator.list_tools()
        available_tools: List[ChatCompletionToolParam] | None = [
            ChatCompletionToolParam(
                type="function",
                function={
                    "name": tool.name,
                    "description": tool.description if tool.description else "",
                    "parameters": self.adjust_schema(tool.inputSchema),
                },
            )
            for tool in response.tools
        ]

        # we do NOT send "stop sequences" as this causes errors with mutlimodal processing
        for i in range(request_params.max_iterations):
            self._log_chat_progress(self.chat_turn(), model=self.default_request_params.model)

            response = await self._vllm_client().post(
                f"{self.base_url}/generate",
                cast_to=ChatCompletion,
                body=GenerateRequest(
                    prompts=message["content"],
                    sampling_params=request_params.sampling_params,
                ).model_dump(),
            )

            # Track usage if response is valid and has usage data
            if (
                hasattr(response, "usage")
                and response.usage
                and not isinstance(response, BaseException)
            ):
                try:
                    model_name = self.default_request_params.model
                    turn_usage = TurnUsage.from_openai(response.usage, model_name)
                    self._finalize_turn_usage(turn_usage)
                except Exception as e:
                    self.logger.warning(f"Failed to track usage: {e}")

            self.logger.debug(
                "OpenAI completion response:",
                data=response,
            )

            if isinstance(response, BaseException):
                self.logger.error(f"Error: {response}")
                break

            if not response.choices or len(response.choices) == 0:
                # No response from the model, we're done
                break

            choice = response.choices[0]
            message = choice.message
            # prep for image/audio gen models
            if message.content:
                responses.append(TextContent(type="text", text=message.content))

            # ParsedChatCompletionMessage is compatible with ChatCompletionMessage
            # since it inherits from it, so we can use it directly
            messages.append(message)

            message_text = message.content
            if choice.finish_reason in ["tool_calls", "function_call"] and message.tool_calls:
                if message_text:
                    await self.show_assistant_message(
                        message_text,
                        message.tool_calls[
                            0
                        ].function.name,  # TODO support displaying multiple tool calls
                    )
                else:
                    await self.show_assistant_message(
                        Text(
                            "the assistant requested tool calls",
                            style="dim green italic",
                        ),
                        message.tool_calls[0].function.name,
                    )

                tool_results = []
                for tool_call in message.tool_calls:
                    self.show_tool_call(
                        available_tools,
                        tool_call.function.name,
                        tool_call.function.arguments,
                    )
                    tool_call_request = CallToolRequest(
                        method="tools/call",
                        params=CallToolRequestParams(
                            name=tool_call.function.name,
                            arguments={}
                            if not tool_call.function.arguments
                            or tool_call.function.arguments.strip() == ""
                            else from_json(tool_call.function.arguments, allow_partial=True),
                        ),
                    )
                    result = await self.call_tool(tool_call_request, tool_call.id)
                    self.show_oai_tool_result(str(result))

                    tool_results.append((tool_call.id, result))
                    responses.extend(result.content)
                messages.extend(OpenAIConverter.convert_function_results_to_openai(tool_results))

                self.logger.debug(
                    f"Iteration {i}: Tool call results: {str(tool_results) if tool_results else 'None'}"
                )
            elif choice.finish_reason == "length":
                # We have reached the max tokens limit
                self.logger.debug(f"Iteration {i}: Stopping because finish_reason is 'length'")
                if request_params and request_params.maxTokens is not None:
                    message_text = Text(
                        f"the assistant has reached the maximum token limit ({request_params.maxTokens})",
                        style="dim green italic",
                    )
                else:
                    message_text = Text(
                        "the assistant has reached the maximum token limit",
                        style="dim green italic",
                    )

                await self.show_assistant_message(message_text)
                break
            elif choice.finish_reason == "content_filter":
                # The response was filtered by the content filter
                self.logger.debug(
                    f"Iteration {i}: Stopping because finish_reason is 'content_filter'"
                )
                break
            elif choice.finish_reason == "stop":
                self.logger.debug(f"Iteration {i}: Stopping because finish_reason is 'stop'")
                if message_text:
                    await self.show_assistant_message(message_text, "")
                break

        if request_params.use_history:
            # Get current prompt messages
            prompt_messages = self.history.get(include_completion_history=False)

            # Calculate new conversation messages (excluding prompts)
            new_messages = messages[len(prompt_messages) :]

            if system_prompt:
                new_messages = new_messages[1:]

            self.history.set(new_messages)

        self._log_chat_finished(model=self.default_request_params.model)

        return responses

    async def _apply_prompt_provider_specific(
        self,
        multipart_messages: List["PromptMessageMultipart"],
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

        # TODO -- this looks like a defect from previous apply_prompt implementation.
        self.history.extend(converted, is_prompt=is_template)

        if "assistant" == last_message.role:
            return last_message

        # For assistant messages: Return the last message (no completion needed)
        message_param: OpenAIMessage = OpenAIConverter.convert_to_openai(last_message)
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
