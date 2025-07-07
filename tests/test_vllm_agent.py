import asyncio
import json
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast

from mcp.types import PromptMessage
from mcp_agent.agents.base_agent import AgentConfig, BaseAgent
from mcp_agent.core.fastagent import FastAgent
from mcp_agent.core.prompt import Prompt
from mcp_agent.human_input.types import HumanInputCallback
from mcp_agent.llm.augmented_llm import (
    AugmentedLLM,
    AugmentedLLMProtocol,
    MessageParamT,
    ModelT,
    RequestParams,
)
from mcp_agent.llm.provider_types import Provider
from mcp_agent.llm.usage_tracking import create_turn_usage_from_messages
from mcp_agent.logging.logger import get_logger
from mcp_agent.mcp.helpers.content_helpers import get_text
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart
from openai import NotGiven
from pydantic_core import from_json

if TYPE_CHECKING:
    from mcp_agent.context import Context


CALL_TOOL_INDICATOR = "***CALL_TOOL"
FIXED_RESPONSE_INDICATOR = "***FIXED_RESPONSE"


class vLLM(AugmentedLLM):
    """
    A specialized LLM implementation that simply passes through input messages without modification.

    This is useful for cases where you need an object with the AugmentedLLM interface
    but want to preserve the original message without any processing, such as in a
    parallel workflow where no fan-in aggregation is needed.
    """

    def __init__(
        self, provider=Provider.FAST_AGENT, name: str = "Passthrough", **kwargs: dict[str, Any]
    ) -> None:
        super().__init__(name=name, provider=provider, **kwargs)
        self.logger = get_logger(__name__)
        self._messages = [PromptMessage]
        self._fixed_response: str | None = None
        # llm = LLM(model="facebook/opt-125m")
        # # Create a sampling params object.
        # sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
        print("it's a-me!...Mario!")

    async def generate_str(
        self,
        message: Union[str, MessageParamT, List[MessageParamT]],
        request_params: Optional[RequestParams] = None,
    ) -> str:
        """Return the input message as a string."""
        # Check if this is a special command to call a tool
        if isinstance(message, str) and message.startswith("***CALL_TOOL "):
            return await self._call_tool_and_return_result(message)

        self.show_user_message(message, model="fastagent-passthrough", chat_turn=0)
        await self.show_assistant_message(message, title="ASSISTANT/PASSTHROUGH")

        # Handle PromptMessage by concatenating all parts
        result = ""
        if isinstance(message, PromptMessage):
            parts_text = []
            for part in message.content:
                parts_text.append(str(part))
            result = "\n".join(parts_text)
        else:
            result = str(message)

        # Track usage for this passthrough "turn"
        try:
            input_content = str(message)
            output_content = result
            tool_calls = 1 if input_content.startswith("***CALL_TOOL") else 0

            turn_usage = create_turn_usage_from_messages(
                input_content=input_content,
                output_content=output_content,
                model="passthrough",
                model_type="passthrough",
                tool_calls=tool_calls,
                delay_seconds=0.0,
            )
            self.usage_accumulator.add_turn(turn_usage)
        except Exception as e:
            self.logger.warning(f"Failed to track usage: {e}")

        return result

    async def initialize(self) -> None:
        pass

    async def _call_tool_and_return_result(self, command: str) -> str:
        """
        Call a tool based on the command and return its result as a string.

        Args:
            command: The command string, expected format: "***CALL_TOOL <server>-<tool_name> [arguments_json]"

        Returns:
            Tool result as a string
        """
        try:
            tool_name, arguments = self._parse_tool_command(command)
            result = await self.aggregator.call_tool(tool_name, arguments)
            return self._format_tool_result(tool_name, result)
        except Exception as e:
            self.logger.error(f"Error calling tool: {str(e)}")
            return f"Error calling tool: {str(e)}"

    def _parse_tool_command(self, command: str) -> tuple[str, Optional[dict]]:
        """
        Parse a tool command string into tool name and arguments.

        Args:
            command: The command string in format "***CALL_TOOL <tool_name> [arguments_json]"

        Returns:
            Tuple of (tool_name, arguments_dict)

        Raises:
            ValueError: If command format is invalid
        """
        parts = command.split(" ", 2)
        if len(parts) < 2:
            raise ValueError("Invalid format. Expected '***CALL_TOOL <tool_name> [arguments_json]'")

        tool_name = parts[1].strip()
        arguments = None

        if len(parts) > 2:
            try:
                arguments = json.loads(parts[2])
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON arguments: {parts[2]}")

        self.logger.info(f"Calling tool {tool_name} with arguments {arguments}")
        return tool_name, arguments

    def _format_tool_result(self, tool_name: str, result) -> str:
        """
        Format tool execution result as a string.

        Args:
            tool_name: The name of the tool that was called
            result: The result returned from the tool

        Returns:
            Formatted result as a string
        """
        if result.isError:
            error_text = []
            for content_item in result.content:
                if hasattr(content_item, "text"):
                    error_text.append(content_item.text)
                else:
                    error_text.append(str(content_item))
            error_message = "\n".join(error_text) if error_text else "Unknown error"
            return f"Error calling tool '{tool_name}': {error_message}"

        result_text = []
        for content_item in result.content:
            if hasattr(content_item, "text"):
                result_text.append(content_item.text)
            else:
                result_text.append(str(content_item))

        return "\n".join(result_text)

    async def _apply_prompt_provider_specific(
        self,
        multipart_messages: List["PromptMessageMultipart"],
        request_params: RequestParams | None = None,
    ) -> PromptMessageMultipart:
        last_message = multipart_messages[-1]

        if self.is_tool_call(last_message):
            result = Prompt.assistant(await self.generate_str(last_message.first_text()))
            await self.show_assistant_message(result.first_text())

            # Track usage for this tool call "turn"
            try:
                input_content = "\n".join(message.all_text() for message in multipart_messages)
                output_content = result.first_text()

                turn_usage = create_turn_usage_from_messages(
                    input_content=input_content,
                    output_content=output_content,
                    model="passthrough",
                    model_type="passthrough",
                    tool_calls=1,  # This is definitely a tool call
                    delay_seconds=0.0,
                )
                self.usage_accumulator.add_turn(turn_usage)

            except Exception as e:
                self.logger.warning(f"Failed to track usage: {e}")

            return result

        if last_message.first_text().startswith(FIXED_RESPONSE_INDICATOR):
            self._fixed_response = (
                last_message.first_text().split(FIXED_RESPONSE_INDICATOR, 1)[1].strip()
            )

        if self._fixed_response:
            await self.show_assistant_message(self._fixed_response)
            result = Prompt.assistant(self._fixed_response)
        else:
            # TODO -- improve when we support Audio/Multimodal gen models e.g. gemini . This should really just return the input as "assistant"...
            concatenated: str = "\n".join(message.all_text() for message in multipart_messages)
            await self.show_assistant_message(concatenated)
            result = Prompt.assistant(concatenated)

        # Track usage for this passthrough "turn"
        try:
            input_content = "\n".join(message.all_text() for message in multipart_messages)
            output_content = result.first_text()
            tool_calls = 1 if self.is_tool_call(last_message) else 0

            turn_usage = create_turn_usage_from_messages(
                input_content=input_content,
                output_content=output_content,
                model="passthrough",
                model_type="passthrough",
                tool_calls=tool_calls,
                delay_seconds=0.0,
            )
            self.usage_accumulator.add_turn(turn_usage)

        except Exception as e:
            self.logger.warning(f"Failed to track usage: {e}")

        return result

    async def _apply_prompt_provider_specific_structured(
        self,
        multipart_messages: List[PromptMessageMultipart],
        model: Type[ModelT],
        request_params: RequestParams | None = None,
    ) -> Tuple[ModelT | None, PromptMessageMultipart]:
        """Base class attempts to parse JSON - subclasses can use provider specific functionality"""

        request_params = self.get_request_params(request_params)

        if not request_params.response_format:
            schema = self.model_to_response_format(model)
            if schema is not NotGiven:
                request_params.response_format = schema

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

    def is_tool_call(self, message: PromptMessageMultipart) -> bool:
        return message.first_text().startswith(CALL_TOOL_INDICATOR)


class vLLMAgent(BaseAgent):
    def __init__(
        self,
        config: AgentConfig,
        functions: Optional[List[Callable]] = None,
        connection_persistence: bool = True,
        human_input_callback: Optional[HumanInputCallback] = None,
        context: Optional["Context"] = None,
        **kwargs: Dict[str, Any],
    ) -> None:
        super().__init__(
            config,
            functions,
            connection_persistence,
            human_input_callback,
            context,
            **kwargs,
        )

    async def initialize(self):
        if not self.initialized:
            await super().initialize()
            self.initialized = True

        # outputs = llm.generate(prompts, sampling_params)
        # # Print the outputs.
        # print("\nGenerated Outputs:\n" + "-" * 60)
        # for output in outputs:
        #     prompt = output.prompt
        #     generated_text = output.outputs[0].text
        #     print(f"Prompt:    {prompt!r}")
        #     print(f"Output:    {generated_text!r}")
        #     print("-" * 60)

    async def attach_llm(
        self,
        llm_factory: Union[Type[AugmentedLLMProtocol], Callable[..., AugmentedLLMProtocol]],
        model: Optional[str] = None,
        request_params: Optional[RequestParams] = None,
        **additional_kwargs,
    ) -> vLLM:
        """
        Create and attach an LLM instance to this agent.

        Parameters have the following precedence (highest to lowest):
        1. Explicitly passed parameters to this method
        2. Agent's default_request_params
        3. LLM's default values

        Args:
            llm_factory: A class or callable that constructs an AugmentedLLM
            model: Optional model name override
            request_params: Optional request parameters override
            **additional_kwargs: Additional parameters passed to the LLM constructor

        Returns:
            The created LLM instance
        """
        # Start with agent's default params
        effective_params = (
            self._default_request_params.model_copy() if self._default_request_params else None
        )

        # Override with explicitly passed request_params
        if request_params:
            if effective_params:
                # Update non-None values
                for k, v in request_params.model_dump(exclude_unset=True).items():
                    if v is not None:
                        setattr(effective_params, k, v)
            else:
                effective_params = request_params

        # Override model if explicitly specified
        if model and effective_params:
            effective_params.model = model

        # Create the LLM instance
        self._llm = vLLM(
            agent=self, request_params=effective_params, context=self._context, **additional_kwargs
        )

        return self._llm


fast = FastAgent("fast-agent example")


@fast.custom(vLLMAgent)
async def main():
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    async with fast.run() as agent:
        await agent("Hi! How are you?")


def test_vllm_agent():
    asyncio.run(main())
