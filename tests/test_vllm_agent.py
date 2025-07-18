import asyncio
from typing import Callable, Type, Union

from mcp_agent.agents.base_agent import BaseAgent
from mcp_agent.core.fastagent import FastAgent
from mcp_agent.llm.augmented_llm import (
    AugmentedLLMProtocol,
)

from oath_keepers.vllm_client import vLLM


class LocalAgent(BaseAgent):
    async def attach_llm(
        self,
        llm_factory: Union[Type[AugmentedLLMProtocol], Callable[..., AugmentedLLMProtocol]],
        **kwargs,
    ) -> AugmentedLLMProtocol:
        return await super().attach_llm(
            llm_factory=vLLM,  # override factory with vLLM
            **kwargs,
        )


agents = FastAgent("fast-agent example")


@agents.custom(LocalAgent, instruction="You are a helpful assistant.")
async def main():
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    async with agents.run() as agent:
        await agent("Hi! How are you?")


def test_vllm_agent():
    asyncio.run(main())


if __name__ == "__main__":
    test_vllm_agent()
