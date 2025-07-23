import asyncio

from mcp_agent.core.fastagent import FastAgent

from oath_keepers.vllm_client import LocalAgent

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
