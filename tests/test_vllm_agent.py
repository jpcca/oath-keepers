import asyncio

import pytest

try:  # Skip cleanly if fast-agent-mcp is not installed
    from mcp_agent.core.fastagent import FastAgent  # type: ignore
except Exception:  # pragma: no cover - unit env without heavy deps
    pytest.skip(
        "fast-agent-mcp not installed; skipping vLLM integration test", allow_module_level=True
    )

try:
    from oath_keepers.vllm_client import LocalAgent
except Exception:  # pragma: no cover - unit env without vLLM client deps
    pytest.skip("LocalAgent unavailable; skipping vLLM integration test", allow_module_level=True)


agents = FastAgent("fast-agent example")


@agents.custom(LocalAgent, instruction="You are a helpful assistant.")
async def main():
    async with agents.run() as agent:
        await agent("Hi! How are you?")


def test_vllm_agent():
    asyncio.run(main())


if __name__ == "__main__":
    test_vllm_agent()
