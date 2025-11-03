import asyncio

import pytest
from mcp_agent.core.prompt import Prompt
from pydantic import BaseModel

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


@agents.custom(
    LocalAgent,
    name="structured_test",
    instruction="You are a helpful assistant that returns structured data..",
)
async def main():
    async with agents.run() as agent:

        class StructuredOutputResponse(BaseModel):
            name: str
            condition: str

        response, _ = await agent.structured_test.structured(
            [Prompt.user("Hi, Jane. How are you? Return structured data")],
            model=StructuredOutputResponse,
        )

        assert isinstance(response, StructuredOutputResponse)
        assert response.name.lower() == "jane"
        assert response.condition


def test_vllm_structured_output():
    asyncio.run(main())


if __name__ == "__main__":
    test_vllm_structured_output()
