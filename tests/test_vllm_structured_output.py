import asyncio

from mcp_agent.core.fastagent import FastAgent
from mcp_agent.core.prompt import Prompt
from pydantic import BaseModel

from oath_keepers.vllm_client import LocalAgent

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
