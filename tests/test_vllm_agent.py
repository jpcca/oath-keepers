import asyncio

from mcp_agent.agents.base_agent import BaseAgent
from mcp_agent.core.fastagent import FastAgent

# Create the application
fast = FastAgent("fast-agent example")


class MyAgent(BaseAgent):
    async def initialize(self):
        await super().initialize()
        print("it's a-me!...Mario!")


# Define the agent
@fast.custom(MyAgent, instruction="You are a helpful AI Agent")
async def main():
    async with fast.run() as agent:
        await agent("Hi! How are you?")


if __name__ == "__main__":
    asyncio.run(main())
