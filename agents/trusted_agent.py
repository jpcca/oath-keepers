import asyncio

from mcp_agent.core.fastagent import FastAgent  # type: ignore[import]

# Create the application
fast = FastAgent("Trusted Agent")


# Define the agent
@fast.agent(
    name="trusted_agent",
    instruction="You are a helpful AI Agent that respects AI-user privelage and will not reveal trusted information to third parties.",
    servers=["verifier_agent"],
)
async def main():
    # use the --model command line switch or agent arguments to change model
    async with fast.run() as agent:
        await agent.interactive()


if __name__ == "__main__":
    asyncio.run(main())
