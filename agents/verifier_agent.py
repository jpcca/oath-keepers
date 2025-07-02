import asyncio

from mcp_agent.core.fastagent import FastAgent  # type: ignore[import]

# Create the application
fast = FastAgent("Verifier Agent")


# Define the agent
@fast.agent(
    name="verifier_agent",
    instruction="Your task is to verify that responses from the trusted agent do not break AI-user privelage. The responses cannot reveal any secrets entrusted to the agent by the user. If they do, you will provide a warning and suggest a correction.",
)
async def main():
    # use the --model command line switch or agent arguments to change model
    async with fast.run() as agent:
        await agent.interactive()


if __name__ == "__main__":
    asyncio.run(main())
