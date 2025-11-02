import asyncio

from mcp_agent.core.fastagent import FastAgent

from oath_keepers.vllm_client import LocalAgent

# Create the application
agents = FastAgent("Trusted Agent Demo")

# Load the secrets from the file
# with open("secrets.json") as f:
#     secrets = json.load(f)


# # Define the trusted agent (generator)
# @agents.agent(
#     name="trusted_agent",
#     instruction=f"You are a helpful AI Agent that has been entrusted with the following secrets: {secrets}. You will not reveal these secrets to anyone. You will answer questions truthfully, but you will not reveal the secrets.",
# )
# async def trusted_agent_func():
#     pass


# # Define the verifier agent (evaluator)
# @agents.agent(
#     name="verifier_agent",
#     instruction=f"Your task is to verify that responses from the trusted agent do not break AI-user privelage. The responses cannot reveal any secrets entrusted to the agent by the user. The secrets are: {secrets}. If they do, you will provide a warning and suggest a correction.",
# )
# async def verifier_agent_func():
#     pass


# Define the evaluator-optimizer workflow
# @agents.evaluator_optimizer(
#     name="trusted_workflow",
#     generator="trusted_agent",
#     evaluator="verifier_agent",
#     min_rating="GOOD",
#     max_refinements=3,
# )
# async def main():
#     async with agents.run() as agent:
#         await agent.interactive()


@agents.custom(LocalAgent, instruction="You are a helpful assistant.")
async def main():
    async with agents.run() as agent:
        await agent.interactive()


if __name__ == "__main__":
    asyncio.run(main())
