import asyncio
import sys
from pathlib import Path
from typing import List
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mcp_agent.core.fastagent import FastAgent
from mcp_agent.core.prompt import Prompt
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart

from oath_keepers.models.age_models import AgeDistribution, AgeBin
from oath_keepers.vllm_client import LocalAgent

# Initialize FastAgent
agents = FastAgent("age-estimation-agent")
base_path = Path(__file__).parent.parent
prompt_path = f"{base_path}/prompts"

@agents.custom(
    LocalAgent,
    name="age_estimator",
    instruction=Path(f"{prompt_path}/age_estimator_prompt.md").read_text(encoding="utf-8"),
    use_history=True, # Maintain history for multi-turn estimation
)
async def estimate_age(messages: List[str]) -> AgeDistribution:
    """
    Estimates age distribution based on a list of messages.
    This function simulates a multi-turn conversation by sending messages one by one
    or as a batch, but for the agent interaction we usually want to maintain state.
    
    However, the requirement says "Input: multi-turn conversation".
    If we want to process a conversation so far, we can send the whole transcript.
    Or if we want to simulate the "after each turn" behavior, we can call this iteratively.
    
    For this implementation, let's assume we are called with the latest message 
    and the agent maintains history internally via `use_history=True`.
    """
    pass # The agent logic is handled by the decorator and the run loop below

def visualize_distribution(distribution: AgeDistribution, filename: str = "age_distribution.png"):
    """
    Generates a bar chart from the age distribution and saves it to a file.
    """
    bins = distribution.bins
    # Sort bins just in case
    bins = sorted(bins, key=lambda x: x.bin_start)
    
    labels = [f"{b.bin_start}-{b.bin_end}" for b in bins]
    probs = [b.p for b in bins]
    
    plt.figure(figsize=(10, 6))
    plt.bar(labels, probs, color='skyblue')
    plt.xlabel('Age Bins')
    plt.ylabel('Probability')
    plt.title('Age Probability Distribution')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Visualization saved to {filename}")

async def run_age_estimation_loop():
    """
    Interactive loop to test the age estimation agent.
    """
    print("Starting Age Estimation Agent. Type 'exit' to quit.")
    
    # Initial distribution (optional, or let the agent generate it on first turn)
    # For now, we just start the loop.
    
    async with agents.run() as agent:
        while True:
            user_input = input("User: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            
            # We use structured() to enforce the output format
            # We send the user input as a prompt
            try:
                result, full_response = await agent.age_estimator.structured(
                    multipart_messages=[Prompt.user(user_input)],
                    model=AgeDistribution
                )
                
                if result:
                    print("AI: Age Distribution Updated")
                    # Print a simplified view or the full JSON
                    # For visualization, let's print the top bins or just the raw JSON
                    print(result.model_dump_json(indent=2))
                    visualize_distribution(result)
                else:
                    print("AI: Failed to generate valid distribution.")
                    if full_response:
                        print("Raw response:", full_response.last_text())
                    else:
                        print("No response received.")
            except Exception as e:
                print(f"Error during interaction: {e}")

if __name__ == "__main__":
    asyncio.run(run_age_estimation_loop())
