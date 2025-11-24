import asyncio
from pathlib import Path
from typing import List

import matplotlib
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mcp_agent.core.fastagent import FastAgent
from mcp_agent.core.prompt import Prompt

from oath_keepers.models.age_models import AgeDistribution
from oath_keepers.vllm_client import LocalAgent

# Initialize FastAgent
agents = FastAgent("age-estimator-agent")
base_path = Path(__file__).parent.parent
output_dir = base_path / "outputs" / "age_estimator"
output_dir.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists
prompt_dir = base_path / "prompts"


def visualize_distribution(distribution: AgeDistribution, turn: int, user_input: str = "", filename: str = None):
    """
    Generates a bar chart from the age distribution and saves it to a file.
    """
    if filename is None:
        filename = output_dir / f"age_distribution_turn_{turn}.png"

    bins = distribution.bins
    # Sort bins just in case
    bins = sorted(bins, key=lambda x: x.bin_start)

    labels = [f"{b.bin_start}-{b.bin_end}" for b in bins]
    probs = [b.p for b in bins]

    plt.figure(figsize=(10, 6))
    plt.bar(labels, probs, color="skyblue")
    plt.xlabel("Age Bins")
    plt.ylabel("Probability")
    plt.title(f"Age Probability Distribution (Turn {turn})\nUser: {user_input}")
    plt.xticks(rotation=45)
    plt.ylim(0, 1.0)  # Fix y-axis to 0-1 for consistency
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Visualization saved to {filename}")
    return filename


def create_gif(
    image_files: List[str], output_file: str = str(output_dir / "age_estimation_progress.gif")
):
    """
    Creates a GIF from a list of image files.
    """
    if not image_files:
        print("No images to create GIF.")
        return

    images = []
    for filename in image_files:
        try:
            images.append(Image.open(filename))
        except IOError:
            print(f"Could not open {filename}")

    if images:
        images[0].save(
            output_file,
            save_all=True,
            append_images=images[1:],
            duration=1000,  # 1 second per frame
            loop=0,
        )
        print(f"GIF saved to {output_file}")


@agents.custom(
    LocalAgent,
    name="age_estimator",
    instruction=Path(f"{prompt_dir}/age_estimator_prompt.md").read_text(encoding="utf-8"),
    use_history=True,  # Maintain history for multi-turn estimation
)
async def run_age_estimation_loop():
    """
    Interactive loop to test the age estimation agent.
    """
    print("Starting Age Estimator Agent. Type 'exit' to quit.")

    turn = 0
    image_files = []

    async with agents.run() as agent:
        while True:
            user_input = input("User: ")
            if user_input.lower() in ["exit", "quit"]:
                break

            turn += 1

            # We use structured() to enforce the output format
            # We send the user input as a prompt
            try:
                result, full_response = await agent.age_estimator.structured(
                    multipart_messages=[Prompt.user(user_input)], model=AgeDistribution
                )

                if result:
                    print("AI: Age Distribution Updated")
                    print(f"Reasoning: {result.reasoning}")
                    # Print a simplified view or the full JSON
                    # For visualization, let's print the top bins or just the raw JSON
                    print(result.model_dump_json(indent=2, exclude={"reasoning"}))

                    img_file = visualize_distribution(result, turn, user_input)
                    image_files.append(img_file)
                else:
                    print("AI: Failed to generate valid distribution.")
                    if full_response:
                        print("Raw response:", full_response.last_text())
                    else:
                        print("No response received.")
            except Exception as e:
                print(f"Error during interaction: {e}")

    # Create GIF on exit
    if image_files:
        print("Creating progress GIF...")
        create_gif(image_files, output_file=str(output_dir / "age_estimation_progress.gif"))


if __name__ == "__main__":
    asyncio.run(run_age_estimation_loop())
