import pytest
from mcp_agent.core.prompt import Prompt

from oath_keepers.agents.age_estimator_agent import agents
from oath_keepers.models.age_models import AgeDistribution


@pytest.mark.asyncio
async def test_age_estimation_flow():
    """
    Test the age estimation agent with the example conversation turns.
    """
    turns = [
        "I grew up playing on a Nintendo Wii with my siblings.",
        "I’ve already switched careers once after a few years in my first job.",
        "I still remember burning CDs for my car when I first started driving.",
    ]

    async with agents.run() as agent:
        for i, turn in enumerate(turns):
            print(f"\n--- Turn {i + 1} ---")
            print(f"User: {turn}")

            result, full_response = await agent.age_estimator.structured(
                multipart_messages=[Prompt.user(turn)], model=AgeDistribution
            )

            if result is None:
                print(f"Raw response: {full_response.last_text()}")

            assert result is not None, "Agent failed to return a structured response"
            assert isinstance(result, AgeDistribution), "Result is not of type AgeDistribution"
            assert result.reasoning, "Reasoning field is empty"
            assert len(result.reasoning) <= 140, (
                f"Reasoning too long: {len(result.reasoning)} chars"
            )
            print(f"Reasoning: {result.reasoning}")

            # Validate sum of probabilities
            total_prob = sum(bin.p for bin in result.bins)
            assert abs(total_prob - 1.0) < 1e-4, f"Total probability {total_prob} is not 1.0"

            # Print distribution for manual inspection
            print("Age Distribution:")
            for bin in sorted(result.bins, key=lambda x: x.bin_start):
                if bin.p > 0.01:  # Only print significant probabilities
                    print(f"  [{bin.bin_start},{bin.bin_end}): {bin.p:.4f}")

            # Basic sanity checks based on the turns
            # Turn 1: Wii -> likely 20s-30s (born 90s-00s)
            # Turn 2: Switched careers -> likely mid-late 20s or 30s
            # Turn 3: Burning CDs -> likely 30s-40s (born 80s-90s)

            # We can add some loose assertions if we want, but the main goal is to ensure
            # the mechanism works and output is valid.
