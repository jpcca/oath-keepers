import asyncio
from datetime import datetime
from pathlib import Path

from mcp_agent.core.fastagent import FastAgent
from mcp_agent.core.prompt import Prompt

from oath_keepers.utils.typing import CandidateResponse, ResponseType
from oath_keepers.vllm_client import LocalAgent

agents = FastAgent("medical-symptom-clarifier")
base_path = Path(__file__).parent.parent
log_path = f"{base_path}/log"
prompt_path = f"{base_path}/prompts"


def get_filepath() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(f"{log_path}/conversation_{timestamp}.txt")


@agents.custom(
    LocalAgent,
    name="clarifier_agent",
    instruction=Path(f"{prompt_path}/clarifier_prompt.md").read_text(encoding="utf-8"),
    use_history=True,
)
async def clarifier_assistant() -> Path | None:
    conversation_log_path = get_filepath()

    async with agents.run() as agent:
        print("=== Medical Symptom Clarification Assistant ===")
        print(
            "This AI will help you organize your symptoms before making an appointment with your doctor.\n"
        )

        turn_count = 0
        while True:
            if turn_count == 0:
                patient_input = "Call received. Please greet the patient on the call."
            else:
                patient_input = input("Patient: ").strip()
            if patient_input.lower() in ["quit", "exit", "bye"]:
                break

            try:
                # Get AI response
                result, messages = await agent.clarifier_agent.structured(
                    multipart_messages=[Prompt.user(patient_input)], model=CandidateResponse
                )
                if result is None:
                    raise ValueError("Failed to parse structured response")

                print(f"Assistant: {result.response}")
                turn_count += 1

                # Check if conversation should end
                if result.response_type is ResponseType.closing:
                    break

            except Exception as e:
                print(f"Error: {e}")
                print("result:", locals().get("result"))
                print("messages:", locals().get("messages"))
                continue

        # Closing
        print(f"\nConversation completed at turn {turn_count}.")
        await agent.send(f"***SAVE_HISTORY {conversation_log_path}")
        return conversation_log_path


if __name__ == "__main__":
    asyncio.run(clarifier_assistant())
