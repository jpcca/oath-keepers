import asyncio
from datetime import datetime
from pathlib import Path

from mcp_agent.core.fastagent import FastAgent
from mcp_agent.core.prompt import Prompt

from oath_keepers.utils.typing import CandidateResponse, ResponseType
from oath_keepers.vllm_client import LocalAgent
from typing import Callable, Awaitable

agents = FastAgent("medical-symptom-clarifier", quiet=True)
base_path = Path(__file__).parent.parent
prompt_path = f"{base_path}/prompts"


@agents.custom(
    LocalAgent,
    name="clarifier_agent",
    instruction=Path(f"{prompt_path}/clarifier_prompt.md").read_text(encoding="utf-8"),
    use_history=True,
)
async def clarifier_assistant(
    conversation_log_path: Path,
    extractor_assistant: Callable[[Path], Awaitable[Path]]
) -> Path | None:
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

                # write conversation history and summarize result to file each turn
                # may want to run extractor asynchronouslynothi
                asyncio.create_task(agent.send(f"***SAVE_HISTORY {conversation_log_path}"))
                asyncio.create_task(extractor_assistant(conversation_log_path))

                # Check if conversation should end
                if result.response_type is ResponseType.cohmlosing:
                    break

            except Exception as e:
                print(f"Error: {e}")
                print("result:", locals().get("result"))
                print("messages:", locals().get("messages"))
                continue

        # Closing
        print(f"\nConversation completed at turn {turn_count}.")

if __name__ == "__main__":
    asyncio.run(clarifier_assistant())
