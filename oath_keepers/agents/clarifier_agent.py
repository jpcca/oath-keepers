import asyncio
from datetime import datetime
from enum import Enum
from pathlib import Path

from mcp_agent.core.fastagent import FastAgent
from mcp_agent.core.prompt import Prompt
from pydantic import BaseModel

from oath_keepers.vllm_client import LocalAgent

agents = FastAgent("medical-symptom-clarifier")
base_path = Path(__file__).parent.parent
log_path = f"{base_path}/log"
prompt_path = f"{base_path}/prompts"


class ResponseType(str, Enum):
    greeting = "greeting"
    questioning = "questioning"
    confirming = "confirming"
    closing = "closing"


class CandidateResponse(BaseModel):
    response: str
    response_type: ResponseType
    reason: str


def get_filename():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{log_path}/conversation_{timestamp}.txt"


@agents.custom(
    LocalAgent,
    name="clarifier_agent",
    instruction=Path(f"{prompt_path}/clarifier_prompt.md").read_text(encoding="utf-8"),
    use_history=True,
)

# To run with MedGemma or other model via Ollama, comment out `@agents.custom` above and uncomment here
# @agents.agent(
#     name="clarifier_agent",
#     instruction=Path(f"{prompt_path}/clarifier_prompt.md").read_text(encoding="utf-8"),
#     model="generic.alibayram/medgemma:27b",  # or other models e.g. "generic.gemma3:12b"
#     use_history=True,
# )

async def clarifier_assistant():
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
                print(
                    "\nThank you for using the symptom clarification assistant. Good luck with your appointment!"
                )
                await agent.send(f"***SAVE_HISTORY {get_filename()}")
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
                    print(f"\nConversation completed at turn {turn_count}.")
                    await agent.send(f"***SAVE_HISTORY {get_filename()}")
                    break

            except Exception as e:
                print(f"Error: {e}")
                print("result:", locals().get("result"))
                print("messages:", locals().get("messages"))
                continue


if __name__ == "__main__":
    asyncio.run(clarifier_assistant())
