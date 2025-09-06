import asyncio
from datetime import datetime
from pathlib import Path

from mcp_agent.core.fastagent import FastAgent
from mcp_agent.core.prompt import Prompt
from pydantic import BaseModel

from oath_keepers.vllm_client import LocalAgent

agents = FastAgent("medical-symptom-clarifier")

BASE_PATH = "oath_keepers"
LOG_PATH = f"{BASE_PATH}/log"
PROMPT_PATH = f"{BASE_PATH}/prompts"


class CandidateResponse(BaseModel):
    Response: str
    Type: str
    Reason: str


def get_filename():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{LOG_PATH}/conversation_{timestamp}.txt"


@agents.custom(
    LocalAgent,
    name="clarifier_agent",
    instruction=Path(f"{PROMPT_PATH}/clarifier_prompt.md").read_text(encoding="utf-8"),
    use_history=True,
)
async def clarifier_assistant():
    async with agents.run() as agent:
        print("=== Medical Symptom Clarification Assistant ===")
        print("This AI will help you organize your symptoms before meeting with your doctor.\n")

        conversation_history = []

        while True:
            patient_input = input("Patient: ").strip()

            if patient_input.lower() in ["quit", "exit", "bye"]:
                print(
                    "\nThank you for using the symptom clarification assistant. Good luck with your appointment!"
                )
                await agent.send(f"***SAVE_HISTORY {get_filename()}")
                break

            # Add patient input to history for context
            conversation_history.append(f"Patient: {patient_input}")

            # Create context-aware prompt
            context = "\n".join(conversation_history[-8:])  # Keep last 6 exchanges
            full_prompt = f"""
            You are a compassionate medical intake assistant AI designed to help patients articulate their symptoms more clearly before meeting with their doctor.
            You are NOT a medical professional and cannot provide diagnoses, medical advice, or treatment recommendations.
            Please provide 3 information below.
            - Response: [Your actual response to the patient]
            - Type: [greeting/questioning/confirming/closing]
            - Reason: [Your reasoning for this response approach and why this type is appropriate now within 70 letters]
            
            Conversation so far:\n{context}\n
            """

            try:
                # Get AI response
                response, messages = await agent.clarifier_agent.structured(
                    multipart_messages=[Prompt.user(full_prompt)], model=CandidateResponse
                )

                (_, response_text), (_, response_type), (_, reason) = response

                # Add AI response to history
                conversation_history.append(f"AI: {response_text}")

                # Check if conversation should end
                if response_type.lower() == "closing":
                    print(
                        "\nConversation completed. Your symptoms have been organized for your doctor visit."
                    )
                    await agent.send(f"***SAVE_HISTORY {get_filename()}")
                    break

            except Exception as e:
                print(f"Error: {e}")
                print("response:", response)
                print("messages:", messages)
                continue


if __name__ == "__main__":
    asyncio.run(clarifier_assistant())
