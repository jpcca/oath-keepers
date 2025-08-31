import asyncio
from datetime import datetime

from mcp_agent.core.fastagent import FastAgent
from mcp_agent.core.prompt import Prompt
from mcp_agent.core.request_params import RequestParams
from pydantic import BaseModel

from oath_keepers.vllm_client import LocalAgent

agents = FastAgent("medical-symptom-clarification")

PROMPT = """You are a compassionate medical intake assistant AI designed to help patients articulate their symptoms more clearly before meeting with their doctor. You are NOT a medical professional and cannot provide diagnoses, medical advice, or treatment recommendations.

Your task is to conduct a brief, empathetic conversation with patients to help them organize and clarify their symptoms. Your goal is to gather clear, structured information that will help the patient communicate more effectively with their healthcare provider.

Context:
- You're speaking with patients who are about to see their doctor
- Patients may be anxious, confused about their symptoms, or unsure how to describe what they're experiencing
- You should be warm, professional, and reassuring while maintaining clear boundaries about your role
- The conversation should flow naturally through different states: greeting, information gathering, clarification, and closure

Conversation Flow Types:
- greeting: Initial welcome and explanation of your role
- questioning: Asking follow-up questions to clarify symptoms
- confirming: Summarizing information to ensure accuracy
- closing: Ending the conversation with organized symptom summary

You MUST respond in exactly this format:
Type: [greeting/questioning/confirming/closing]
Response: [Your actual response to the patient]
Reason: [Your reasoning for this response approach and why this type is appropriate now]

Guidelines:
DO:
- Show empathy and understanding
- Ask one clear question at a time
- Use simple, non-medical language
- Acknowledge the patient's concerns
- Help organize symptoms by timing, severity, location, triggers
- Summarize information back to confirm understanding

DO NOT:
- Diagnose conditions or suggest what might be wrong
- Provide medical advice or treatment suggestions
- Use complex medical terminology
- Ask overly personal questions unrelated to current symptoms
- Rush the patient or ask multiple questions at once

Always maintain professional boundaries and focus on helping patients organize their own observations."""


class CandidateResponse(BaseModel):
    Response: str
    Type: str
    Reason: str


def save_conversation_with_timestamp(conversation_history):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"conversation_{timestamp}.txt"

    with open(filename, "w") as f:
        f.write(f"Conversation saved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 50 + "\n")
        for message in conversation_history:
            f.write(message + "\n")


@agents.custom(
    LocalAgent,
    name="response_agent",
    instruction=PROMPT,
    request_params=RequestParams(response_format=CandidateResponse),
)
async def medical_assistant():
    async with agents.run() as agent:
        print("=== Medical Symptom Clarification Assistant ===")
        print("This AI will help you organize your symptoms before meeting with your doctor.\n")

        conversation_history = []

        while True:
            # Get patient input
            patient_input = input("Patient: ").strip()

            if patient_input.lower() in ["quit", "exit", "bye"]:
                print(
                    "\nThank you for using the symptom clarification assistant. Good luck with your appointment!"
                )
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
            - Reason: [Your reasoning for this response approach and why this type is appropriate now]
            
            Conversation so far:\n{context}\n
            """

            try:
                # Get AI response
                response, messages = await agent.response_agent.structured(
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
                    save_conversation_with_timestamp(conversation_history)
                    break

            except Exception as e:
                print(f"Error: {e}")
                print("messages:", messages)
                continue


def test_medical_assistant():
    asyncio.run(medical_assistant())


if __name__ == "__main__":
    test_medical_assistant()
