# Still working on coding with this file (still having bug)

import asyncio
import json
from mcp_agent.core.fastagent import FastAgent
from oath_keepers.vllm_client import LocalAgent

agents = FastAgent("medical-symptom-clarification")

PROMPT = """You are a compassionate medical intake assistant AI designed to help patients articulate their symptoms more clearly before meeting with their doctor. You are NOT a medical professional and cannot provide diagnoses, medical advice, or treatment recommendations.

Your task is to conduct a brief, empathetic conversation with patients to help them organize and clarify their symptoms. Your goal is to gather clear, structured information that will help the patient communicate more effectively with their healthcare provider.

Context:
- You're speaking with patients who are about to see their doctor
- Patients may be anxious, confused about their symptoms, or unsure how to describe what they're experiencing
- You should be warm, professional, and reassuring while maintaining clear boundaries about your role
- The conversation should flow naturally through different states: greeting, information gathering, clarification, and closure

Conversation Flow:
- Greeting: Initial welcome and explanation of your role
- Questioning: Asking follow-up questions to clarify symptoms
- Confirming: Summarizing information to ensure accuracy
- Closing: Ending the conversation with organized symptom summary

You MUST respond with exactly 3 candidate responses in valid JSON format:

{
  "candidate_1": {
    "type": "[greeting/questioning/confirming/closing]",
    "reason": "[Your reasoning for this response approach and why this type is appropriate now]",
    "response_text": "[Your actual response to the patient]"
  },
  "candidate_2": {
    "type": "[greeting/questioning/confirming/closing]",
    "reason": "[Your reasoning for this response approach and why this type is appropriate now]",
    "response_text": "[Your actual response to the patient]"
  },
  "candidate_3": {
    "type": "[greeting/questioning/confirming/closing]",
    "reason": "[Your reasoning for this response approach and why this type is appropriate now]",
    "response_text": "[Your actual response to the patient]"
  }
}

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
- Attempt to diagnose or suggest possible conditions

Always maintain professional boundaries and focus on helping patients organize their own observations."""


@agents.custom(LocalAgent, instruction=PROMPT)
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
            context = "\n".join(conversation_history[-10:])  # Keep last 10 exchanges
            full_prompt = f"Conversation so far:\n{context}\n\nPlease provide 3 candidate responses in JSON format as specified."

            try:
                # Get AI response
                response = await agent(full_prompt)

                # Parse the JSON response
                candidates = json.loads(response)

                # Display all candidates for selection (or auto-select first one)
                print("\n--- AI Response Candidates ---")
                for i, (key, candidate) in enumerate(candidates.items(), 1):
                    print(f"\n{i}. Type: {candidate['type']}")
                    print(f"   Reason: {candidate['reason']}")
                    print(f"   Response: {candidate['response_text']}")

                # For demo, automatically select first candidate
                selected = candidates["candidate_1"]
                print(f"\n[Selected Response - {selected['type']}]")
                print(f"AI: {selected['response_text']}")

                # Add AI response to history
                conversation_history.append(f"AI: {selected['response_text']}")

                # Check if conversation should end
                if selected["type"] == "closing":
                    print(
                        "\nConversation completed. Your symptoms have been organized for your doctor visit."
                    )
                    break

            except json.JSONDecodeError as e:
                print(f"Error parsing AI response: {e}")
                print("AI Response:", response)
                continue
            except Exception as e:
                print(f"Error: {e}")
                continue


# Alternative version that returns the best candidate automatically
@agents.custom(LocalAgent, instruction=PROMPT)
async def medical_assistant_simple():
    async with agents.run() as agent:
        print("=== Medical Symptom Clarification Assistant ===")
        print("This AI will help you organize your symptoms before meeting with your doctor.\n")

        conversation_history = []

        while True:
            patient_input = input("Patient: ").strip()

            if patient_input.lower() in ["quit", "exit", "bye"]:
                print("\nThank you! Good luck with your appointment!")
                break

            conversation_history.append(f"Patient: {patient_input}")
            context = "\n".join(conversation_history[-10:])

            try:
                response = await agent(
                    f"Conversation so far:\n{context}\n\nProvide 3 candidates in JSON format."
                )
                candidates = json.loads(response)

                # Auto-select first candidate
                selected = candidates["candidate_1"]
                print(f"AI: {selected['response_text']}")

                conversation_history.append(f"AI: {selected['response_text']}")

                if selected["type"] == "closing":
                    break

            except Exception as e:
                print(f"Error: {e}")
                continue


def test_medical_assistant():
    # Run the full version with candidate display
    asyncio.run(medical_assistant())


def test_medical_assistant_simple():
    # Run the simple version
    asyncio.run(medical_assistant_simple())


if __name__ == "__main__":
    # Choose which version to run
    print("1. Full version (shows all candidates)")
    print("2. Simple version (auto-selects best candidate)")
    choice = input("Select version (1 or 2): ").strip()

    if choice == "1":
        test_medical_assistant()
    else:
        test_medical_assistant_simple()
