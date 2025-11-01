import textwrap
from pathlib import Path

from oath_keepers.utils.extractor_parser import parse_conversation_log


def write_log(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "conversation.log"
    p.write_text(textwrap.dedent(content).lstrip("\n"), encoding="utf-8")
    return p


TEST_CONTENT = """
---USER
Call received. Please greet the patient on the call.
---ASSISTANT
{
  "response": "Hello! Thank you for calling. I'm here to help you prepare for your doctor's appointment by organizing your symptoms."
}
---USER
I have a headache.
---ASSISTANT
{
  "response": "How long have you had it?"
}
---USER
Since yesterday.
---ASSISTANT
It started yesterday? Any other symptoms?
"""


def test_parse_conversation_log_skip_first_user(tmp_path):
    log = write_log(tmp_path, TEST_CONTENT)

    transcript = parse_conversation_log(log, skip_first_user=True)

    assert transcript == "\n\n".join(
        [
            # First user reply is skipped
            "Assistant: Hello! Thank you for calling. I'm here to help you prepare for your doctor's appointment by organizing your symptoms.",
            "Patient: I have a headache.",
            "Assistant: How long have you had it?",
            "Patient: Since yesterday.",
            "Assistant: It started yesterday? Any other symptoms?",
        ]
    )


def test_parse_conversation_log_include_first_user(tmp_path):
    log = write_log(tmp_path, TEST_CONTENT)

    transcript = parse_conversation_log(log, skip_first_user=False)

    assert transcript == "\n\n".join(
        [
            "Patient: Call received. Please greet the patient on the call.",
            "Assistant: Hello! Thank you for calling. I'm here to help you prepare for your doctor's appointment by organizing your symptoms.",
            "Patient: I have a headache.",
            "Assistant: How long have you had it?",
            "Patient: Since yesterday.",
            "Assistant: It started yesterday? Any other symptoms?",
        ]
    )


def test_assistant_block_without_json_is_kept(tmp_path):
    log = write_log(
        tmp_path,
        """
        ---USER
        Hi.
        ---ASSISTANT
        Plain text assistant reply without JSON.
        """,
    )

    transcript = parse_conversation_log(log, skip_first_user=False)
    assert transcript == "\n\n".join(
        [
            "Patient: Hi.",
            "Assistant: Plain text assistant reply without JSON.",
        ]
    )
