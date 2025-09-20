import asyncio
import json
import sys
from pathlib import Path
from typing import Union

from mcp_agent.core.fastagent import FastAgent
from mcp_agent.core.prompt import Prompt

from oath_keepers.utils.typing import ExtractionResult
from oath_keepers.vllm_client import LocalAgent

agents = FastAgent("medical-symptom-extractor")
base_path = Path(__file__).parent.parent
prompt_path = f"{base_path}/prompts"


def _parse_conversation_log(path: Union[str, Path], *, skip_first_user: bool = True) -> str:
    """Convert a saved conversation log file into an Assistant/Patient transcript.

    - Extracts only the `response` field from assistant JSON blocks when present.
    - Skips the very first user turn when `skip_first_user` is True.
    - Groups by turn: Assistant then Patient, separated by blank lines.
    """

    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    state = None  # 'USER' or 'ASSISTANT'
    buf: list[str] = []
    pairs: list[tuple[str, str]] = []
    current_user = None
    current_assistant = None

    def flush_user():
        nonlocal current_user, buf
        content = "\n".join(buf).strip()
        current_user = content

    def flush_assistant():
        nonlocal current_assistant, buf
        raw = "\n".join(buf).strip()
        resp = None
        # Try JSON parse to extract only the "response" field
        try:
            if raw.startswith("{") and raw.endswith("}"):
                obj = json.loads(raw)
                if isinstance(obj.get("response"), str):
                    resp = obj["response"]
        except Exception:
            resp = None
        current_assistant = resp if isinstance(resp, str) else raw

    for line in lines:
        if line.startswith("---USER"):
            if state == "ASSISTANT":
                flush_assistant()
                if current_user is not None and current_assistant is not None:
                    pairs.append((current_user, current_assistant))
                    current_user = None
                    current_assistant = None
            if state == "USER":
                flush_user()
            state = "USER"
            buf = []
            continue
        if line.startswith("---ASSISTANT"):
            if state == "USER":
                flush_user()
            if state == "ASSISTANT":
                flush_assistant()
            state = "ASSISTANT"
            buf = []
            continue
        buf.append(line)

    # Flush at EOF
    if state == "ASSISTANT":
        flush_assistant()
        if current_user is not None and current_assistant is not None:
            pairs.append((current_user, current_assistant))
    elif state == "USER":
        flush_user()

    # Optionally skip only the first Patient turn while keeping the first Assistant
    if skip_first_user and pairs:
        first_user, first_assistant = pairs[0]
        pairs[0] = ("", first_assistant)

    # Build transcript grouped by turn
    chunks: list[str] = []
    for u, a in pairs:
        if a:
            chunks.append(f"Assistant: {a}")
        if u:
            chunks.append(f"Patient: {u}")
    return "\n\n".join(chunks)


@agents.custom(
    LocalAgent,
    name="extractor_agent",
    instruction=Path(f"{prompt_path}/extractor_prompt.md").read_text(encoding="utf-8"),
    use_history=False,
)
async def extractor_assistant(log_path: Union[str, Path]) -> Path:
    """Run the extractor on a conversation log file path and write JSON next to it.

    Returns the output file Path.
    """
    transcript = _parse_conversation_log(log_path, skip_first_user=True)
    try:
        async with agents.run() as agent:
            result, messages = await agent.extractor_agent.structured(
                multipart_messages=[Prompt.user(transcript)], model=ExtractionResult
            )
            if result is None:
                raise ValueError("Extractor returned no result")
            # Normalize to JSON string
            try:
                # Pydantic model
                json_text = result.model_dump_json(indent=2)  # type: ignore[attr-defined]
            except Exception:
                json_text = None
            try:
                # Dict/list
                if json_text is None:
                    json_text = json.dumps(result, indent=2, ensure_ascii=False)
            except Exception:
                # Fallback to string
                if json_text is None:
                    json_text = str(result)

            # Write next to the log file
            lp = Path(log_path)
            out_path = lp.with_name(lp.stem + "_extracted.json")
            out_path.write_text(json_text, encoding="utf-8")
            print(f"Saved extracted findings to {out_path}")
            return out_path
    except Exception as e:
        print(f"Error: {e}")
        print("result:", locals().get("result"))
        print("messages:", locals().get("messages"))
        return Path("")


async def main(argv: list[str] | None = None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("Usage: python -m oath_keepers.agents.extractor_agent <conversation_log_path>")
        return
    log_path = argv[0]
    await extractor_assistant(log_path)


if __name__ == "__main__":
    asyncio.run(main())
