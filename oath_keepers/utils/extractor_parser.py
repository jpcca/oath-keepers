import json
from pathlib import Path
from typing import Union


def parse_conversation_log(path: Union[str, Path], *, skip_first_user: bool = True) -> str:
    """Parse a conversation log into a concise transcript.

    - Reads blocks delimited by lines starting with `---USER` / `---ASSISTANT`.
    - From assistant blocks, prefer the JSON `response` field if present.
    - Emits turns as "Assistant:" then "Patient:", separated by blank lines.
    - When `skip_first_user` is True, omits the first Patient message.
    """

    def extract_assistant_text(raw: str) -> str:
        raw = raw.strip()
        if raw.startswith("{") and raw.endswith("}"):
            try:
                obj = json.loads(raw)
                if isinstance(obj.get("response"), str):
                    return obj["response"]
            except Exception:
                pass
        return raw

    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    blocks: list[tuple[str, str]] = []  # (role, content)
    role: str | None = None
    buf: list[str] = []

    def flush() -> None:
        nonlocal role, buf
        if role is not None:
            content = "\n".join(buf).strip()
            blocks.append((role, content))
        buf = []

    for line in lines:
        if line.startswith("---USER"):
            flush()
            role = "USER"
            continue
        if line.startswith("---ASSISTANT"):
            flush()
            role = "ASSISTANT"
            continue
        buf.append(line)
    flush()

    # Output transcript exactly in order, ignoring only the first user block
    chunks: list[str] = []
    skipped_first_user = False
    for r, content in blocks:
        if r == "USER":
            if skip_first_user and not skipped_first_user:
                skipped_first_user = True
                continue
            if content:
                chunks.append(f"Patient: {content}")
        elif r == "ASSISTANT":
            text = extract_assistant_text(content)
            if text:
                chunks.append(f"Assistant: {text}")
    return "\n\n".join(chunks)
