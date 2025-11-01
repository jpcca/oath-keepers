import asyncio
import json
import sys
from pathlib import Path
from typing import Union

from mcp_agent.core.fastagent import FastAgent
from mcp_agent.core.prompt import Prompt

from oath_keepers.utils.extractor_parser import parse_conversation_log
from oath_keepers.utils.typing import ExtractionResult
from oath_keepers.vllm_client import LocalAgent

agents = FastAgent("medical-symptom-extractor")
base_path = Path(__file__).parent.parent
prompt_path = f"{base_path}/prompts"


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
    transcript = parse_conversation_log(log_path, skip_first_user=True)
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
