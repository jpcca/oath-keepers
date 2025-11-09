import asyncio
from pathlib import Path
from datetime import datetime


from oath_keepers.agents import clarifier_assistant, extractor_assistant

log_path = Path(__file__).parent / "oath_keepers" / "log"

def get_filepath() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(f"{log_path}/conversation_{timestamp}.txt")


async def run_workflow() -> None:
    conversation_log_path = get_filepath()
    await clarifier_assistant(conversation_log_path, extractor_assistant)

    print(f"Saved log to: {conversation_log_path}")


if __name__ == "__main__":
    asyncio.run(run_workflow())
