import asyncio

from oath_keepers.agents import clarifier_assistant, extractor_assistant


async def run_workflow() -> None:
    # 1) Run clarifier. It saves a conversation log on exit.
    log_path = await clarifier_assistant()

    # If clarifier didn't return a specific path, report and stop.
    if not log_path:
        print("Conversation history was not saved properly; skipping extraction.")
        return

    # 2) Run extractor which writes JSON next to the log and returns the path.
    out_path = await extractor_assistant(log_path)
    if out_path:
        print(f"Saved extracted findings to {out_path}")


if __name__ == "__main__":
    asyncio.run(run_workflow())
