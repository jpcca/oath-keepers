import argparse

import uvicorn
from fastapi import FastAPI
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage
from vllm import LLM
from vllm.logger import init_logger
from vllm.utils import random_uuid

from oath_keepers.utils.typing import GenerateRequest

logger = init_logger("vllm_server")
app = FastAPI()


@app.post("/generate", response_model=ChatCompletion)
async def generate(request: GenerateRequest) -> ChatCompletion:
    sampling_params = llm.get_default_sampling_params()
    request_id = random_uuid()

    for key, value in request.sampling_params.model_dump().items():
        setattr(sampling_params, key, value)

    if request.sampling_params.guided_decoding:
        sampling_params.guided_decoding = request.sampling_params.guided_decoding

    response = llm.generate(
        f"""
            <start_of_turn>user
            {request.prompts}<end_of_turn>
            <start_of_turn>model
        """,
        sampling_params=sampling_params,
        use_tqdm=False,
    )[0]

    return ChatCompletion(
        choices=[
            Choice(
                message=ChatCompletionMessage(
                    content=output.text,
                    role="assistant",
                    tool_calls=None,
                    function_call=None,
                    refusal=None,
                    annotations=None,
                    audio=None,
                ),
                finish_reason="stop",
                index=output.index,
            )
            for output in response.outputs
        ],
        id=request_id,
        model="vLLM",
        object="chat.completion",
        created=0,
        usage=CompletionUsage(
            completion_tokens=sum(len(output.token_ids) for output in response.outputs),
            prompt_tokens=len(response.prompt_token_ids),
            total_tokens=(
                sum(len(output.token_ids) for output in response.outputs)
                + len(response.prompt_token_ids)
            ),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="google/gemma-3-12b-it")
    args = parser.parse_args()

    llm = LLM(model=args.model)
    uvicorn.run(app, host="0.0.0.0", port=8000)
