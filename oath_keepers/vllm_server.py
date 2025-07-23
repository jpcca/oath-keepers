import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage
from vllm import LLM
from vllm.entrypoints.utils import with_cancellation
from vllm.logger import init_logger
from vllm.utils import random_uuid

logger = init_logger("vllm_server")

llm = LLM(model="google/gemma-3-1b-it")
app = FastAPI()


@app.post("/generate")
async def generate(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    request_dict = await request.json()
    return await _generate(request_dict, raw_request=request)


@with_cancellation
async def _generate(request_dict: dict, raw_request: Request) -> Response:
    prompts = request_dict.pop("prompts")
    sampling_params = llm.get_default_sampling_params()
    request_id = random_uuid()

    for key, value in request_dict.pop("sampling_params").items():
        setattr(sampling_params, key, value)

    response = llm.generate(
        f"""
            <start_of_turn>user
            {prompts}<end_of_turn>
            <start_of_turn>model
        """,
        sampling_params=sampling_params,
        use_tqdm=False,
    )[0]

    return JSONResponse(
        ChatCompletion(
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
        ).model_dump()
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
