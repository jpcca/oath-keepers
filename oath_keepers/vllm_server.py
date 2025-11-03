import argparse
import asyncio
import logging

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage
from vllm import LLM
from vllm.logger import init_logger
from vllm.utils import random_uuid
from whisperlivekit import AudioProcessor, TranscriptionEngine, get_inline_ui_html

from oath_keepers.utils.typing import GenerateRequest

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger().setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

vllm_logger = init_logger("vllm_server")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def get():
    return HTMLResponse(get_inline_ui_html())


async def handle_websocket_results(websocket, results_generator):
    """Consumes results from the audio processor and sends them via WebSocket."""
    try:
        async for response in results_generator:
            await websocket.send_json(response.to_dict())
        # when the results_generator finishes it means all audio has been processed
        logger.info("Results generator finished. Sending 'ready_to_stop' to client.")
        await websocket.send_json({"type": "ready_to_stop"})
    except WebSocketDisconnect:
        logger.info(
            "WebSocket disconnected while handling results (client likely closed connection)."
        )
    except Exception as e:
        logger.exception(f"Error in WebSocket results handler: {e}")


@app.websocket("/asr")
async def websocket_endpoint(websocket: WebSocket):
    global transcription_engine
    audio_processor = AudioProcessor(
        transcription_engine=transcription_engine,
    )
    await websocket.accept()
    logger.info("WebSocket connection opened.")

    try:
        await websocket.send_json({"type": "config", "useAudioWorklet": bool(args.pcm_input)})
    except Exception as e:
        logger.warning(f"Failed to send config to client: {e}")

    results_generator = await audio_processor.create_tasks()
    websocket_task = asyncio.create_task(handle_websocket_results(websocket, results_generator))

    try:
        while True:
            message = await websocket.receive_bytes()
            await audio_processor.process_audio(message)
    except KeyError as e:
        if "bytes" in str(e):
            logger.warning("Client has closed the connection.")
        else:
            logger.error(f"Unexpected KeyError in websocket_endpoint: {e}", exc_info=True)
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected by client during message receiving loop.")
    except Exception as e:
        logger.error(f"Unexpected error in websocket_endpoint main loop: {e}", exc_info=True)
    finally:
        logger.info("Cleaning up WebSocket endpoint...")
        if not websocket_task.done():
            websocket_task.cancel()
        try:
            await websocket_task
        except asyncio.CancelledError:
            logger.info("WebSocket results handler task was cancelled.")
        except Exception as e:
            logger.warning(f"Exception while awaiting websocket_task completion: {e}")

        await audio_processor.cleanup()
        logger.info("WebSocket endpoint cleaned up successfully.")


@app.post("/generate", response_model=ChatCompletion)
async def generate(request: GenerateRequest) -> ChatCompletion:
    sampling_params = llm.get_default_sampling_params()
    request_id = random_uuid()

    for key, value in request.sampling_params.model_dump().items():
        setattr(sampling_params, key, value)

    if request.sampling_params.structured_outputs:
        sampling_params.structured_outputs = request.sampling_params.structured_outputs

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


# patched version of whisperlivekit.parse_args
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--warmup-file", type=str, default=None, dest="warmup_file")
    parser.add_argument("--confidence-validation", action="store_true")
    parser.add_argument("--diarization", action="store_true", default=False)
    parser.add_argument("--punctuation-split", action="store_true", default=False)
    parser.add_argument("--segmentation-model", type=str, default="pyannote/segmentation-3.0")
    parser.add_argument("--embedding-model", type=str, default="pyannote/embedding")
    parser.add_argument(
        "--diarization-backend", type=str, default="sortformer", choices=["sortformer", "diart"]
    )
    parser.add_argument("--no-transcription", action="store_true")
    parser.add_argument("--disable-punctuation-split", action="store_true")
    parser.add_argument("--min-chunk-size", type=float, default=0.5)
    parser.add_argument(
        "--model", type=str, default="google/gemma-3-12b-it"
    )  # patching in Gemma-3 model as default
    parser.add_argument("--model_cache_dir", type=str, default=None)
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--lan", "--language", type=str, default="auto", dest="lan")
    parser.add_argument(
        "--task", type=str, default="transcribe", choices=["transcribe", "translate"]
    )

    parser.add_argument("--target-language", type=str, default="", dest="target_language")
    parser.add_argument(
        "--backend",
        type=str,
        default="simulstreaming",
        choices=[
            "faster-whisper",
            "whisper_timestamped",
            "mlx-whisper",
            "openai-api",
            "simulstreaming",
        ],
    )
    parser.add_argument("--no-vac", action="store_true", default=False)
    parser.add_argument("--vac-chunk-size", type=float, default=0.04)
    parser.add_argument("--no-vad", action="store_true")
    parser.add_argument(
        "--buffer_trimming", type=str, default="segment", choices=["sentence", "segment"]
    )
    parser.add_argument("--buffer_trimming_sec", type=float, default=15)
    parser.add_argument(
        "-l",
        "--log-level",
        dest="log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="DEBUG",
    )
    parser.add_argument("--ssl-certfile", type=str, default=None)
    parser.add_argument("--ssl-keyfile", type=str, default=None)
    parser.add_argument("--forwarded-allow-ips", type=str, default=None)
    parser.add_argument("--pcm-input", action="store_true", default=False)
    simulstreaming_group = parser.add_argument_group(
        "SimulStreaming arguments (only used with --backend simulstreaming)"
    )
    simulstreaming_group.add_argument(
        "--disable-fast-encoder", action="store_true", default=False, dest="disable_fast_encoder"
    )
    simulstreaming_group.add_argument("--custom-alignment-heads", type=str, default=None)
    simulstreaming_group.add_argument(
        "--frame-threshold", type=int, default=25, dest="frame_threshold"
    )
    simulstreaming_group.add_argument("--beams", "-b", type=int, default=1)
    simulstreaming_group.add_argument(
        "--decoder", type=str, default=None, dest="decoder_type", choices=["beam", "greedy"]
    )
    simulstreaming_group.add_argument(
        "--audio-max-len", type=float, default=30.0, dest="audio_max_len"
    )
    simulstreaming_group.add_argument(
        "--audio-min-len", type=float, default=0.0, dest="audio_min_len"
    )
    simulstreaming_group.add_argument(
        "--cif-ckpt-path", type=str, default=None, dest="cif_ckpt_path"
    )
    simulstreaming_group.add_argument(
        "--never-fire", action="store_true", default=False, dest="never_fire"
    )
    simulstreaming_group.add_argument("--init-prompt", type=str, default=None, dest="init_prompt")
    simulstreaming_group.add_argument(
        "--static-init-prompt", type=str, default=None, dest="static_init_prompt"
    )
    simulstreaming_group.add_argument(
        "--max-context-tokens", type=int, default=None, dest="max_context_tokens"
    )
    simulstreaming_group.add_argument("--model-path", type=str, default=None, dest="model_path")
    simulstreaming_group.add_argument(
        "--preload-model-count", type=int, default=1, dest="preload_model_count"
    )
    simulstreaming_group.add_argument("--nllb-backend", type=str, default="ctranslate2")
    simulstreaming_group.add_argument("--nllb-size", type=str, default="600M")
    args = parser.parse_args()

    args.transcription = not args.no_transcription
    args.vad = not args.no_vad
    delattr(args, "no_transcription")
    delattr(args, "no_vad")

    return args


if __name__ == "__main__":
    args = parse_args()

    llm = LLM(model=args.model)

    args.model = "tiny"  # always use the same model for transcription engine
    transcription_engine = TranscriptionEngine(
        **vars(args),  # errors out for some reason if not passed this way
    )

    uvicorn.run(app, host="0.0.0.0", port=8000)
