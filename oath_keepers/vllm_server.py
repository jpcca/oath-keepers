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


def parse_args():
    parser = argparse.ArgumentParser(description="Whisper FastAPI Online Server")
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="The host address to bind the server to.",
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="The port number to bind the server to."
    )
    parser.add_argument(
        "--warmup-file",
        type=str,
        default=None,
        dest="warmup_file",
        help="""
        The path to a speech audio wav file to warm up Whisper so that the very first chunk processing is fast.
        If not set, uses https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav.
        If empty, no warmup is performed.
        """,
    )

    parser.add_argument(
        "--confidence-validation",
        action="store_true",
        help="Accelerates validation of tokens using confidence scores. Transcription will be faster but punctuation might be less accurate.",
    )

    parser.add_argument(
        "--diarization",
        action="store_true",
        default=False,
        help="Enable speaker diarization.",
    )

    parser.add_argument(
        "--punctuation-split",
        action="store_true",
        default=False,
        help="Use punctuation marks from transcription to improve speaker boundary detection. Requires both transcription and diarization to be enabled.",
    )

    parser.add_argument(
        "--segmentation-model",
        type=str,
        default="pyannote/segmentation-3.0",
        help="Hugging Face model ID for pyannote.audio segmentation model.",
    )

    parser.add_argument(
        "--embedding-model",
        type=str,
        default="pyannote/embedding",
        help="Hugging Face model ID for pyannote.audio embedding model.",
    )

    parser.add_argument(
        "--diarization-backend",
        type=str,
        default="sortformer",
        choices=["sortformer", "diart"],
        help="The diarization backend to use.",
    )

    parser.add_argument(
        "--no-transcription",
        action="store_true",
        help="Disable transcription to only see live diarization results.",
    )

    parser.add_argument(
        "--disable-punctuation-split",
        action="store_true",
        help="Disable the split parameter.",
    )

    parser.add_argument(
        "--min-chunk-size",
        type=float,
        default=0.5,
        help="Minimum audio chunk size in seconds. It waits up to this time to do processing. If the processing takes shorter time, it waits, otherwise it processes the whole segment that was received by this time.",
    )

    parser.add_argument("--model", type=str, default="google/gemma-3-12b-it")

    parser.add_argument(
        "--model_cache_dir",
        type=str,
        default=None,
        help="Overriding the default model cache dir where models downloaded from the hub are saved",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Dir where Whisper model.bin and other files are saved. This option overrides --model and --model_cache_dir parameter.",
    )
    parser.add_argument(
        "--lan",
        "--language",
        type=str,
        default="auto",
        dest="lan",
        help="Source language code, e.g. en,de,cs, or 'auto' for language detection.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="transcribe",
        choices=["transcribe", "translate"],
        help="Transcribe or translate.",
    )

    parser.add_argument(
        "--target-language",
        type=str,
        default="",
        dest="target_language",
        help="Target language for translation. Not functional yet.",
    )

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
        help="Load only this backend for Whisper processing.",
    )
    parser.add_argument(
        "--no-vac",
        action="store_true",
        default=False,
        help="Disable VAC = voice activity controller.",
    )
    parser.add_argument(
        "--vac-chunk-size", type=float, default=0.04, help="VAC sample size in seconds."
    )

    parser.add_argument(
        "--no-vad",
        action="store_true",
        help="Disable VAD (voice activity detection).",
    )

    parser.add_argument(
        "--buffer_trimming",
        type=str,
        default="segment",
        choices=["sentence", "segment"],
        help='Buffer trimming strategy -- trim completed sentences marked with punctuation mark and detected by sentence segmenter, or the completed segments returned by Whisper. Sentence segmenter must be installed for "sentence" option.',
    )
    parser.add_argument(
        "--buffer_trimming_sec",
        type=float,
        default=15,
        help="Buffer trimming length threshold in seconds. If buffer length is longer, trimming sentence/segment is triggered.",
    )
    parser.add_argument(
        "-l",
        "--log-level",
        dest="log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the log level",
        default="DEBUG",
    )
    parser.add_argument(
        "--ssl-certfile", type=str, help="Path to the SSL certificate file.", default=None
    )
    parser.add_argument(
        "--ssl-keyfile", type=str, help="Path to the SSL private key file.", default=None
    )
    parser.add_argument(
        "--forwarded-allow-ips", type=str, help="Allowed ips for reverse proxying.", default=None
    )
    parser.add_argument(
        "--pcm-input",
        action="store_true",
        default=False,
        help="If set, raw PCM (s16le) data is expected as input and FFmpeg will be bypassed. Frontend will use AudioWorklet instead of MediaRecorder.",
    )
    # SimulStreaming-specific arguments
    simulstreaming_group = parser.add_argument_group(
        "SimulStreaming arguments (only used with --backend simulstreaming)"
    )

    simulstreaming_group.add_argument(
        "--disable-fast-encoder",
        action="store_true",
        default=False,
        dest="disable_fast_encoder",
        help="Disable Faster Whisper or MLX Whisper backends for encoding (if installed). Slower but helpful when GPU memory is limited",
    )

    simulstreaming_group.add_argument(
        "--custom-alignment-heads",
        type=str,
        default=None,
        help="Use your own alignment heads, useful when `--model-dir` is used",
    )

    simulstreaming_group.add_argument(
        "--frame-threshold",
        type=int,
        default=25,
        dest="frame_threshold",
        help="Threshold for the attention-guided decoding. The AlignAtt policy will decode only until this number of frames from the end of audio. In frames: one frame is 0.02 seconds for large-v3 model.",
    )

    simulstreaming_group.add_argument(
        "--beams",
        "-b",
        type=int,
        default=1,
        help="Number of beams for beam search decoding. If 1, GreedyDecoder is used.",
    )

    simulstreaming_group.add_argument(
        "--decoder",
        type=str,
        default=None,
        dest="decoder_type",
        choices=["beam", "greedy"],
        help="Override automatic selection of beam or greedy decoder. If beams > 1 and greedy: invalid.",
    )

    simulstreaming_group.add_argument(
        "--audio-max-len",
        type=float,
        default=30.0,
        dest="audio_max_len",
        help="Max length of the audio buffer, in seconds.",
    )

    simulstreaming_group.add_argument(
        "--audio-min-len",
        type=float,
        default=0.0,
        dest="audio_min_len",
        help="Skip processing if the audio buffer is shorter than this length, in seconds. Useful when the --min-chunk-size is small.",
    )

    simulstreaming_group.add_argument(
        "--cif-ckpt-path",
        type=str,
        default=None,
        dest="cif_ckpt_path",
        help="The file path to the Simul-Whisper's CIF model checkpoint that detects whether there is end of word at the end of the chunk. If not, the last decoded space-separated word is truncated because it is often wrong -- transcribing a word in the middle. The CIF model adapted for the Whisper model version should be used. Find the models in https://github.com/backspacetg/simul_whisper/tree/main/cif_models . Note that there is no model for large-v3.",
    )

    simulstreaming_group.add_argument(
        "--never-fire",
        action="store_true",
        default=False,
        dest="never_fire",
        help="Override the CIF model. If True, the last word is NEVER truncated, no matter what the CIF model detects. If False: if CIF model path is set, the last word is SOMETIMES truncated, depending on the CIF detection. Otherwise, if the CIF model path is not set, the last word is ALWAYS trimmed.",
    )

    simulstreaming_group.add_argument(
        "--init-prompt",
        type=str,
        default=None,
        dest="init_prompt",
        help="Init prompt for the model. It should be in the target language.",
    )

    simulstreaming_group.add_argument(
        "--static-init-prompt",
        type=str,
        default=None,
        dest="static_init_prompt",
        help="Do not scroll over this text. It can contain terminology that should be relevant over all document.",
    )

    simulstreaming_group.add_argument(
        "--max-context-tokens",
        type=int,
        default=None,
        dest="max_context_tokens",
        help="Max context tokens for the model. Default is 0.",
    )

    simulstreaming_group.add_argument(
        "--model-path",
        type=str,
        default=None,
        dest="model_path",
        help="Direct path to the SimulStreaming Whisper .pt model file. Overrides --model for SimulStreaming backend.",
    )

    simulstreaming_group.add_argument(
        "--preload-model-count",
        type=int,
        default=1,
        dest="preload_model_count",
        help="Optional. Number of models to preload in memory to speed up loading (set up to the expected number of concurrent instances).",
    )

    simulstreaming_group.add_argument(
        "--nllb-backend",
        type=str,
        default="ctranslate2",
        help="transformers or ctranslate2",
    )

    simulstreaming_group.add_argument(
        "--nllb-size",
        type=str,
        default="600M",
        help="600M or 1.3B",
    )

    args = parser.parse_args()

    args.transcription = not args.no_transcription
    args.vad = not args.no_vad
    delattr(args, "no_transcription")
    delattr(args, "no_vad")

    return args


if __name__ == "__main__":
    args = parse_args()

    llm = LLM(model=args.model)

    args.model = "tiny"
    transcription_engine = TranscriptionEngine(
        **vars(args),
    )

    uvicorn.run(app, port=8000)
