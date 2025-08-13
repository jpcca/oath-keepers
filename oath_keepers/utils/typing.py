from typing import Annotated, Any, Optional, Sequence, Union

from pydantic import BaseModel, ConfigDict, Field


class SamplingParams(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    n: int = 1
    best_of: Optional[int] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    seed: Optional[int] = None
    stop: Optional[Union[str, list[str]]] = None
    stop_token_ids: Optional[list[int]] = None
    ignore_eos: bool = False
    max_tokens: Optional[int] = 16
    min_tokens: int = 0
    logprobs: Optional[int] = None
    prompt_logprobs: Optional[int] = None
    detokenize: bool = True
    skip_special_tokens: bool = True
    spaces_between_special_tokens: bool = True
    logits_processors: Optional[Any] = None
    include_stop_str_in_output: bool = False
    truncate_prompt_tokens: Optional[Annotated[int, Field(ge=1)]] = None
    output_kind: int = 0
    guided_decoding: Optional[Any] = None
    logit_bias: Optional[dict[int, float]] = None
    allowed_token_ids: Optional[list[int]] = None
    extra_args: Optional[dict[str, Any]] = None
    bad_words: Optional[list[str]] = None


class GenerateRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompts: str | Sequence[str]
    sampling_params: SamplingParams | Sequence[SamplingParams]
