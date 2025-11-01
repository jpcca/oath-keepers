from dataclasses import dataclass
from enum import Enum
from typing import Annotated, Any, Optional, Sequence, Union

from pydantic import BaseModel, ConfigDict, Field


@dataclass
class GuidedDecodingParams:
    """One of these fields will be used to build a logit processor."""

    json: Optional[Union[str, dict]] = None
    regex: Optional[str] = None
    choice: Optional[list[str]] = None
    grammar: Optional[str] = None
    json_object: Optional[bool] = None
    """These are other options that can be set"""
    backend: Optional[str] = None
    backend_was_auto: bool = False
    disable_fallback: bool = False
    disable_any_whitespace: bool = False
    disable_additional_properties: bool = False
    whitespace_pattern: Optional[str] = None
    structural_tag: Optional[str] = None

    @staticmethod
    def from_optional(
        json: Optional[Union[dict, BaseModel, str]] = None,
        regex: Optional[str] = None,
        choice: Optional[list[str]] = None,
        grammar: Optional[str] = None,
        json_object: Optional[bool] = None,
        backend: Optional[str] = None,
        whitespace_pattern: Optional[str] = None,
        structural_tag: Optional[str] = None,
    ) -> Optional["GuidedDecodingParams"]:
        if all(arg is None for arg in (json, regex, choice, grammar, json_object, structural_tag)):
            return None
        # Extract json schemas from pydantic models
        if isinstance(json, (BaseModel, type(BaseModel))):
            json = json.model_json_schema()
        return GuidedDecodingParams(
            json=json,
            regex=regex,
            choice=choice,
            grammar=grammar,
            json_object=json_object,
            backend=backend,
            whitespace_pattern=whitespace_pattern,
            structural_tag=structural_tag,
        )

    def __post_init__(self):
        """Validate that some fields are mutually exclusive."""
        guide_count = sum(
            [
                self.json is not None,
                self.regex is not None,
                self.choice is not None,
                self.grammar is not None,
                self.json_object is not None,
            ]
        )
        if guide_count > 1:
            raise ValueError(
                "You can only use one kind of guided decoding but multiple are "
                f"specified: {self.__dict__}"
            )


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
    guided_decoding: Optional[GuidedDecodingParams] = None
    logit_bias: Optional[dict[int, float]] = None
    allowed_token_ids: Optional[list[int]] = None
    extra_args: Optional[dict[str, Any]] = None
    bad_words: Optional[list[str]] = None


class GenerateRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompts: str | Sequence[str]
    sampling_params: SamplingParams | Sequence[SamplingParams]


class ResponseType(str, Enum):
    greeting = "greeting"
    questioning = "questioning"
    confirming = "confirming"
    closing = "closing"


class CandidateResponse(BaseModel):
    response: str
    response_type: ResponseType
    reason: str


# Extractor JSON schema models
class ClinicalFinding(BaseModel):
    """Single extracted clinical finding from a transcript."""

    location: Optional[str] = None
    symptom: str
    details: Optional[str] = None


class ExtractionResult(BaseModel):
    """Top-level structure returned by the extractor agent."""

    findings: list[ClinicalFinding] = Field(default_factory=list)
