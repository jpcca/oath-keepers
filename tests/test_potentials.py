import torch
import asyncio
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
)
from genlm.control import Potential, PromptedLLM, direct_token_sampler  # type: ignore
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import log_softmax


class SentimentAnalysis(Potential):
    def __init__(
        self,
        model=DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased-finetuned-sst-2-english"
        ),
        tokenizer=DistilBertTokenizer.from_pretrained(
            "distilbert-base-uncased-finetuned-sst-2-english"
        ),
        sentiment="POSITIVE",
    ):
        self.model = model
        self.tokenizer = tokenizer

        self.sentiment_idx = model.config.label2id.get(sentiment, None)
        if self.sentiment_idx is None:
            raise ValueError(f"Sentiment {sentiment} not found in model labels")

        super().__init__(vocabulary=list(range(256)))  # Defined over bytes

    def _forward(self, contexts):
        strings = [
            bytes(context).decode("utf-8", errors="ignore") for context in contexts
        ]
        inputs = self.tokenizer(strings, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        return logits.log_softmax(dim=-1)[:, self.sentiment_idx].cpu().numpy()

    async def prefix(self, context):
        return self._forward([context])[0].item()

    async def complete(self, context):
        return self._forward([context])[0].item()

    async def batch_complete(self, contexts):
        return self._forward(contexts)

    async def batch_prefix(self, contexts):
        return self._forward(contexts)


class EntailmentPotential(Potential):
    """
    A potential that uses a Natural Language Inference (NLI) model
    to steer generation away from a privileged text.
    """

    def __init__(
        self,
        privileged_text: str,
        model_name: str = "roberta-large-mnli",
    ):
        # Load a pre-trained NLI model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()  # Set to evaluation mode

        self.privileged_text = privileged_text

        # Get the IDs for the NLI labels
        self.entailment_id = self.model.config.label2id["ENTAILMENT"]
        self.neutral_id = self.model.config.label2id["NEUTRAL"]
        self.contradiction_id = self.model.config.label2id["CONTRADICTION"]

        # This potential operates on byte sequences
        super().__init__(vocabulary=list(range(256)))

    def _forward(self, contexts: list[list[int]]) -> np.ndarray:
        """
        Calculates the log-probability that the contexts do NOT entail the secret.
        """
        if not contexts:
            return np.array([])

        # Decode byte contexts to strings to form hypotheses
        hypotheses = [
            bytes(context).decode("utf-8", errors="ignore") for context in contexts
        ]
        # The premise is always the privileged text
        premises = [self.privileged_text] * len(hypotheses)

        # Tokenize premise-hypothesis pairs
        inputs = self.tokenizer(
            premises, hypotheses, return_tensors="pt", padding=True, truncation=True
        )

        with torch.no_grad():
            logits = self.model(**inputs).logits
            # Get log-probabilities for all classes
            log_probs = log_softmax(logits, dim=-1)

        # We want to steer away from entailment.
        # So, we reward sequences that are neutral or contradictory.
        # We calculate log(P(neutral) + P(contradiction)) using logsumexp for stability.
        safe_log_probs = torch.stack(
            [log_probs[:, self.neutral_id], log_probs[:, self.contradiction_id]], dim=1
        )
        log_weight = torch.logsumexp(safe_log_probs, dim=1)

        return log_weight.cpu().numpy()

    async def prefix(self, context: list[int]) -> float:
        return self._forward([context])[0].item()

    async def complete(self, context: list[int]) -> float:
        return self._forward([context])[0].item()

    async def batch_prefix(self, contexts: list[list[int]]) -> np.ndarray:
        return self._forward(contexts)

    async def batch_complete(self, contexts: list[list[int]]) -> np.ndarray:
        return self._forward(contexts)

    def spawn(self):
        # Required for multiprocessing
        return EntailmentPotential(self.privileged_text, self.model.name_or_path)


async def sentiment_analysis():

    # Initialize sentiment analysis potential
    sentiment_analysis = SentimentAnalysis(
        sentiment="NEGATIVE",  # or "POSITIVE"
    )

    # Test the potential
    print("\nSentiment analysis test:")
    print(
        "sentiment_analysis.prefix(b'so good') =",
        await sentiment_analysis.prefix(b"so good"),
    )
    print(
        "sentiment_analysis.prefix(b'so bad') =",
        await sentiment_analysis.prefix(b"so bad"),
    )

    # Load and prompt gpt2 (or any other HuggingFace model)
    llm = PromptedLLM.from_name("gpt2", temperature=0.5, eos_tokens=[b"."])
    llm.set_prompt_from_str("Montreal is")

    # Run SMC using the sentiment analysis potential as a critic
    token_sampler = direct_token_sampler(llm)
    sequences = await token_sampler.smc(
        n_particles=5,
        max_tokens=25,
        ess_threshold=0.5,
        critic=sentiment_analysis.coerce(
            token_sampler.target, f=b"".join
        ).to_autobatched(),
    )

    print(sequences.decoded_posterior)


def test_sentiment_analysis():
    asyncio.run(sentiment_analysis())


async def entailment_potential():

    secret = "The sky is blue."
    critic = EntailmentPotential(privileged_text=secret)

    # Test the potential
    print("\nEntailment potential test:")
    print(
        "critic.prefix(b'The sky is blue.') =",
        await critic.prefix(b"The sky is blue."),
    )
    print(
        "critic.prefix(b'The sky is red.') =",
        await critic.prefix(b"The sky is red."),
    )

    llm = PromptedLLM.from_name(
        "meta-llama/Llama-3.2-1B-Instruct", temperature=0.5, eos_tokens=[b"."]
    )
    llm.set_prompt_from_str("What colour is the sky?")
    token_sampler = direct_token_sampler(llm)

    # 4. Run SMC using the NLI potential as a critic
    sequences = await token_sampler.smc(
        n_particles=50,
        max_tokens=25,
        ess_threshold=0.5,
        critic=critic.coerce(token_sampler.target, f=b"".join).to_autobatched(),
    )

    print("\n--- NLI Critic Generation ---")
    print(f"Prompt: '{llm.prompt}'")
    print(f"Avoiding: '{secret}'")
    print("\nGenerated sequences posterior:")
    print(sequences.decoded_posterior)


def test_entailment_potential():
    asyncio.run(entailment_potential())


if __name__ == "__main__":
    test_entailment_potential()
