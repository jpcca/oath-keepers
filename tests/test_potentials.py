import torch
import asyncio
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
)
from genlm.control import Potential, PromptedLLM, direct_token_sampler  # type: ignore


# Create our own custom potential for sentiment analysis.
# Custom potentials must subclass `Potential` and implement the `prefix` and `complete` methods.
# They can also override other methods, like `batch_prefix`, and `batch_complete` for improved performance.
# Each Potential needs to specify its vocabulary of tokens; this potential has a vocabulary of individual bytes.
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


if __name__ == "__main__":
    test_sentiment_analysis()
