import asyncio
from genlm.control import PromptedLLM, BoolFSA, AWRS  # type: ignore


def test_regex():
    # Create a language model potential.
    llm = PromptedLLM.from_name("gpt2")
    llm.set_prompt_from_str("Here is my honest opinion:")

    # Create a finite-state automaton potential using a regular expression.
    fsa = BoolFSA.from_regex(r" SMC is (🔥🔥|😍😍|🤌🤌) with LMs")

    # Coerce the FSA so that it operates on the token type of the LLM.
    coerced_fsa = fsa.coerce(llm, f=b"".join)

    # Create a token sampler that combines the language model and FSA.
    token_sampler = AWRS(llm, coerced_fsa)

    # Generate text using SMC. Generation is asynchronous
    sequences = asyncio.run(
        token_sampler.smc(
            n_particles=10,  # Number of candidate sequences to maintain
            ess_threshold=0.5,  # Threshold for resampling
            max_tokens=30,  # Maximum sequence length
            verbosity=1,  # Print particles at each step
        )
    )

    sequences.decoded_posterior
    # Example output:
    # {
    #   ' SMC is 🔥🔥 with LMs': 1.0,
    # }
