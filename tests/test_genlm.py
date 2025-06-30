import asyncio
from genlm.control import PromptedLLM, JsonSchema, BoolFSA, AWRS  # type: ignore
import json


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


def test_json_schema():
    person_schema = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "enum": ["Alice", "Bob", "Charlie"],
                "description": "The name of the person",
            },
            "age": {
                "type": "integer",
                "minimum": 20,
                "maximum": 80,
                "description": "The age of the person",
            },
        },
    }

    book_schema = {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "minLength": 1,
                "description": "The title of the book",
            },
            "pages": {
                "type": "integer",
                "minimum": 1,
                "maximum": 2000,
                "description": "The number of pages in the book",
            },
            "genre": {
                "type": "string",
                "enum": ["fiction", "non-fiction", "mystery"],
                "description": "The genre of the book",
            },
        },
    }

    # Create a language model potential.
    # Since this task is harder, we use a larger model.
    llm = PromptedLLM.from_name(
        "meta-llama/Llama-3.2-1B-Instruct",
        eos_tokens=[b"<|eom_id|>", b"<|eot_id|>"],
        temperature=0.8,
    )

    # Set the prompt for the language model.
    # Since we are using an instruction-tuned model, we use the chat template.
    # The prompt contains an example of a schema and a generated object,
    # followed by the schema we want to match.
    llm.prompt_ids = llm.model.tokenizer.apply_chat_template(
        conversation=[
            {
                "role": "system",
                "content": "You need to generate a JSON object that matches the schema below. Only generate the JSON object on a single line with no other text.",
            },
            {"role": "user", "content": json.dumps(person_schema)},
            {"role": "assistant", "content": '{"name": "Alice", "age": 30}'},
            {"role": "user", "content": json.dumps(book_schema)},
        ],
        tokenize=True,
        add_generation_prompt=True,
    )

    # Create a schema potential.
    schema_potential = JsonSchema(book_schema)

    # Coerce the schema potential so that it operates on the token type of the language model.
    coerced_schema = schema_potential.coerce(llm, f=b"".join)

    # Create a token sampler that combines the language model and the schema potential.
    token_sampler = AWRS(llm, coerced_schema)

    # Generate text using SMC.
    # Generation is asynchronous; use `await` if calling in an async context (like in an async
    # function or in a Jupyter notebook) and `asyncio.run(token_sampler.smc(...))` otherwise.
    sequences = asyncio.run(
        token_sampler.smc(
            n_particles=2,  # Number of candidate sequences to maintain
            ess_threshold=0.5,  # Threshold for resampling
            max_tokens=30,  # Maximum sequence length
            verbosity=0,  # Print particles at each step
        )
    )

    # Show the inferred posterior distribution over complete UTF-8 decodable sequences.
    sequences.decoded_posterior
    # Example output:
    # {
    #   '{"title": "The Lord of the Rings", "pages": 1200, "genre": "fiction"}': 0.5008318164809697,
    #   '{"title": "The Great Gatsby", "pages": 178, "genre": "fiction"}': 0.49916818351903025,
    # }
