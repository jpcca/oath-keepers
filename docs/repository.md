Directory structure:
└── genlm-genlm-control/
    ├── README.md
    ├── DEVELOPING.md
    ├── pyproject.toml
    ├── docs/
    │   ├── gen_reference_page.py
    │   ├── getting_started.md
    │   ├── index.md
    │   ├── performance.md
    │   ├── potentials.md
    │   ├── samplers.md
    │   └── javascripts/
    │       └── mathjax.js
    ├── genlm/
    │   └── control/
    │       ├── __init__.py
    │       ├── constant.py
    │       ├── typing.py
    │       ├── util.py
    │       ├── viz.py
    │       ├── html/
    │       │   └── smc.html
    │       ├── potential/
    │       │   ├── __init__.py
    │       │   ├── autobatch.py
    │       │   ├── base.py
    │       │   ├── coerce.py
    │       │   ├── multi_proc.py
    │       │   ├── operators.py
    │       │   ├── product.py
    │       │   ├── stateful.py
    │       │   ├── streaming.py
    │       │   ├── testing.py
    │       │   └── built_in/
    │       │       ├── __init__.py
    │       │       ├── canonical.py
    │       │       ├── json.py
    │       │       ├── llm.py
    │       │       ├── wcfg.py
    │       │       └── wfsa.py
    │       └── sampler/
    │           ├── __init__.py
    │           ├── sequence.py
    │           ├── set.py
    │           └── token.py
    ├── tests/
    │   ├── conftest.py
    │   ├── test_constant.py
    │   ├── test_setups.py
    │   ├── test_typing.py
    │   ├── test_util.py
    │   ├── test_viz.py
    │   ├── potential/
    │   │   ├── test_autobatch.py
    │   │   ├── test_base.py
    │   │   ├── test_canonical.py
    │   │   ├── test_coerce.py
    │   │   ├── test_json.py
    │   │   ├── test_llm.py
    │   │   ├── test_mp.py
    │   │   ├── test_operators.py
    │   │   ├── test_product.py
    │   │   ├── test_stateful.py
    │   │   ├── test_testing.py
    │   │   ├── test_wcfg.py
    │   │   └── test_wfsa.py
    │   └── sampler/
    │       ├── test_awrs.py
    │       ├── test_seq_sampler.py
    │       ├── test_sequences.py
    │       ├── test_set_sampler.py
    │       └── test_token_sampler.py


================================================
FILE: README.md
================================================
![Logo](logo.png)

<div align="center">

[![Docs](https://github.com/genlm/genlm-control/actions/workflows/docs.yml/badge.svg)](https://genlm.github.io/genlm-control/)
[![Tests](https://github.com/genlm/genlm-control/actions/workflows/pytest.yml/badge.svg)](https://genlm.github.io/genlm-control/)
[![codecov](https://codecov.io/github/genlm/genlm-control/graph/badge.svg?token=665ffkDXvZ)](https://codecov.io/github/genlm/genlm-control)
[![PyPI](https://img.shields.io/pypi/v/genlm-control?label=pypi)](https://pypi.org/project/genlm-control/)

</div>

GenLM Control is a library for controlled generation from language models using programmable constraints. It leverages sequential Monte Carlo (SMC) methods to efficiently generate text that satisfies constraints or preferences encoded by arbitrary potential functions.

See the [docs](https://genlm.github.io/genlm-control) for details.


## Quick Start

This library requires python>=3.11 and can be installed using pip:

```bash
pip install genlm-control
```

For faster and less error-prone installs, consider using [`uv`](https://github.com/astral-sh/uv):

```bash
uv pip install genlm-control
```

See [DEVELOPING.md](DEVELOPING.md) for details on how to install the project for development.

## Examples

**Note**: If you are running the examples below at the top-level in a regular Python script or REPL (as opposed to a Jupyter notebook), replace any `await token_sampler.smc(...)` calls with `asyncio.run(token_sampler.smc(...))`. See also the [Async primer](https://github.com/genlm/genlm-control?tab=readme-ov-file#async-primer) below for more details on running asynchronous functions.

### Controlling an LLM with a regular expression

This example demonstrates how to constrain an LLM using a regular expression.

```python
from genlm.control import PromptedLLM, BoolFSA, AWRS

# Create a language model potential.
llm = PromptedLLM.from_name("gpt2")
llm.set_prompt_from_str("Here is my honest opinion:")

# Create a finite-state automaton potential using a regular expression.
fsa = BoolFSA.from_regex(r" SMC is (🔥🔥|😍😍|🤌🤌) with LMs")

# Coerce the FSA so that it operates on the token type of the language model.
coerced_fsa = fsa.coerce(llm, f=b"".join)

# Create a token sampler that combines the language model and FSA.
token_sampler = AWRS(llm, coerced_fsa)

# Generate text using SMC.
# Generation is asynchronous; use `await` if calling in an async context (like in an async
# function or in a Jupyter notebook) and `asyncio.run(token_sampler.smc(...))` otherwise.
sequences = await token_sampler.smc(
    n_particles=10, # Number of candidate sequences to maintain
    ess_threshold=0.5, # Threshold for resampling
    max_tokens=30, # Maximum sequence length
    verbosity=1 # Print particles at each step
)

sequences.decoded_posterior
# Example output:
# {
#   ' SMC is 🔥🔥 with LMs': 1.0,
# }
```

### Controlling an LLM with a JSON schema

This example demonstrates how to control an LLM to generate JSON objects that match a given schema.

```python
import json
from genlm.control import PromptedLLM, JsonSchema, AWRS

person_schema = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "enum": ["Alice", "Bob", "Charlie"],
            "description": "The name of the person"
        },
        "age": {
            "type": "integer",
            "minimum": 20,
            "maximum": 80,
            "description": "The age of the person"
        },
    },
}

book_schema = {
    "type": "object",
    "properties": {
        "title": {
            "type": "string",
            "minLength": 1,
            "description": "The title of the book"
        },
        "pages": {
            "type": "integer",
            "minimum": 1,
            "maximum": 2000,
            "description": "The number of pages in the book"
        },
        "genre": {
            "type": "string",
            "enum": ["fiction", "non-fiction", "mystery"],
            "description": "The genre of the book"
        }
    },
}

# Create a language model potential.
# Since this task is harder, we use a larger model.
# (You will need to login via the Hugging Face CLI and have access to the model.)
llm = PromptedLLM.from_name(
    "meta-llama/Llama-3.2-1B-Instruct",
    eos_tokens=[b"<|eom_id|>", b"<|eot_id|>"],
    temperature=0.8
)

# Set the prompt for the language model.
# Since we are using an instruction-tuned model, we use the chat template.
# The prompt contains an example of a schema and a generated object,
# followed by the schema we want to match.
llm.prompt_ids = llm.model.tokenizer.apply_chat_template(
    conversation=[
        {"role": "system", "content": "You need to generate a JSON object that matches the schema below. Only generate the JSON object on a single line with no other text."},
        {"role": "user", "content": json.dumps(person_schema)},
        {"role": "assistant", "content": '{"name": "Alice", "age": 30}'},
        {"role": "user", "content": json.dumps(book_schema)},
    ],
    tokenize=True,
    add_generation_prompt=True
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
sequences = await token_sampler.smc(
    n_particles=2, # Number of candidate sequences to maintain
    ess_threshold=0.5, # Threshold for resampling
    max_tokens=30, # Maximum sequence length
    verbosity=1 # Print particles at each step
)

# Show the inferred posterior distribution over complete UTF-8 decodable sequences.
sequences.decoded_posterior
# Example output:
# {
#   '{"title": "The Lord of the Rings", "pages": 1200, "genre": "fiction"}': 0.5008318164809697,
#   '{"title": "The Great Gatsby", "pages": 178, "genre": "fiction"}': 0.49916818351903025,
# }
```

### Async primer

`genlm-control` makes use of asynchronous programming; the sampling method `token_sampler.smc(...)` in the examples below returns a coroutine that must be awaited.

If you're running code inside an `async def` function or in a Jupyter notebook (which supports top-level await), you can use `await` directly:
    
```python
sequences = await token_sampler.smc(...)
```

If you're writing a regular Python script (e.g., a .py file), you can't use `await` at the top level. In that case, wrap the call with `asyncio.run(...)` to run it inside an event loop:
    
```python
import asyncio
sequences = asyncio.run(token_sampler.smc(...))
```
This distinction is important so your code doesn't raise a `SyntaxError` (if you use `await` at the top level) or `RuntimeError` (if you call `asyncio.run()` from inside an already-running event loop).

### More examples

See the [docs](https://genlm.github.io/genlm-control/getting_started) for more examples.

## Development

See [DEVELOPING.md](DEVELOPING.md) for details on how to install the project locally.



================================================
FILE: DEVELOPING.md
================================================
# Developer's Guide

This guide describes how to complete various tasks you'll encounter when working
on the `genlm-control` codebase.

## Requirements

- Python >= 3.11
- The core dependencies listed in the `pyproject.toml` file of the repository.

## Installation

Clone the repository:
```bash
git clone git@github.com:genlm/genlm-control.git
cd genlm-control
```
and install with pip:

```bash
pip install -e ".[test,docs]"
```

This installs the dependencies needed for testing (test) and documentation (docs).

For faster and less error-prone installs, consider using [`uv`](https://github.com/astral-sh/uv):

```bash
uv pip install -e ".[test,docs]"
```

It is also recommended to use a dedicated environment.

With conda:
```bash
conda create -n genlm python=3.11
conda activate genlm
uv pip install -e ".[test,docs]"
```

With uv:
```bash
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[test,docs]"
```


## Testing

When test dependencies are installed, the test suite can be run via:

```bash
pytest tests
```

## Documentation

Documentation is generated using [mkdocs](https://www.mkdocs.org/) and hosted on GitHub Pages. To build the documentation, run:

```bash
mkdocs build
```

To serve the documentation locally, run:

```bash
mkdocs serve
```

## Commit Hooks

We use [pre-commit](https://pre-commit.com/) to manage a series of git
pre-commit hooks for the project; for example, each time you commit code, the
hooks will make sure that your python is formatted properly. If your code isn't,
the hook will format it, so when you try to commit the second time you'll get
past the hook.

All hooks are defined in `.pre-commit-config.yaml`. To install these hooks,
install `pre-commit` if you don't yet have it. I prefer using
[pipx](https://github.com/pipxproject/pipx) so that `pre-commit` stays globally
available.

```bash
pipx install pre-commit
```

Then install the hooks with this command:

```bash
pre-commit install
```

Now they'll run on every commit. If you want to run them manually, run the
following command:

```bash
pre-commit run --all-files
```

================================================
FILE: pyproject.toml
================================================
[build-system]
requires = ["setuptools>=64.0", "setuptools-scm>=8"]
build-backend = "setuptools.build_meta"

[tool.setuptools_scm]

[project]
name = "genlm-control"
dynamic = ["version"]
description = "Controlled generation from LMs using programmable constraints"
readme = "README.md"
requires-python = ">=3.11"
authors = [
    { name = "Ben LeBrun", email = "benlebrun1@gmail.com" },
    { name = "The GenLM Team" },
]
dependencies = [
    "genlm-grammar>=0.2.0",
    "genlm-backend>=0.1.1",
    "llamppl",
    "arsenal>=3.1.3",
    "IPython",
    "numpy",
    "torch",
    "json-stream",
    "jsonschema[format-nongpl]",
]

[project.optional-dependencies]
test = [
    "coverage",
    "pytest",
    "pytest-benchmark",
    "pytest-asyncio",
    "pytest-timeout",
    "pytest-cov",
    "pytest-mock",
    "hypothesis==6.130.13",
    "hypothesis-jsonschema",
]
docs = [
    "mkdocs",
    "mkdocstrings[python]",
    "mkdocs-material",
    "mkdocs-gen-files",
    "mkdocs-literate-nav",
    "mkdocs-section-index",
]

[tool.setuptools.packages.find]
include = ["genlm", "genlm/control"]

[tool.setuptools]
package-data = { "genlm.control" = ["html/*"] }


================================================
FILE: docs/gen_reference_page.py
================================================
from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

for path in sorted(Path("genlm/control").rglob("*.py")):
    if any(part.startswith(".") for part in path.parts):
        continue

    module_path = path.relative_to(".").with_suffix("")
    doc_path = path.relative_to(".").with_suffix(".md")
    full_doc_path = Path("reference", doc_path)

    parts = tuple(module_path.parts)

    if parts[-1] == "__init__":
        print(f"init, making parts {parts[:-1]}")
        parts = parts[:-1]
    elif parts[-1] == "__main__":
        continue

    nav[parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        ident = ".".join(parts)
        fd.write(f"::: {ident}")

    mkdocs_gen_files.set_edit_path(full_doc_path, path)

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())



================================================
FILE: docs/getting_started.md
================================================
# Getting Started with GenLM Control

This example demonstrates how to use the `genlm-control` library, starting with basic usage and building up to more complex scenarios. It's a good starting point for understanding how to build increasingly complex genlm-control programs, even though the actual example is somewhat contrived.

## Basic LLM Sampling

First, let's look at basic language model sampling using a [`PromptedLLM`][genlm.control.PromptedLLM]:

```python
from genlm.control import PromptedLLM, direct_token_sampler

# Load gpt2 (or any other HuggingFace model)
mtl_llm = PromptedLLM.from_name("gpt2", temperature=0.5, eos_tokens=[b'.'])

# Set the fixed prompt prefix for the language model
# All language model predictions will be conditioned on this prompt
mtl_llm.set_prompt_from_str("Montreal is")

# Load a sampler that proposes tokens by sampling directly from the LM's distribution
token_sampler = direct_token_sampler(mtl_llm)

# Run SMC with 5 particles, a maximum of 25 tokens, and an ESS threshold of 0.5
sequences = await token_sampler.smc(n_particles=5, max_tokens=25, ess_threshold=0.5)

# Show the posterior over token sequences
sequences.posterior

# Show the posterior over complete UTF-8 decodable sequences
sequences.decoded_posterior
```

Note: Sequences are lists of `bytes` objects because each token in the language model's vocabulary is represented as a bytes object.

## Prompt Intersection

Next, we'll look at combining prompts from multiple language models using a [`Product`][genlm.control.potential.Product] potential:

```python
# Spawn a new language model (shallow copy, sharing the same underlying model)
bos_llm = mtl_llm.spawn()
bos_llm.set_prompt_from_str("Boston is")

# Take the product of the two language models
# This defines a `Product` potential which is the element-wise product of the two LMs
product = mtl_llm * bos_llm

# Create a sampler that proposes tokens by sampling directly from the product
token_sampler = direct_token_sampler(product)

sequences = await token_sampler.smc(n_particles=5, max_tokens=25, ess_threshold=0.5)

sequences.posterior

sequences.decoded_posterior
```

## Adding Regex Constraints

We can add regex constraints to our `product` using a [`BoolFSA`][genlm.control.potential.built_in.wfsa.BoolFSA] and the [`AWRS`][genlm.control.sampler.token.AWRS] token sampler:

```python
from genlm.control import BoolFSA, AWRS

# Create a regex constraint that matches sequences containing the word "the"
# followed by either "best" or "worst" and then anything else
best_fsa = BoolFSA.from_regex(r"\sthe\s(best|worst).*")

# BoolFSA's are defined over individual bytes by default
# Their `prefix` and `complete` methods are called on byte sequences
print("best_fsa.prefix(b'the bes') =", await best_fsa.prefix(b"the bes"))
print(
    "best_fsa.complete(b'the best city') =",
    await best_fsa.complete(b"the best city"),
)

# Coerce the FSA to work with the LLM's vocabulary
coerced_fsa = best_fsa.coerce(product, f=b"".join)

# Use the AWRS token sampler; it will only call the fsa on a subset of the product vocabulary
token_sampler = AWRS(product, coerced_fsa)

sequences = await token_sampler.smc(n_particles=5, max_tokens=25, ess_threshold=0.5)

sequences.posterior

sequences.decoded_posterior
```

## Custom Sentiment Analysis Potential

Now we'll create a custom potential by subclassing [`Potential`][genlm.control.potential.base.Potential] and use it as a **critic** to further guide generation:

```python
import torch
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
)
from genlm.control import Potential

# Create our own custom potential for sentiment analysis.
# Custom potentials must subclass `Potential` and implement the `prefix` and `complete` methods.
# They can also override other methods, like `batch_prefix`, and `batch_complete` for improved performance.
# Each Potential needs to specify its vocabulary of tokens; this potential has a vocabulary of individual bytes.
class SentimentAnalysis(Potential):
    def __init__(self, model, tokenizer, sentiment="POSITIVE"):
        self.model = model
        self.tokenizer = tokenizer

        self.sentiment_idx = model.config.label2id.get(sentiment, None)
        if self.sentiment_idx is None:
            raise ValueError(f"Sentiment {sentiment} not found in model labels")

        super().__init__(vocabulary=list(range(256)))  # Defined over bytes

    def _forward(self, contexts):
        strings = [bytes(context).decode("utf-8", errors="ignore") for context in contexts]
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

# Initialize sentiment analysis potential
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
sentiment_analysis = SentimentAnalysis(
    model=DistilBertForSequenceClassification.from_pretrained(model_name),
    tokenizer=DistilBertTokenizer.from_pretrained(model_name),
    sentiment="POSITIVE",
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

# Verify the potential satisfies required properties
await sentiment_analysis.assert_logw_next_consistency(b"the best", top=5)
await sentiment_analysis.assert_autoreg_fact(b"the best")

# Set up efficient sampling with the sentiment analysis potential
token_sampler = AWRS(product, coerced_fsa)
critic = sentiment_analysis.coerce(token_sampler.target, f=b"".join)

# Run SMC using the sentiment analysis potential as a critic
sequences = await token_sampler.smc(
    n_particles=5,
    max_tokens=25,
    ess_threshold=0.5,
    critic=critic, # Pass the critic to the SMC sampler; this will reweight samples at each step based on their positivity
)

# Show the posterior over complete UTF-8 decodable sequences
sequences.decoded_posterior
```

## Optimizing with Autobatching

Finally, we can optimize performance using autobatching. During generation, all requests to the sentiment analysis potential are made to the instance methods (`prefix`, `complete`). We can take advantage of the fact that we have parallelized batch versions of these methods using the [`to_autobatched`][genlm.control.potential.operators.PotentialOps.to_autobatched] method.

```python
from arsenal.timer import timeit

# Create an autobatched version of the critic
# This creates a new potential that automatically batches concurrent
# requests to the instance methods (`prefix`, `complete`, `logw_next`)
# and processes them using the batch methods (`batch_complete`, `batch_prefix`, `batch_logw_next`).
autobatched_critic = critic.to_autobatched()

# Run SMC with timing for comparison
with timeit("Timing sentiment-guided sampling with autobatching"):
    sequences = await token_sampler.smc(
        n_particles=10,
        max_tokens=25,
        ess_threshold=0.5,
        critic=autobatched_critic, # Pass the autobatched critic to the SMC sampler
    )

sequences.decoded_posterior

# The autobatched version should be significantly faster than this version
with timeit("Timing sentiment-guided sampling without autobatching"):
    sequences = await token_sampler.smc(
        n_particles=10,
        max_tokens=25,
        ess_threshold=0.5,
        critic=critic,
    )

sequences.decoded_posterior
```



================================================
FILE: docs/index.md
================================================
![Logo](logo.png)

[![Docs](https://github.com/genlm/genlm-control/actions/workflows/docs.yml/badge.svg)](https://genlm.github.io/genlm-control/)
[![Tests](https://github.com/genlm/genlm-control/actions/workflows/pytest.yml/badge.svg)](https://genlm.github.io/genlm-control/)
[![codecov](https://codecov.io/github/genlm/genlm-control/graph/badge.svg?token=665ffkDXvZ)](https://codecov.io/github/genlm/genlm-control)

GenLM Control is a library for controlled generation from language models using programmable constraints. It leverages sequential Monte Carlo (SMC) methods to efficiently generate text that satisfies constraints or preferences encoded by arbitrary potential functions.

## Quick Start

This library can be installed using pip:

```bash
pip install genlm-control
```

See [DEVELOPING.md](https://github.com/genlm/genlm-control/tree/main/DEVELOPING.md) for details on how to install the project for development.

## Examples

### Controlling an LLM with a regular expression

This example demonstrates how to constrain an LLM using a regular expression.

```python
from genlm.control import PromptedLLM, BoolFSA, AWRS

# Create a language model potential.
llm = PromptedLLM.from_name("gpt2")
llm.set_prompt_from_str("Here is my honest opinion:")

# Create a finite-state automaton potential using a regular expression.
fsa = BoolFSA.from_regex(r" SMC is (🔥🔥|😍😍|🤌🤌) with LMs")

# Coerce the FSA so that it operates on the token type of the language model.
coerced_fsa = fsa.coerce(llm, f=b"".join)

# Create a token sampler that combines the language model and FSA.
token_sampler = AWRS(llm, coerced_fsa)

# Generate text using SMC.
# Generation is asynchronous; use `await` if calling in an async context (like in an async
# function or in a Jupyter notebook) and `asyncio.run(token_sampler.smc(...))` otherwise.
sequences = await token_sampler.smc(
    n_particles=10, # Number of candidate sequences to maintain
    ess_threshold=0.5, # Threshold for resampling
    max_tokens=30, # Maximum sequence length
    verbosity=1 # Print particles at each step
)

sequences.decoded_posterior
# Example output:
# {
#   ' SMC is 🔥🔥 with LMs': 1.0,
# }
```

### Controlling an LLM with a JSON schema

This example demonstrates how to control an LLM to generate JSON objects that match a given schema.

```python
import json
from genlm.control import PromptedLLM, JsonSchema, AWRS

person_schema = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "enum": ["Alice", "Bob", "Charlie"],
            "description": "The name of the person"
        },
        "age": {
            "type": "integer",
            "minimum": 20,
            "maximum": 80,
            "description": "The age of the person"
        },
    },
}

book_schema = {
    "type": "object",
    "properties": {
        "title": {
            "type": "string",
            "minLength": 1,
            "description": "The title of the book"
        },
        "pages": {
            "type": "integer",
            "minimum": 1,
            "maximum": 2000,
            "description": "The number of pages in the book"
        },
        "genre": {
            "type": "string",
            "enum": ["fiction", "non-fiction", "mystery"],
            "description": "The genre of the book"
        }
    },
}

# Create a language model potential.
# Since this task is harder, we use a larger model.
# (You will need to login via the Hugging Face CLI and have access to the model.)
llm = PromptedLLM.from_name(
    "meta-llama/Llama-3.2-1B-Instruct",
    eos_tokens=[b"<|eom_id|>", b"<|eot_id|>"],
    temperature=0.8
)

# Set the prompt for the language model.
# Since we are using an instruction-tuned model, we use the chat template.
# The prompt contains an example of a schema and a generated object,
# followed by the schema we want to match.
llm.prompt_ids = llm.model.tokenizer.apply_chat_template(
    conversation=[
        {"role": "system", "content": "You need to generate a JSON object that matches the schema below. Only generate the JSON object on a single line with no other text."},
        {"role": "user", "content": json.dumps(person_schema)},
        {"role": "assistant", "content": '{"name": "Alice", "age": 30}'},
        {"role": "user", "content": json.dumps(book_schema)},
    ],
    tokenize=True,
    add_generation_prompt=True
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
sequences = await token_sampler.smc(
    n_particles=2, # Number of candidate sequences to maintain
    ess_threshold=0.5, # Threshold for resampling
    max_tokens=30, # Maximum sequence length
    verbosity=1 # Print particles at each step
)

# Show the inferred posterior distribution over complete UTF-8 decodable sequences.
sequences.decoded_posterior
# Example output:
# {
#   '{"title": "The Lord of the Rings", "pages": 1200, "genre": "fiction"}': 0.5008318164809697,
#   '{"title": "The Great Gatsby", "pages": 178, "genre": "fiction"}': 0.49916818351903025,
# }
```

### More examples

See [getting_started.md](getting_started.md) to get an overview of the full range of features, including how to specify custom potential functions.

## Development

See [DEVELOPING.md](https://github.com/genlm/genlm-control/tree/main/DEVELOPING.md) for details on how to install the project locally.



================================================
FILE: docs/performance.md
================================================
# Performance Optimizations

The `genlm-control` library offers two key performance optimizations for instances of the `Potential` class:

- **Autobatching**: Automatically batches concurrent requests to the potential's instances methods
- **Multiprocessing**: Runs multiple instances of a `Potential` in parallel across CPU cores


## Auto-batching

Auto-batching improves performance when a `Potential` class's batch methods (`batch_complete`,  `batch_prefix`, `batch_logw_next`, `batch_score`) are more efficient than sequentially running individual instance methods.

### Usage

To enable auto-batching, use the `to_autobatched()` method:

```python
autobatched_potential = potential.to_autobatched()
# Use it exactly like a regular potential - batching happens automatically
results = await asyncio.gather(
    *(autobatched.complete(seq) for seq in sequences) # These will batched and processed by batch_complete
)
```

This creates a new potential that is a wrapper ([`AutoBatchedPotential`][genlm.control.potential.autobatch]) around the original potential. The wrapper automatically collects concurrent requests in the background and processes them together using the potential's batch methods. This happens transparently without requiring changes to your code structure.

## Multiprocessing

CPU parallelization can significantly improve performance for compute-intensive `Potential` classes. This is particularly useful when methods like `complete`, `prefix`, or `logw_next` involve heavy computation.

### Usage

To enable multiprocessing, use the `to_multiprocess()` method:

```python
# Create a multiprocess wrapper with desired number of workers
mp_potential = potential.to_multiprocess(num_workers=2)
# Use it like a regular potential - requests are distributed across workers
results = await asyncio.gather(
    *(mp_potential.complete(seq) for seq in sequences) # These will be distributed across workers
)
```

This creates a new potential that is a wrapper ([`MultiProcPotential`][genlm.control.potential.multi_proc]) around the original potential. The wrapper asynchronously distributes requests across multiple processes (in a non-blocking manner). This allows you to scale your computations across multiple cores without changing your code structure.

### Requirements

For multiprocessing to work, the potential must implement a picklable `spawn()` method that creates a new instance of the potential. Only some built-in `Potential` classes support this by default. Custom potentials need to implement their own `spawn()` method.

### Performance Benefits

Multiprocessing improves performance for both batched methods (`batch_complete`, `batch_prefix`, `batch_logw_next`) and unbatched methods (`complete`, `prefix`, `logw_next`).

In the batched case, requests within a batch are processed in parallel across workers. For individual method calls, requests are distributed to available worker processes and are executed asynchronously.

## When to use each optimization

> **Note:** Built-in `Potential` classes that can benefit from auto-batching support (e.g., `PromptedLLM`) will have auto-batching enabled by default.

- Use auto-batching when the potential's batch operations are more efficient than sequential operations
- Use multiprocessing when the potential's operations are compute-intensive and can benefit from parallel processing
- Consider the overhead of each optimization when deciding which to use. Multiprocessing in particular incurs a significant overhead when the potential's operations are not compute-intensive.



================================================
FILE: docs/potentials.md
================================================
# Potentials

[Potentials][genlm.control.potential] are the core object in `genlm-control`. A potential encodes constraints or preferences by assigning non-negative weights to sequences of tokens.

Potentials guide text generation by:

* Acting as components of [**samplers**](samplers.md), which serve to propose new tokens at each step of the generation process.
* Serving as **critics**, which serve to reweight sequences based on whether they satisfy the constraint encoded by the potential at each step of the generation process.

## Key concepts

### Vocabulary

Each potential has a **vocabulary** which defines the set of tokens it operates on. Most built-in potentials operate on vocabularies whose tokens are `bytes` or `int` objects (the latter often representing individual bytes).

### Weight assignment

Potentials assign weights to sequences of tokens from their vocabulary. These weights are always non-negative real numbers, though they are computed in log space for numerical stability.

A potential defines two core weighting functions:

1. `complete` - Assigns weights to sequences that are considered "finished" or "complete". For example, a potential enforcing grammatical correctness would assign positive weights to grammatically valid sentences and zero weights (negative infinity in log space) to invalid ones.

2. `prefix` - Assigns weights to partial sequences that could potentially be extended into valid complete sequences. For example, a potential enforcing grammatical correctness would assign positive weights to prefixes of grammatically valid sequences.

    Given a complete method, there are many possible prefix methods that could be used, providing as much or as little information as desired. The key requirement is that if a prefix has zero weight, then all of its extensions and completions must also have zero weight - in other words, prefix cannot rule out sequences that could later become valid.

The relationship between complete and prefix weights is formalized in the [Formalization](#formalization) section.

### Next-token weights

Potentials also implement a `logw_next` method, which computes weights for each possible next token in the potential's vocabulary (and a reserved end-of-sequence token) given a context sequence. These weights are crucial for controlled text generation as they can be used to guide the selection of the next token at each step.

The `logw_next` method is implemented by default in terms of the `complete` and `prefix` methods. Potentials will often override this method to provide a more efficient implementation. However, `logw_next` must satisfy a contract with `complete`/`prefix`, specified in [Formalization](#formalization).

### Batch methods

For improved performance with large batches of inputs, potentials support batch operations:

* `batch_complete(contexts)`
* `batch_prefix(contexts)`
* `batch_logw_next(contexts)`
* `batch_score(contexts)`

By default, these methods simply call the corresponding non-batch method for all inputs, but potentials can override them to provide more efficient implementations. They can be used in conjunction with [auto batching](performance.md#auto-batching) for improved performance during generation.

## Built-in potentials

`genlm-control` comes with a number of built-in potentials that can be used in controlled text generation.

### Language models

[`PromptedLLM`][genlm.control.potential.built_in.llm.PromptedLLM] represents a language model conditioned on a fixed prompt prefix.

```python
# Load GPT-2 with temperature 0.5
llm = PromptedLLM.from_name("gpt2", temperature=0.5)

# Set a prompt prefix that all generations will be conditioned on
llm.set_prompt_from_str("Montreal is")
```

`PromptedLLM`s have a vocabulary of `bytes` tokens, obtained from the language model's tokenizer.

### Finite-state automata

`genlm-control` provides two [FSA implementations][genlm.control.potential.built_in.wfsa]:

1. `WFSA` (Weighted Finite-State Automata) - For weighted constraints:
```python
# Create a WFSA from a regex pattern
# Transitions are automatically normalized to form probability distributions
wfsa = WFSA.from_regex(r"\sthe\s(best|worst).*😎")
```

2. `BoolFSA` (Boolean Finite-State Automata) - For hard constraints:
```python
# Create a boolean FSA from a regex pattern
# Transitions are binary (0 or -inf in log space)
fsa = BoolFSA.from_regex(r"\sthe\s(best|worst).*😎")
```

Both FSAs:

* Support regex patterns with standard syntax
* Operate on byte-level sequences by default
* Can be combined with other potentials via products

### Context-free grammars

Similar to FSAs, `genlm-control` provides two [CFG implementations][genlm.control.potential.built_in.wcfg]:

1. `WCFG` (Weighted Context-Free Grammar).
```python
cfg = WCFG.from_string("""
    1.0: S -> NP VP
    0.5: NP -> the N
    0.5: NP -> a N
    1.0: VP -> V NP
    0.5: N -> cat
    0.5: N -> dog
    0.5: V -> saw
    0.5: V -> chased
""")
```

2. `BoolCFG` (Boolean Context-Free Grammar).
```python
# Create a boolean CFG from a Lark grammar string
cfg = BoolCFG.from_lark("""
    start: np vp
    np: ("the" | "a") WS n
    vp: WS v WS np
    n: "cat" | "dog"
    v: "saw" | "chased"
    %import common.WS
""")
```

`BoolCFG`s support grammar specification via [Lark syntax](https://lark-parser.readthedocs.io/en/latest/grammar.html).

Both CFGs:

* Use Earley parsing for efficient recognition
* Can be combined with other potentials
* Operate on byte-level sequences by default

> **Note:** It is recommended to specify grammars via lark syntax. The `from_string` method is provided for convenience, but it is not as flexible and robust.

## Custom potentials

You can create custom potentials to implement specialized constraints or preferences that aren't covered by the built-in options.

### Creating a custom potential

To define a custom potential:

1. Create a subclass of `Potential`
2. Implement the `complete` and `prefix` methods
3. Optionally override `logw_next` and the batch methods for performance optimization

When implementing custom potentials, the key is understanding the relationship between `complete` and `prefix`. Consider the following example of a potential that only allows sequences of a given length:

```python
class LengthPotential(Potential):
    """ A potential that only allows sequences of a given length. """
    def __init__(self, vocabulary, length):
        # Initialize the superclass with the potential's vocabulary.
        super().__init__(vocabulary)
        self.length = length

    async def complete(self, context):
        # Note: 0.0 = log(1.0) and float('-inf') = log(0.0)
        return 0.0 if len(context) == self.length else float('-inf')

    async def prefix(self, context):
        # Note: 0.0 = log(1.0) and float('-inf') = log(0.0)
        return 0.0 if len(context) <= self.length else float('-inf')

length_potential = LengthPotential(vocabulary=[b'the', b'a', b'cat', b'dog', b'saw', b'chased'], length=5)
```

This example illustrates the key difference between `complete` and `prefix`: the `complete` method only allows sequences of exactly the target length, while the `prefix` method allows any sequence that could potentially reach the target length (i.e., any sequence not exceeding the target length).

### Common pitfalls

When implementing custom potentials, be aware of these common issues:

1. **Inconsistent complete/prefix relationship** - If your `prefix` method assigns zero weight to a sequence, all extensions must also have zero weight.

2. **Inefficient implementations** - For complex potentials, consider overriding `logw_next` with a more efficient implementation than the default.

3. **Not handling async properly** - All potential methods are asynchronous. Make sure to use `await` when calling them and define your methods with `async def`.

### Testing your custom potential

Potentials automatically inherit from the [`PotentialTests`][genlm.control.potential.testing] mixin, which provides a number of tests for validating the correctness of the potential's implementation.

```python
# These will raise an exception if the potential implementation does not satisfy the properties
await potential.assert_logw_next_consistency(context)
await potential.assert_autoreg_fact(context)
await potential.assert_batch_consistency(contexts)
```

## Complex usage

### Products of potentials

The [`Product`][genlm.control.potential.product] class allows you to combine two potentials. A `Product` is itself is a potential, meaning that it implements all potential methods and that it is possible to chain products to combine more than two potentials.

```python
# Example: Prompt intersection
mtl_llm = PromptedLLM.from_name("gpt2")
mtl_llm.set_prompt_from_str("Montreal is")

bos_llm = mtl_llm.spawn()
bos_llm.set_prompt_from_str("Boston is")

# Create product using multiplication operator
product = mtl_llm * bos_llm
```

The product potential operates on the intersection of the two potentials' vocabularies. For a product potential:

- The vocabulary $\A$ is the intersection of the two potentials' vocabularies: $\A = \A_1 \cap \A_2$.
- The prefix potential $\prefix$ is the product (sum in log space) of the individual prefix potentials: $\log \prefix(\xx) = \log \prefix_1(\xx) + \log \prefix_2(\xx)$.
- The complete potential $\complete$ is the product (sum in log space) of the individual complete potentials: $\log \complete(\xx) = \log \complete_1(\xx) + \log \complete_2(\xx)$.
- The next-token potential $\pot(\cdot \mid \xx)$ is the product (sum in log space) of the individual next-token potentials: $\log \pot(x \mid \xx) = \log \pot_1(x \mid \xx) + \log \pot_2(x \mid \xx)$ for $x \in (\A_1 \cap \A_2) \cup \{\eos\}$

> **Warning:** Be careful when taking products of potentials with minimal vocabulary overlap, as the resulting potential will only operate on tokens present in both vocabularies. A warning will be raised if the vocabulary overlap is less than 10% of either potential's vocabulary.


### Coerced potentials

The [`Coerced`][genlm.control.potential.coerce] class allows you to adapt a potential to work with a different vocabulary using a coercion function. The coercion function must map between sequences in the new vocabulary and sequences in the potential's original vocabulary. This is particularly useful when combining potentials that operate on different types of tokens.

```python
# Example: Coercing a byte-level FSA to work with a language model's tokens
fsa = BoolFSA.from_regex(r"\sthe\s(best|worst).*")  # Works on bytes
llm = PromptedLLM.from_name("gpt2")  # Works on byte sequences

# Coerce the FSA to work with the LLM's tokens by joining tokens into bytes
coerced_fsa = fsa.coerce(llm, f=b''.join)

# Now we can combine them using the product operator!
product = llm * coerced_fsa
```

Common use cases for coercion include:

- Adapting byte-level constraints (like FSAs) to work with token-level language models (which have vocabularies of byte *sequences*)
- Implementing constraints that operate on processed versions of the tokens (e.g., lowercase text)
- Converting between different tokenization schemes

> **Performance Note:** The coercion operation can impact performance, especially when mapping from a coarser token type to a finer token type (e.g., byte sequences to individual bytes). To sample tokens from a coerced product, consider using specialized samplers (e.g., `eager_token_sampler`, `topk_token_sampler`).

### Performance optimizations

`genlm-control` provides a number of performance optimizations for potentials, described in the [performance](performance.md) section.


## Formalization

This section provides a formal definition of potentials and the relationships between their complete, prefix, and next-token potentials.

**Notation** Let $\A$ be a vocabulary of tokens and $\eos$ a specialized end-of-sequence token. Let $\A^*$ denote the set of all sequences of tokens which can be built from $\A$ (including the empty sequence $\epsilon$) and $\A^*{\eos} = \{\xx\eos : \xx \in \A^*\}$ the set of $\eos$-terminated sequences. We refer to $\A^*$ as the set of *prefix* sequences and $\A^*{\eos}$ the set of *complete* sequences.

A potential $\pot$ is a function $\pot: \A^* \cup\A^*{\eos} \rightarrow \mathbb{R}_{\geq 0}$ which assigns a non-negative real number to prefix and complete sequences from its vocabulary $\A$:

$$
\pot(\xx) = \begin{cases}
    \prefix(\xx) & \text{if } \xx \in \A^* \\
    \complete(\yy) & \text{if } \xx = \yy\eos, \yy \in \A^*
\end{cases}
$$

where

* $\prefix : \A^* \rightarrow \mathbb{R}_{\geq 0}$ is the **prefix potential**
* $\complete : \A^* \rightarrow \mathbb{R}_{\geq 0}$ is the **complete potential**

The complete and prefix potentials are related by the following equality:

$$
\prefix(\xx) = 0 \implies \complete(\xx\yy) = 0 \, \forall \xx,\yy \text{ such that } \xx\yy \in \A^*
$$

Intuitively, this means that the prefix potential cannot rule out a sequence which can later on turn out to be valid according to the complete potential.

Finally, we define the **next-token weights function** $\pot(x \mid \xx) : \A \cup \{\eos\} \rightarrow \mathbb{R}_{\geq 0}$, which assigns a non-negative real number to each token $x \in \A \cup \{\eos\}$ given a sequence $\xx \in \A^*$:

$$
\pot(x \mid \xx) = \frac{\pot(\xx x)}{\prefix(\xx)} = \begin{cases}
    \frac{\prefix(\xx x)}{\prefix(\xx)} & \text{if } x \in \A \\
    \frac{\complete(\xx)}{\prefix(\xx)} & \text{if } x = \eos
\end{cases}
$$

$\pot(\cdot \mid \xx)$ is related to the complete and prefix potentials according to the following autoregressive factorization:

$$
\frac{\complete(\xx)}{\prefix(\epsilon)} = \pot(\eos \mid \xx) \prod_{x \in \xx} \pot(x \mid \xx)
$$

### Correspondance with the `Potential` class

Each of the quantities above directly corresponds to a method or attribute of the `Potential` class:

| Method/Attribute | Mathematical Quantity | Description |
|-----------------|----------------------|-------------|
| `vocab` | $\A$ | The vocabulary of the potential. |
| `eos` | $\eos$ | The end-of-sequence token. |
| `vocab_eos` | $\A \cup \{\eos\}$ | The vocabulary of the potential including the end-of-sequence token. |
| `complete(self, context)` | $\log \complete(\xx)$ | The complete potential for a given sequence. |
| `prefix(self, context)` | $\log \prefix(\xx)$ | The prefix potential for a given sequence. |
| `logw_next(self, context)` | $\log \pot(\cdot \mid \xx)$ | The next-token potential for a given prefix sequence. |
| `score(self, context)` | $\log \pot(\xx)$ | The potential, dispatching to `complete` for eos-terminated sequences and `prefix` otherwise. |



================================================
FILE: docs/samplers.md
================================================
# Token Samplers

[TokenSamplers][genlm.control.sampler.token.TokenSampler]  are the objects that propose new tokens during generation. They generate individual tokens $x$ given a `context` sequence. Each sample $x$ is attached with a log importance weight $w$.[^1]

[^1]: Tokens samplers also return a log-probability which corresponds to the log-probability of all the random choices made by the sampler. It is returned for testing purposes and is not used during generation.

## Direct Token Sampling

The simplest token sampler is the [`DirectTokenSampler`][genlm.control.sampler.token.DirectTokenSampler], which samples directly from the normalized version of a potential's `logw_next` method:

```python
# Create a direct token sampler for a potential
sampler = DirectTokenSampler(potential)

# Sample a token
token, logw, logp = await sampler.sample(context)
```

`DirectTokenSampler` is efficient when the potential's `logw_next` method is efficient (e.g., for language models). However, for potentials with large vocabularies or expensive `logw_next` computations, other sampling strategies may be more appropriate.


## Adaptive Weighted Rejection Sampling

When attempting to sample from the product of a potential (e.g., a language model) and a *boolean* constraint potential (e.g., a [CFG][genlm.control.potential.built_in.wcfg] or [JSON schema][genlm.control.potential.built_in.json] potential), the most efficient and lowest variance sampler is [`AWRS`][genlm.control.sampler.token.AWRS].[^2] This framework is described in detail in [Lipkin et al. (2025)](https://arxiv.org/abs/2504.05410).

```python
# Create a AWRS token sampler from an llm and a cfg
token_sampler = AWRS(llm, cfg)
# Sample a token and weight
token, logw, _ = await token_sampler.sample(context)
```

[^2]: "Higher variance" refers to the variance of the estimator, which is influenced by the variance of the importance weights. When a sampler has high variance, the importance weights can vary dramatically across different samples, leading to unstable estimates in downstream tasks. While high-variance samplers may generate samples efficiently, they often require more samples to achieve the same level of accuracy as lower-variance alternatives.

## Set-based Token Sampling

A [`SetTokenSampler`][genlm.control.sampler.token.SetTokenSampler] samples tokens by first sampling a weighted subset of tokens using a [`SetSampler`][genlm.control.sampler.set.SetSampler], and then selects one token from the set proportional to its weight. These samplers are commonly used to sample tokens from a language model while enforcing non-boolean byte-level constraints. This algorithm is described in Appendix C of [Loula et al. (2025)](https://openreview.net/pdf?id=xoXn62FzD0).

### Set Samplers

SetTokenSamplers wrap a SetSampler, which is responsible for sampling a weighted subset of tokens. Currently, `genlm-control` provides two set samplers:

1. [`EagerSetSampler`][genlm.control.sampler.set.EagerSetSampler] - Eagerly samples a set of tokens by sampling one "subtoken" (e.g., byte) at a time.
2. [`TopKSetSampler`][genlm.control.sampler.set.TopKSetSampler] - Lazily enumerates the top $K$ tokens by weight and samples an additional "wildcard" token to ensure absolute continuity. This sampler is typically slower than `EagerSetSampler`.

Both of these set samplers are designed to work with two types of potentials:

1. An **iterable potential** which has a vocabulary of iterable tokens (e.g., over byte sequences)
2. An **item potential** which has a vocabulary of items which form the elements of iterable tokens (e.g., over individual bytes)

In common scenarios, the iterable potential is a language model and the item potential is a byte-level potential.

```python
# Create a set-based token sampler using a set sampler
set_sampler = EagerSetSampler(llm, fsa)
token_sampler = SetTokenSampler(set_sampler)

# Sample a token and weight
token, logw, _ = await token_sampler.sample(context)
```

### Factory methods

For convenience, we provide factory methods for creating set token samplers from potentials.

```python
from genlm.control.sampler import topk_token_sampler, eager_token_sampler

topk_sampler = topk_token_sampler(llm, fsa, K=10)

eager_sampler = eager_token_sampler(llm, fsa)
```

## Sampler Selection Guide for Controlled Generation

The following table provides general guidelines for selecting a sampler in the context of controlled generation from an LLM. Note that the best sampler may vary depending on the specific controlled generation task.

| Scenario | Recommended Sampler | Notes |
|----------|-------------------|--------|
| No token-level constraints | `DirectTokenSampler` | Basic LM sampling; used when all constraints are enforced using `critics` |
| Boolean constraints (e.g., FSA, CFG, JSON schema) | `AWRS` | Efficient, low-variance, and exact sampling from product of a LLM and constraint |
| Byte-level non-boolean constraints| `eager_token_sampler` | Generally less efficient than `AWRS`, but more flexible |

## Custom Token Samplers

It is also possible to implement custom token samplers by subclassing the [`TokenSampler`][genlm.control.sampler.token.TokenSampler] class and implementing the [`sample`][genlm.control.sampler.token.TokenSampler.sample] method. These implementations must satisfy the following contract.

### Token Sampler Contract

All token samplers in `genlm-control` must generate properly weighted samples with respect to a target potential's next-token weights $\pot(\cdot \mid \bm{x})$ given a context $\xx$:

A weighted sample $(x, w) \sim q(\cdot \mid \xx)$ is properly weighted with respect to $\pot(\cdot \mid \xx)$ if, for any function $f$,

$$
\mathbb{E}_{(x,w) \sim q(\cdot \mid \xx)}[w f(x)] = \sum_{x \in \A \cup \{\eos\}} f(x)\cdot\pot(x \mid \xx)
$$

where $\mathcal{A}$ is the vocabulary of the target potenital $\pot$.



================================================
FILE: docs/javascripts/mathjax.js
================================================
window.MathJax = {
    tex: {
      macros: {
        bm: ["\\boldsymbol{#1}", 1],
        prefix: "\\psi",
        complete: "\\phi",
        pot: "\\Phi",
        xx: "\\bm{x}",
        A: "\\mathcal{A}",
        eos: "\\textsf{eos}",
        yy: "\\bm{y}"
      }
    },
    options: {
      processHtmlClass: 'arithmatex'
    }
  };



================================================
FILE: genlm/control/__init__.py
================================================
from .constant import EOS, EOT
from .potential import Potential, PromptedLLM, BoolCFG, BoolFSA, WFSA, WCFG, JsonSchema, CanonicalTokenization
from .sampler import (
    SMC,
    direct_token_sampler,
    eager_token_sampler,
    topk_token_sampler,
    AWRS,
)
from .viz import InferenceVisualizer

__all__ = [
    "EOS",
    "EOT",
    "SMC",
    "Potential",
    "PromptedLLM",
    "WCFG",
    "BoolCFG",
    "WFSA",
    "BoolFSA",
    "JsonSchema",
    "CanonicalTokenization",
    "AWRS",
    "direct_token_sampler",
    "eager_token_sampler",
    "topk_token_sampler",
    "InferenceVisualizer",
]



================================================
FILE: genlm/control/constant.py
================================================
class EndOfSequence:
    """End-of-sequence tokens."""

    def __init__(self, type_="EOS"):
        self.type_ = type_

    def __repr__(self):
        return self.type_

    def __eq__(self, other):
        return isinstance(other, EndOfSequence) and self.type_ == other.type_

    def __radd__(self, other):
        if isinstance(other, (str, bytes)):
            return [*list(other), self]
        elif isinstance(other, (list, tuple)):
            return type(other)(list(other) + [self])
        else:
            raise TypeError(f"Cannot concatenate {type(other)} with {type(self)}")

    def __hash__(self):
        return hash(self.type_)

    def __iter__(self):
        return iter([self])

    def __len__(self):
        return 1


EOS = EndOfSequence("EOS")
EOT = EndOfSequence("EOT")



================================================
FILE: genlm/control/typing.py
================================================
from dataclasses import dataclass
from collections.abc import Sequence as SequenceABC

from genlm.control.constant import EndOfSequence


@dataclass
class TokenType:
    """Base class representing the type of a token"""

    def check(self, value):
        """Check if a value matches this type"""
        raise NotImplementedError()  # pragma: no cover

    def is_iterable_of(self, element_type):
        """Check if this type can be interpreted as an iterable of element_type.

        Args:
            element_type (TokenType): The type to check if this is an iterable of

        Examples:
            >>> Sequence(Atomic(int)).is_iterable_of(Atomic(int))
            True
            >>> Atomic(bytes).is_iterable_of(Atomic(int))
            True
        """
        if isinstance(self, Sequence):
            return self.element_type == element_type

        if isinstance(self, Atomic):
            # Special cases for built-in iterables
            if (
                self.type is bytes
                and isinstance(element_type, Atomic)
                and element_type.type is int
            ):
                return True
            if (
                self.type is str
                and isinstance(element_type, Atomic)
                and element_type.type is str
            ):
                return True

        return False


@dataclass
class Atomic(TokenType):
    """Represents a simple type like int or str"""

    type: type  # The Python type (int, str, etc.)

    def check(self, value):
        return isinstance(value, self.type) or isinstance(value, EndOfSequence)

    def convert(self, value):
        return self.type(value)

    def __repr__(self):
        return f"Atomic({self.type.__name__})"


@dataclass
class Sequence(TokenType):
    """Represents a list/sequence of another type"""

    element_type: TokenType  # The type of elements in the sequence

    def check(self, value):
        return isinstance(value, (list, tuple)) and all(
            self.element_type.check(x) for x in value
        )

    def convert(self, value):
        return tuple(self.element_type.convert(x) for x in value)

    def __repr__(self):
        return f"Sequence({self.element_type!r})"


def infer_type(value):
    """Infer the TokenType from a value.

    Args:
        value (Any): A sample value to infer type from

    Returns:
        (TokenType): The inferred type

    Examples:
        >>> infer_type(42)
        Atomic(type=int)
        >>> infer_type([1, 2, 3])
        Sequence(element_type=Atomic(type=int))
        >>> infer_type([[1, 2], [3, 4]])
        Sequence(element_type=Sequence(element_type=Atomic(type=int)))
    """
    if isinstance(value, SequenceABC) and not isinstance(value, (bytes, str)):
        if not value:
            raise ValueError("Cannot infer type from empty sequence")
        element_type = infer_type(value[0])
        if not all(element_type.check(x) for x in value):
            raise ValueError("Inconsistent types in sequence")
        return Sequence(element_type)

    return Atomic(type(value))


def infer_vocabulary_type(vocabulary):
    """Infer the TokenType from a vocabulary.

    Args:
        vocabulary (List[Any]): A list of tokens to infer type from

    Returns:
        (TokenType): The inferred type

    Raises:
        ValueError: If vocabulary is empty or contains inconsistent types

    Examples:
        >>> infer_vocabulary_type([1, 2, 3])
        Atomic(type=int)
        >>> infer_vocabulary_type([[1, 2], [3, 4]])
        Sequence(element_type=Atomic(type=int))
    """
    if not vocabulary:
        raise ValueError("Cannot infer type from empty vocabulary")

    token_type = infer_type(vocabulary[0])
    if not all(token_type.check(x) for x in vocabulary):
        raise ValueError("Inconsistent types in vocabulary")

    return token_type



================================================
FILE: genlm/control/util.py
================================================
import numpy as np
from genlm.grammar import Float, Log
from arsenal.maths import logsumexp


class LazyWeights:
    """
    A class to represent weights in a lazy manner, allowing for efficient operations
    on potentially large weight arrays without immediate materialization.

    Attributes:
        weights (np.ndarray): The weights associated with the tokens.
        encode (dict): A mapping from tokens to their corresponding indices in the weights array.
        decode (list): A list of tokens corresponding to the weights.
        is_log (bool): A flag indicating whether the weights are in log space.
    """

    def __init__(self, weights, encode, decode, log=True):
        """
        Initialize the LazyWeights instance.

        Args:
            weights (np.ndarray): The weights associated with the tokens.
            encode (dict): A mapping from tokens to their corresponding indices in the weights array.
            decode (list): A list of tokens corresponding to the weights.
            log (bool, optional): Indicates if the weights are in log space. Defaults to True.

        Raises:
            AssertionError: If the lengths of weights and decode or encode do not match.
        """
        assert len(weights) == len(decode)
        assert len(encode) == len(decode)

        self.weights = weights
        self.encode = encode
        self.decode = decode
        self.is_log = log

    def __getitem__(self, token):
        """
        Retrieve the weight for a given token.

        Args:
            token (Any): The token for which to retrieve the weight.

        Returns:
            (float): The weight of the token, or -inf/0 if the token is not found.
        """
        if token not in self.encode:
            return float("-inf") if self.is_log else 0
        return self.weights[self.encode[token]]

    def __len__(self):
        return len(self.weights)

    def __array__(self):
        raise NotImplementedError(
            "LazyWeights cannot be converted to a numpy array. "
            "If you want to combine multiple LazyWeights, use their weights attribute directly."
        )

    def keys(self):
        """Return the list of tokens (keys) in the vocabulary."""
        return self.decode

    def values(self):
        """Return the weights associated with the tokens."""
        return self.weights

    def items(self):
        """Return a zip of tokens and weights."""
        return zip(self.keys(), self.values())

    def normalize(self):
        """
        Normalize the weights.

        Normalization is performed using log-space arithmetic when weights are logarithmic,
        or standard arithmetic otherwise.

        Returns:
            (LazyWeights): A new LazyWeights instance with normalized weights.
        """
        if self.is_log:
            return self.spawn(self.weights - logsumexp(self.weights))
        else:
            return self.spawn(self.weights / np.sum(self.weights))

    def exp(self):
        """
        Exponentiate the weights. This operation can only be performed when weights are in log space.

        Returns:
            (LazyWeights): A new LazyWeights instance with exponentiated weights.

        Raises:
            AssertionError: If the weights are not in log space.
        """
        assert self.is_log, "Weights must be in log space to exponentiate"
        return self.spawn(np.exp(self.weights), log=False)

    def log(self):
        """
        Take the logarithm of the weights. This operation can only be performed when weights are in regular space.

        Returns:
            (LazyWeights): A new LazyWeights instance with logarithmic weights.

        Raises:
            AssertionError: If the weights are already in log space.
        """
        assert not self.is_log, "Weights must be in regular space to take the logarithm"
        return self.spawn(np.log(self.weights), log=True)

    def sum(self):
        """
        Sum the weights.

        Summation is performed using log-space arithmetic when weights are logarithmic,
        or standard arithmetic otherwise.

        Returns:
            (float): The sum of the weights, either in log space or regular space.
        """
        if self.is_log:
            return logsumexp(self.weights)
        else:
            return np.sum(self.weights)

    def spawn(self, new_weights, log=None):
        """
        Create a new LazyWeights instance over the same vocabulary with new weights.

        Args:
            new_weights (np.ndarray): The new weights for the LazyWeights instance.
            log (bool, optional): Indicates if the new weights are in log space. Defaults to None.

        Returns:
            (LazyWeights): A new LazyWeights instance.
        """
        if log is None:
            log = self.is_log
        return LazyWeights(
            weights=new_weights, encode=self.encode, decode=self.decode, log=log
        )

    def materialize(self, top=None):
        """
        Materialize the weights into a chart.

        Args:
            top (int, optional): The number of top weights to materialize. Defaults to None.

        Returns:
            (Chart): A chart representation of the weights.
        """
        weights = self.weights
        if top is not None:
            top_ws = weights.argsort()[-int(top) :]
        else:
            top_ws = weights.argsort()

        semiring = Log if self.is_log else Float

        chart = semiring.chart()
        for i in reversed(top_ws):
            chart[self.decode[i]] = weights[i]

        return chart

    def __repr__(self):
        return repr(self.materialize())

    def assert_equal(self, other, **kwargs):
        """
        Assert that two LazyWeights instances are equal.

        This method asserts that the two LazyWeights instances have the same vocabulary
        (in identical order) and that their weights are numerically close.

        Args:
            other (LazyWeights): The other LazyWeights instance to compare.
            **kwargs (dict): Additional arguments for np.testing.assert_allclose (e.g., rtol, atol).
        """
        assert self.decode == other.decode
        np.testing.assert_allclose(self.weights, other.weights, **kwargs)

    def assert_equal_unordered(self, other, **kwargs):
        """
        Assert that two LazyWeights instances are equal, ignoring vocabularyorder.

        Args:
            other (LazyWeights): The other LazyWeights instance to compare.
            **kwargs (dict): Additional arguments for np.isclose (e.g., rtol, atol).
        """
        assert set(self.decode) == set(other.decode), "keys do not match"

        for x in self.decode:
            have, want = self[x], other[x]
            assert np.isclose(have, want, **kwargs), f"{x}: {have} != {want}"


def load_trie(V, backend=None, **kwargs):
    """
    Load a TokenCharacterTrie.

    Args:
        V (list): The vocabulary.
        backend (str, optional): The backend to use for trie construction. Defaults to None.
        **kwargs (dict): Additional arguments for the trie construction.

    Returns:
        (TokenCharacterTrie): A trie instance.
    """
    import torch

    if backend is None:
        backend = "parallel" if torch.cuda.is_available() else "sequential"

    if backend == "parallel":
        from genlm.backend.trie import ParallelTokenCharacterTrie

        return ParallelTokenCharacterTrie(V, **kwargs)
    else:
        from genlm.backend.trie import TokenCharacterTrie

        return TokenCharacterTrie(V, **kwargs)


def load_async_trie(V, backend=None, **kwargs):
    """
    Load an AsyncTokenCharacterTrie. This is a TokenCharacterTrie that
    automatically batches weight_sum and weight_max requests.

    Args:
        V (list): The vocabulary.
        backend (str, optional): The backend to use for trie construction. Defaults to None.
        **kwargs (dict): Additional arguments for the trie construction.

    Returns:
        (AsyncTokenCharacterTrie): An async trie instance.
    """
    from genlm.backend.trie import AsyncTokenCharacterTrie

    return AsyncTokenCharacterTrie(load_trie(V, backend, **kwargs))


def fast_sample_logprobs(logprobs: np.ndarray, size: int = 1) -> np.ndarray:
    """Sample indices from an array of log probabilities using the Gumbel-max trick.

    Args:
        logprobs: Array of log probabilities
        size (int): Number of samples to draw

    Returns:
        (np.ndarray): Array of sampled indices

    Note:
        This is much faster than np.random.choice for large arrays since it avoids
        normalizing probabilities and uses vectorized operations.
    """
    noise = -np.log(-np.log(np.random.random((size, len(logprobs)))))
    return (logprobs + noise).argmax(axis=1)


def fast_sample_lazyweights(lazyweights):
    """Sample a token from a LazyWeights instance using the Gumbel-max trick.

    Args:
        lazyweights (LazyWeights): A LazyWeights instance

    Returns:
        (Any): Sampled token
    """
    assert lazyweights.is_log
    token_id = fast_sample_logprobs(lazyweights.weights, size=1)[0]
    return lazyweights.decode[token_id]



================================================
FILE: genlm/control/viz.py
================================================
import webbrowser
import http.server
import socketserver
import threading
import tempfile
import shutil
from pathlib import Path


class InferenceVisualizer:
    """Web-based visualization server for SMC inference results.

    This class is intended to be used in conjunction with the `InferenceEngine` class.

    Example:
        ```python
        from genlm.control import InferenceVisualizer
        # create the visualizer
        viz = InferenceVisualizer()
        # run inference and save the record to a JSON file
        sequences = await token_sampler.smc(
            n_particles=10,
            max_tokens=20,
            ess_threshold=0.5,
            json_path="smc_record.json" # save the record to a JSON file
        )
        # visualize the inference run
        viz.visualize("smc_record.json")
        # clean up visualization server
        viz.shutdown_server()
        ```
    """

    def __init__(self, port=8000, serve_dir=None):
        """Initialize the visualization server.

        Args:
            port (int): Port to run the server on.
            serve_dir (str | Path, optional): Directory to serve files from.
                If None, creates a temporary directory.

        Raises:
            OSError: If the port is already in use
        """
        self._server = None
        self._server_thread = None
        self._port = port
        self._html_dir = Path(__file__).parent / "html"

        # Set up serve directory
        if serve_dir is None:
            self._serve_dir = Path(tempfile.mkdtemp(prefix="smc_viz_"))
            self._using_temp_dir = True
        else:
            self._serve_dir = Path(serve_dir).resolve()
            self._using_temp_dir = False
            self._serve_dir.mkdir(exist_ok=True)

        # Create handler that serves from both directories
        class Handler(http.server.SimpleHTTPRequestHandler):
            def translate_path(self_, path):
                # Remove query parameters for file lookup
                clean_path = path.split("?")[0]
                # HTML files come from package
                if clean_path.endswith(".html"):
                    return str(self._html_dir / clean_path.lstrip("/"))
                # JSON files come from serve directory
                return str(self._serve_dir / clean_path.lstrip("/"))

        self._start_server(Handler)

    def visualize(self, json_path, auto_open=False):
        """Visualize the inference run in a browser.

        Args:
            json_path (str | Path): Path to the JSON file to visualize. If the file is not
                in the serve directory, it will be copied there. For efficiency, you can
                write JSON files directly to the serve directory
            auto_open (bool): Whether to automatically open in browser

        Returns:
            (str): URL where visualization can be accessed
        """
        if self._server is None:
            raise RuntimeError("Server is not running")

        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_path}")

        # If file isn't in serve directory, copy it there
        dest_path = self._serve_dir / json_path.name
        if json_path.resolve() != dest_path.resolve():
            shutil.copy2(json_path, dest_path)

        url = f"http://localhost:{self._port}/smc.html?path={json_path.name}"

        if auto_open:
            webbrowser.open(url)

        return url

    def _start_server(self, handler_class):
        """Start the HTTP server."""
        try:
            self._server = socketserver.TCPServer(
                ("", self._port), handler_class, bind_and_activate=False
            )
            self._server.allow_reuse_address = True
            self._server.server_bind()
            self._server.server_activate()
        except OSError as e:
            if e.errno == 48 or e.errno == 98:  # Address already in use
                raise OSError(f"Port {self._port} is already in use") from None
            raise

        self._server_thread = threading.Thread(target=self._server.serve_forever)
        self._server_thread.daemon = True
        self._server_thread.start()

    def shutdown_server(self):
        """Shut down the visualization server."""
        if self._server is not None:
            if self._server_thread is not None and self._server_thread.is_alive():
                self._server.shutdown()
                self._server_thread.join()
            self._server.server_close()
            self._server = None
            self._server_thread = None

        # Clean up any temporary files
        if self._using_temp_dir and self._serve_dir.exists():
            shutil.rmtree(self._serve_dir)

    def __del__(self):
        """Ensure server is shut down when object is deleted."""
        self.shutdown_server()



================================================
FILE: genlm/control/html/smc.html
================================================
<!--
    This file is a modified version of the original smc.html file from the llamppl library.
    It is used to visualize the SMC inference results.
    The original file can be found at https://github.com/probcomp/llamppl/blob/main/html/smc.html
-->

<html>
<meta charset="utf-8">

<head>
    <title>SMC Visualization</title>
    <link rel="icon" href="data:,">
    <style>
        #frame-background {
            fill: rgb(241, 241, 241);
            stroke: gray;
        }

        circle.particle {
            fill: darkblue;
            cursor: pointer
        }

        circle.particle.highlighted {
            fill: rgb(216, 184, 0);
        }

        circle.particle.zeroweight {
            fill: #ccc;
        }

        path.parentline {
            fill: none;
            stroke: #BBE;
            stroke-width: 2px;
            marker-end: url(#arrowhead)
        }

        path.parentline.rejuv {
            stroke: #EBB;
        }

        path.parentline.highlighted {
            stroke: rgb(216, 184, 0);
            stroke-width: 8px;
            /* opacity: .5; */
            marker-end: none
        }

        text.program {
            font-size: 18px;
            font-family: "Gill Sans";
            /* fill: rgb(86, 86, 86); */
            fill: darkblue;
            cursor: pointer;
        }

        tspan.modified-expr {
            font-weight: bold;
        }

        text.program.zeroweight {
            fill: #ccc;
        }

        text.program.highlighted {
            fill: rgb(117, 102, 16);
        }


        line.dotted {
            stroke: #777;
            stroke-dasharray: 2, 6;
        }

        line.dotted.highlighted {
            stroke: rgb(117, 102, 16);
            stroke-width: 2px;
            stroke-dasharray: 4, 6;
        }




        .hover {
            font-size: 10px;
            font-family: "Gill Sans";
        }

        rect.hover_rect {
            fill: rgb(238, 238, 238);
            stroke: black;
            stroke-width: 1;
            rx: 5;
            ry: 5;
        }

        rect.hover_rect_header {
            stroke: none;
            rx: 5;
            ry: 5;
        }

        line.highlight {
            stroke: blue;
            stroke-width: 2px;
            opacity: .5;
        }

        text {
            font-size: 18px;
            font-family: "Gill Sans";
        }
    </style>

</head>

<body>

    <h1>Sequential Monte Carlo - Visualization</h1>

    <div id="svg">
        <svg>
            <defs>
                <marker id="arrowhead" markerWidth="5" markerHeight="3.5" refX="0" refY="1.75" orient="auto">
                    <polygon points="0 0, 5 1.75, 0 3.5" />
                </marker>
            </defs>
            <rect id="frame-background" />
            <g id="frame-foreground"></g>
        </svg>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>

    <script>
        "use strict";

        let load_path = ""

        const url_params = new URLSearchParams(window.location.search)
        if (url_params.has("path"))
            load_path = url_params.get("path")
        load_by_path()

        function load_by_path(path) {
            if (path == "")
                return
            d3.json('../' + load_path, { cache: "no-store" })
                .then(data => show_data(data))
                .catch(error => console.log(error));
        }

        const frame_background = d3.select('#frame-background')
        const frame_foreground = d3.select('#frame-foreground')

        const zoom_control = d3.zoom().on('zoom', e => frame_foreground.attr('transform', e.transform));
        frame_background.call(zoom_control);

        function show_data(data) {
            const svg_margin = 50
            const SVG_WIDTH = ((window.innerWidth > 0) ? window.innerWidth : screen.width) - svg_margin * 2;
            const SVG_HEIGHT = ((window.innerHeight > 0) ? window.innerHeight : screen.height) - 200;

            let history = data
            window.svg = history
            console.log(history)
            const num_particles = history[0].particles.length

            const particle_yspace = 30
            const state_yspace = particle_yspace * num_particles + 50
            const particle_xspace = 30
            const state_xspace = particle_xspace * num_particles

            // modify `data` in place in any ways you need
            for (let i = 0; i < history.length; i++) {
                let particles = history[i].particles
                history[i].logweight_total = particles.reduce((acc, p) => logaddexp(acc, p.logweight), -Infinity)
                console.log("Total log weight step " + i + ": " + history[i].logweight_total)
                for (let j = 0; j < particles.length; j++) {
                    for (const [k, v] of Object.entries(particles[j]))
                        particles[j][k] = null_to_neginf(v)

                    const particle = particles[j]
                    particle.x = j * particle_xspace
                    particle.y = i * state_yspace + j * particle_yspace
                    particle.relative_weight = Math.exp(particle.logweight - history[i].logweight_total)
                    particle.prefix = particle.contents.split("<<<")[0]
                    particle.suffix = particle.contents.split(">>>")[1]
                    particle.highlighted = particle.contents.split(">>>")[0].split("<<<")[1]
                    // if (Number.isFinite(particle.likelihood) && particle.likelihood > 0)
                    //     min_likelihood = Math.min(min_likelihood, particle.likelihood)
                    // if (Number.isFinite(particle.posterior) && particle.posterior > 0)
                    //     min_posterior = Math.min(min_posterior, particle.posterior)
                    // if (Number.isFinite(particle.prior) && particle.prior > 0)
                    // min_prior = Math.min(min_prior, particle.prior)
                    particle.parent = undefined
                    particle.children = []
                }
            }


            const svg = d3.select("#svg svg")
                .attr("width", SVG_WIDTH)
                .attr("height", SVG_HEIGHT)
                .attr("transform", `translate(${svg_margin},0)`)

            frame_foreground.selectAll("*").remove()

            frame_background
                .attr("width", SVG_WIDTH)
                .attr("height", SVG_HEIGHT)


            const link = d3.linkVertical()
                .x(d => d.x)
                .y(d => d.y)

            for (let i = 1; i < history.length; i++) {
                for (let j = 0; j < num_particles; j++) {
                    const particle = history[i].particles[j]
                    let parent
                    if (history[i].mode == "resample") {
                        parent = history[i - 1].particles[history[i].ancestors[j]]
                    } else {
                        parent = history[i - 1].particles[j]
                    }
                    particle.parent = parent
                    particle.parent.children.push(particle)
                    particle.parent_line = frame_foreground
                        .append("path")
                        .classed("parentline", true)
                        .classed(particle.mode, true) // e.g. ".rejuv"
                        .attr("d", link({ source: { x: parent.x, y: parent.y + 10 }, target: { x: particle.x, y: particle.y - 18.5 } }))
                }
            }

            for (const state of history) {
                const particles = state.particles
                const largest_relweight = particles.reduce((acc, p) => Math.max(acc, p.relative_weight), 0)
                state.x = state.particles[0].x
                state.y = state.particles[0].y

                // show "SMC Step" or "Resample" etc
                frame_foreground.append("text")
                    .attr("transform", `translate(${state.x - 180},${state.y})`)
                    .text(`Step ${state.step}`)
                    .attr("text-anchor", "middle")
                    .style("font-size", 40)

                // show "SMC Step" or "Resample" etc
                // frame_foreground.append("text")
                //     .attr("transform", `translate(${state.x - 160 - 300 - 180},${state.y + 20})`)
                //     .text("(" + state.mode.replace(/_/g, ' ') + ")")
                //     .attr("text-anchor", "middle")
                //     .style("font-size", 40)

                for (const particle of particles) {
                    particle.g = frame_foreground
                        .append("g")
                        .attr("transform", `translate(${particle.x},${particle.y})`)
                        .on("click", () => {
                            set_click_highlight(particle)
                            console.log(particle)
                        })
                    const r = 10
                    particle.circle = particle.g
                        .append("circle")
                        .classed("particle", true)
                        .classed("zeroweight", particle.relative_weight == 0)
                        .attr("r", r * Math.sqrt(particle.relative_weight / largest_relweight) + 3)


                    particle.text = particle.g
                        .append("text")
                        .classed("program", true)
                        // .attr("x", r * 2)
                        .attr("x", state_xspace - particle.x)
                        .attr("y", r / 2)
                        .classed("zeroweight", particle.relative_weight == 0)
                    particle.text
                        .append("tspan")
                        .text(particle.prefix)
                    particle.text
                        .append("tspan")
                        .classed("modified-expr", true)
                        .text(particle.highlighted)
                    particle.text
                        .append("tspan")
                        .text(particle.suffix)

                    const left_side = -particle.x - r * 2

                    particle.g.append("text")
                        .attr("x", 2 * r)
                        .attr("y", -r)
                        .style("font-size", 10)
                        .style("fill", "#888")
                        .text("w/Σw=" + particle.relative_weight.toFixed(2) + " " + "w=" + show_prob(Math.exp(particle.logweight)) + " Δw (" + show_prob(Math.exp(particle.weight_incr)) + ")")

                    particle.dotted_line = particle.g.append("line")
                        .classed("dotted", true)
                        .attr("x1", left_side)
                        .attr("x2", state_xspace - particle.x)
                        .lower()

                }
            }
        }

        let curr_highlighted = undefined
        /// Highlights the history leading up to this particle
        function set_click_highlight(particle, highlight = true) {

            if (highlight && curr_highlighted != undefined) {
                set_click_highlight(curr_highlighted, false)
            }
            if (highlight && particle === curr_highlighted) {
                curr_highlighted = undefined
                return
            }
            if (highlight)
                curr_highlighted = particle

            const to_highlight = [particle]
            for (let ancestor = particle; ancestor != undefined; ancestor = ancestor.parent) {
                to_highlight.push(ancestor)
            }
            const worklist = [particle]
            while (worklist.length > 0) {
                let descendant = worklist.pop()
                worklist.push(...descendant.children)
                to_highlight.push(...descendant.children)
            }

            for (const p of to_highlight) {
                p.circle.classed("highlighted", highlight)
                p.dotted_line.classed("highlighted", highlight)
                p.text.classed("highlighted", highlight)
                if (p.parent_line) {
                    p.parent_line.classed("highlighted", highlight)
                }
            }

        }

        function show_prob(prob, digits = 0) {
            if (prob == 0)
                return "0"
            if (prob == 1)
                return "1"
            if (prob >= 1e-3 && prob <= 1e3)
                return prob.toPrecision(Math.max(digits, 1))
            return prob.toExponential(digits)
        }

        function logaddexp(x, y) {
            if (x == -Infinity)
                return y
            if (y == -Infinity)
                return x
            let answer = Math.max(x, y) + Math.log1p(Math.exp(-Math.abs(x - y)))

            return Math.max(x, y) + Math.log1p(Math.exp(-Math.abs(x - y)))
        }

        // json maps NaN and -Inf and Inf to `null` so we undo that
        function null_to_neginf(x) {
            return x == null ? -Infinity : x
        }

    </script>

</body>

</html>



================================================
FILE: genlm/control/potential/__init__.py
================================================
from .base import Potential as Potential
from .autobatch import AutoBatchedPotential
from .multi_proc import MultiProcPotential
from .operators import PotentialOps
from .product import Product
from .coerce import Coerced

from .built_in import PromptedLLM, WCFG, BoolCFG, WFSA, BoolFSA, JsonSchema, CanonicalTokenization

__all__ = [
    "Potential",
    "PotentialOps",
    "Product",
    "PromptedLLM",
    "JsonSchema",
    "WCFG",
    "BoolCFG",
    "WFSA",
    "BoolFSA",
    "CanonicalTokenization",
    "AutoBatchedPotential",
    "MultiProcPotential",
    "Coerced",
]



================================================
FILE: genlm/control/potential/autobatch.py
================================================
import asyncio
from typing import NamedTuple, Callable
from collections import defaultdict

from genlm.control.potential.base import Potential


class Request(NamedTuple):
    batch_method_name: str
    args_accumulator: Callable
    future: asyncio.Future


class AutoBatchedPotential(Potential):
    """
    AutoBatchedPotential is a wrapper around a Potential that enables automatic batching of concurrent requests.

    This class manages a background loop that collects concurrent requests to instance methods
    (`complete`, `prefix`, `score`, `logw_next`) and batches them together before
    delegating to the corresponding batch methods of the underlying potential
    (`batch_complete`, `batch_prefix`, `batch_score`, `batch_logw_next`).

    This class inherits all methods from [`Potential`][genlm.control.potential.base.Potential].

    Attributes:
        potential (Potential): The underlying potential instance that is being wrapped.
        background_loop (AsyncBatchLoop): An asynchronous loop that manages batch requests.
    """

    def __init__(self, potential):
        self.potential = potential
        self.background_loop = AsyncBatchLoop(potential)
        self.background_loop.start()
        super().__init__(potential.vocab)

    async def complete(self, context):
        return await self.background_loop.queue_request(
            "batch_complete", lambda args: ([*args[0], context],)
        )

    async def prefix(self, context):
        return await self.background_loop.queue_request(
            "batch_prefix", lambda args: ([*args[0], context],)
        )

    async def score(self, context):
        return await self.background_loop.queue_request(
            "batch_score", lambda args: ([*args[0], context],)
        )

    async def logw_next(self, context):
        return await self.background_loop.queue_request(
            "batch_logw_next", lambda args: ([*args[0], context],)
        )

    async def batch_complete(self, contexts):
        return await self.potential.batch_complete(contexts)

    async def batch_prefix(self, contexts):
        return await self.potential.batch_prefix(contexts)

    async def batch_score(self, contexts):
        return await self.potential.batch_score(contexts)

    async def batch_logw_next(self, contexts):
        return await self.potential.batch_logw_next(contexts)

    def spawn(self, *args, **kwargs):
        # creates a new background loop.
        return AutoBatchedPotential(self.potential.spawn(*args, **kwargs))

    def __repr__(self):
        return f"{self.__class__.__name__}({self.potential!r})"

    async def cleanup(self):
        """Async cleanup - preferred method"""
        await self.background_loop.cleanup()

    def __del__(self):
        if loop := getattr(self, "background_loop", None):
            loop.close()


class AsyncBatchLoop:
    """Asynchronous batch processing loop for potential methods."""

    def __init__(self, potential, history=None):
        self.potential = potential
        self.q = asyncio.Queue()
        self.task = None
        self.history = history

    def start(self):
        """Start the background processing task."""
        self.task = asyncio.create_task(self._background_loop())

    def queue_request(self, batch_method_name, arg_accumulator):
        """Queue a request for batch processing."""
        future = asyncio.Future()
        self.q.put_nowait(Request(batch_method_name, arg_accumulator, future))
        return future

    async def _background_loop(self):
        """Background task that processes queued requests."""
        while True:
            try:
                method_groups = defaultdict(list)
                req = await self.q.get()
                method_groups[req.batch_method_name].append(req)

                try:
                    while True:
                        req = self.q.get_nowait()
                        method_groups[req.batch_method_name].append(req)
                except asyncio.QueueEmpty:
                    pass

                for method_name, requests in method_groups.items():
                    try:
                        batch_args = ([],)
                        for req in requests:
                            batch_args = req.args_accumulator(batch_args)

                        results = await getattr(self.potential, method_name)(
                            *batch_args
                        )

                        assert len(results) == len(requests)
                        for i, req in enumerate(requests):
                            req.future.set_result(results[i])

                    except Exception as e:
                        for req in requests:
                            if not req.future.done():
                                req.future.set_exception(e)

            except asyncio.CancelledError:
                break

    def close(self):
        """Stop the background processing task and cleanup resources."""
        if task := getattr(self, "task", None):
            try:
                task.cancel()
            except RuntimeError:  # pragma: no cover
                pass  # pragma: no cover
            self.task = None

    async def cleanup(self):
        """Async cleanup - preferred method"""
        if self.task and not self.task.done():
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
            self.task = None

    def __del__(self):
        self.close()



================================================
FILE: genlm/control/potential/base.py
================================================
import asyncio
import numpy as np
from abc import ABC, abstractmethod

from genlm.control.constant import EOS, EndOfSequence
from genlm.control.util import LazyWeights
from genlm.control.typing import TokenType, infer_vocabulary_type
from genlm.control.potential.operators import PotentialOps
from genlm.control.potential.testing import PotentialTests


class Potential(ABC, PotentialOps, PotentialTests):
    """Abstract base class for potentials.

    A Potential is a function that maps sequences of tokens in a vocabulary to non-negative real numbers (weights).

    Potentials assign weights to sequences of tokens based on whether they are complete sequences or prefixes of complete sequences.

    - `complete`: Assess the log weight of a sequence of tokens in the vocabulary as a complete sequence.
    - `prefix`: Assess the log weight of a sequence of tokens in the vocabulary as a prefix.

    Potentials additionally implement a `logw_next` method:

    - `logw_next`: Compute the next-token log weights of each token in the vocabulary and a special EOS (end-of-sequence) token given a context.

    Subclasses must minimally implement `complete` and `prefix`. `logw_next` and batched versions of the above methods
    come with default implementations, but may be overridden by subclasses for improved performance.

    All Potentials must satisfy a set of properties which can be tested using [PotentialTests][genlm.control.potential.testing.PotentialTests].

    Attributes:
        token_type (TokenType): The type of tokens in the vocabulary.
        vocab (list): List of tokens making up the vocabulary.
        eos (EndOfSequence): Special token to use as end-of-sequence.
        vocab_eos (list): List of tokens in `vocab` and `eos`. `eos` is assumed to be the last token in `vocab_eos`.
        lookup (dict): Mapping from tokens and `eos` to their indices in `vocab_eos`.
    """

    def __init__(self, vocabulary, token_type=None, eos=None):
        """
        Initialize the potential.

        Args:
            vocabulary (list): List of tokens that make up the vocabulary.
            token_type (TokenType, optional): Optional TokenType of all elements of the vocabulary.
                If None, will be inferred from vocabulary.
            eos (EndOfSequence, optional): Special token to use as end-of-sequence. Defaults to `EOS`.
                In general, this should not be set by users.

        Raises:
            ValueError: If vocabulary is empty.
            TypeError: If vocabulary contains tokens which are not of `token_type`.
        """
        if not vocabulary:
            raise ValueError("vocabulary cannot be empty")

        if token_type is None:
            token_type = infer_vocabulary_type(vocabulary)
        elif not isinstance(token_type, TokenType):
            raise ValueError(f"token_type must be a TokenType, got {token_type!r}.")

        if not all(token_type.check(x) for x in vocabulary):
            raise TypeError(f"Tokens in vocabulary must be of type {token_type}.")

        if eos and not isinstance(eos, EndOfSequence):
            raise ValueError(f"EOS must be an instance of EndOfSequence, got {eos!r}.")

        self.eos = eos or EOS

        self.token_type = token_type
        self.vocab = vocabulary
        self.vocab_eos = self.vocab + [self.eos]
        self.lookup = {}
        for i, x in enumerate(vocabulary):
            if x in self.lookup:
                raise ValueError(f"Duplicate token {x!r} found in vocabulary")
            self.lookup[x] = i
        self.lookup[self.eos] = len(self.vocab)

    ####################
    # Instance methods #
    ####################

    @abstractmethod
    async def complete(self, context):
        """Assess the weight of `context` as a complete sequence.

        Args:
            context (list): Sequence of tokens.

        Returns:
            (float): Log weight of the context under the language.
        """
        pass  # pragma: no cover

    @abstractmethod
    async def prefix(self, context):
        """Assess the weight of `context` as a prefix.

        Args:
            context (list): Sequence of tokens.

        Returns:
            (float): Log weight of the context as a prefix.
        """
        pass  # pragma: no cover

    async def score(self, context):
        """Assess the weight of `context` based on EOS-termination.

        This is a convenience method which dispatches to `complete` if `context` ends with `self.eos`, otherwise to `prefix`.

        Args:
            context (list): Sequence of tokens.

        Returns:
            (float): Log weight of the context, either as a prefix or complete sequence.
        """
        if context and context[-1] == self.eos:
            return await self.complete(context[:-1])
        else:
            return await self.prefix(context)

    async def logw_next(self, context):
        """Compute the next-token weights of each token in `self.vocab_eos` given `context`.

        Args:
            context (list): Sequence of tokens.

        Returns:
            (LazyWeights): Weights of each token in the vocabulary and EOS.
        """
        ctx_log_w = await self.prefix(context)

        if ctx_log_w == float("-inf"):
            raise ValueError(f"Context {context!r} has weight zero under `prefix`.")

        scores = await self.batch_score([[*context, x] for x in self.vocab_eos])
        logws = scores - ctx_log_w

        return self.make_lazy_weights(logws)

    ###################
    # Batched methods #
    ###################

    async def batch_complete(self, contexts):
        """Batched equivalent to `complete`.

        Assess the weight of each context as a complete sequence.

        Args:
            contexts (list): List of sequences of tokens.

        Returns:
            (np.array): Array of log weights for each context.
        """
        if not contexts:
            raise ValueError("Contexts must be non-empty.")

        return np.array(
            await asyncio.gather(*[self.complete(context) for context in contexts])
        )

    async def batch_prefix(self, contexts):
        """Batched equivalent to `prefix`.

        Assess the weight of each context as a prefix.

        Args:
            contexts (list): List of sequences of tokens.

        Returns:
            (np.array): Array of log weights for each context.
        """
        if not contexts:
            raise ValueError("Contexts must be non-empty.")

        return np.array(
            await asyncio.gather(*[self.prefix(context) for context in contexts])
        )

    async def batch_score(self, contexts):
        """Batched equivalent to `score`.

        Assess the weight of each context based on EOS-termination.

        Args:
            contexts (list): List of sequences of tokens.

        Returns:
            (np.array): Array of log weights for each context.
        """
        if not contexts:
            raise ValueError("Contexts must be non-empty.")

        complete, prefix = [], []
        complete_indices, prefix_indices = [], []

        for i, context in enumerate(contexts):
            # We want == here instead of `is`.
            if context and context[-1] == self.eos:
                complete.append(context[:-1])
                complete_indices.append(i)
            else:
                prefix.append(context)
                prefix_indices.append(i)

        complete_scores = (
            await self.batch_complete(complete) if complete else np.array([])
        )
        prefix_scores = await self.batch_prefix(prefix) if prefix else np.array([])

        results = np.empty(len(contexts))
        if len(complete_scores) > 0:
            results[complete_indices] = complete_scores
        if len(prefix_scores) > 0:
            results[prefix_indices] = prefix_scores

        return results

    async def batch_logw_next(self, contexts):
        """Batched equivalent to `logw_next`.

        Computes the next-token weights of each token in `self.vocab_eos` given each context in the batch.

        Args:
            contexts (list): List of sequences of tokens.

        Returns:
            (list): List of LazyWeights objects, one for each context.

        Raises:
            ValueError: If any context has zero weight (log weight of -inf) under `prefix`.
        """
        if not contexts:
            raise ValueError("Contexts must be non-empty.")

        return await asyncio.gather(*[self.logw_next(context) for context in contexts])

    #############
    # Utilities #
    #############

    def make_lazy_weights(self, weights, log=True):
        """Helper method to create a LazyWeights object over the potential's vocabulary and EOS.

        Args:
            weights (np.array): Array of weights.
            log (bool, optional): Whether the weights are in log space. Defaults to True.

        Returns:
            (LazyWeights): LazyWeights object defined over `self.vocab_eos`.
        """
        return LazyWeights(
            weights=weights, encode=self.lookup, decode=self.vocab_eos, log=log
        )

    def alloc_logws(self, default=float("-inf")):
        """Allocate a new array of log weights for the potential's vocabulary and EOS.

        Args:
            default (float, optional): Default log weight. Defaults to -inf.

        Returns:
            (np.array): Array of length `len(self.vocab_eos)` filled with `default`.
        """
        return np.full((len(self.vocab_eos),), default)

    def spawn(self):
        """
        Spawn a fresh instance of the potential.

        This method is not required by default, but may be implemented by subclasses
        to support CPU-parallelism using (`MultiProcPotential`)[genlm.control.potential.multi_proc.MultiProcPotential].
        """
        raise NotImplementedError(
            "Potential.spawn() must be implemented by subclasses."
        )

    async def cleanup(self):
        """
        Cleanup the potential.

        This method may be implemented by subclasses to release resources.
        """
        pass



================================================
FILE: genlm/control/potential/coerce.py
================================================
from genlm.control.potential import Potential
from itertools import chain
import asyncio


class Coerced(Potential):
    """
    Coerce a potential to operate on another vocabulary.

    This class allows a potential to be adapted to work with a different set of tokens,
    defined by a target vocabulary and coersion function.

    This class inherits all methods from [`Potential`][genlm.control.potential.base.Potential].
    Each method delegates to the corresponding method of the underlying potential, but first
    maps any input token sequences from the target vocabulary to the original potential's vocabulary
    using the coercion function.

    Formally, if $f$ is the coercion function, then for any sequence $x_1, \\ldots, x_n$ of tokens from the target vocabulary,
    $$
    \\textsf{Coerced.prefix}(x_1, \\ldots, x_n) = \\textsf{Coerced.potential.prefix}(f(x_1, \\ldots, x_n))
    $$

    $$
    \\textsf{Coerced.complete}(x_1, \\ldots, x_n) = \\textsf{Coerced.potential.complete}(f(x_1, \\ldots, x_n))
    $$

    Attributes:
        potential (Potential): The original potential instance that is being coerced.
        f (callable): A function that maps sequences of tokens from the target vocabulary to sequences of tokens from
            the original potential's vocabulary.

    Note:
        The coerced potential's vocabulary will by default be pruned to only include tokens that can be mapped to the original potential's vocabulary
        via the coercion function (i.e. `set(f([x])) <= set(potential.vocab)`). If no such tokens are found, a `ValueError` is raised.
        This behavior can be overridden by setting `prune=False`, in which case the coerced potential's vocabulary will include all tokens from the target vocabulary.
    """

    def __init__(self, potential, target_vocab, f, prune=True):
        """
        Initialize a Coerced potential.

        Args:
            potential (Potential): The original potential instance that is being coerced.
            target_vocab (list): The target vocabulary that the potential will operate on.
                Each element of `target_vocab` must be hashable.
            f (callable): A function that maps iterables of tokens from the target vocabulary
                to the original potential's vocabulary.
            prune (bool): Whether to prune the coerced potential's vocabulary to only include tokens that can be mapped to the original potential's vocabulary.
                If `False`, the coerced potential's vocabulary will include all tokens from the target vocabulary.

        Raises:
            ValueError: If no valid tokens are found in the target vocabulary that can be mapped to the original potential's vocabulary.
        """
        self.potential = potential
        self.f = f

        if prune:
            tokens = []
            for target_token in target_vocab:
                base_token = f([target_token])
                if set(base_token) <= set(potential.vocab):
                    tokens.append(target_token)
        else:
            tokens = target_vocab

        if not tokens:
            raise ValueError("No valid tokens found in target vocabulary")

        super().__init__(tokens)

    def _batch_f(self, contexts):
        return [self.f(context) for context in contexts]

    async def complete(self, context):
        return await self.potential.complete(context=self.f(context))

    async def prefix(self, context):
        return await self.potential.prefix(context=self.f(context))

    async def logw_next(self, context):
        Ws = self.alloc_logws()
        ctx = self.f(context)
        ctx_w = await self.potential.prefix(ctx)
        Ws[-1] = await self.potential.complete(ctx) - ctx_w
        exts = [self.f(chain(context, [x])) for x in self.vocab]  # slow!!
        Ws[:-1] = await self.potential.batch_prefix(exts) - ctx_w
        return self.make_lazy_weights(Ws)

    async def batch_complete(self, contexts):
        return await self.potential.batch_complete(contexts=self._batch_f(contexts))

    async def batch_prefix(self, contexts):
        return await self.potential.batch_prefix(contexts=self._batch_f(contexts))

    async def batch_logw_next(self, contexts):
        return await asyncio.gather(*[self.logw_next(context) for context in contexts])

    def __repr__(self):
        return f"{self.__class__.__name__}({self.potential!r})"



================================================
FILE: genlm/control/potential/multi_proc.py
================================================
import asyncio
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from genlm.control.potential.base import Potential


class MultiProcPotential(Potential):
    """A Potential that adds parallel processing capabilities to any base Potential implementation.

    Creates a process pool of worker processes, each containing an instance of the potential.

    This class inherits all methods from [`Potential`][genlm.control.potential.base.Potential].
    Each method delegates to a corresponding method of the potential instances running in the
    worker processes, distributing work across multiple processes for improved performance.
    """

    def __init__(self, potential_factory, factory_args, num_workers=2):
        """
        Initialize the MultiProcPotential.

        Args:
            potential_factory (callable): A factory function that creates a potential instance.
            factory_args (tuple): Arguments to pass to the potential factory.
            num_workers (int): The number of worker processes to spawn. Each will contain an instance of the potential.
        """
        self.num_workers = num_workers
        self.executor = ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=self._init_worker,
            initargs=(potential_factory, factory_args),
        )
        # Get vocab and eos from one of the workers
        vocab, eos = self.executor.submit(self._get_vocab_and_eos).result()
        super().__init__(vocab, eos=eos)

    @staticmethod
    def _init_worker(factory, args):
        global _worker_potential, _worker_event_loop
        _worker_potential = factory(*args)
        _worker_event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_worker_event_loop)

    @staticmethod
    def _get_vocab_and_eos():
        return _worker_potential.vocab, _worker_potential.eos

    @staticmethod
    def _run_coroutine(coroutine):
        global _worker_event_loop
        return _worker_event_loop.run_until_complete(coroutine)

    @staticmethod
    def _worker_logw_next(context):
        return MultiProcPotential._run_coroutine(
            _worker_potential.logw_next(context)
        ).weights

    @staticmethod
    def _worker_prefix(context):
        return MultiProcPotential._run_coroutine(_worker_potential.prefix(context))

    @staticmethod
    def _worker_complete(context):
        return MultiProcPotential._run_coroutine(_worker_potential.complete(context))

    # @staticmethod
    # def _worker_score(context):
    #    return MultiProcPotential._run_coroutine(_worker_potential.score(context))

    async def _run_in_executor(self, func, *args):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, func, *args)

    async def logw_next(self, context):
        result = await self._run_in_executor(self._worker_logw_next, context)
        return self.make_lazy_weights(result)

    async def prefix(self, context):
        return await self._run_in_executor(self._worker_prefix, context)

    async def complete(self, context):
        return await self._run_in_executor(self._worker_complete, context)

    async def batch_logw_next(self, contexts):
        results = await asyncio.gather(
            *(
                self._run_in_executor(self._worker_logw_next, context)
                for context in contexts
            )
        )
        return [self.make_lazy_weights(result) for result in results]

    async def batch_complete(self, contexts):
        results = await asyncio.gather(
            *(
                self._run_in_executor(self._worker_complete, context)
                for context in contexts
            )
        )
        return np.array(results)

    async def batch_prefix(self, contexts):
        results = await asyncio.gather(
            *(
                self._run_in_executor(self._worker_prefix, context)
                for context in contexts
            )
        )
        return np.array(results)

    def __del__(self):
        if self.executor is not None:
            self.executor.shutdown()
            self.executor = None

    def __repr__(self):
        return f"{self.__class__.__name__}({self.num_workers=})"

    def spawn(self):
        raise ValueError("MultiProcPotentials are not spawnable.")



================================================
FILE: genlm/control/potential/operators.py
================================================
class PotentialOps:
    """Mixin providing operations for potential functions:

    1. Product (`*`): Take the product of two potentials.\n
    2. Coercion (`coerce`): Coerce the potential to operate on another potential's vocabulary.\n
    3. Auto-batching (`to_autobatched`): Create a version that automatically batches concurrent requests to the instance methods.\n
    4. Parallelization (`to_multiprocess`): Create a version that parallelizes operations over multiple processes.\n
    """

    def __mul__(self, other):
        """Take the product of two potentials.

        See [`Product`][genlm.control.potential.product.Product] for more details.

        Args:
            other (Potential): Another potential instance to take the product with.

        Returns:
            (Product): A Product instance representing the unnormalized product of the two potentials.

        Note:
            Potentials must operate on the same token type and the intersection of their vocabularies must be non-empty.
        """
        from genlm.control.potential.product import Product

        return Product(self, other)

    def coerce(self, other, f, prune=True):
        """Coerce the current potential to operate on the vocabulary of another potential.

        See [`Coerced`][genlm.control.potential.coerce.Coerced] for more details.

        Args:
            other (Potential): The potential instance whose vocabulary will be used.
            f (callable): A function mapping sequences of tokens from self's vocab to sequences of tokens from other's vocab.
            prune (bool): Whether to prune the coerced potential's vocabulary to only include tokens that can be mapped to the original potential's vocabulary.
                If `False`, the coerced potential's vocabulary will include all tokens from the target vocabulary.

        Returns:
            (Coerced): A Potential that operates on the vocabulary of `other`.
        """
        from genlm.control.potential.coerce import Coerced

        return Coerced(self, other.vocab, f=f, prune=prune)

    def to_autobatched(self):
        """Create a new potential instance that automatically batches concurrent requests to the instance methods.

        See [`AutoBatchedPotential`][genlm.control.potential.autobatch.AutoBatchedPotential] for more details.

        Returns:
            (AutoBatchedPotential): A new potential instance that wraps the current potential and automatically batches concurrent requests to the instance methods.
        """
        from genlm.control.potential.autobatch import AutoBatchedPotential

        return AutoBatchedPotential(self)

    def to_multiprocess(self, num_workers=2, spawn_args=None):
        """Create a new potential instance that parallelizes operations using multiprocessing.

        See [`MultiProcPotential`][genlm.control.potential.multi_proc.MultiProcPotential] for more details.

        Args:
            num_workers (int): The number of workers to use in the multiprocessing pool.
            spawn_args (tuple): The positional arguments to pass to the potential's `spawn` method.

        Returns:
            (MultiProcPotential): A new potential instance that wraps the current potential and uses multiprocessing to parallelize operations.

        Note:
            For this method to be used, the potential must implement a picklable `spawn` method.
        """
        from genlm.control.potential.multi_proc import MultiProcPotential

        factory_args = spawn_args or ()
        return MultiProcPotential(
            potential_factory=self.spawn,
            factory_args=factory_args,
            num_workers=num_workers,
        )



================================================
FILE: genlm/control/potential/product.py
================================================
import asyncio
import warnings
from genlm.control.potential.base import Potential


class Product(Potential):
    """
    Combine two potential instances via element-wise multiplication (sum in log space).

    This class creates a new potential that is the element-wise product of two potentials:
    ```
    prefix(xs) = p1.prefix(xs) + p2.prefix(xs)
    complete(xs) = p1.complete(xs) + p2.complete(xs)
    logw_next(x | xs) = p1.logw_next(x | xs) + p2.logw_next(x | xs)
    ```

    The new potential's vocabulary is the intersection of the two potentials' vocabularies.

    This class inherits all methods from [`Potential`][genlm.control.potential.base.Potential],
    see there for method documentation.

    Attributes:
        p1 (Potential): The first potential instance.
        p2 (Potential): The second potential instance.
        token_type (str): The type of tokens that this product potential operates on.
        vocab (list): The common vocabulary shared between the two potentials.

    Warning:
        Be careful when taking products of potentials with minimal vocabulary overlap.
        The resulting potential will only operate on tokens present in both vocabularies.
    """

    def __init__(self, p1, p2):
        """Initialize a Product potential.

        Args:
            p1 (Potential): First potential
            p2 (Potential): Second potential
        """
        self.p1 = p1
        self.p2 = p2

        if self.p1.token_type == self.p2.token_type:
            self.token_type = self.p1.token_type
        else:
            raise ValueError(
                "Potentials in product must have the same token type. "
                f"Got {self.p1.token_type} and {self.p2.token_type}."
                + (
                    "\nMaybe you forgot to coerce the potentials to the same token type? See `Coerce`."
                    if (
                        self.p1.token_type.is_iterable_of(self.p2.token_type)
                        or self.p2.token_type.is_iterable_of(self.p1.token_type)
                    )
                    else ""
                )
            )

        common_vocab = list(set(p1.vocab) & set(p2.vocab))
        if not common_vocab:
            raise ValueError("Potentials in product must share a common vocabulary")

        # Check for small vocabulary overlap
        threshold = 0.1
        for potential, name in [(p1, "p1"), (p2, "p2")]:
            overlap_ratio = len(common_vocab) / len(potential.vocab)
            if overlap_ratio < threshold:
                warnings.warn(
                    f"Common vocabulary ({len(common_vocab)} tokens) is less than {threshold * 100}% "
                    f"of {name}'s ({potential!r}) vocabulary ({len(potential.vocab)} tokens). "
                    "This Product potential only operates on this relatively small subset of tokens.",
                    RuntimeWarning,
                )

        super().__init__(common_vocab, token_type=self.token_type)

        # For fast products of weights
        self.v1_idxs = [p1.lookup[token] for token in self.vocab_eos]
        self.v2_idxs = [p2.lookup[token] for token in self.vocab_eos]

    async def prefix(self, context):
        w1 = await self.p1.prefix(context)
        if w1 == float("-inf"):
            return float("-inf")
        w2 = await self.p2.prefix(context)
        return w1 + w2

    async def complete(self, context):
        w1 = await self.p1.complete(context)
        if w1 == float("-inf"):
            return float("-inf")
        w2 = await self.p2.complete(context)
        return w1 + w2

    async def batch_complete(self, contexts):
        W1, W2 = await asyncio.gather(
            self.p1.batch_complete(contexts), self.p2.batch_complete(contexts)
        )
        return W1 + W2

    async def batch_prefix(self, contexts):
        W1, W2 = await asyncio.gather(
            self.p1.batch_prefix(contexts), self.p2.batch_prefix(contexts)
        )
        return W1 + W2

    async def logw_next(self, context):
        W1, W2 = await asyncio.gather(
            self.p1.logw_next(context), self.p2.logw_next(context)
        )
        return self.make_lazy_weights(
            W1.weights[self.v1_idxs] + W2.weights[self.v2_idxs]
        )

    async def batch_logw_next(self, contexts):
        Ws1, Ws2 = await asyncio.gather(
            self.p1.batch_logw_next(contexts), self.p2.batch_logw_next(contexts)
        )
        return [
            self.make_lazy_weights(
                Ws1[n].weights[self.v1_idxs] + Ws2[n].weights[self.v2_idxs]
            )
            for n in range(len(contexts))
        ]

    def spawn(self, p1_opts=None, p2_opts=None):
        return Product(
            self.p1.spawn(**(p1_opts or {})),
            self.p2.spawn(**(p2_opts or {})),
        )

    def __repr__(self):
        return f"Product({self.p1!r}, {self.p2!r})"



================================================
FILE: genlm/control/potential/stateful.py
================================================
import bisect
from collections import defaultdict
import heapq
import asyncio
from abc import ABC, abstractmethod, abstractproperty
import numpy as np

from genlm.control.potential.base import Potential


def make_immutable(context):
    if isinstance(context, (str, bytes, tuple)):
        return context
    try:
        return bytes(context)
    except (ValueError, TypeError):
        return tuple(context)


class ParticleState(ABC):
    def __init__(self, owner):
        self.owner = owner
        self.finished = False
        self.context = []

    async def update_context(self, incremental_context):
        """Update the context with more data that has come in."""
        if self.finished:
            return
        self.context.extend(incremental_context)
        await self.impl_update_context(incremental_context)

    async def finish(self):
        """Mark this state as finished, clearing up any associated
        state, and updating the current score to reflect whether
        this is a valid string in the associated language."""
        if self.finished:
            return
        self.finished = True
        await self.impl_finish()

    @abstractproperty
    def current_score(self):
        """The current score associated with this potential, which
        will reflect whether the current context is a suitable member
        of the language if this has been finished, or whether it is a
        suitable prefix if it has not."""

    @abstractmethod
    async def impl_update_context(self, incremental_context): ...

    @abstractmethod
    async def impl_finish(self): ...

    async def clone(self):
        if self.finished:
            return self
        result = self.owner.new_state()
        await result.update_context(self.context)
        assert self.context == result.context
        assert self.current_score == result.current_score
        return result


class StatefulPotential(Potential):
    def __init__(
        self, vocabulary, token_type=None, eos=None, state_class=None, cache_size=100
    ):
        super().__init__(vocabulary=vocabulary, token_type=token_type, eos=eos)
        self.__state_class = state_class

        self.__cache_size = cache_size
        self.__state_count = 0

        self.__state_pool = defaultdict(list)
        self.__known_contexts = []

        self.__eviction_heap = []
        self.__ages = dict()
        self.__epoch = 0

    def __tick(self):
        self.__epoch += 1
        return self.__epoch

    def new_state(self) -> ParticleState:
        if self.__state_class is None:
            raise NotImplementedError()
        return self.__state_class(self)

    async def cleanup(self):
        await asyncio.gather(
            *[state.finish() for pool in self.__state_pool.values() for state in pool]
        )
        self.__state_pool.clear()
        self.__known_contexts.clear()
        self.__ages.clear()
        self.__eviction_heap.clear()

    async def __look_up_state(self, context):
        context = make_immutable(context)

        state = None

        i = bisect.bisect_left(self.__known_contexts, context)
        if i < len(self.__known_contexts):
            existing = self.__known_contexts[i]
            if context[: len(existing)] == existing:
                pool = self.__state_pool[existing]
                if pool:
                    state = pool.pop()
                    if not pool:
                        del self.__known_contexts[i]
                    self.__state_count -= 1
                    assert self.__state_count >= 0
        if state is None:
            state = self.new_state()
        if len(context) > len(state.context):
            await state.update_context(context[len(state.context) :])
        assert len(state.context) == len(context)
        assert list(state.context) == list(context)
        return state

    def __return_state(self, state):
        assert not state.finished
        context = make_immutable(state.context)
        i = bisect.bisect_left(self.__known_contexts, context)
        if i >= len(self.__known_contexts):
            self.__known_contexts.append(context)
        elif self.__known_contexts[i] != context:
            self.__known_contexts.insert(i, context)
        self.__state_pool[context].append(state)
        self.__state_count += 1
        age = self.__tick()
        heapq.heappush(self.__eviction_heap, (age, context))
        self.__ages[context] = age
        while self.__state_count > self.__cache_size:
            assert len(self.__eviction_heap) >= self.__state_count
            seen_age, to_evict = heapq.heappop(self.__eviction_heap)
            if seen_age == self.__ages[to_evict]:
                i = bisect.bisect_left(self.__known_contexts, to_evict)
                assert self.__known_contexts[i] == to_evict
                pool = self.__state_pool.pop(to_evict, ())
                assert pool
                self.__state_count -= len(pool)
                del self.__known_contexts[i]

    async def prefix(self, context):
        state = await self.__look_up_state(context)
        result = state.current_score
        self.__return_state(state)
        return result

    async def complete(self, context):
        state = await self.__look_up_state(context)
        await state.finish()
        return state.current_score

    async def logw_next(self, context):
        """Compute the next-token weights of each token in `self.vocab_eos` given `context`.

        Args:
            context (list): Sequence of tokens.

        Returns:
            (LazyWeights): Weights of each token in the vocabulary and EOS.
        """
        state = await self.__look_up_state(context)
        assert not state.finished
        ctx_log_w = state.current_score

        if ctx_log_w == float("-inf"):
            raise ValueError(f"Context {context!r} has weight zero under `prefix`.")

        async def step_score(x):
            local_state = await state.clone()
            await local_state.update_context([x])

            if x == self.eos:
                await local_state.finish()
                return local_state.current_score
            else:
                result = local_state.current_score
                await local_state.finish()
                return result

        scores = np.array(
            await asyncio.gather(*[step_score(x) for x in self.vocab_eos])
        )

        logws = scores - ctx_log_w

        return self.make_lazy_weights(logws)



================================================
FILE: genlm/control/potential/streaming.py
================================================
from genlm.control.potential.stateful import StatefulPotential, ParticleState
from abc import ABC, abstractmethod
from typing import Any, Iterable
from queue import SimpleQueue
from enum import Enum, auto
from threading import Thread
import asyncio
import random
import time


class Responses(Enum):
    INCOMPLETE = auto()
    COMPLETE = auto()
    ERROR = auto()


class UniqueIdentifier:
    def __init__(self, name):
        self.__name = name

    def __repr__(self):
        return self.__name


PING_TOKEN = UniqueIdentifier("PING_TOKEN")
SHUTDOWN_TOKEN = UniqueIdentifier("SHUTDOWN_TOKEN")


class Timeout(Exception):
    pass


def timeout_sequence():
    start = time.time()
    # Initially we just yield to the the event loop
    for _ in range(3):
        yield 0.0
    # Then we do a series of short sleeps
    for _ in range(3):
        yield random.random() * 0.01
    sleep = 0.015
    while time.time() < start + 30:
        yield random.random() * sleep
        sleep = min(sleep * 1.1, 1)
    raise Timeout(f"Timed out after {time.time() - start:.2f}s")


class RunningInThread:
    def __init__(self, function):
        self.incoming_data = SimpleQueue()
        self.responses = SimpleQueue()
        self.last_message = None
        self.running = False
        self.complete = False
        self.function = function

    def __chunks(self):
        while True:
            self.last_message, chunk = self.incoming_data.get()
            if chunk is SHUTDOWN_TOKEN:
                break
            yield chunk
            self.responses.put((self.last_message, Responses.INCOMPLETE))

    def run(self):
        assert not self.running
        try:
            self.running = True
            self.last_message, chunk = self.incoming_data.get()
            assert chunk == PING_TOKEN
            self.responses.put((self.last_message, Responses.INCOMPLETE))
            result = self.function(self.__chunks())
        except Exception as e:
            self.responses.put((self.last_message, Responses.ERROR, e))
        else:
            self.responses.put((self.last_message, Responses.COMPLETE, result))
        finally:
            self.running = False
            self.complete = True


class StreamingState(ParticleState):
    def __init__(self, owner):
        super().__init__(owner)
        self.__token = 0
        self.__background = None
        self.__background_thread = None
        self.__score = 0.0
        self.__shut_down = False

    def __new_token(self):
        self.__token += 1
        return self.__token

    async def __initialize_background(self):
        if self.__background is None:
            self.__background = RunningInThread(self.owner.calculate_score_from_stream)

            # Sometimes, especially in consistency check tests, we have too many threads
            # running and need to wait before we're able to start a new thread.
            for t in timeout_sequence():
                try:
                    self.__background_thread = Thread(
                        target=self.__background.run, daemon=True
                    )
                    self.__background_thread.start()
                    break
                except RuntimeError:
                    await asyncio.sleep(t)
            await self.__send_message(PING_TOKEN)
            assert self.__background.running or self.__background.complete
        assert self.__background is not None

    async def impl_update_context(self, incremental_context):
        await self.__initialize_background()
        await self.__send_message(incremental_context)

    async def impl_finish(self):
        await self.__initialize_background()
        self.shutdown()

    @property
    def current_score(self):
        return self.__score

    async def __send_message(self, message):
        if self.__background.complete:
            return
        token = self.__new_token()
        self.__background.incoming_data.put((token, message))

        for timeout in timeout_sequence():
            if not self.__background.responses.empty():
                break
            await asyncio.sleep(timeout)
        self.__receive_response(token)

    def __receive_response(self, token):
        response_token, response_type, *payload = self.__background.responses.get()
        assert token == response_token
        match response_type:
            case Responses.INCOMPLETE:
                pass
            case Responses.COMPLETE:
                self.__score = payload[0] or 0.0
            case Responses.ERROR:
                self.__score = -float("inf")

    def shutdown(self):
        if self.__shut_down:
            return
        self.__shut_down = True
        if self.__background_thread is not None and self.__background_thread.is_alive():
            token = self.__new_token()
            self.__background.incoming_data.put((token, SHUTDOWN_TOKEN))
            # Should in fact terminate very fast. Long timeout here for debugging purposes
            # only - we want a log if it hangs.
            self.__background_thread.join(timeout=1.0)
            self.__receive_response(token)

    def __del__(self):
        self.shutdown()


class StreamingPotential(StatefulPotential, ABC):
    def __init__(self, vocabulary, token_type=None, eos=None):
        super().__init__(
            vocabulary=vocabulary,
            token_type=token_type,
            eos=eos,
            state_class=StreamingState,
        )

    @abstractmethod
    def calculate_score_from_stream(self, stream: Iterable[Any]) -> float: ...


# This should be an async generator really but async generators
# are fundamentally broken. See https://peps.python.org/pep-0789/
# I kept running into problems with this during implementation, so
# ended up finding it easier to just hand roll implementations of
# this rather than trying to use yield based generators.
class AsyncSource(ABC):
    @abstractmethod
    async def more(self): ...


class Chunks(AsyncSource):
    def __init__(self, running_in_task):
        self.running_in_task = running_in_task
        self.__first = True

    async def more(self):
        if not self.__first:
            await self.running_in_task.responses.put(
                (self.running_in_task.last_message, Responses.INCOMPLETE)
            )
        self.__first = False
        (
            self.running_in_task.last_message,
            chunk,
        ) = await self.running_in_task.incoming_data.get()
        if chunk is SHUTDOWN_TOKEN:
            raise StopAsyncIteration()
        return chunk


class RunningInTask:
    def __init__(self, function):
        self.incoming_data = asyncio.Queue()
        self.responses = asyncio.Queue()
        self.last_message = None
        self.running = False
        self.complete = False
        self.function = function

    async def run(self):
        assert not self.running
        try:
            self.running = True
            self.last_message, chunk = await self.incoming_data.get()
            assert chunk == PING_TOKEN
            await self.responses.put((self.last_message, Responses.INCOMPLETE))
            chunks = Chunks(self)
            result = await self.function(chunks)
        except Exception as e:
            await self.responses.put((self.last_message, Responses.ERROR, e))
        else:
            await self.responses.put((self.last_message, Responses.COMPLETE, result))
        finally:
            self.running = False
            self.complete = True


# This is sortof insane, but asyncio will get *very* upset with you if your task
# objects are garbage collected before they're complete. This keeps a set of them
# around until they're completed.
KEEP_ALIVE_SET = set()


class AsyncStreamingState(ParticleState):
    def __init__(self, owner):
        super().__init__(owner)
        self.__token = 0
        self.__background = None
        self.__score = 0.0

    def __new_token(self):
        self.__token += 1
        return self.__token

    async def __initialize_background(self):
        if self.__background is None:
            self.__background = RunningInTask(self.owner.calculate_score_from_stream)
            self.__background_task = asyncio.create_task(self.__background.run())
            await self.__send_message(PING_TOKEN)
            KEEP_ALIVE_SET.add(self.__background_task)
            self.__background_task.add_done_callback(KEEP_ALIVE_SET.discard)
        assert self.__background is not None

    async def impl_update_context(self, incremental_context):
        await self.__initialize_background()
        await self.__send_message(incremental_context)

    async def impl_finish(self):
        await self.__initialize_background()
        await self.shutdown()

    @property
    def current_score(self):
        return self.__score

    async def __send_message(self, message):
        if self.__background.complete:
            return
        token = (self.__new_token(), message)
        await self.__background.incoming_data.put((token, message))

        (
            response_token,
            response_type,
            *payload,
        ) = await self.__background.responses.get()

        assert token == response_token
        match response_type:
            case Responses.INCOMPLETE:
                pass
            case Responses.COMPLETE:
                self.__score = payload[0] or 0.0
            case Responses.ERROR:
                self.__score = -float("inf")

    async def shutdown(self):
        if self.__background is not None:
            await self.__send_message(SHUTDOWN_TOKEN)


class AsyncStreamingPotential(StatefulPotential, ABC):
    def __init__(self, vocabulary, token_type=None, eos=None):
        super().__init__(
            vocabulary=vocabulary,
            token_type=token_type,
            eos=eos,
            state_class=AsyncStreamingState,
        )

    @abstractmethod
    async def calculate_score_from_stream(self, stream: AsyncSource) -> float: ...



================================================
FILE: genlm/control/potential/testing.py
================================================
import asyncio
import numpy as np


class PotentialTests:
    """A mixin class providing testing utilities for validating Potential implementations.

    This class provides methods to verify the mathematical consistency and correctness
    of Potential implementations through various assertions:

    - logw_next consistency: Verifies that token-level log weights are consistent with
      prefix and complete scores.
    - Autoregressive factorization: Validates that complete scores factor correctly as
      a sum of log token weights (with an additional correction term corresponding to the
      prefix weight of the empty sequence).
    - Batch consistency: Ensures batch operations produce identical results to
      their non-batch counterparts.

    All Potential instances inherit from this class and thus automatically gain access to these
    testing utilities.
    """

    colors = {
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "red": "\033[91m",
        "reset": "\033[0m",
    }

    async def assert_logw_next_consistency(
        self, context, rtol=1e-3, atol=1e-5, top=None, verbosity=0, method_args=()
    ):
        """
        Assert that `logw_next` is consistent with `prefix` and `complete`.

        For a `context` of tokens $x_1, \\ldots, x_{n-1}$, this checks (in log space) whether:

        $$
        \\textsf{logw\\_next}(x_n | x_1, \\ldots, x_{n-1}) = \\textsf{score}(x_1, \\ldots, x_n) - \\textsf{prefix}(x_1, \\ldots, x_{n-1})
        $$
        for $x_n \\in \\textsf{vocab_eos}$, i.e., the potential's vocabulary and end-of-sequence token.

        Args:
            context (list): Context to test.
            rtol (float): Relative tolerance for floating point comparison.
            atol (float): Absolute tolerance for floating point comparison.
            top (int):If specified, only test the top-k tokens by log weight. If None, test all tokens.
            verbosity (int): Verbosity level.
            method_args (tuple): Positional arguments to pass to `logw_next`, `prefix`, and `batch_score`.
                Defaults to empty tuple.

        Raises:
            AssertionError: If `logw_next` is not consistent with `prefix` and `complete`.
        """
        top_logw_next = (await self.logw_next(context, *method_args)).materialize(
            top=top
        )
        tokens = list(top_logw_next.keys())
        extended = [[*context, x] for x in tokens]

        context_w = await self.prefix(context, *method_args)
        extended_ws = np.array(
            await asyncio.gather(*[self.score(e, *method_args) for e in extended])
        )

        wants = np.array([top_logw_next[x] for x in tokens])
        haves = extended_ws - context_w

        errors, valids = [], []
        for i, (want, have) in enumerate(zip(wants, haves)):
            abs_diff, rel_diff = self._compute_diff(want, have)
            info = (want, have, abs_diff, rel_diff, tokens[i])
            (valids if abs_diff <= atol and rel_diff <= rtol else errors).append(info)

        if valids and verbosity > 0:
            print(
                f"{self.colors['green']}logw_next consistency with context={context!r} satisfied for tokens:{self.colors['reset']}\n"
            )
            for valid in valids:
                want, have, abs_diff, rel_diff, token = valid
                print(
                    self._format_diff(want, have, abs_diff, rel_diff, atol, rtol, token)
                )

        if errors:
            error_msg = f"{self.colors['red']}logw_next consistency with context={context!r} not satisfied for tokens:{self.colors['reset']}\n\n"
            for error in errors:
                want, have, abs_diff, rel_diff, token = error
                error_msg += self._format_diff(
                    want, have, abs_diff, rel_diff, atol, rtol, token
                )
            raise AssertionError(error_msg)

    async def assert_autoreg_fact(
        self, context, rtol=1e-3, atol=1e-5, verbosity=0, method_args=()
    ):
        """
        Assert that `complete` factors as an autoregressive sum of `logw_next`s.

        For a `context` of tokens $x_1, \\ldots, x_n$, this checks (in log space) whether:

        $$
        \\textsf{complete}(x_1, \\ldots, x_n) - \\textsf{prefix}(\\epsilon) = \\textsf{logw\\_next}(\\textsf{eos} \\mid x_1, \\ldots, x_{n}) + \\sum_{i=1}^{n} \\textsf{logw_next}(x_i \\mid x_1, \\ldots, x_{i-1})
        $$
        where $\\epsilon$ is the empty sequence.

        Args:
            context (list): Context to test.
            rtol (float): Relative tolerance for floating point comparison.
            atol (float): Absolute tolerance for floating point comparison.
            verbosity (int): Verbosity level.
            method_args (tuple): Positional arguments to pass to `complete`, `prefix`, and `logw_next`.
                Defaults to empty tuple.

        Raises:
            AssertionError: If the autoregressive factorization is not satisfied.
        """
        want = (await self.complete(context, *method_args)) - (
            await self.prefix([], *method_args)
        )

        logw_next_results = await asyncio.gather(
            *[self.logw_next(context[:i], *method_args) for i in range(len(context))],
            self.logw_next(context, *method_args),
        )

        have = (
            sum(logw_next_results[i][context[i]] for i in range(len(context)))
            + logw_next_results[-1][self.eos]
        )

        abs_diff, rel_diff = self._compute_diff(want, have)
        if abs_diff > atol or rel_diff > rtol:
            error_msg = (
                f"{self.colors['red']}Factorization not satisfied for context {context!r}:{self.colors['reset']}\n"
                + self._format_diff(want, have, abs_diff, rel_diff, atol, rtol)
            )
            raise AssertionError(error_msg)

        if verbosity > 0:
            print(
                f"{self.colors['green']}Factorization property satisfied for context {context}:{self.colors['reset']}\n"
            )
            print(self._format_diff(want, have, abs_diff, rel_diff, atol, rtol))

    async def assert_batch_consistency(
        self,
        contexts,
        rtol=1e-3,
        atol=1e-5,
        verbosity=0,
        batch_method_args=(),
        method_args=(),
    ):
        """
        Assert that batch results are equal to non-batch results.

        Args:
            contexts (list[list]): Contexts to test.
            rtol (float): Relative tolerance for floating point comparison.
            atol (float): Absolute tolerance for floating point comparison.
            verbosity (int): Verbosity level.
            batch_method_args (tuple): Positional arguments to pass to batch methods.
                Defaults to empty tuple.
            method_args (tuple): Positional arguments to pass to underlying potential methods.
                Defaults to empty tuple.

        Raises:
            AssertionError: If the batch results are not equal to the non-batch results.
        """
        batch_logw_nexts = await self.batch_logw_next(contexts, *batch_method_args)
        batch_scores = await self.batch_score(contexts, *batch_method_args)

        for i, context in enumerate(contexts):
            logw_next = await self.logw_next(context, *method_args)
            try:
                np.testing.assert_allclose(
                    batch_logw_nexts[i].weights, logw_next.weights, rtol=rtol, atol=atol
                )
                if verbosity > 0:
                    print(
                        f"{self.colors['green']}Batch logw_next consistency satisfied for context {context}:{self.colors['reset']}"
                    )
                    print(
                        f"{self.colors['green']}Non-batched: {logw_next.weights}\n"
                        + f"{self.colors['green']}Batched:     {batch_logw_nexts[i].weights}{self.colors['reset']}\n"
                    )
            except AssertionError:
                raise AssertionError(
                    f"{self.colors['red']}Batch logw_next mismatch for context {context}:{self.colors['reset']}\n"
                    + f"{self.colors['green']}Non-batched: {logw_next.weights}\n"
                    + f"{self.colors['red']}Batched:     {batch_logw_nexts[i].weights}{self.colors['reset']}"
                )

            score = await self.score(context, *method_args)
            abs_diff, rel_diff = self._compute_diff(score, batch_scores[i])
            if abs_diff > atol or rel_diff > rtol:
                raise AssertionError(
                    f"{self.colors['red']}Batch score mismatch for context {context}:{self.colors['reset']}\n"
                    + f"{self.colors['green']}Non-batched: {score}\n"
                    + f"{self.colors['red']}Batched:     {batch_scores[i]}{self.colors['reset']}"
                )
            elif verbosity > 0:
                print(
                    f"{self.colors['green']}Batch score consistency satisfied for context {context}:{self.colors['reset']}"
                )
                print(
                    f"{self.colors['green']}Non-batched: {score}\n"
                    + f"{self.colors['green']}Batched:     {batch_scores[i]}{self.colors['reset']}\n"
                )

    def _compute_diff(self, want, have):
        is_inf = want == float("-inf") and have == float("-inf")
        abs_diff = 0 if is_inf else abs(want - have)
        if want == 0:
            rel_diff = 0 if have == 0 else float("inf")
        else:
            rel_diff = 0 if is_inf else abs((want - have) / want)
        return abs_diff, rel_diff

    def _format_diff(self, want, have, abs_diff, rel_diff, atol, rtol, token=None):
        abs_diff_str = (
            f"{self.colors['cyan']}Abs Diff: {abs_diff:.6f} <= {atol=}\033[0m"
        )
        rel_diff_str = (
            f"{self.colors['magenta']}Rel Diff: {rel_diff:.6f} <= {rtol=}\033[0m"
        )

        want_str = f"{self.colors['green']}Expected: {want:.6f}{self.colors['reset']}"
        have_clr = (
            self.colors["yellow"]
            if abs_diff <= atol and rel_diff <= rtol
            else self.colors["red"]
        )
        have_str = f"{have_clr}Actual:   {have:.6f}{self.colors['reset']}"

        if abs_diff <= atol:
            abs_diff_str = f"{self.colors['green']}Abs Diff: {abs_diff:.6f} <= {atol=}{self.colors['reset']}"
        else:
            abs_diff_str = f"{self.colors['red']}Abs Diff: {abs_diff:.6f} > {atol=}{self.colors['reset']}"

        if rel_diff <= rtol:
            rel_diff_str = f"{self.colors['green']}Rel Diff: {rel_diff:.6f} <= {rtol=}{self.colors['reset']}"
        else:
            rel_diff_str = f"{self.colors['red']}Rel Diff: {rel_diff:.6f} > {rtol=}{self.colors['reset']}"

        token_str = (
            f"{self.colors['blue']}Token:    {token}{self.colors['reset']}\n"
            if token
            else ""
        )
        return f"{token_str}{want_str}\n{have_str}\n{abs_diff_str}\n{rel_diff_str}\n\n"



================================================
FILE: genlm/control/potential/built_in/__init__.py
================================================
from .llm import PromptedLLM
from .wcfg import WCFG, BoolCFG
from .wfsa import WFSA, BoolFSA
from .json import JsonSchema
from .canonical import CanonicalTokenization

__all__ = [
    "PromptedLLM",
    "JsonSchema",
    "WCFG",
    "BoolCFG",
    "WFSA",
    "BoolFSA",
    "CanonicalTokenization",
]



================================================
FILE: genlm/control/potential/built_in/canonical.py
================================================
import json
import numpy as np
import scipy.sparse as sp
from collections import defaultdict
from genlm.control.potential.base import Potential
from genlm.backend.tokenization import decode_vocab
from genlm.control.potential.built_in.llm import PromptedLLM

VERYLARGE = 10000000


def _extract_bpe_merges(tokenizer):
    """
    Attempts to extract the ordered BPE merge rules from various tokenizer types.

    Args:
        tokenizer: Tokenizer instance.

    Returns:
        list[tuple[int, int, int]]: A list of merge rules as (u_id, v_id, uv_id) tuples,
                                     ordered by application priority. Returns empty list
                                     if merges cannot be extracted.
    """
    _merges = []
    V = tokenizer.get_vocab()  # Get token string -> ID map

    def _map_merges(merge_list_str):
        """Helper to convert string pairs to ID triples."""
        mapped = []
        for u_str, v_str in merge_list_str:
            u_id = V.get(u_str)
            v_id = V.get(v_str)
            uv_id = V.get(u_str + v_str)
            if u_id is not None and v_id is not None and uv_id is not None:
                mapped.append((u_id, v_id, uv_id))
            # else: ID mapping failed
        return mapped

    # fast tokenizer
    if tokenizer.is_fast and hasattr(tokenizer, "_tokenizer"):
        # fast tokenizer + direct access
        if hasattr(tokenizer._tokenizer, "model") and hasattr(
            tokenizer._tokenizer.model, "merges"
        ):
            hf_merges_list = tokenizer._tokenizer.model.merges
            _merges = _map_merges(hf_merges_list)
            if _merges or not hf_merges_list:
                return _merges
                # else: Accessed direct merges, but ID mapping failed for ALL pairs.
        elif hasattr(tokenizer._tokenizer, "to_str"):
            subtokenizer_dict = json.loads(tokenizer._tokenizer.to_str())
            if "model" in subtokenizer_dict and "merges" in subtokenizer_dict["model"]:
                hf_merges_list = subtokenizer_dict["model"]["merges"]
                _merges = _map_merges(hf_merges_list)
                if (
                    _merges or not hf_merges_list
                ):  # Return if successful or if there were no merges to begin with
                    return _merges
                # else: Parsed JSON merges, but ID mapping failed for ALL pairs.

    # slow tokenizer
    if not _merges and hasattr(
        tokenizer, "bpe_ranks"
    ):  # Only try if fast methods failed
        hf_merges_dict = tokenizer.bpe_ranks  # dict: (u_str, v_str) -> rank
        if hf_merges_dict:
            # Sort by rank to get merge order
            sorted_merges_str = sorted(
                hf_merges_dict.keys(), key=lambda p: hf_merges_dict[p]
            )
            _merges = _map_merges(sorted_merges_str)
            if _merges or not hf_merges_dict:
                return _merges
            # else: Tokenizer had bpe_ranks, but ID mapping failed for ALL pairs

    if not _merges:
        raise ValueError("Could not determine BPE merges.")


class FastCanonicalityFilterBPE:
    def __init__(self, _merges, _encode, _decode, _encode_byte, eos_token_ids):
        self._encode_byte = _encode_byte
        self._merges = _merges
        self._encode = _encode
        self._decode = _decode
        self.V = len(_decode)  # token vocabulary size

        # priority dict might still be useful if merges aren't strictly ordered
        # or for potential future optimizations, keep it for now.
        # self.priority = {(u, v): -i for i, (u, v, _) in enumerate(self._merges)}
        self.make_derivation_table()  # Call the rewritten method

        self.__left_spine, max_left_spine_width = self._left_spine_table()
        self.__right_spine, max_right_spine_width = self._right_spine_table()

        self.left_spine_vector = self.vectorize_spine(
            self.__left_spine, max_left_spine_width
        )
        self.right_spine_vector = self.vectorize_spine(
            self.__right_spine, max_right_spine_width
        )

        self.indices = np.array(
            [
                (index, j)
                for index in range(self.V)
                for j in range(len(self.__left_spine[index]) - 1)
            ]
        )

        self.vector_r = self.left_spine_vector[self.indices[:, 0], self.indices[:, 1]]
        self.vector_rp = self.left_spine_vector[
            self.indices[:, 0], self.indices[:, 1] + 1
        ]

        tmp = sp.dok_matrix((self.V, self.V), dtype=np.int32)
        for u, v, uv in _merges:
            tmp[u, v] = uv + 1  # +1 to avoid zero-indexing

        self.parent_l_matrix = tmp.tocsr()
        self.parent_l_matrix = self.parent_l_matrix[:, self.vector_r]

        self.eos_token_ids = set(eos_token_ids)
        self.overrides = defaultdict(lambda: set())

    def __call__(self, context):
        if context == ():
            mask = np.ones(self.V, dtype=bool)
        else:
            (_, last_token) = context
            try:
                left_id = self._encode[last_token]  # Get the ID of the last token
            except KeyError as e:
                raise KeyError(
                    f"Last token {last_token!r} not found in encode map."
                ) from e

            mask = self._vectorized_conflicting_next_tokens(
                left_id
            )  # Get base mask from BPE rules

            # Apply overrides: Ensure overridden tokens are allowed (True)
            if left_id in self.overrides:
                override_ids = [oid for oid in self.overrides[left_id] if oid < self.V]
                mask[override_ids] = True

            eos_indices = [e for e in self.eos_token_ids if e < self.V]
            mask[eos_indices] = True
        return mask

    def make_derivation_table(self):
        # Initialize left and right child lookup tables
        self._left = [None] * self.V
        self._right = [None] * self.V

        # Populate _left and _right based on the ordered merges
        # Assumes self._merges is ordered by priority (highest priority first) because of the way we build it in extract_bpe_merges
        for u, v, uv in self._merges:
            # Only record the first (highest priority) merge that forms uv
            if self._left[uv] is None and self._right[uv] is None:
                self._left[uv] = u
                self._right[uv] = v

    def vectorize_spine(self, spine, max_spine_width):
        new_spine = [
            [s[i] if i < len(s) else -VERYLARGE for i in range(max_spine_width)]
            for s in spine
        ]
        return np.array(new_spine, dtype=np.int32)

    def _left_spine_table(self):
        "Closure of the left tables."
        max_width = 0
        left_spine = [None] * self.V
        left = self._left
        for i in range(self.V):
            spine = [VERYLARGE, i]
            x = i
            while True:
                x = left[x]
                if x is None:
                    break
                spine.append(x)
            spine.reverse()
            left_spine[i] = spine
            max_width = max(max_width, len(spine))
        return left_spine, max_width

    def _right_spine_table(self):
        "Closure of the right tables."
        max_width = 0
        right_spine = [None] * self.V
        right = self._right
        for i in range(self.V):
            spine = [VERYLARGE, i]
            x = i
            while True:
                x = right[x]
                if x is None:
                    break
                spine.append(x)
            spine.reverse()
            right_spine[i] = spine
            max_width = max(max_width, len(spine))
        return right_spine, max_width

    def set_overrides(self, model_name):
        if "gpt2" in model_name:
            for left, right in [(198, 198), (2637, 82)]:
                self.overrides[left].add(right)

    def _vectorized_conflicting_next_tokens(self, left: int):
        spine_left = self.__right_spine[left]

        L = len(spine_left) - 1  # inf padding

        mask = np.ones(self.V, dtype=bool)

        np_matrix = self.parent_l_matrix[spine_left[:L]].toarray()

        for i in range(L):
            lp = spine_left[i + 1]

            vector_k = np_matrix[i]
            # convert 0 in vector_k to VERYLARGE
            vector_k = np.where(vector_k != 0, vector_k - 1, VERYLARGE)

            conflict_mask = vector_k < VERYLARGE
            conflict_mask &= vector_k <= self.vector_rp
            conflict_mask &= vector_k < lp
            mask[self.indices[conflict_mask][:, 0]] = False

        return mask

    @classmethod
    def from_tokenizer(cls, tokenizer, eos_token_ids=None):
        _decode, _ = decode_vocab(tokenizer)
        if len(_decode) != len(set(_decode)):
            raise ValueError(
                "Duplicate byte sequences found in vocabulary. Cannot create unique byte->ID mapping (_encode)."
            )

        _merges = _extract_bpe_merges(tokenizer)

        # Build _encode (bytes -> token_id map) from _decode
        _encode = {b: i for i, b in enumerate(_decode) if b is not None}

        # Build _encode_byte (single byte -> token_id map)
        _encode_byte = [None] * 256
        for i in range(256):
            byte_val = bytes([i])
            if byte_val in _encode:
                _encode_byte[i] = _encode[byte_val]

        if not eos_token_ids:
            eos_token_ids = [tokenizer.eos_token_id]

        return cls(_merges, _encode, _decode, _encode_byte, eos_token_ids)


class CanonicalTokenization(Potential):
    """
    A custom potential that enforces canonical BPE tokenization.

    This potential ensures that tokens follow the canonical tokenization rules
    by using the FastCanonicalityFilterBPE under the hood.
    """

    def __init__(self, canonicality_filter):
        """
        Initialize the Canonical Potential

        Args:
            canonicality_filter (FastCanonicalityFilterBPE): An initialized FastCanonicalityFilterBPE instance.
        """
        # Store the pre-initialized filter and tokenizer
        self.canonicality_filter = canonicality_filter

        # IMPORTANT: In the base Potential class, EOS will be added to vocab automatically
        # So we should NOT add it ourselves to the vocabulary we pass to super().__init__
        vocabulary = self.canonicality_filter._decode
        super().__init__(vocabulary)

    @classmethod
    def from_llm(cls, llm):
        """
        Factory method to create CanonicalTokenization from a PromptedLLM instance.

        Args:
            llm (PromptedLLM): An instance of PromptedLLM containing the model and tokenizer.

        Returns:
            (CanonicalTokenization): An initialized CanonicalTokenization instance.
        """
        if not isinstance(llm, PromptedLLM):
            raise TypeError(
                f"Expected llm to be an instance of PromptedLLM, got {type(llm)}"
            )

        # Extract necessary components from llm
        tokenizer = llm.model.tokenizer
        eos_token_ids = llm.token_maps.eos_idxs
        model_name = tokenizer.name_or_path

        # Create the filter using its factory method
        canonicality_filter = FastCanonicalityFilterBPE.from_tokenizer(
            tokenizer, eos_token_ids
        )

        # Set overrides on the filter
        canonicality_filter.set_overrides(model_name)

        # Call __init__ with the created filter and tokenizer
        return cls(canonicality_filter)

    async def complete(self, context):
        """
        Assess if a complete sequence follows canonical tokenization.

        Args:
            context (list): Sequence of tokens

        Returns:
            (float): 0.0 if canonical, float('-inf') otherwise
        """
        # Empty sequences are considered canonical
        if not context:
            return 0.0

        # Check if the sequence is canonical
        is_canonical = self._check_canonicality(context)
        return 0.0 if is_canonical else float("-inf")

    async def prefix(self, context):
        """
        Assess if a prefix sequence could potentially extend to a canonical sequence.
        For canonicality, this is the same as complete.

        Args:
            context (list): Sequence of tokens

        Returns:
            (float): 0.0 if potentially canonical, float('-inf') otherwise
        """
        return await self.complete(context)

    async def logw_next(self, context):
        """
        Compute weights for each possible next token given the context.

        Args:
            context (list): Sequence of tokens

        Returns:
            (LazyWeights): Weights for each token in the vocabulary and EOS
        """
        # Get the prefix weight (to check if context itself is canonical)
        ctx_log_w = await self.prefix(context)

        if ctx_log_w == float("-inf"):
            raise ValueError("Context is non-canonical")
        else:
            if context:
                t = (None, context[-1])
                filter_mask = self.canonicality_filter(t)
            else:
                filter_mask = np.ones(len(self.canonicality_filter._decode), dtype=bool)

            # Create log weights directly instead of using np.log(filter_mask)
            # This is more efficient, avoids torch (with torch can't combine with other potentials!)
            logws_no_eos = np.where(filter_mask, 0.0, float("-inf")).astype(np.float32)

            # append eos to the logws, always allow eos.
            # NOTE: concat is because ._decode does not include eos while .vocab_eos does
            logws = np.concatenate([logws_no_eos, np.array([0.0], dtype=np.float32)])

        return self.make_lazy_weights(logws)

    def _check_canonicality(self, context):
        """
        Check if a sequence follows canonical tokenization.

        Args:
            context (list): Sequence of tokens

        Returns:
            (bool): True if the sequence is canonical, False otherwise
        """
        # If we're checking a single token, it's always canonical
        if len(context) <= 1:
            return True

        # Check all adjacent token pairs for canonicality
        for i in range(1, len(context)):
            prev_token = context[i - 1]
            current_token = context[i]

            # Format expected by the filter: (None, previous_token)
            t = (None, prev_token)
            mask = self.canonicality_filter(t)
            # print("percent of mask: ", np.sum(mask)*100 / len(mask))

            # Find token_id in the canonicality filter's vocabulary
            token_id = self.canonicality_filter._encode[current_token]
            if not mask[token_id]:
                return False

        return True



================================================
FILE: genlm/control/potential/built_in/json.py
================================================
import json_stream
import json
import regex
from typing import Generic, TypeVar, Union, Any, Callable
from jsonschema import Draft7Validator
from jsonschema import _types
from typing import Iterable, AsyncIterator
from genlm.control.potential import Potential
from contextlib import contextmanager
from genlm.control.potential.streaming import (
    StreamingPotential,
    AsyncStreamingPotential,
    AsyncSource,
)
from array import array


def is_sequence(checker, instance):
    from collections.abc import Sequence, Mapping

    return isinstance(instance, Sequence) and not isinstance(
        instance, (str, bytes, bytearray, Mapping)
    )


def is_object(checker, instance):
    from json_stream.base import StreamingJSONObject
    from collections.abc import Mapping

    return isinstance(instance, (Mapping, StreamingJSONObject))


# We're using a streaming JSON library that doesn't return proper lists
# and dicts. In theory we could use jsonschema's custom typechecker logic
# here. In practice, this works until it encounters an explicitly specified
# schema type, at which point it creates a new validator that ignores the
# type checker. There is probably a sensible official way to fix this (I hope)
# but I couldn't figure it out and this was expedient and probably won't
# cause too many problems (I hope) - DRMacIver.
_types.is_array.__code__ = is_sequence.__code__
_types.is_object.__code__ = is_object.__code__


# Ideally we would be using Draft202012Validator for compatibility with
# jsonschemabench, but something about the way it's written makes it worse
# at lazy validation, so we're using an older draft for now.
LazyCompatibleValidator = Draft7Validator


UTF8_START_BYTE_MASKS = [
    (0b00000000, 0b10000000),
    (0b11000000, 0b11100000),
    (0b11100000, 0b11110000),
    (0b11110000, 0b11111000),
]


def is_utf8_start_byte(n: int) -> bool:
    """Checks if this is a byte that can appear at the
    start of a UTF-8 character."""
    assert 0 <= n < 256
    for prefix, mask in UTF8_START_BYTE_MASKS:
        if n & mask == prefix:
            return True
    return False


BAD_WHITESPACE = regex.compile(rb"(?:\n\s+\n)|(?:\n\n\n)", regex.MULTILINE)


def chunk_to_complete_utf8(byte_blocks):
    for s in chunk_bytes_to_strings(byte_blocks):
        yield s.encode("utf-8")


def chunk_bytes_to_strings(byte_blocks):
    buffer = bytearray()
    for block in byte_blocks:
        buffer.extend(block)
        try:
            yield buffer.decode("utf-8")
            buffer.clear()
            continue
        except UnicodeDecodeError:
            for i in range(1, min(5, len(buffer) + 1)):
                if is_utf8_start_byte(buffer[-i]):
                    block = buffer[:-i]
                    if block:
                        yield block.decode("utf-8")
                        del buffer[:-i]
                    break
            else:
                raise


class StreamingJsonSchema(StreamingPotential):
    def __init__(self, schema):
        super().__init__(
            vocabulary=list(range(256)),
        )
        self.schema = schema
        self.validator = LazyCompatibleValidator(
            self.schema, format_checker=Draft7Validator.FORMAT_CHECKER
        )
        self.parser = json_schema_parser(schema)

    def calculate_score_from_stream(self, stream: Iterable[Any]) -> float:
        x = json_stream.load(chunk_to_complete_utf8(stream), persistent=True)
        self.validator.validate(x)
        if hasattr(x, "read_all"):
            x.read_all()
        return 0.0


class ValidateJSON(Potential):
    def __init__(self):
        super().__init__(
            vocabulary=list(range(256)),
        )

    async def prefix(self, context):
        # Sometimes a model can get itself into a position where it can't
        # generate any valid tokens, but it can keep generating whitespace
        # indefinitely.
        context = bytes(context)
        if BAD_WHITESPACE.search(context):
            return float("-inf")
        return 0.0

    async def complete(self, context):
        context = bytes(context)
        prefix = await self.prefix(context)
        if prefix == float("-inf"):
            return float("-inf")

        # json-stream will just read a JSON object off the start of
        # the stream and then stop, so we reparse the whole string
        # with the normal JSON parser to validate it at the end, or
        # we will allow JSON values to be followed by arbitrary nonsense.
        # This should only fire when we've successfully created a valid
        # JSON value and want to terminate the sequence.
        try:
            json.loads(context)
            return 0.0
        except json.JSONDecodeError:
            return float("-inf")


def JsonSchema(schema):
    return (
        StreamingJsonSchema(schema)
        * ValidateJSON()
        * ParserPotential(json_schema_parser(schema))
    )


class StringSource(AsyncSource):
    def __init__(self, byte_source):
        self.byte_source = byte_source
        self.buffer = bytearray()

    async def more(self):
        while True:
            # Might raise but that's fine, we're done then.
            block = await self.byte_source.more()
            self.buffer.extend(block)
            try:
                result = self.buffer.decode("utf-8")
                self.buffer.clear()
                return result
            except UnicodeDecodeError:
                for i in range(1, min(5, len(self.buffer) + 1)):
                    if is_utf8_start_byte(self.buffer[-i]):
                        block = self.buffer[:-i]
                        if block:
                            del self.buffer[:-i]
                            return block.decode("utf-8")
                        break
                else:
                    raise


class ParserPotential(AsyncStreamingPotential):
    def __init__(self, parser):
        super().__init__(
            vocabulary=list(range(256)),
        )
        self.parser = parser

    async def calculate_score_from_stream(self, stream: AsyncSource) -> float:
        rechunked = StringSource(stream)
        input = Input(rechunked)
        await input.parse(self.parser)
        return 0.0


S = TypeVar("S")
T = TypeVar("T")


class ParseError(Exception):
    pass


class Incomplete(Exception):
    pass


class Input:
    """Convenience wrapper to provide a stateful stream-like interface
    that makes it easier to write parsers."""

    def __init__(self, incoming: AsyncIterator[str]):
        self.__incoming = incoming
        self.__finished = False
        # There's no textarray equivalent, so we store the growable
        # string as an array of integer codepoints.
        self.buffer = array("I")
        self.index = 0

    async def __read_more(self):
        if self.__finished:
            return False
        try:
            next_block = await self.__incoming.more()
            self.buffer.extend([ord(c) for c in next_block])
            return True
        except StopAsyncIteration:
            self.__finished = True
            return False

    async def __read_until(self, condition):
        while True:
            if condition():
                break
            if not await self.__read_more():
                raise Incomplete()

    async def read_pattern(self, pattern, group=0):
        await self.__read_until(lambda: self.index < len(self.buffer))
        while True:
            # Having to convert the whole thing to a string here is really
            # annoying, but in practice the inefficiency is dwarfed by the LLM
            # so hopefully we don't have to worry about it.
            buffer = "".join(chr(i) for i in self.buffer[self.index :])
            match = pattern.match(buffer, pos=0, partial=True)
            if match is None or (result := match.group(group)) is None:
                raise ParseError()
            elif match.partial:
                if not await self.__read_more():
                    raise Incomplete()
            else:
                self.index += match.end()
                return result

    async def current_char(self):
        await self.__read_until(lambda: self.index < len(self.buffer))
        return chr(self.buffer[self.index])

    async def read(self, n) -> str:
        await self.__read_until(lambda: self.index + n <= len(self.buffer))
        result = self.buffer[self.index : self.index + n]
        assert len(result) == n
        self.index += n
        return "".join(map(chr, result))

    async def expect(self, expected: str):
        actual = await self.read(len(expected))
        if actual != expected:
            raise ParseError(
                f"Expected: {expected} but got {actual} at index {self.index}"
            )

    @contextmanager
    def preserving_index(self):
        """Only advance the index if the operation in the context block does
        not error."""
        start = self.index
        try:
            yield
        except Exception:
            self.index = start
            raise

    async def parse(self, parser: "Parser[T]") -> T:
        with self.preserving_index():
            return await parser.parse(self)

    async def skip_whitespace(self):
        if self.index == len(self.buffer):
            if not await self.__read_more():
                return
        # TODO: Given inefficiencies with regex, maybe worth a more direct
        # implementation here?
        await self.parse(WHITESPACE_PARSER)


class TrivialSource(AsyncSource):
    def __init__(self, value):
        self.value = value
        self.__called = False

    async def more(self):
        if not self.__called:
            self.__called = True
            return self.value
        else:
            raise StopAsyncIteration()


class Parser(Generic[T]):
    """Very basic parser combinators for mostly unambiguous grammars."""

    async def parse(self, input: Input) -> T: ...

    async def parse_string(self, s: str) -> T:
        return await Input(TrivialSource(s)).parse(self)

    def __floordiv__(self, other: Generic[S]) -> "Parser[Union[T, S]]":
        return AltParser(self, other)

    def drop_result(self) -> "Parser[None]":
        return self.map(lambda x: None)

    def map(self, apply: Callable[[T], S]) -> "Parser[S]":
        return MapParser(self, apply)


class MapParser(Parser[T]):
    def __init__(self, base: Parser[S], apply: Callable[[S], T]):
        self.base = base
        self.apply = apply

    async def parse(self, input: Input) -> T:
        return self.apply(await input.parse(self.base))

    def __repr__(self):
        return f"{self.base}.map({self.apply})"


class AltParser(Parser[Union[S, T]]):
    def __init__(self, left: Parser[S], right: Parser[T]):
        self.left = left
        self.right = right

    async def parse(self, input: Input) -> T:
        try:
            with input.preserving_index():
                return await self.left.parse(input)
        except ParseError:
            return await self.right.parse(input)


class RegexParser(Parser[str]):
    def __init__(self, pattern, group=0, options=regex.MULTILINE | regex.UNICODE):
        self.pattern = regex.compile(pattern, options)
        self.group = group

    async def parse(self, input: Input) -> str:
        return await input.read_pattern(self.pattern, group=self.group)

    def __repr__(self):
        return f"RegexParser({self.pattern})"


FLOAT_REGEX_PARSER: Parser[float] = RegexParser(
    r"-?((0|([1-9][0-9]*))((\.[0-9]+)?)([eE][+-]?[0-9]+)?)"
).map(json.loads)


class FloatParser(Parser[float]):
    async def parse(self, input: Input) -> float:
        start = input.index
        preliminary_result = await input.parse(FLOAT_REGEX_PARSER)
        try:
            next_char = await input.read(1)
        except Incomplete:
            return preliminary_result

        if next_char == ".":
            await input.read(1)
        elif next_char in "eE":
            next_next_char = await input.read(1)
            if next_next_char in "-+":
                await input.read(1)

        try:
            while (await input.read(1)) in "0123456789":
                continue
        except Incomplete:
            pass

        input.index = start
        return await input.parse(FLOAT_REGEX_PARSER)


FLOAT_PARSER = FloatParser()

INTEGER_PARSER: Parser[float] = RegexParser(
    r"-?((0|([1-9][0-9]*))([eE]+?[0-9]+)?)"
).map(json.loads)


STRING_LITERAL_PARSER = RegexParser(r'"([^\\"]|\\"|\\[^"])*"').map(json.loads)

NULL_PARSER = RegexParser("null").drop_result()

BOOL_PARSER = RegexParser("false|true").map(json.loads)

WHITESPACE_PARSER = RegexParser(r"\s*")


class ObjectSchemaParser(Parser[Any]):
    def __init__(self, schema):
        self.schema = schema

        properties = self.schema.get("properties", {})
        self.child_parsers = {k: json_schema_parser(v) for k, v in properties.items()}
        if schema.get("additionalProperties", False):
            self.key_parser = STRING_LITERAL_PARSER
        else:
            # TODO: Something is going wrong here with regex escape codes
            self.key_parser = RegexParser(
                "|".join(
                    f"({regex.escape(json.dumps(k, ensure_ascii=b))})"
                    for k in properties
                    for b in [False, True]
                )
            ).map(json.loads)
        self.required_keys = frozenset(schema.get("required", ()))

    def __repr__(self):
        return f"ObjectSchemaParser({self.schema})"

    async def parse(self, input: Input):
        await input.skip_whitespace()

        await input.expect("{")

        result = {}

        keys_seen = set()

        first = True

        while True:
            await input.skip_whitespace()
            if await input.current_char() == "}":
                await input.read(1)
                break
            if not first:
                await input.expect(",")
                await input.skip_whitespace()
            first = False
            key = await input.parse(self.key_parser)
            assert isinstance(key, str)
            if key in keys_seen:
                raise ParseError(f"Duplicated key {repr(key)}")
            keys_seen.add(key)
            await input.skip_whitespace()
            await input.expect(":")
            await input.skip_whitespace()
            value_parser = self.child_parsers.get(key, ARBITRARY_JSON)
            result[key] = await input.parse(value_parser)
        return result


class ArraySchemaParser(Parser[Any]):
    def __init__(self, schema):
        self.schema = schema
        if "items" in schema:
            self.items_parser = json_schema_parser(schema["items"])
        else:
            self.items_parser = None

    def __repr__(self):
        return f"ArraySchemaParser({self.schema})"

    async def parse(self, input: Input):
        await input.skip_whitespace()

        await input.expect("[")

        if self.items_parser is None:
            items_parser = ARBITRARY_JSON
        else:
            items_parser = self.items_parser

        result = []

        first = True

        while True:
            await input.skip_whitespace()
            if await input.current_char() == "]":
                await input.read(1)
                break
            if not first:
                await input.expect(",")
                await input.skip_whitespace()
            first = False
            result.append(await input.parse(items_parser))
        return result


ARBITRARY_JSON = (
    NULL_PARSER
    // BOOL_PARSER
    // FLOAT_PARSER
    // STRING_LITERAL_PARSER
    // ArraySchemaParser({})
    // ObjectSchemaParser({"additionalProperties": True})
)


def json_schema_parser(schema):
    if "type" not in schema:
        return ARBITRARY_JSON
    elif schema["type"] == "number":
        return FLOAT_PARSER
    elif schema["type"] == "integer":
        return INTEGER_PARSER
    elif schema["type"] == "null":
        return NULL_PARSER
    elif schema["type"] == "boolean":
        return BOOL_PARSER
    elif schema["type"] == "string":
        return STRING_LITERAL_PARSER
    elif schema["type"] == "object" and schema.get("properties"):
        return ObjectSchemaParser(schema)
    elif schema["type"] == "array":
        return ArraySchemaParser(schema)
    else:
        return ARBITRARY_JSON



================================================
FILE: genlm/control/potential/built_in/llm.py
================================================
import torch
import warnings
from typing import NamedTuple
from genlm.control.potential.base import Potential


def load_model_by_name(name, backend, **kwargs):
    if backend == "vllm":
        from genlm.backend.llm import AsyncVirtualLM  # pragma: no cover

        model_cls = AsyncVirtualLM  # pragma: no cover
    elif backend == "hf":
        from genlm.backend.llm import AsyncTransformer

        model_cls = AsyncTransformer
    else:
        raise ValueError(
            f"Unknown backend: {backend}. Must be one of ['vllm', 'hf']"
        )  # pragma: no cover

    return model_cls.from_name(name, **kwargs)


class TokenMappings(NamedTuple):
    """
    Container for token mappings between bytes and tokens IDs in a language model.

    This mapping is generally different from the `decode` and `encode` mappings in the `PromptedLLM` class (see notes on EOS token handling).
    """

    decode: list[bytes]  # token_id -> bytes
    encode: dict[bytes, int]  # bytes -> token_id
    eos_idxs: list[int]  # IDs of EOS tokens

    @classmethod
    def create(cls, decode, eos_tokens):
        encode = {x: i for i, x in enumerate(decode)}
        if not all(eos in encode for eos in eos_tokens):
            raise ValueError("EOS token not in language model vocabulary")
        eos_idxs = [encode[eos] for eos in eos_tokens]
        return cls(decode=decode, encode=encode, eos_idxs=eos_idxs)


class PromptedLLM(Potential):
    """A potential representing a language model conditioned on a fixed prompt prefix.

    `PromptedLLM`s operate on byte sequences.

    Notes on EOS Token Handling:\n
    - Tokens to treat as end-of-sequence tokens are specified via the `eos_tokens` argument.\n
    - These tokens are excluded from the potential's vocabulary and as such do not appear in the `vocab` attribute.\n
        This means they cannot appear in any input contexts to the potential nor in the output of `logw_next`. They can be used in the prompt however.\n
    - The log probability assigned to the `genlm.control`'s reserved `EOS` token is the sum of the log probabilities of all the specified EOS tokens.\n

    This class wraps an `AsyncLM` instance.
    """

    def __init__(self, llm, prompt_ids=None, eos_tokens=None, temperature=1):
        """`
        Initializes the PromptedLLM potential.

        Args:
            llm (AsyncLM): The language model to use.
            prompt_ids (list[int], optional): Optional prompt to use as a prompt prefix for all input contexts.
                Must be a list of token IDs. Defaults to None. The prompt ids can be set post-init via `prompt` or `prompt_ids`.
            eos_tokens (list[bytes], optional): List of tokens to treat as end-of-sequence tokens.
                Defaults to the EOS token of the language model's tokenizer.
            temperature (float, optional): The temperature to apply to the language model's logits. Defaults to 1.

        Raises:
            ValueError: If any EOS token is not in the language model vocabulary.
        """
        self.model = llm
        self.prompt_ids = prompt_ids or []

        if not eos_tokens:
            self._eos_tokens = [llm.byte_vocab[self.model.tokenizer.eos_token_id]]
        else:
            self._eos_tokens = eos_tokens

        assert len(set(self._eos_tokens)) == len(self._eos_tokens), (
            "duplicate eos tokens"
        )

        self.token_maps = TokenMappings.create(
            decode=llm.byte_vocab, eos_tokens=self._eos_tokens
        )

        self.temperature = temperature

        V = [x for x in self.token_maps.decode if x not in self._eos_tokens]

        super().__init__(vocabulary=V)

    @classmethod
    def from_name(
        cls,
        name,
        backend=None,
        eos_tokens=None,
        prompt_ids=None,
        temperature=1.0,
        **kwargs,
    ):
        """Create a `PromptedLLM` from a HugginFace model name.

        Args:
            name (str): Name of the model to load
            backend (str, optional): `AsyncLM` backend to use:\n
                * 'vllm' to instantiate an `AsyncVirtualLM`; ideal for GPU usage\n
                * 'hf' for an `AsyncTransformer`; ideal for CPU usage\n
                * 'mock' for a `MockAsyncLM`; ideal for testing.\n
                Defaults to 'vllm' if CUDA is available, otherwise 'hf'.
            eos_tokens (list[bytes], optional): List of tokens to treat as end-of-sequence tokens.
                Defaults to the EOS token of the language model's tokenizer.
            prompt_ids (list[int], optional): Optional prompt to use as a prompt prefix for all input contexts.
                Must be a list of token IDs. Defaults to None. The prompt ids can be set post-init via `set_prompt_from_str` or `prompt_ids`.
            temperature (float, optional): The temperature to apply to the language model's logits. Defaults to 1.
            **kwargs (dict): Additional arguments passed to AsyncLM constructor

        Returns:
            (PromptedLLM): An instance of PromptedLLM
        """
        backend = backend or ("vllm" if torch.cuda.is_available() else "hf")
        model = load_model_by_name(name, backend=backend, **kwargs)
        return cls(
            model, prompt_ids=prompt_ids, eos_tokens=eos_tokens, temperature=temperature
        )

    @property
    def eos_tokens(self):
        return self._eos_tokens

    @eos_tokens.setter
    def eos_tokens(self, value):
        raise ValueError(
            "Cannot reset eos_tokens after initialization. "
            "Use spawn_new_eos(new_eos_tokens) instead."
        )

    @property
    def prompt(self):
        """
        Get the current prompt as a list of byte sequences corresponding to the prompt token IDs.

        Returns:
            (list[bytes]|None): The current prompt as a list of bytes sequences or None if no prompt_ids are set.
        """
        if not self.prompt_ids:
            return  # pragma: no cover
        return [self.token_maps.decode[x] for x in self.prompt_ids]

    def set_prompt_from_str(self, prompt_str):
        """Set the fixed prompt from a string.

        Modifies `prompt_ids` to be the token IDs of the input prompt according to the language model's tokenizer.

        Args:
            prompt_str (str): The prompt to set.
        """
        # TODO: Handle race condition where prompt_ids reset concurrently.
        if not isinstance(prompt_str, str):
            raise ValueError(
                f"Prompt must a string got {type(prompt_str)}. "
                f"To set the prompt from a list of token IDs, use prompt_ids."
            )

        if prompt_str.endswith(" "):
            warnings.warn(
                "Prompt ends with whitespace, which may affect tokenization. "
                "Consider removing trailing whitespace.",
                stacklevel=2,
            )

        self.prompt_ids = self.model.tokenizer.encode(prompt_str)

    def encode_tokens(self, tokens):
        """Encode a list of byte tokens to a list of token IDs in
        the underlying language model's vocabulary.

        Args:
            tokens (list[bytes]): List of byte tokens to encode

        Returns:
            (list[int]): A list of token IDs corresponding to the input tokens.

        Raises:
            ValueError: If any token is not in the vocabulary
        """
        try:
            return [self.token_maps.encode[x] for x in tokens]
        except KeyError as e:
            raise ValueError(f"Token {e.args[0]} not in vocabulary") from e

    def decode_tokens(self, ids):
        """
        Decode a list of token IDs in the language model's vocabulary to a list of byte tokens.

        Args:
            ids (list[int]): A list of token IDs in the language model's vocabulary.

        Returns:
            (list[bytes]): A list of byte tokens corresponding to the input token IDs.
        """
        return [self.token_maps.decode[x] for x in ids]

    def tokenize(self, context_str):
        """Tokenize a string to a list of `bytes` objects, each corresponding to a token in the vocabulary.

        Uses the language model's tokenizer to map `context_str` to a list of token IDs, and then decodes the token IDs to bytes.

        Args:
            context_str (str): A string to encode

        Returns:
            (List[bytes]): A list of byte tokens corresponding to the input string.
        """
        return self.decode_tokens(self.model.tokenizer.encode(context_str))

    async def log_probability(self, context):
        """
        Compute the log probability of `context` given the prompt.

        Args:
            context (list[bytes]): A sequence of bytes tokens.

        Returns:
            (float): The log probability of `context`.
        """
        if not context:
            return 0

        context_ids = self.encode_tokens(context)
        return await self._log_probability(context_ids)

    async def _log_probability(self, context_ids):
        prefixes = [self.prompt_ids + context_ids[:i] for i in range(len(context_ids))]
        log_ps = self._maybe_temper(
            await self.model.batch_next_token_logprobs(prefixes)
        )
        target_ids = torch.tensor(context_ids, device=log_ps.device)
        with torch.no_grad():
            token_logprobs = torch.gather(log_ps, 1, target_ids.unsqueeze(1))
            total_logprob = token_logprobs.sum().item()

        return total_logprob

    def _maybe_temper(self, logps):
        if self.temperature == 1:
            return logps
        return torch.log_softmax(logps / self.temperature, dim=-1)

    async def prefix(self, context):
        """
        Compute the log probability of `context` given the prompt.

        Args:
            context (list[bytes]): A sequence of bytes tokens.

        Returns:
            (float): The log probability of `context`.
        """
        return await self.log_probability(context)

    async def complete(self, context):
        """
        Compute the log probability of `context` and the eos tokens given the prompt.

        If the model has multiple eos tokens, their probabilities will be summed.

        Args:
            context (list[bytes]): A sequence of bytes tokens.

        Returns:
            (float): The log probability of the context.
        """
        context_ids = self.encode_tokens(context)
        logp_context = await self._log_probability(context_ids)
        logp_next = self._maybe_temper(
            await self.model.next_token_logprobs(self.prompt_ids + context_ids)
        )
        logp_eos = torch.logsumexp(logp_next[self.token_maps.eos_idxs], dim=0).item()
        return logp_context + logp_eos

    def _process_logw_next(self, logw_next):
        """Process the log probabilities for the next tokens.

        This function rearranges the log probabilities such that the end-of-sequence (EOS) token's log probability
        is the sum of the log probabilities of `self.eos_tokens`.

        Args:
            logw_next (torch.tensor): The log probabilities for the next tokens.

        Returns:
            (LazyWeights): Processed log probabilities for the next tokens.
        """
        # This is ugly, but it's useful for all potentials to adhere to the convention
        # of keeping the EOS token at the end of the weights array.
        logw_next = logw_next[: len(self.token_maps.decode)]
        logw_next = logw_next.log_softmax(dim=0)
        _logw_next = torch.full((len(self.vocab) + 1,), float('-inf'), dtype=logw_next.dtype, device=logw_next.device)
        _logw_next[: len(self.vocab)] = logw_next[
            ~torch.isin(torch.arange(len(logw_next)), torch.tensor(self.token_maps.eos_idxs))
        ]
        _logw_next[-1] = torch.logsumexp(logw_next[self.token_maps.eos_idxs], dim=0).item()
        return self.make_lazy_weights(_logw_next.float().cpu().numpy())

    async def logw_next(self, context):
        """Get log probabilities for next tokens given the prompt and `context`.

        Args:
            context (List[bytes]): A sequence of bytes tokens.

        Returns:
            (LazyWeights): Log probabilities for next tokens and EOS.
        """
        logw_next = self._maybe_temper(
            await self.model.next_token_logprobs(
                self.prompt_ids + self.encode_tokens(context)
            )
        )
        return self._process_logw_next(logw_next)

    async def batch_logw_next(self, contexts):
        """Get log probabilities for next tokens given the prompt and `context`, for a batch of contexts.

        Args:
            contexts (list[list[bytes]]): A list of sequences of bytes tokens.

        Returns:
            (List[LazyWeights]): Log probabilities for next tokens and EOS for each context.
        """
        logw_nexts = self._maybe_temper(
            await self.model.batch_next_token_logprobs(
                [self.prompt_ids + self.encode_tokens(context) for context in contexts]
            )
        )
        return [
            self._process_logw_next(logw_next)
            for logw_next in logw_nexts
        ]

    def __repr__(self):
        return f"PromptedLLM(prompt={self.prompt!r})"

    def spawn(self):
        """
        Spawn a new PromptedLLM with the same prompt and eos tokens.

        Returns:
            (PromptedLLM): A new PromptedLLM with the same prompt and eos tokens.

        Note:
            This is a shallow copy. The new PromptedLLM will share the underlying AsyncLM instance.
        """
        return PromptedLLM(
            self.model,
            prompt_ids=self.prompt_ids.copy(),
            eos_tokens=self._eos_tokens.copy(),
            temperature=self.temperature,
        )

    def spawn_new_eos(self, eos_tokens):
        """
        Create a new PromptedLLM with a different set of end-of-sequence tokens.

        Args:
            eos_tokens (list[bytes]): A list of tokens to treat as end-of-sequence tokens.

        Returns:
            (PromptedLLM): A new PromptedLLM with the specified end-of-sequence tokens.
                The new model will have the same prompt_ids as `self`.
        """
        return PromptedLLM(
            self.model,
            prompt_ids=self.prompt_ids.copy(),
            eos_tokens=eos_tokens.copy(),
            temperature=self.temperature,
        )

    def to_autobatched(self):
        raise ValueError("PromptedLLMs are autobatched by default.")



================================================
FILE: genlm/control/potential/built_in/wcfg.py
================================================
import numpy as np
from genlm.grammar import CFG, Earley, Float, Boolean
from genlm.grammar.lark_interface import LarkStuff
from genlm.grammar.cfglm import _gen_nt

from genlm.control.constant import EOS
from genlm.control.potential.base import Potential


def _add_eos(cfg, eos):
    S = _gen_nt("<START>")
    cfg_eos = cfg.spawn(S=S)
    cfg_eos.V.add(eos)
    cfg_eos.add(cfg.R.one, S, cfg.S, eos)
    for r in cfg:
        cfg_eos.add(r.w, r.head, *r.body)
    return cfg_eos


class WCFG(Potential):
    """
    A weighted context-free grammar potential.

    This class wraps a `genlm_grammar.CFG` and provides methods for computing the log-weight of a sequence,
    the prefix log-weight of a sequence, and the log-weights of the next token given a sequence.
    """

    def __init__(self, cfg):
        """
        Initialize the WCFG potential.

        Args:
            cfg (genlm_grammar.CFG): The context-free grammar configuration to use.
                The CFG must in the Float semiring.
        """
        # TODO: convert to LogSemiring to handle underflow
        if cfg.R is not Float:
            raise ValueError("cfg semiring must be Float")
        self.cfg = cfg  # cfg before prefix transform
        self.cfg_eos = _add_eos(cfg, EOS)  # augmented with eos
        self.model = Earley(self.cfg_eos.prefix_grammar)
        super().__init__(vocabulary=list(cfg.V))

    @classmethod
    def from_string(cls, grammar, to_bytes=True, **kwargs):
        """Create a WCFG from a string.

        Args:
            grammar (str): The string grammar specification to create the WCFG from.
            to_bytes (bool, optional): Whether to convert the WCFG terminals to indivudual bytes.
                Defaults to True.
            **kwargs (dict): Additional arguments passed to the WCFG constructor.

        Returns:
            (WCFG): The created WCFG.
        """
        cfg = CFG.from_string(grammar, Float)
        if to_bytes:
            cfg = cfg.to_bytes()
        return cls(cfg, **kwargs)

    async def complete(self, context):
        """
        Compute the log weight of `context` under the WCFG.

        For example, if the WCFG accepts "cat" and "car" with weights $w_{cat}$ and $w_{car}$:\n
        - `complete("c")` returns $-\\infty$ since this sequence is not accepted by the WCFG\n
        - `complete("cat")` returns $\\log(w_{cat})$\n
        - `complete("d")` returns $-\\infty$ since this sequence is not accepted by the WCFG

        Args:
            context (list): A sequence of tokens in the WCFG's alphabet.

        Returns:
            (float): The log weight of `context` under the WCFG.
        """
        w = self.model([*context, EOS])
        return np.log(w) if w > 0 else float("-inf")

    async def prefix(self, context):
        """
        Compute the log prefix weight of `context` under the WCFG.

        This corresponds to the log of the sum of the weights of all sequences with prefix `context`.

        For example, if the WCFG accepts "cat" and "car" with weights $w_{cat}$ and $w_{car}$:\n
        - `prefix("c")` returns $\\log(w_{cat} + w_{car})$\n
        - `prefix("cat")` returns $\\log(w_{cat})$\n
        - `prefix("d")` returns $-\\infty$ since the WCFG does not accept any sequences with prefix "d"

        Args:
            context (list): A sequence of tokens in the WCFG's alphabet.

        Returns:
            (float): The log prefix weight of `context` under the WCFG.
        """
        w = self.model(context)
        return np.log(w) if w > 0 else float("-inf")

    async def logw_next(self, context):
        """
        Compute the next token log weights given `context`.

        Args:
            context (list): A sequence of tokens in the WCFG's alphabet.

        Returns:
            (LazyWeights): The log weights for the next tokens and EOS given `context`.
        """
        ws = self.model.next_token_weights(self.model.chart(context))
        ws = ws.trim().normalize()

        ws_array = np.array([ws[x] for x in self.vocab_eos])
        mask = ws_array > 0
        log_ws = np.full_like(ws_array, float("-inf"), dtype=np.float64)
        log_ws[mask] = np.log(ws_array[mask])

        return self.make_lazy_weights(log_ws)

    def clear_cache(self):
        """Clear the internal cache of the parser."""
        self.model.clear_cache()

    def __repr__(self):
        return f"WCFG(cfg={self.cfg!r})"

    def _repr_html_(self):
        return self.cfg._repr_html_()

    def spawn(self):
        """Spawn a new WCFG."""
        return WCFG(self.cfg)


class BoolCFG(Potential):
    """BoolCFG represents a boolean context-free grammar."""

    def __init__(self, cfg):
        if cfg.R != Boolean:
            cfg = cfg.map_values(lambda x: Boolean(x > 0), Boolean)
        self.cfg = cfg  # cfg before prefix transform
        self.cfg_eos = _add_eos(cfg, EOS)  # augmented with eos
        self.model = Earley(self.cfg_eos.prefix_grammar)
        super().__init__(vocabulary=list(cfg.V))

    @classmethod
    def from_lark(cls, lark_string, charset="core"):
        """
        Create a BoolCFG instance from a Lark grammar string.

        The output grammar will be defined at the byte-level.

        Args:
            lark_string (str): The Lark grammar string to parse. See Lark documentation for correct syntax.
            charset (str): The character set to use. Defaults to "core".
                See `genlm-grammar` documentation for more details.

        Returns:
            (BoolCFG): An instance of BoolCFG created from the provided Lark grammar.
        """
        byte_cfg = LarkStuff(lark_string).byte_cfg(charset=charset)
        return cls(byte_cfg)

    async def complete(self, context):
        """
        Checks whether the context is accepted by the CFG.

        Args:
            context (list): A sequence of tokens in the CFG's alphabet.

        Returns:
            (float): Log weight for whether `context` is accepted by the CFG.
        """
        w = self.model([*context, EOS])
        return 0 if w.score else float("-inf")

    async def prefix(self, context):
        """
        Checks whether `context` is accepted as a prefix by the CFG, i.e.,
        whether there exists a completion to `context` that is accepted by the CFG.

        Args:
            context (list): A sequence of tokens in the CFG's alphabet.

        Returns:
            (float): Log weight for whether `context` is accepted as a prefix by the CFG.
        """
        if not context:  # FIX: this is a hack to handle the empty string because genlm-grammar doesn't support it
            return 0
        w = self.model(context)
        return 0 if w.score else float("-inf")

    async def logw_next(self, context):
        """
        Compute the next token log weights given `context`.

        Args:
            context (list): A sequence of tokens in the CFG's alphabet.

        Returns:
            (LazyWeights): The log weights for the next tokens and EOS given `context`.
        """
        ws = self.model.next_token_weights(self.model.chart(context))
        log_ws = np.array([0 if ws[x].score else float("-inf") for x in self.vocab_eos])
        return self.make_lazy_weights(log_ws)

    async def batch_logw_next(self, contexts):
        """
        Batch version of `logw_next`.

        Args:
            contexts (list): A list of sequences of tokens in the CFG's alphabet.

        Returns:
            (list): A list of log-weights for next token, one per context.
        """
        Ws = []
        for context in contexts:
            ws = self.model.next_token_weights(self.model.chart(context))
            log_ws = np.array(
                [0 if ws[x].score else float("-inf") for x in self.vocab_eos]
            )
            Ws.append(self.make_lazy_weights(log_ws))
        return Ws

    def spawn(self):
        """Spawn a new BoolCFG."""
        return BoolCFG(self.cfg)

    def clear_cache(self):
        """Clear the internal cache of the parser."""
        self.model.clear_cache()

    def __repr__(self):
        return f"BoolCFG(cfg={self.cfg!r})"

    def _repr_html_(self):
        return self.cfg._repr_html_()



================================================
FILE: genlm/control/potential/built_in/wfsa.py
================================================
import string
import numpy as np
from arsenal.maths import logsumexp

from genlm.grammar import Float, Log, WFSA as BaseWFSA
from genlm.grammar.lark_interface import interegular_to_wfsa

from genlm.control.potential.base import Potential


class WFSA(Potential):
    """
    A weighted finite state automaton (WFSA) potential.

    This class wraps a `genlm_grammar.WFSA` and provides methods for computing the log-weight of a context,
    the prefix log-weight of a context, and the log-weights of the next token given a context.

    Attributes:
        wfsa (genlm_grammar.WFSA): The weighted finite state automaton used for potential calculations.
    """

    def __init__(self, wfsa):
        """
        Initializes the WFSA potential.

        Args:
            wfsa (genlm_grammar.WFSA): The weighted finite state automaton.

        Raises:
            ValueError: If the semiring of the provided WFSA is not Float or Log.

        Note:
            The WFSA will be converted to the Log semiring to avoid underflow if the semiring is Float.
        """
        if wfsa.R not in (Float, Log):
            raise ValueError(f"Unsupported semiring: {wfsa.R}")

        if wfsa.R is Float:
            self.wfsa = self._convert_to_log(wfsa)
        else:
            self.wfsa = wfsa

        self.cache = {(): self.wfsa.epsremove.start}
        super().__init__(vocabulary=list(self.wfsa.alphabet))

    @classmethod
    def from_regex(cls, pattern, charset=None, to_bytes=True):
        """
        Create a WFSA from a regex pattern.

        Args:
            pattern (str): The regex pattern to convert into a WFSA.
            charset (set): The character set to use for negative character classes.
                Defaults to characters in string.printable.
            to_bytes (bool): Whether to convert the WFSA transitions to bytes.
                Defaults to True. When set to False, the WFSA transitions will be strings.

        Returns:
            (WFSA): An instance of the WFSA class.

        Note:
            The transition weights are automatically normalized to form a probability distribution.
            For each state, the weights of all outgoing transitions (including final state transitions)
            sum to 1.0. This means if a state has n possible transitions, each transition will have
            weight 1/n. To create a WFSA from a regex with non-probabilistic transitions, use `BoolFSA`.
        """
        charset = charset or set(string.printable)
        wfsa = interegular_to_wfsa(pattern, charset=charset)
        if to_bytes:
            wfsa = wfsa.to_bytes()
        return cls(wfsa=wfsa)

    @staticmethod
    def _convert_to_log(wfsa):
        """Convert a WFSA from the Float semiring to the Log semiring."""
        assert wfsa.R is Float
        assert isinstance(wfsa, BaseWFSA)
        new = BaseWFSA(Log)

        for i, w in wfsa.I:
            new.add_I(i, Log(np.log(w)))

        for i, w in wfsa.F:
            new.add_F(i, Log(np.log(w)))

        for i, a, j, w in wfsa.arcs():
            new.add_arc(i, a, j, Log(np.log(w)))

        return new

    def _consume(self, bs):
        # XXX implement cache eviction
        bs = tuple(bs)

        try:
            return self.cache[bs]
        except KeyError:
            pass

        wfsa = self.wfsa.epsremove
        curr = wfsa.R.chart()
        prev = self._consume(bs[:-1])
        for i in prev:
            for j, w in wfsa.arcs(i, bs[-1]):
                curr[j] += prev[i] * w

        self.cache[bs] = curr

        return curr

    async def complete(self, context):
        """
        Computes the log weight of the context under the weighted language represented by the WFSA.

        For example, if the WFSA accepts "cat" and "car" with weights $w_{cat}$ and $w_{car}$:\n
        - `complete("c")` returns $-\\infty$ since this sequence is not accepted by the WFSA\n
        - `complete("cat")` returns $\\log(w_{cat})$\n
        - `complete("d")` returns $-\\infty$ since this sequence is not accepted by the WFSA

        Args:
            context (list): A sequence of tokens in the WFSA's alphabet.

        Returns:
            (float): Log weight of context under the WFSA.
        """
        # TODO: optimize to use _consume cache
        return self.wfsa(context).score

    def _prefix(self, context):
        curr = self._consume(context)

        if not curr:
            return float("-inf"), curr

        bkwd = self.wfsa.epsremove.backward
        log_ctx_w = logsumexp([(curr[i] * bkwd[i]).score for i in curr])

        if np.isnan(log_ctx_w):
            return float("-inf"), curr

        return log_ctx_w, curr

    async def prefix(self, context):
        """
        Computes the prefix log weight of `context` under the WFSA.

        This corresponds to the log of the sum of the weights of all sequences with prefix `context`.

        For example, if the WFSA accepts "cat" and "car" with weights $w_{cat}$ and $w_{car}$:\n
        - `prefix("c")` returns $\\log(w_{cat} + w_{car})$\n
        - `prefix("ca")` returns $\\log(w_{cat})$\n
        - `prefix("d")` returns $-\\infty$ since the WFSA does not accept any sequences with prefix "d"

        Args:
            context (list): A sequence of tokens in the WFSA's alphabet.

        Returns:
            (float): Log weight of `context` as a prefix under the WFSA.
        """
        return self._prefix(context)[0]

    async def logw_next(self, context):
        """Returns next token log weights given `context`.

        Args:
            context (list): A sequence of tokens in the WFSA's alphabet.

        Returns:
            (LazyWeights): Log-weights for next token and EOS.
        """
        log_ctx_w, curr = self._prefix(context)

        if log_ctx_w == float("-inf"):
            raise ValueError(f"Context {context!r} has zero weight.")

        bkwd = self.wfsa.epsremove.backward

        ws = self.wfsa.R.chart()
        for i in curr:
            for b, j, w in self.wfsa.epsremove.arcs(i=i):
                ws[b] += curr[i] * w * bkwd[j]

        ws[self.eos] = self.wfsa.R.zero
        for j, w in self.wfsa.epsremove.F:
            ws[self.eos] += curr[j] * w

        log_ws = np.array([ws[b].score for b in self.vocab_eos]) - log_ctx_w

        return self.make_lazy_weights(log_ws)

    def _repr_svg_(self):
        return self.wfsa._repr_svg_()

    def __repr__(self):
        return f"WFSA(wfsa={self.wfsa!r})"

    def spawn(self):
        cls = type(self)
        return cls(wfsa=self.wfsa)

    def clear_cache(self):
        self.cache = {(): self.wfsa.epsremove.start}


class BoolFSA(WFSA):
    """Boolean FSA potential."""

    async def prefix(self, context):
        """
        Computes whether the context is accepted as a prefix by the FSA.

        Args:
            context (list): A sequence of tokens in the WFSA's alphabet.

        Returns:
            (float): `0` if the context is accepted as a prefix, `-inf` otherwise.
        """
        prefix_w = await super().prefix(context)
        if prefix_w > float("-inf"):
            return 0
        return float("-inf")

    async def complete(self, context):
        """
        Computes whether the context is accepted by the FSA.

        Args:
            context (list): A sequence of tokens in the WFSA's alphabet.

        Returns:
            (float): `0` if the context is accepted, `-inf` otherwise.
        """
        complete_w = await super().complete(context)
        if complete_w > float("-inf"):
            return 0
        return float("-inf")

    async def logw_next(self, context):
        """
        Returns next token log weights given `context`.

        Args:
            context (list): A sequence of tokens in the WFSA's alphabet.

        Returns:
            (LazyWeights): Boolean log-weights for next token.
        """
        logw_next = await super().logw_next(context)
        return logw_next.spawn(
            new_weights=np.where(
                logw_next.weights > float("-inf"), 0, logw_next.weights
            )
        )

    async def batch_logw_next(self, contexts):
        """
        Returns next token log weights for a batch of contexts.

        Args:
            contexts (list): The list of contexts.

        Returns:
            (list): List of log-weights for next token, one per context.
        """
        logw_nexts = await super().batch_logw_next(contexts)
        return [
            logw_next.spawn(
                new_weights=np.where(
                    logw_next.weights > float("-inf"), 0, logw_next.weights
                )
            )
            for logw_next in logw_nexts
        ]

    def __repr__(self):
        return f"BoolFSA(wfsa={self.wfsa!r})"



================================================
FILE: genlm/control/sampler/__init__.py
================================================
from .token import DirectTokenSampler, SetTokenSampler, AWRS
from .set import EagerSetSampler, TopKSetSampler
from .sequence import SMC, SequenceModel
from genlm.control.potential import Potential


def direct_token_sampler(potential):
    """Create a `DirectTokenSampler` that samples directly from a potential's vocabulary.

    See `DirectTokenSampler` for more details.

    Args:
        potential (Potential): The potential function to sample from. Should have an efficient logw_next method.

    Returns:
        (DirectTokenSampler): A sampler that directly samples tokens from the potential's vocabulary.
    """
    assert isinstance(potential, Potential)
    return DirectTokenSampler(potential)


def eager_token_sampler(iter_potential, item_potential):
    """Create a `SetTokenSampler` that uses the `EagerSetSampler` to sample a set of tokens.

    See `EagerSetSampler` for more details.

    Args:
        iter_potential (Potential): A potential function defined over a vocabulary of iterables.
        item_potential (Potential): A potential function defined over a vocabulary of items which are elements of the iterables.

    Returns:
        (SetTokenSampler): A sampler that wraps an `EagerSetSampler`.

    Note:
        This is the fastest sampler in most cases.
    """
    return SetTokenSampler(EagerSetSampler(iter_potential, item_potential))


def topk_token_sampler(iter_potential, item_potential, K):
    """Create a `SetTokenSampler` that uses the `TopKSetSampler` to sample a set of tokens.

    See `TopKSetSampler` for more details.

    Args:
        iter_potential (Potential): A potential function defined over a vocabulary of iterables.
        item_potential (Potential): A potential function defined over a vocabulary of items which are elements of the iterables.
        K (int|None): The `K` parameter for the `TopKSetSampler`.

    Returns:
        (SetTokenSampler): A sampler that wraps an `TopKSetSampler`.
    """
    return SetTokenSampler(TopKSetSampler(iter_potential, item_potential, K))


__all__ = [
    "AWRS",
    "direct_token_sampler",
    "eager_token_sampler",
    "topk_token_sampler",
    "DirectTokenSampler",
    "EagerSetSampler",
    "TopKSetSampler",
    "SetTokenSampler",
    "Importance",
    "SMC",
    "SequenceModel",
]



================================================
FILE: genlm/control/sampler/sequence.py
================================================
import numpy as np
from genlm.grammar import Float
from arsenal.maths import logsumexp
from functools import cached_property
from dataclasses import dataclass
from arsenal import colors

from llamppl import Model
from llamppl import smc_standard

from genlm.control.potential import Potential
from genlm.control.constant import EOS, EndOfSequence
from genlm.control.sampler.token import TokenSampler


class SMC:
    """This class implements sequential Monte Carlo (SMC) inference for controlled text generation.
    The generation process works as follows:

    1. Token Sampling: At each step, the `unit_sampler` is used to extend each particle (candidate sequence)
       by sampling a new token. This grows all sequences by one token at a time. The sampler also outputs
       an importance weight with each extension to correct for the myopic nature of token-by-token sampling.

    2. Critic Evaluation: If a `critic` is provided, it scores the updated sequences (via it's `score` method),
       reweighting the particles based on how well they satisfy the constraints encoded by the critic.

    3. Resampling: When the effective sample size (ESS) falls below the threshold,
       particles are resampled according to their weights. This helps focus computation
       on more promising sequences.

    4. Termination: The process continues until either:\n
        - All sequences reach an end-of-sequence (EOS) token\n
        - The maximum token length is reached

    If a critic is provided, the resulting sequences are properly weighted with respect to the product of the unit sampler's
    target potential and the critic potential (`unit_sampler.target * critic`). If a critic is not provided,
    the resulting sequences are weighted with respect to the unit sampler's target potential.

    Args:
        unit_sampler (TokenSampler): The sampler that generates tokens.
        critic (Potential, optional): A potential function that guides the generation process
            by scoring candidate sequences. Must have the same token type as the unit_sampler.

    Raises:
        ValueError: If unit_sampler is not a TokenSampler, if critic is not a Potential,
            or if the token types of unit_sampler and critic don't match.
    """

    def __init__(self, unit_sampler, critic=None):
        if not isinstance(unit_sampler, TokenSampler):
            raise ValueError("`unit_sampler` must be a TokenSampler")

        if critic:
            if not isinstance(critic, Potential):
                raise ValueError("`critic` must be a Potential")
            if not unit_sampler.token_type == critic.token_type:
                raise ValueError(
                    "`critic` must have the same token type as the `unit_sampler`. "
                    f"Got {unit_sampler.token_type} and {critic.token_type}."
                    + (
                        "\nMaybe you forgot to coerce the critic to the token type of the unit sampler? See `Coerce`."
                        if unit_sampler.token_type.is_iterable_of(critic.token_type)
                        else ""
                    )
                )

        self.unit_sampler = unit_sampler
        self.critic = critic

    async def __call__(
        self,
        n_particles,
        ess_threshold,
        max_tokens,
        verbosity=0,
        json_path=None,
        **kwargs,
    ):
        """Generate sequences using sequential Monte Carlo inference.

        Args:
            n_particles (int): Number of particles (candidate sequences) to maintain during
                generation. Higher values provide better exploration but require more
                computation.
            ess_threshold (float): Effective sample size threshold for resampling,
                expressed as a fraction of the number of particles. When ESS falls below
                this value, particles are resampled according to their weights. Should be between 0 and 1.
                Higher values lead to more frequent resampling. Note that when ess_threshold = 0,
                the critic is only applied at the end of the generation (if it is provided).
            max_tokens (int): Maximum number of tokens to generate per sequence. Generation
                may terminate earlier if all sequences reach an EOS token.
            verbosity (int, optional): Verbosity level for the SMC algorithm. 0 is silent, 1 prints the
                particles at each step. Default is 0.
            json_path (str, optional): JSON file path for saving a record of the inference run.
                This can be used in conjunction with the `InferenceVisualizer` to visualize the inference run.
            **kwargs (dict): Additional keyword arguments to pass to the SMC algorithm.
                See the `llamppl.inference.smc_standard` documentation for more details.

        Returns:
            (Sequences): A container holding the generated sequences, their importance weights, and
                other metadata from the generation process.
        """
        model = SequenceModel(
            unit_sampler=self.unit_sampler,
            critic=self.critic,
            max_tokens=max_tokens,
            verbosity=verbosity,
            twist_with_critic=ess_threshold > 0,
        )

        particles = await smc_standard(
            model=model,
            n_particles=n_particles,
            ess_threshold=ess_threshold,
            json_file=json_path,
            **kwargs,
        )

        return Sequences(*_unpack_particles(particles))

    async def cleanup(self):
        """Clean up resources used by the inference engine.

        This method should be called when the InferenceEngine is no longer needed.

        Example:
            ```python
            sampler = SequenceSampler(unit_sampler, critic)
            try:
                sequences = await sampler(n_particles=10, ess_threshold=0.5, max_tokens=20)
            finally:
                await sampler.cleanup()
            ```
        """
        await self.unit_sampler.cleanup()
        if self.critic:
            await self.critic.cleanup()


@dataclass
class Sequences:
    """Container for sequence samples with their weights and probabilities.

    Args:
        contexts (list): List of token sequences generated by the sampler.
        log_weights (list): Log importance weights for each sequence.

    Attributes:
        size (int): Number of sequences in the container.
        logp (float): Sum of log probabilities across all sequences.
        log_total (float): Log of the sum of importance weights.
        log_ml (float): Log marginal likelihood estimate.
        log_normalized_weights (list): Log weights normalized to sum to 1.
        log_ess (float): Log of the effective sample size.
        ess (float): Effective sample size of the particle population.
    """

    contexts: list
    log_weights: list

    def __post_init__(self):
        assert len(self.contexts) == len(self.log_weights)

        if not isinstance(self.log_weights, np.ndarray):
            self.log_weights = np.array(self.log_weights)

        self.size = len(self.contexts)

        # Handle case where all weights are -inf
        if np.all(np.isneginf(self.log_weights)):
            self.log_total = float("-inf")
            self.log_ml = float("-inf")
            self.log_normalized_weights = np.full_like(self.log_weights, float("-inf"))
            self.log_ess = float("-inf")
            self.ess = 0.0
            return

        self.log_total = logsumexp(self.log_weights)
        max_weight = max(self.log_weights)
        self.log_ml = (
            np.log(np.mean(np.exp(self.log_weights - max_weight))) + max_weight
        )
        self.log_normalized_weights = self.log_weights - self.log_total
        self.log_ess = -logsumexp(2 * self.log_normalized_weights)
        self.ess = np.exp(self.log_ess)

    @cached_property
    def posterior(self):
        """Compute the estimated posterior distribution over sequences.

        The probability of a sequence corresponds to its normalized weight. The probabilities
        of duplicate sequences are summed.

        Returns:
            (Float.chart): A normalized chart mapping sequences to their posterior probabilities,
                sorted in descending order by probability.
        """
        posterior = Float.chart()
        for sequence, prob in zip(self.contexts, self.normalized_weights):
            posterior[tuple(sequence)] += prob
        return posterior.normalize().sort_descending()

    @cached_property
    def decoded_posterior(self):
        """Compute posterior distribution over completed UTF-8 decodable sequences.

        Filters for sequences that:\n
        1. End with an EndOfSequence token\n
        2. Can be decoded as UTF-8 strings

        The probability of each sequence corresponds to its normalized weight among completed and decodable sequences.
        Probabilities of duplicate sequences (after decoding) are summed.

        To obtain the posterior distribution over all byte sequences, use `self.posterior`.

        Returns:
            (Float.chart): A normalized chart mapping decoded string sequences to their
                posterior probabilities, sorted in descending order by probability.
                Only includes sequences that meet both filtering criteria.
        """
        posterior = Float.chart()
        for sequence, w in zip(self.contexts, np.exp(self.log_weights)):
            if sequence and isinstance(sequence[-1], EndOfSequence):
                try:
                    string_sequence = b"".join(sequence[:-1]).decode("utf-8")
                    posterior[string_sequence] += w
                except UnicodeDecodeError:
                    pass
        return posterior.normalize().sort_descending()

    @property
    def normalized_weights(self):
        """Return exponential of normalized log weights."""
        if np.all(np.isneginf(self.log_weights)):
            return np.full_like(self.log_weights, 0.0)
        return np.exp(self.log_normalized_weights)

    def __len__(self):
        return self.size

    def __iter__(self):
        return iter(zip(self.contexts, self.log_weights))

    def __getitem__(self, i):
        return self.contexts[i], self.log_weights[i]

    def __str__(self):
        return str(self.decoded_posterior)

    def _repr_html_(self):
        return self.decoded_posterior._repr_html_()

    def __repr__(self):
        return str(self.decoded_posterior)

    def show(self):
        for p in sorted(self, reverse=True):
            print(p)


class SequenceModel(Model):
    def __init__(
        self,
        unit_sampler,
        critic=None,
        max_tokens=float("inf"),
        verbosity=0,
        twist_with_critic=True,
    ):
        assert max_tokens > 0

        super().__init__()
        self.token_ctx = []
        self.unit_sampler = unit_sampler
        self.max_tokens = max_tokens
        self.critic = critic
        self.logp = 0
        self.verbosity = verbosity
        self.twist_with_critic = twist_with_critic

    async def start(self):
        start_w = await self.unit_sampler.start_weight()
        if start_w == float("-inf"):
            raise ValueError(
                "Start weight is -inf (log(0)). This is likely because a potential assigns zero weight to "
                "the empty sequence under `prefix`, which violates the potential contract."
            )
        self.score(start_w)

    async def step(self):
        unit = await self.call(self.unit_sampler)
        self.token_ctx.append(unit)

        inf_weight = self.weight == float("-inf")
        if inf_weight:
            if self.critic:
                assert self.twist_amount != float("-inf")
            self.finish()
            return

        if self.critic and self.twist_with_critic:
            twist_amt = await self.critic.score(self.token_ctx)
            if twist_amt != float("-inf"):
                self.twist(twist_amt)
            else:
                self.score(twist_amt)
                self.finish()
                return

        if self.verbosity > 0:
            print(self.__repr__())

        self.max_tokens -= 1
        if self.max_tokens == 0 or self.token_ctx[-1] is EOS:
            self.finish()
            if self.critic:
                if not self.twist_with_critic:
                    twist_amt = await self.critic.score(self.token_ctx)
                self.score(twist_amt)
            return

    def __repr__(self):
        return (
            f"{self.weight:.2f}:\t"
            + colors.magenta % "["
            + (colors.magenta % "|").join(repr(y) for y in self.token_ctx)
            + colors.magenta % "]"
        )

    def string_for_serialization(self):
        return "|".join(repr(y) for y in self.token_ctx)

    def immutable_properties(self):
        return set(["unit_sampler", "critic"])


def _unpack_particles(particles):
    contexts, logws = map(
        list,
        zip(
            *[
                (p.token_ctx, float("-inf") if np.isnan(p.weight) else p.weight)
                for p in particles
            ]
        ),
    )
    return contexts, logws



================================================
FILE: genlm/control/sampler/set.py
================================================
import numpy as np
from genlm.grammar import Float
from arsenal.maths import sample_dict
from arsenal.datastructures import LocatorMaxHeap
from abc import ABC, abstractmethod

from genlm.control.util import load_async_trie


class SetSampler(ABC):
    """Base class for set samplers.

    A set sampler samples a weighted set of tokens from a the vocabulary of a `target` potential.

    Given a context of tokens $x_1, \\ldots, x_{n-1}$ in the target potential's vocabulary and a sampled set of tokens $S \\subseteq \\textsf{target.vocab_eos}$,
    the log-weight associated with each token $x_n$ must correspond to:

    $$
        \\textsf{target.logw_next}(x_n | x_1, \\ldots, x_{n-1}) - \\log \\Pr(x_n \\in S)
    $$

    where $\\Pr(x_n \\in S)$ is the probability the token was included in a sampled set.

    Attributes:
        target (Potential): The target potential with respect to which the set's weights are computed.
    """

    def __init__(self, target):
        self.target = target

    @abstractmethod
    async def sample_set(self, context):
        """Sample a weighted set of tokens from the target potential's vocabulary."""
        pass  # pragma: no cover

    async def cleanup(self):
        pass  # pragma: no cover


class TrieSetSampler(SetSampler):
    """
    TrieSetSampler is a specialized set sampler that utilizes a trie data structure to efficiently sample a weighted set of tokens.

    This sampler is designed to work with two potentials:\n
    - a potential over a vocabulary of iterables (`iter_potential`) and\n
    - a potential over a vocabulary of items which are the elements of the iterables (`item_potential`).

    For example, if `iter_potential` is a potential over byte sequences, then `item_potential` is a potential over bytes.

    The target potential is the product of `iter_potential` and the `item_potential` coerced to operate on the token type of `iter_potential`. Thus,
    `TrieSetSampler`s sample tokens from the `iter_potential`'s vocabulary.
    """

    def __init__(self, iter_potential, item_potential):
        """
        Initialize the `TrieSetSampler`.

        Args:
            iter_potential (Potential): The potential defined over a vocabulary of iterables.
            item_potential (Potential): The potential defined over a vocabulary of items.

        Raises:
            ValueError: If the token type of `iter_potential` is not an iterable of the token type of `item_potential`.
        """
        if not iter_potential.token_type.is_iterable_of(item_potential.token_type):
            raise ValueError(
                "Token type of `iter_potential` must be an iterable of token type of `item_potential`. "
                f"Got {iter_potential.token_type} and {item_potential.token_type}."
            )
        self.iter_potential = iter_potential
        self.item_potential = item_potential
        self.f = lambda context: [item for items in context for item in items]

        super().__init__(
            iter_potential * item_potential.coerce(iter_potential, f=self.f)
        )

        self.trie_executor = load_async_trie(
            self.iter_potential.vocab_eos, backend="parallel"
        )
        self.trie = self.trie_executor.trie

        vocab_eos = self.target.vocab_eos
        word2leaf = self.trie.word2leaf
        lookup = self.target.lookup

        common_tokens = set(vocab_eos) & set(word2leaf)

        self.leaf_to_token_id = dict(
            (word2leaf[token], lookup[token]) for token in common_tokens
        )

    async def sample_set(self, context):
        """
        Sample a weighted set of tokens given a context.

        Args:
            context (list): The sequence to condition on.

        Returns:
            (LazyWeights, float): A weighted set of tokens and the log-probability of the sampled set.

        Raises:
            NotImplementedError: If the method is not implemented in subclasses.
        """
        raise NotImplementedError(
            "Subclasses must implement sample_set"
        )  # pragma: no cover

    async def cleanup(self):
        """
        Cleanup the TrieSetSampler. It is recommended to call this method at the end of usage.
        """
        await self.trie_executor.cleanup()


class EagerSetSampler(TrieSetSampler):
    """
    A trie-based set sampler that implements an eager sampling strategy
    for generating a set of tokens.

    An `EagerSetSampler` samples tokens by incrementally sampling items from the item-wise product of the `iter_potential` and `item_potential`.
    The sampled set is the set of sequences of items that correspond to valid tokens in `iter_potential`'s vocabulary.
    """

    async def sample_set(self, context, draw=None):
        """
        Sample a set of tokens given a context.

        Args:
            context (list): A sequence of tokens in the `iter_potential`'s vocabulary.

        Returns:
            (LazyWeights, float): A weighted set of tokens and the log-probability of the sampled set.
        """
        if draw is None:
            draw = sample_dict
        iter_logws = await self.iter_potential.logw_next(context)
        item_ws = await self.trie_executor.weight_sum(iter_logws.exp().weights)

        logws = self.target.alloc_logws()
        curr = self.trie.root
        coerced_ctx = self.f(context)
        subtokens = []
        logp, logw = 0, 0

        while True:
            children = self.trie.children[curr]
            item_w_curr = item_ws[curr]
            item_ws1 = Float.chart(
                {a: item_ws[c] / item_w_curr for a, c in children.items()}
            )

            if None in item_ws1:
                leaf = children[None]
                token = self.trie.leaf2word[leaf]
                token_id = self.leaf_to_token_id[leaf]
                logws[token_id] = iter_logws[token] + logw - logp

            item_logws2 = await self.item_potential.logw_next(coerced_ctx + subtokens)
            item_ws2 = item_logws2.exp().materialize()
            w_next = (item_ws1 * item_ws2).trim()

            if not w_next:
                break

            ps = w_next.normalize()
            b = draw(ps)
            logp += np.log(ps[b])
            logw += item_logws2[b]

            if b == self.target.eos:
                assert not subtokens, "subtokens should be empty at EOS."
                logws[-1] = iter_logws[self.target.eos] + logw - logp
                break

            subtokens.append(b)
            curr = children[b]

        return self.target.make_lazy_weights(logws), logp


class TopKSetSampler(TrieSetSampler):
    """
    A trie-based set sampler that lazily enumerates the top K tokens by weight in the target,
    and samples an additional "wildcard" token to ensure absolute continuity.

    Warning:
        This sampler is not guaranteed to be correct if the `item_potential`'s
        prefix weights do not monotonically decrease with the length of the context.
        That is, $\\textsf{item_potential.prefix}(x) \\leq \\textsf{item_potential.prefix}(xy)$ for all sequences of items $x, y$.
    """

    def __init__(self, iter_potential, item_potential, K):
        """
        Initialize the TopKSetSampler.

        Args:
            iter_potential (Potential): The potential defined over a vocabulary of iterables.
            item_potential (Potential): The potential defined over a vocabulary of items.
            K (int|None): The number of top tokens to enumerate. If None, all tokens are enumerated.
        """
        if K is not None and K <= 0:
            raise ValueError("K must be greater than 0 or None")
        super().__init__(iter_potential, item_potential)
        self.K = K

    async def sample_set(self, context, draw=None):
        """
        Sample a set of tokens given a context.

        Args:
            context (list): A sequence of tokens in the `iter_potential`'s vocabulary.

        Returns:
            (LazyWeights, float): A weighted set of tokens and the log-probability of the sampled set.
        """
        if draw is None:
            draw = sample_dict
        iter_logws = await self.iter_potential.logw_next(context)
        max_logws = await self.trie_executor.weight_max(iter_logws.weights)

        k = 0
        logws = self.target.alloc_logws()
        sampled = self.target.alloc_logws(default=False)

        async for token_id, logw in self._lazy_enum(context, max_logws):
            logws[token_id] = logw
            sampled[token_id] = True
            k += 1
            if self.K is not None and k >= self.K:
                break

        logp_wc = 0
        if self.K is not None and k == self.K:
            # Get the distribution over wildcard tokens
            iter_ws = iter_logws.exp()
            W_wc = Float.chart(
                {
                    token_id: iter_ws[token]
                    for token_id, token in enumerate(self.target.vocab_eos)
                    if not sampled[token_id]
                }
            )

            # if W_wc is non-empty, sample a wildcard token to ensure absolute continuity
            if W_wc:
                P_wc = W_wc.normalize()
                wc_id = draw(P_wc)
                logp_wc = np.log(P_wc[wc_id])
                wc = self.target.vocab_eos[wc_id]
                item_ctx = self.f(context)
                prefix_w = await self.item_potential.prefix(item_ctx)
                if wc == self.target.eos:
                    w_guide_wc = await self.item_potential.complete(item_ctx) - prefix_w
                else:
                    w_guide_wc = (
                        await self.item_potential.prefix(self.f(context + [wc]))
                        - prefix_w
                    )
                logws[wc_id] = np.log(W_wc[wc_id]) + w_guide_wc - logp_wc

        return self.target.make_lazy_weights(logws), logp_wc

    async def _lazy_enum(self, context, max_logws):
        agenda = LocatorMaxHeap()

        W = Float.chart()

        # initial conditions
        (token, node) = ((), self.trie.root)
        agenda[token, node, False] = max_logws[node]
        W[node] = 0

        children = self.trie.children
        coerced_ctx = self.f(context)

        curr_priority = float("inf")
        prev_best = float("inf")
        while agenda:
            (token, node, done), score = agenda.popitem()

            assert score <= curr_priority, (
                "Monotonicity assumption violated. "
                "`item_potential` prefix weight must be monotonically decreasing."
            )
            curr_priority = score

            # terminal state
            if done:
                value = W[node] + max_logws[node]
                assert prev_best >= value
                prev_best = value
                yield (self.leaf_to_token_id[node], value)
                continue

            logws = None
            for x, y in children[node].items():
                if x is None:
                    W_y = W[node]
                    W[y] = W_y
                    agenda[token, y, True] = W_y + max_logws[y]
                else:
                    if logws is None:
                        logws = await self.item_potential.logw_next(
                            coerced_ctx + list(token)
                        )
                    W_y = W[node] + logws[x]
                    if W_y == float("-inf"):
                        continue
                    W[y] = W_y
                    agenda[(*token, x), y, False] = W_y + max_logws[y]



================================================
FILE: genlm/control/sampler/token.py
================================================
import numpy as np
from arsenal import colors
from llamppl import SubModel
from arsenal.maths import log1mexp, logsumexp

from genlm.control.util import fast_sample_lazyweights
from genlm.control.sampler.set import SetSampler


class TokenSampler(SubModel):
    """Base class for sampling a token from a potential's vocabulary.

    `TokenSampler`s generate properly weighted samples with respect to a `target` potential.

    Given a context of tokens $x_1, \\ldots, x_{n-1}$ in the target potential's vocabulary,
    a `TokenSampler` samples a token $x_n \\in \\textsf{target.vocab_eos}$ and weight $w$.

    The sampled token and weight are properly weighted with respect to
    $$
    \\textsf{target.logw_next}(x_n | x_1, \\ldots, x_{n-1})
    $$

    Args:
        target (Potential): The potential that samples are properly weighted with respect to.
    """

    def __init__(self, target):
        super().__init__()
        self.target = target
        self.token_type = self.target.token_type

    async def start_weight(self):
        """Compute the weight of the empty sequence under the target potential."""
        return await self.target.prefix([])

    async def forward(self):
        parent = self.parent  # For some reason, need to hold onto this reference.
        token, logw, logp = await self.sample(parent.token_ctx)
        parent.score(logw)
        parent.logp += logp
        return token

    async def sample(self, context, draw):
        """Sample a token and weight from the `target`potential's vocabulary.

        Args:
            context (list[int]): A sequence of tokens in the `target` potential's vocabulary.
            draw (callable): A callable that draws a sample from a distribution.

        Returns:
            (token, weight, logp): A tuple containing the sampled token, weight, and log-probability of the sampled token.
        """
        raise NotImplementedError(
            "Subclasses must implement sample method"
        )  # pragma: no cover

    async def cleanup(self):
        pass  # pragma: no cover

    async def smc(self, n_particles, ess_threshold, max_tokens, critic=None, **kwargs):
        """Generate sequences using sequential Monte Carlo (SMC) inference with this token sampler and an optional critic.

        This method is a convenience wrapper around [`SMC`][genlm.control.sampler.sequence.SMC].
        See [`SMC`][genlm.control.sampler.sequence.SMC] for more details on the generation process.

        Args:
            n_particles (int): The number of particles to use in the SMC algorithm.
            ess_threshold (float): The threshold for the effective sample size (ESS).
            max_tokens (int): The maximum number of tokens to generate.
            critic (Potential, optional): A potential function that guides the generation process
                by scoring candidate sequences. Must have the same token type as the token sampler.
            **kwargs (dict): Additional keyword arguments to pass to `SMC`'s `__call__` method.
        """
        from genlm.control.sampler.sequence import SMC

        return await SMC(self, critic)(
            n_particles=n_particles,
            ess_threshold=ess_threshold,
            max_tokens=max_tokens,
            **kwargs,
        )


class DirectTokenSampler(TokenSampler):
    """Samples individual tokens directly from the log-normalized `logw_next` function
    of a potential.

    Args:
        potential (Potential): The potential function to sample from

    Warning:
        Only use this sampler if the potential's `logw_next` method is efficient. This is the case
        for potentials like `PromptedLLM`, but for custom potentials with a large vocabulary size,
        the default implementation of `logw_next` generally will not be efficient, and thus this
        sampler will be slow.
    """

    def __init__(self, potential):
        super().__init__(target=potential)
        self.potential = potential

    async def sample(self, context, draw=None):
        """Sample a token and weight that are properly weighted with respect to the target potential's `logw_next` method.

        Given a context of tokens $x_1, \\ldots, x_{n-1}$ in the target potential's vocabulary,
        this method samples a token $x_n \\in \\textsf{target.vocab_eos}$ and weight $w$.

        The sampled token and weight are properly weighted with respect to
        $$
        \\textsf{target.logw_next}(x_n | x_1, \\ldots, x_{n-1})
        $$

        The returned weight corresponds to the log normalizing constant of $\\textsf{target.logw_next}(x_n | x_1, \\ldots, x_{n-1})$.

        Returns:
            (token, weight, logp): A tuple containing the sampled token, weight, and log-probability of the sampled token.
        """
        logws = await self.potential.logw_next(context)
        logps = logws.normalize()
        if draw is None:
            # fast sampling from logps using gumbel-max trick
            token = fast_sample_lazyweights(logps)
        else:
            token = draw(logps.exp().materialize())
        return token, logws.sum(), logps[token]

    async def cleanup(self):
        pass  # pragma: no cover


class SetTokenSampler(TokenSampler):
    """Samples individual tokens by sampling a weighted set of tokens and then selecting one
    proportional to its weight.

    This class wraps a `SetSampler`.

    Args:
        set_sampler (SetSampler): The set sampler to sample from
    """

    def __init__(self, set_sampler):
        assert isinstance(set_sampler, SetSampler)
        super().__init__(set_sampler.target)
        self.set_sampler = set_sampler

    async def sample(self, context, draw=None):
        """Sample a token and weight by sampling a weighted set of tokens from the `set_sampler`
        and then selecting one proportional to its weight.

        Given a context of tokens $x_1, \\ldots, x_{n-1}$ in the vocabulary of the set sampler's target potential,
        this method samples a token $x_n \\in \\textsf{set_sampler.target.vocab_eos}$ and a weight.

        The sampled token and weight are properly weighted with respect to
        $$
        \\textsf{set_sampler.target.logw_next}(x_n | x_1, \\ldots, x_{n-1})
        $$

        The returned weight corresponds to the sum of the weights of the sampled set.

        Args:
            context (list[int]): A sequence of tokens in the vocabulary of the set sampler's target potential.

        Returns:
            (token, weight, logp): A tuple containing the sampled token, weight, and log-probability of the random
                choices made in sampling that token.

        Note:
            For properly weighted sampling, the `set_sampler` must assign correct weights to each token. See
            `SetSampler` for more details.
        """
        logws, logp = await self.set_sampler.sample_set(context, draw=draw)
        logps = logws.normalize()
        if draw is None:
            token = fast_sample_lazyweights(logps)
        else:
            token = draw(logps.exp().materialize())
        return token, logws.sum(), logp + logps[token]

    async def cleanup(self):
        """Clean up the sampler.

        This method should be called when the sampler is no longer needed.
        """
        await self.set_sampler.cleanup()


class AWRS(TokenSampler):
    """Samples individual tokens through an adaptive weighted rejection sampling algorithm.

    This sampler is based on the algorithm described in [Fast Controlled Generation from Language Models with Adaptive Weighted Rejection Sampling](https://arxiv.org/abs/2504.05410)

    It draws properly weighted samples from the product of a non-boolean potential and a boolean condition.

    Args:
        potential (Potential): The non-boolean potential.
        condition (Potential): The boolean condition. This potential must only output boolean values (0 or -inf in log-space).
        seed (int): The seed for the random number generator.
        prune_logws (bool): Whether to prune the logws to only include the tokens in the intersection of the potential and condition vocabularies
        proper_weights (bool): Whether to return properly weighted samples.
            If False, the sampler will only run one round of adaptive rejection sampling.
    """

    def __init__(
        self, potential, condition, seed=42, prune_logws=True, proper_weights=True
    ):
        super().__init__(target=potential * condition)
        self.potential = potential
        self.condition = condition

        self.prune_logws = prune_logws
        self.proper_weights = proper_weights
        self.valid_idxs = np.array(
            [self.potential.lookup[t] for t in self.target.vocab_eos]
        )

        self.vocab_eos_set = set(self.target.vocab_eos)
        self.V = len(self.potential.vocab_eos)
        self.rng = np.random.default_rng(seed=seed)

    def _prune_logws(self, logws):
        # Prune the logws to only include the tokens in the
        # target vocabulary. (This zeros-out tokens which we know a priori
        # will be rejected.) Note: We need an additional correction term
        # to account for the fact that we're throwing away some probability mass.
        # This should be handled in `sample`.
        pruned = self.potential.alloc_logws()
        pruned[self.valid_idxs] = logws.weights[self.valid_idxs]
        logws.weights = pruned
        return logws

    async def _accept(self, context, token, verbosity=0):
        if self.prune_logws or token in self.vocab_eos_set:
            if token is self.target.eos:
                logscore = await self.condition.complete(context)
            else:
                logscore = await self.condition.prefix(context + [token])
            assert logscore in {-np.inf, 0}, "`condition` must be Boolean"
        else:
            logscore = -np.inf

        do_accept = logscore == 0

        if verbosity > 0:
            if do_accept:
                print(colors.green % f". {repr(token)}")
            else:
                print(colors.red % ".", end="")

        return do_accept

    async def sample(self, context, verbosity=0):
        """Sample a token and weight that are properly weighted with respect to the target potential's `logw_next` method via adaptive weighted rejection sampling.

        The returned weight corresponds to the log normalizing constant of $\\textsf{target.logw_next}(x_n | x_1, \\ldots, x_{n-1})$.

        Returns:
            (token, weight, np.nan): A tuple containing the sampled token, weight, and a dummy value for the log-probability of the sampled token.
        """
        logws = await self.potential.logw_next(context)
        if self.prune_logws:
            logws = self._prune_logws(logws)

        logZ = logsumexp(logws.weights)
        logps = logws.weights - logZ
        toks = logws.decode

        tok, nrej, logp0 = None, 0, []
        for _ in range(2):
            keys = logps - np.log(-np.log(self.rng.random((self.V,))))
            order = np.argsort(-keys)
            for rank in range(logps.size):
                item = order[rank]
                if keys[item] == -np.inf:
                    break
                if await self._accept(context, toks[item], verbosity):
                    if tok is None:
                        tok = toks[item]
                    break
                else:
                    nrej += 1
                    if tok is None:
                        logp0.append(logps[item])
                    logps[item] = -np.inf

            if not self.proper_weights:
                if tok is None:
                    return self.target.eos, float("-inf"), np.nan
                return tok, 0, np.nan

        if tok is None:  # No token was accepted, return EOS and kill the particle.
            return self.target.eos, float("-inf"), np.nan

        if not logp0:  # Success on first try.
            logw = logZ - np.log(nrej + 1)
        else:
            logw = logZ + log1mexp(logsumexp(logp0)) - np.log(nrej + 1)

        return tok, logw, np.nan



================================================
FILE: tests/conftest.py
================================================
import html
import numpy as np
from arsenal import Integerizer, colors
from arsenal.maths import sample, logsumexp
from graphviz import Digraph
from genlm.grammar import Float
from genlm.control.potential import Potential
from hypothesis import strategies as st


class MockPotential(Potential):
    def __init__(self, vocab, next_token_logws):
        self.next_token_logws = np.array(next_token_logws)
        super().__init__(vocab)

    def _logw(self, context):
        return sum([self.next_token_logws[self.lookup[i]] for i in context])

    async def prefix(self, context):
        return self._logw(context)

    async def complete(self, context):
        return self._logw(context) + self.next_token_logws[-1]

    async def logw_next(self, context):
        return self.make_lazy_weights(self.next_token_logws)


@st.composite
def mock_vocab(draw):
    item_strategy = draw(
        st.sampled_from(
            (
                st.text(min_size=1),
                st.binary(min_size=1),
            )
        )
    )

    # Sample vocabulary of iterables.
    vocab = draw(st.lists(item_strategy, min_size=1, max_size=10, unique=True))
    return vocab


@st.composite
def mock_vocab_and_ws(draw, max_w=1e3):
    vocab = draw(mock_vocab())
    ws = draw(
        st.lists(
            st.floats(1e-5, max_w),
            min_size=len(vocab) + 1,
            max_size=len(vocab) + 1,
        )
    )
    return vocab, ws


@st.composite
def mock_params(draw, max_w=1e3):
    iter_vocab, iter_next_token_ws = draw(mock_vocab_and_ws())

    # Sample context from iter_vocab
    context = draw(st.lists(st.sampled_from(iter_vocab), min_size=0, max_size=10))

    return (iter_vocab, iter_next_token_ws, context)


@st.composite
def iter_item_params(draw, max_iter_w=1e3, max_item_w=1e3):
    iter_vocab, iter_next_token_ws, context = draw(mock_params(max_iter_w))

    item_vocab = set()
    for items in iter_vocab:
        item_vocab.update(items)
    item_vocab = list(item_vocab)

    # Sample weights over item vocabulary and EOS.
    item_next_token_ws = draw(
        st.lists(
            st.floats(1e-5, max_item_w),
            min_size=len(item_vocab) + 1,
            max_size=len(item_vocab) + 1,
        )
    )

    return (iter_vocab, iter_next_token_ws, item_vocab, item_next_token_ws, context)


class WeightedSet(Potential):
    def __init__(self, sequences, weights):
        self.complete_logws = {
            tuple(seq): np.log(w) if w != 0 else float("-inf")
            for seq, w in zip(sequences, weights)
        }

        prefix_ws = {}
        for seq, w in zip(sequences, weights):
            for i in range(0, len(seq) + 1):
                prefix = tuple(seq[:i])
                if prefix not in prefix_ws:
                    prefix_ws[prefix] = 0.0
                prefix_ws[prefix] += w

        self.prefix_log_ws = {
            prefix: np.log(w) if w != 0 else float("-inf")
            for prefix, w in prefix_ws.items()
        }
        total_weight = sum(weights)
        assert np.isclose(
            self.prefix_log_ws[()],
            np.log(total_weight) if total_weight != 0 else float("-inf"),
        )

        super().__init__(list(set(t for seq in sequences for t in seq)))

    async def complete(self, context):
        return self.complete_logws.get(tuple(context), float("-inf"))

    async def prefix(self, context):
        return self.prefix_log_ws.get(tuple(context), float("-inf"))


@st.composite
def weighted_sequence(draw, max_seq_len=5):
    sequence = draw(st.text(min_size=1, max_size=max_seq_len))
    weight = draw(st.floats(min_value=1e-3, max_value=1e3))
    return sequence, weight


@st.composite
def double_weighted_sequence(draw, max_seq_len=5):
    # We use the second weight as the weight assigned to the sequence
    # by the critic.
    sequence = draw(st.text(min_size=1, max_size=max_seq_len))
    weight1 = draw(st.floats(min_value=1e-3, max_value=1e3))
    weight2 = draw(st.floats(min_value=0, max_value=1e3))
    return sequence, weight1, weight2


@st.composite
def weighted_set(draw, item_sampler, max_seq_len=5, max_size=5):
    return draw(
        st.lists(
            item_sampler(max_seq_len),
            min_size=1,
            max_size=max_size,
            unique_by=lambda x: x[0],
        )
    )


def separate_keys_vals(x):
    from genlm.control.util import LazyWeights

    if isinstance(x, LazyWeights):
        return x.keys(), x.values()
    elif isinstance(x, np.ndarray):
        return range(len(x)), x
    else:
        return list(x.keys()), np.array(list(x.values()))


class Tracer:
    """
    This class lazily materializes the probability tree of a generative process by program tracing.
    """

    def __init__(self):
        self.root = Node(idx=-1, mass=1.0, parent=None)
        self.cur = None

    def __call__(self, p, context=None):
        "Sample an action while updating the trace cursor and tree data structure."

        keys, p = separate_keys_vals(p)
        cur = self.cur

        if cur.child_masses is None:
            cur.child_masses = cur.mass * p
            cur.context = context

        if context != cur.context:
            print(colors.light.red % "ERROR: trace divergence detected:")
            print(colors.light.red % "trace context:", self.cur.context)
            print(colors.light.red % "calling context:", context)
            raise ValueError((p, cur))

        a = cur.sample()
        if a not in cur.active_children:
            cur.active_children[a] = Node(
                idx=a,
                mass=cur.child_masses[a],
                parent=cur,
                token=keys[a],
            )
        self.cur = cur.active_children[a]
        return keys[a]


class Node:
    __slots__ = (
        "idx",
        "mass",
        "parent",
        "token",
        "child_masses",
        "active_children",
        "context",
        "_mass",
    )

    def __init__(
        self,
        idx,
        mass,
        parent,
        token=None,
        child_masses=None,
        context=None,
    ):
        self.idx = idx
        self.mass = mass
        self.parent = parent
        self.token = token  # used for visualization
        self.child_masses = child_masses
        self.active_children = {}
        self.context = context
        self._mass = mass  # bookkeeping: remember the original mass

    def sample(self):
        return sample(self.child_masses)

    def p_next(self):
        return Float.chart((a, c.mass / self.mass) for a, c in self.children.items())

    # TODO: untested
    def sample_path(self):
        curr = self
        path = []
        P = 1
        while True:
            p = curr.p_next()
            a = curr.sample()
            P *= p[a]
            curr = curr.children[a]
            if not curr.children:
                break
            path.append(a)
        return (P, path, curr)

    def update(self):
        # TODO: Fennwick tree alternative, sumheap
        # TODO: optimize this by subtracting from masses, instead of resumming
        "Restore the invariant that self.mass = sum children mass."
        if self.parent is not None:
            self.parent.child_masses[self.idx] = self.mass
            self.parent.mass = np.sum(self.parent.child_masses)
            self.parent.update()

    def graphviz(
        self,
        fmt_edge=lambda x, a, y: f"{html.escape(str(a))}/{y._mass / x._mass:.2g}",
        # fmt_node=lambda x: ' ',
        fmt_node=lambda x: (
            f"{x.mass}/{x._mass:.2g}" if x.mass > 0 else f"{x._mass:.2g}"
        ),
    ):
        "Create a graphviz instance for this subtree"
        g = Digraph(
            graph_attr=dict(rankdir="LR"),
            node_attr=dict(
                fontname="Monospace",
                fontsize="10",
                height=".05",
                width=".05",
                margin="0.055,0.042",
            ),
            edge_attr=dict(arrowsize="0.3", fontname="Monospace", fontsize="9"),
        )
        f = Integerizer()
        xs = set()
        q = [self]
        while q:
            x = q.pop()
            xs.add(x)
            if x.child_masses is None:
                continue
            for a, y in x.active_children.items():
                a = y.token if y.token is not None else a
                g.edge(str(f(x)), str(f(y)), label=f"{fmt_edge(x, a, y)}")
                q.append(y)
        for x in xs:
            if x.child_masses is not None:
                g.node(str(f(x)), label=str(fmt_node(x)), shape="box")
            else:
                g.node(str(f(x)), label=str(fmt_node(x)), shape="box", fillcolor="gray")
        return g

    def downstream_nodes(self):
        q = [self]
        while q:
            x = q.pop()
            yield x
            if x.child_masses is None:
                continue
            for y in x.active_children.values():
                q.append(y)


class TraceSWOR(Tracer):
    """
    Sampling without replacement 🤝 Program tracing.
    """

    def __enter__(self):
        self.cur = self.root

    def __exit__(self, *args):
        self.cur.mass = 0  # we will never sample this node again.
        self.cur.update()  # update invariants

    def _repr_svg_(self):
        return self.root.graphviz()._repr_image_svg_xml()

    def sixel_render(self):
        try:
            from sixel import converter
            import sys
            from io import BytesIO

            c = converter.SixelConverter(
                BytesIO(self.root.graphviz()._repr_image_png())
            )
            c.write(sys.stdout)
        except ImportError:
            import warnings

            warnings.warn("Install imgcat or sixel to enable rendering.")
            print(self)

    def __repr__(self):
        return self.root.graphviz().source


async def trace_swor(sampler, context):
    tracer = TraceSWOR()
    logP = sampler.target.alloc_logws()
    while tracer.root.mass > 0:
        with tracer:
            token, logw, logp = await sampler.sample(context, draw=tracer)
            token_id = sampler.target.lookup[token]
            logP[token_id] = logsumexp([logP[token_id], logw + logp])

    return sampler.target.make_lazy_weights(logP)


async def trace_swor_set(sampler, context):
    tracer = TraceSWOR()
    logws = sampler.target.alloc_logws()
    while tracer.root.mass > 0:
        with tracer:
            set_logws, logp = await sampler.sample_set(context, draw=tracer)
            for token_id, logw in enumerate(set_logws.weights):
                if logw == float("-inf"):
                    continue
                logws[token_id] = logsumexp([logws[token_id], logw + logp])

    return sampler.target.make_lazy_weights(logws)



================================================
FILE: tests/test_constant.py
================================================
import pytest
from genlm.control.constant import EndOfSequence, EOS, EOT


def test_init_and_repr():
    eos = EndOfSequence("TEST")
    assert repr(eos) == "TEST"
    assert str(eos) == "TEST"


def test_equality():
    eos1 = EndOfSequence("TEST")
    eos2 = EndOfSequence("TEST")
    eos3 = EndOfSequence("OTHER")

    assert eos1 == eos2
    assert eos1 != eos3
    assert eos1 != "TEST"  # Compare with non-EndOfSequence


def test_radd():
    eos = EndOfSequence("TEST")

    # Test with string
    result = "hello" + eos
    assert isinstance(result, list)
    assert result == ["h", "e", "l", "l", "o", eos]

    # Test with bytes
    result = b"hello" + eos
    assert isinstance(result, list)
    assert result == [104, 101, 108, 108, 111, eos]

    # Test with list and verify type preservation
    input_list = [1, 2, 3]
    result = input_list + eos
    assert isinstance(result, list)
    assert result == [1, 2, 3, eos]
    assert type(result) is type(input_list)

    # Test with tuple and verify type preservation
    input_tuple = (1, 2, 3)
    result = input_tuple + eos
    assert isinstance(result, tuple)
    assert result == (1, 2, 3, eos)
    assert type(result) is type(input_tuple)


def test_radd_error():
    eos = EndOfSequence("TEST")
    with pytest.raises(
        TypeError,
        match=r"Cannot concatenate <class 'int'> with <class '.*EndOfSequence'>",
    ):
        _ = 42 + eos


def test_hash():
    eos1 = EndOfSequence("TEST")
    eos2 = EndOfSequence("TEST")
    eos3 = EndOfSequence("OTHER")

    # Same type should have same hash
    assert hash(eos1) == hash(eos2)
    # Different type should have different hash
    assert hash(eos1) != hash(eos3)

    # Test can be used in sets/dicts
    test_set = {eos1, eos2, eos3}
    assert len(test_set) == 2


def test_iter():
    eos = EndOfSequence("TEST")
    assert list(iter(eos)) == [eos]


def test_len():
    eos = EndOfSequence("TEST")
    assert len(eos) == 1


def test_predefined_constants():
    assert isinstance(EOS, EndOfSequence)
    assert EOS.type_ == "EOS"
    assert isinstance(EOT, EndOfSequence)
    assert EOT.type_ == "EOT"



================================================
FILE: tests/test_setups.py
================================================
import pytest
import numpy as np
from genlm.control import SMC
from genlm.control.potential import Potential, PromptedLLM, BoolFSA
from genlm.control.sampler import (
    AWRS,
    direct_token_sampler,
    eager_token_sampler,
    topk_token_sampler,
)
from genlm.control.sampler.token import TokenSampler
from unittest.mock import Mock


@pytest.fixture(scope="module")
def llm():
    return PromptedLLM.from_name("gpt2", backend="hf", temperature=0.5)


@pytest.fixture(scope="module")
def best_fsa():
    return BoolFSA.from_regex(r"\sthe\s(best|greatest).+")


async def assert_engine_run(engine, n_particles, max_tokens, ess_threshold, **kwargs):
    sequences = await engine(
        n_particles=n_particles,
        ess_threshold=ess_threshold,
        max_tokens=max_tokens,
        **kwargs,
    )

    assert len(sequences) == n_particles
    assert all(len(seq) <= max_tokens for seq in sequences)

    print(sequences)

    return sequences


@pytest.mark.asyncio
async def test_with_llm(llm):
    mtl_llm = llm.spawn_new_eos([b"."])
    mtl_llm.set_prompt_from_str("Montreal is")

    sampler = direct_token_sampler(mtl_llm)
    engine = SMC(sampler)

    sequences = await assert_engine_run(
        engine, n_particles=10, max_tokens=25, ess_threshold=0.5
    )

    assert all(b"." not in seq for seq, _ in sequences)

    await engine.cleanup()


@pytest.mark.asyncio
async def test_with_product_llm(llm):
    mtl_llm = llm.spawn_new_eos([b"."])
    mtl_llm.set_prompt_from_str("Montreal is")

    nyc_llm = mtl_llm.spawn()
    nyc_llm.set_prompt_from_str("NYC is")

    sampler = direct_token_sampler(mtl_llm * nyc_llm)
    engine = SMC(sampler)

    await assert_engine_run(
        engine, n_particles=10, max_tokens=25, ess_threshold=0.5, verbosity=1
    )

    await engine.cleanup()


@pytest.mark.asyncio
async def test_with_llm_and_critic(llm):
    mtl_llm = llm.spawn_new_eos([b"."])
    mtl_llm.set_prompt_from_str("Montreal is")

    nyc_llm = mtl_llm.spawn()
    nyc_llm.set_prompt_from_str("NYC is")

    sampler = direct_token_sampler(mtl_llm)
    engine = SMC(sampler, critic=nyc_llm)

    await assert_engine_run(engine, n_particles=10, max_tokens=25, ess_threshold=0.5)

    await engine.cleanup()


@pytest.mark.asyncio
async def test_with_llm_and_critic_no_twist(llm):
    # When the ess_threshold is 0, the critic is only applied at the end of the generation.
    # This is to avoid running the critic at each step for IS.
    # We test that the critic is applied the correct number of times.

    mtl_llm = llm.spawn_new_eos([b"."])
    mtl_llm.set_prompt_from_str("Montreal is")

    n_calls = 0

    class MockCritic(Potential):
        async def prefix(self, context):
            return 0

        async def complete(self, context):
            return 0

        async def score(self, context):
            nonlocal n_calls
            n_calls += 1
            return 0

    sampler = direct_token_sampler(mtl_llm)
    engine = SMC(sampler, critic=MockCritic(mtl_llm.vocab))

    n_particles = 10

    await assert_engine_run(engine, n_particles, max_tokens=5, ess_threshold=0)

    assert n_calls == n_particles

    await engine.cleanup()


@pytest.mark.asyncio
async def test_with_llm_critic_early_stop(llm):
    mtl_llm = llm.spawn_new_eos([b"."])
    n_calls = 0
    n_particles = 10

    class MockSampler(TokenSampler):
        async def sample(self, context):
            nonlocal n_calls
            n_calls += 1
            return b"a", float("-inf"), np.nan

    class MockPotential(Potential):
        async def prefix(self, context):
            return 0

        async def complete(self, context):
            return 0

    sampler = MockSampler(mtl_llm)
    engine = SMC(sampler, critic=MockPotential(mtl_llm.vocab))

    await assert_engine_run(engine, n_particles, max_tokens=5, ess_threshold=0)

    assert n_calls == n_particles

    await engine.cleanup()


@pytest.mark.asyncio
async def test_with_llm_no_critic_early_stop(llm):
    mtl_llm = llm.spawn_new_eos([b"."])
    n_calls = 0
    n_particles = 10

    class MockSampler(TokenSampler):
        async def sample(self, context):
            nonlocal n_calls
            n_calls += 1
            return b"a", float("-inf"), np.nan

    sampler = MockSampler(mtl_llm)
    engine = SMC(sampler)

    await assert_engine_run(engine, n_particles, max_tokens=5, ess_threshold=0)

    assert n_calls == n_particles

    await engine.cleanup()


@pytest.mark.asyncio
async def test_with_llm_and_fsa(llm, best_fsa):
    mtl_llm = llm.spawn_new_eos([b"."])
    mtl_llm.set_prompt_from_str("Montreal is")

    sampler = direct_token_sampler(mtl_llm)

    best_fsa = best_fsa.coerce(mtl_llm, f=b"".join)

    engine = SMC(sampler, critic=best_fsa)
    await assert_engine_run(engine, n_particles=10, max_tokens=25, ess_threshold=0.5)

    nyc_llm = mtl_llm.spawn()
    nyc_llm.set_prompt_from_str("NYC is")
    engine = SMC(sampler, critic=best_fsa * nyc_llm)
    await assert_engine_run(engine, n_particles=10, max_tokens=25, ess_threshold=0.5)

    await engine.cleanup()


@pytest.mark.asyncio
async def test_with_llm_and_fsa_eager_sampler(llm, best_fsa):
    mtl_llm = llm.spawn_new_eos([b"."])
    mtl_llm.set_prompt_from_str("Montreal is")

    sampler = eager_token_sampler(mtl_llm, best_fsa)
    engine = SMC(sampler)

    await assert_engine_run(engine, n_particles=10, max_tokens=25, ess_threshold=0.5)

    nyc_llm = mtl_llm.spawn()
    nyc_llm.set_prompt_from_str("NYC is")

    engine = SMC(sampler, critic=nyc_llm)

    await assert_engine_run(engine, n_particles=10, max_tokens=25, ess_threshold=0.5)

    await engine.cleanup()


@pytest.mark.asyncio
async def test_with_llm_and_fsa_topk_sampler(llm, best_fsa):
    mtl_llm = llm.spawn_new_eos([b"."])
    mtl_llm.set_prompt_from_str("Montreal is")

    sampler = topk_token_sampler(mtl_llm, best_fsa, K=10)
    engine = SMC(sampler)

    await assert_engine_run(engine, n_particles=10, max_tokens=25, ess_threshold=0.5)

    nyc_llm = mtl_llm.spawn()
    nyc_llm.set_prompt_from_str("NYC is")

    engine = SMC(sampler, critic=nyc_llm)

    await assert_engine_run(engine, n_particles=10, max_tokens=25, ess_threshold=0.5)

    await engine.cleanup()


@pytest.mark.asyncio
async def test_with_llm_and_fsa_awrs_sampler(llm, best_fsa):
    mtl_llm = llm.spawn_new_eos([b"."])
    mtl_llm.set_prompt_from_str("Montreal is")

    sampler = AWRS(mtl_llm, best_fsa.coerce(mtl_llm, f=b"".join))
    engine = SMC(sampler)

    await assert_engine_run(engine, n_particles=10, max_tokens=25, ess_threshold=0.5)

    nyc_llm = mtl_llm.spawn()
    nyc_llm.set_prompt_from_str("NYC is")

    engine = SMC(sampler, critic=nyc_llm)

    await assert_engine_run(engine, n_particles=10, max_tokens=25, ess_threshold=0.5)

    await engine.cleanup()


def test_invalids(llm, best_fsa):
    with pytest.raises(ValueError):
        SMC(llm)

    sampler = direct_token_sampler(llm)

    with pytest.raises(ValueError):
        SMC(llm, critic=sampler)

    sampler = direct_token_sampler(llm)
    with pytest.raises(ValueError):
        # Fail to coerce beforehand.
        SMC(sampler, critic=best_fsa)


def test_invalid_critic():
    # Create a mock TokenSampler
    mock_sampler = Mock(spec=TokenSampler)

    # Try to create SMC with an invalid critic (just a string)
    with pytest.raises(ValueError, match="`critic` must be a Potential"):
        SMC(unit_sampler=mock_sampler, critic="not a potential")



================================================
FILE: tests/test_typing.py
================================================
import pytest
from genlm.control.typing import (
    Atomic,
    Sequence,
    infer_type,
    infer_vocabulary_type,
)


def test_atomic_type_check():
    int_type = Atomic(int)
    assert int_type.check(42)
    assert not int_type.check("42")

    str_type = Atomic(str)
    assert str_type.check("hello")
    assert not str_type.check(b"hello")

    bytes_type = Atomic(bytes)
    assert bytes_type.check(b"hello")
    assert not bytes_type.check("hello")


def test_sequence_type_check():
    int_seq = Sequence(Atomic(int))
    assert int_seq.check([1, 2, 3])
    assert not int_seq.check([1, "2", 3])
    assert not int_seq.check(42)

    nested_int_seq = Sequence(Sequence(Atomic(int)))
    assert nested_int_seq.check([[1, 2], [3, 4]])
    assert not nested_int_seq.check([[1, 2], 3])

    bytes_seq = Sequence(Atomic(bytes))
    assert bytes_seq.check([b"hello", b"world"])
    assert not bytes_seq.check(["hello", "world"])


def test_atomic_inference():
    assert infer_type(42) == Atomic(int)
    assert infer_type("hello") == Atomic(str)
    assert infer_type(b"hello") == Atomic(bytes)
    assert infer_type(3.14) == Atomic(float)
    assert infer_type(True) == Atomic(bool)


def test_sequence_inference():
    assert infer_type([1, 2, 3]) == Sequence(Atomic(int))
    assert infer_type(["a", "b"]) == Sequence(Atomic(str))
    assert infer_type([[1, 2], [3, 4]]) == Sequence(Sequence(Atomic(int)))
    assert infer_type([b"AB", b"CD"]) == Sequence(Atomic(bytes))


def test_empty_sequence_error():
    with pytest.raises(ValueError):
        infer_type([])


def test_inconsistent_sequence_error():
    with pytest.raises(ValueError):
        infer_type([1, "2", 3])


def test_is_iterable_of():
    assert Sequence(Atomic(int)).is_iterable_of(Atomic(int))
    assert Sequence(Atomic(str)).is_iterable_of(Atomic(str))
    assert not Sequence(Atomic(int)).is_iterable_of(Atomic(str))

    assert Atomic(bytes).is_iterable_of(Atomic(int))
    assert Atomic(str).is_iterable_of(Atomic(str))

    assert not Atomic(int).is_iterable_of(Atomic(int))
    assert not Atomic(bytes).is_iterable_of(Atomic(str))
    assert not Atomic(str).is_iterable_of(Atomic(int))

    nested_seq = Sequence(Sequence(Atomic(int)))
    assert nested_seq.is_iterable_of(Sequence(Atomic(int)))
    assert not nested_seq.is_iterable_of(Atomic(int))


def test_vocabulary_type_inference():
    """Test the infer_vocabulary_type function"""
    assert infer_vocabulary_type([1, 2, 3]) == Atomic(int)
    assert infer_vocabulary_type(["a", "b"]) == Atomic(str)
    assert infer_vocabulary_type([[1, 2], [3, 4]]) == Sequence(Atomic(int))

    # Test empty vocabulary
    with pytest.raises(ValueError):
        infer_vocabulary_type([])

    # Test inconsistent types
    with pytest.raises(ValueError):
        infer_vocabulary_type([1, "2", 3])


def test_atomic_convert():
    int_type = Atomic(int)
    assert int_type.convert(42) == 42
    assert int_type.convert("42") == 42
    repr(int_type)


def test_sequence_convert():
    int_seq = Sequence(Atomic(int))
    assert int_seq.convert([1, 2, 3]) == (1, 2, 3)
    assert int_seq.convert(["1", "2", "3"]) == (1, 2, 3)
    repr(int_seq)



================================================
FILE: tests/test_util.py
================================================
import pytest
import numpy as np
from genlm.control.util import LazyWeights, load_trie


def test_lazy_weights_basic():
    # Test basic initialization and access
    weights = np.array([0.1, 0.2, 0.3])
    encode = {"a": 0, "b": 1, "c": 2}
    decode = ["a", "b", "c"]

    lw = LazyWeights(weights, encode, decode, log=False)

    assert lw["a"] == 0.1
    assert lw["b"] == 0.2
    assert lw["c"] == 0.3
    assert lw["d"] == 0  # Non-existent token
    assert len(lw) == 3


def test_lazy_weights_log():
    # Test log-space weights
    weights = np.array([0.0, np.log(2), np.log(3)])
    encode = {"a": 0, "b": 1, "c": 2}
    decode = ["a", "b", "c"]

    lw = LazyWeights(weights, encode, decode, log=True)

    assert lw["a"] == 0.0
    assert lw["b"] == np.log(2)
    assert lw["c"] == np.log(3)
    assert lw["d"] == float("-inf")  # Non-existent token


def test_lazy_weights_normalize():
    weights = np.array([0.1, 0.2, 0.3])
    encode = {"a": 0, "b": 1, "c": 2}
    decode = ["a", "b", "c"]

    # Test normal-space normalization
    lw = LazyWeights(weights, encode, decode, log=False)
    normalized = lw.normalize()
    np.testing.assert_allclose(np.sum(normalized.weights), 1.0)

    # Test log-space normalization
    lw_log = LazyWeights(np.log(weights), encode, decode, log=True)
    normalized_log = lw_log.normalize()
    np.testing.assert_allclose(np.exp(normalized_log.weights).sum(), 1.0)


def test_lazy_weights_exp_log():
    weights = np.array([1.0, 2.0, 3.0])
    encode = {"a": 0, "b": 1, "c": 2}
    decode = ["a", "b", "c"]

    # Test exp
    lw_log = LazyWeights(np.log(weights), encode, decode, log=True)
    lw_exp = lw_log.exp()
    np.testing.assert_allclose(lw_exp.weights, weights)

    # Test log
    lw = LazyWeights(weights, encode, decode, log=False)
    lw_log = lw.log()
    np.testing.assert_allclose(lw_log.weights, np.log(weights))


def test_lazy_weights_assertions():
    with pytest.raises(NotImplementedError):
        weights = np.array([1.0, 2.0])
        lw = LazyWeights(weights, {"a": 0, "b": 1}, ["a", "b"])
        np.array(lw)

    with pytest.raises(AssertionError):
        lw = LazyWeights(np.log(weights), {"a": 0, "b": 1}, ["a", "b"], log=True)
        lw.log()  # Can't take log of log weights

    with pytest.raises(AssertionError):
        lw = LazyWeights(weights, {"a": 0, "b": 1}, ["a", "b"], log=False)
        lw.exp()  # Can't take exp of non-log weights


def test_lazy_weights_sum():
    weights = np.array([1.0, 2.0, 3.0])
    encode = {"a": 0, "b": 1, "c": 2}
    decode = ["a", "b", "c"]

    # Test sum in normal space
    lw = LazyWeights(weights, encode, decode, log=False)
    assert lw.sum() == 6.0

    # Test sum in log space
    log_weights = np.log(weights)
    lw_log = LazyWeights(log_weights, encode, decode, log=True)
    np.testing.assert_allclose(lw_log.sum(), np.log(6.0))


def test_lazy_weights_assert_equal():
    w1 = np.array([1.0, 2.0, 3.0])
    w2 = np.array([1.0, 2.0, 3.0])
    w3 = np.array([1.1, 2.0, 3.0])  # Slightly different weights
    encode = {"a": 0, "b": 1, "c": 2}
    decode = ["a", "b", "c"]

    lw1 = LazyWeights(w1, encode, decode, log=False)
    lw2 = LazyWeights(w2, encode, decode, log=False)
    lw3 = LazyWeights(w3, encode, decode, log=False)

    # Test exact equality
    lw1.assert_equal(lw2)

    # Test equality with tolerance
    with pytest.raises(AssertionError):
        lw1.assert_equal(lw3)
    lw1.assert_equal(lw3, rtol=0.2, atol=0.2)  # Should pass with higher tolerance


def test_lazy_weights_assert_equal_unordered():
    w1 = np.array([1.0, 2.0, 3.0])
    w2 = np.array([3.0, 1.0, 2.0])  # Same values, different order

    encode1 = {"a": 0, "b": 1, "c": 2}
    encode2 = {"c": 0, "a": 1, "b": 2}
    decode1 = ["a", "b", "c"]
    decode2 = ["c", "a", "b"]

    lw1 = LazyWeights(w1, encode1, decode1, log=False)
    lw2 = LazyWeights(w2, encode2, decode2, log=False)

    # Test unordered equality
    lw1.assert_equal_unordered(lw2)

    # Test with missing key
    encode3 = {"a": 0, "b": 1, "d": 2}
    decode3 = ["a", "b", "d"]
    lw3 = LazyWeights(w1, encode3, decode3, log=False)

    with pytest.raises(AssertionError, match="keys do not match"):
        lw1.assert_equal_unordered(lw3)

    with pytest.raises(AssertionError, match="keys do not match"):
        lw3.assert_equal_unordered(lw1)


def test_lazy_weights_keys():
    weights = np.array([1.0, 2.0, 3.0])
    encode = {"a": 0, "b": 1, "c": 2}
    decode = ["a", "b", "c"]

    lw = LazyWeights(weights, encode, decode, log=False)
    assert lw.keys() == ["a", "b", "c"]


def test_lazy_weights_values():
    weights = np.array([1.0, 2.0, 3.0])
    encode = {"a": 0, "b": 1, "c": 2}
    decode = ["a", "b", "c"]

    lw = LazyWeights(weights, encode, decode, log=False)
    assert list(lw.values()) == [1.0, 2.0, 3.0]


def test_lazy_weights_items():
    weights = np.array([1.0, 2.0, 3.0])
    encode = {"a": 0, "b": 1, "c": 2}
    decode = ["a", "b", "c"]

    lw = LazyWeights(weights, encode, decode, log=False)
    assert list(lw.items()) == [("a", 1.0), ("b", 2.0), ("c", 3.0)]


def test_load_trie():
    vocab = ["a", "b", "c"]
    trie = load_trie(vocab, backend="sequential")
    assert trie.decode == vocab

    trie = load_trie(vocab, backend="parallel")
    assert trie.decode == vocab

    trie = load_trie(vocab)
    assert trie.decode == vocab


def test_lazy_weights_repr():
    weights = np.array([1.0, 2.0, 3.0])
    encode = {"a": 0, "b": 1, "c": 2}
    decode = ["a", "b", "c"]

    lw = LazyWeights(weights, encode, decode, log=False)
    lw.__repr__()



================================================
FILE: tests/test_viz.py
================================================
import pytest
import json
import socket
import tempfile
import time
import requests
from pathlib import Path
from genlm.control.viz import InferenceVisualizer


@pytest.fixture
def mocker(request):
    """Fixture to provide mocker."""
    return request.getfixturevalue("mocker")


def is_port_in_use(port):
    """Check if a port is in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


@pytest.fixture
def viz():
    """Fixture that provides a visualizer and ensures cleanup."""
    visualizer = InferenceVisualizer()
    yield visualizer
    visualizer.shutdown_server()


@pytest.fixture
def test_data():
    return [
        {
            "step": 1,
            "mode": "init",
            "particles": [
                {
                    "contents": "<<<>>>b'h'",
                    "logweight": "-11.892930183943907",
                    "weight_incr": "-11.892930183943907",
                }
            ],
        },
    ]


def test_server_starts_on_default_port():
    """Test that server starts on the default port (8000)."""
    assert not is_port_in_use(8000)
    viz = InferenceVisualizer()
    try:
        assert is_port_in_use(8000)
    finally:
        viz.shutdown_server()


def test_server_uses_specified_port():
    """Test that server uses the specified port."""
    assert not is_port_in_use(8001)
    viz = InferenceVisualizer(port=8001)
    try:
        assert is_port_in_use(8001)
        assert not is_port_in_use(8000)
    finally:
        viz.shutdown_server()


def test_visualization_with_custom_dir(test_data):
    """Test visualization with a custom serve directory."""
    with tempfile.TemporaryDirectory() as serve_dir:
        viz = InferenceVisualizer(serve_dir=serve_dir)
        try:
            # Create a test JSON file in the serve directory
            json_path = Path(serve_dir) / "test.json"
            with open(json_path, "w") as f:
                json.dump(test_data, f)

            # Should be able to visualize immediately
            response = requests.get(f"http://localhost:8000/{json_path.name}")
            assert response.status_code == 200
            assert response.json() == test_data
        finally:
            viz.shutdown_server()


def test_visualization_with_external_file(test_data):
    """Test visualization with a file outside the serve directory."""
    viz = InferenceVisualizer()
    try:
        # Create a test JSON file in a different directory
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump(test_data, f)
            json_path = Path(f.name)

        # Should copy file and make it available
        viz.visualize(json_path)
        response = requests.get(f"http://localhost:8000/{json_path.name}")
        assert response.status_code == 200
        assert response.json() == test_data

    finally:
        viz.shutdown_server()
        json_path.unlink()


def test_server_cleanup():
    """Test that server cleanup works correctly."""
    viz = InferenceVisualizer()
    temp_dir = viz._serve_dir
    assert temp_dir.exists()

    viz.shutdown_server()
    time.sleep(0.5)  # Give the server a moment to fully shut down

    # Verify server is shut down and temp directory is cleaned up
    assert not is_port_in_use(8000)
    assert not temp_dir.exists()


def test_port_in_use():
    """Test that appropriate error is raised when port is in use."""
    viz1 = InferenceVisualizer(port=8002)
    try:
        with pytest.raises(OSError, match="Port.*already in use"):
            InferenceVisualizer(port=8002)
    finally:
        viz1.shutdown_server()


def test_html_file_request():
    """Test that HTML file requests are handled correctly."""
    viz = InferenceVisualizer()
    try:
        response = requests.get("http://localhost:8000/smc.html")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
    finally:
        viz.shutdown_server()


def test_server_not_running():
    """Test error when server is not running."""
    viz = InferenceVisualizer()
    viz.shutdown_server()
    with pytest.raises(RuntimeError, match="Server is not running"):
        viz.visualize("test.json")


def test_file_not_found():
    """Test error when JSON file doesn't exist."""
    viz = InferenceVisualizer()
    try:
        with pytest.raises(FileNotFoundError, match="JSON file not found"):
            viz.visualize("nonexistent.json")
    finally:
        viz.shutdown_server()


def test_auto_open_browser(mocker):
    """Test auto-opening browser functionality."""
    mock_open = mocker.patch("webbrowser.open")
    viz = InferenceVisualizer()
    try:
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w") as f:
            json.dump([], f)
            f.flush()
            viz.visualize(f.name, auto_open=True)
        mock_open.assert_called_once()
    finally:
        viz.shutdown_server()


def test_other_oserror(mocker):
    """Test handling of OSError other than port in use."""
    with mocker.patch(
        "socketserver.TCPServer.server_bind",
        side_effect=OSError(99, "Some other error"),
    ):
        with pytest.raises(OSError, match="Some other error"):
            InferenceVisualizer()



================================================
FILE: tests/potential/test_autobatch.py
================================================
import pytest
import asyncio
import time
import numpy as np
from genlm.control.potential import Potential


class MockPotential(Potential):
    """Mock potential for testing with controlled delays"""

    def __init__(self):
        super().__init__(list(range(256)))
        self.delay = 0.1  # 100ms delay per operation

    async def complete(self, context):
        time.sleep(self.delay)
        return np.log(len(context))

    async def prefix(self, context):
        time.sleep(self.delay)
        return np.log(len(context) / 2)

    async def batch_complete(self, contexts):
        time.sleep(self.delay)  # Single delay for batch
        return np.array([np.log(len(context)) for context in contexts])

    async def batch_prefix(self, contexts):
        time.sleep(self.delay)  # Single delay for batch
        return np.array([np.log(len(context) / 2) for context in contexts])

    def spawn(self):
        return MockPotential()


@pytest.mark.asyncio
async def test_correctness():
    """Test that autobatched results match sequential results"""
    potential = MockPotential()
    autobatched = potential.to_autobatched()

    sequences = [b"hello", b"world", b"test", b"batch", b"foo"]

    want = await asyncio.gather(*(potential.complete(seq) for seq in sequences))
    have = await asyncio.gather(*(autobatched.complete(seq) for seq in sequences))
    assert want == have, [want, have]

    want = await asyncio.gather(*(potential.prefix(seq) for seq in sequences))
    have = await asyncio.gather(*(autobatched.prefix(seq) for seq in sequences))
    assert want == have, [want, have]

    want = await asyncio.gather(*(potential.score(seq) for seq in sequences))
    have = await asyncio.gather(*(autobatched.score(seq) for seq in sequences))
    assert want == have, [want, have]

    wants = await asyncio.gather(*(potential.logw_next(seq) for seq in sequences))
    haves = await asyncio.gather(*(autobatched.logw_next(seq) for seq in sequences))
    for have, want in zip(haves, wants):
        have.assert_equal(want)

    await autobatched.cleanup()


@pytest.mark.asyncio
async def test_batch_methods():
    """Test that batch methods return expected results (they shouldn't change)"""
    potential = MockPotential()
    autobatched = potential.to_autobatched()

    sequences = [b"hello", b"world", b"test", b"batch", b"foo"]

    want_complete = await potential.batch_complete(sequences)
    have_complete = await autobatched.batch_complete(sequences)
    np.testing.assert_array_equal(want_complete, have_complete)

    want_prefix = await potential.batch_prefix(sequences)
    have_prefix = await autobatched.batch_prefix(sequences)
    np.testing.assert_array_equal(want_prefix, have_prefix)

    want_score = await potential.batch_score(sequences)
    have_score = await autobatched.batch_score(sequences)
    np.testing.assert_array_equal(want_score, have_score)

    want_logw_next = await potential.batch_logw_next(sequences)
    have_logw_next = await autobatched.batch_logw_next(sequences)
    for have, want in zip(have_logw_next, want_logw_next):
        have.assert_equal(want)

    await autobatched.cleanup()


@pytest.mark.asyncio
async def test_performance():
    """Test that autobatched operations are faster than sequential"""
    potential = MockPotential()
    autobatched = potential.to_autobatched()

    sequences = [b"hello", b"world", b"test", b"batch", b"foo"]

    start = time.perf_counter()
    await asyncio.gather(*(potential.complete(seq) for seq in sequences))
    sequential_time = time.perf_counter() - start

    start = time.perf_counter()
    await asyncio.gather(*(autobatched.complete(seq) for seq in sequences))
    autobatched_time = time.perf_counter() - start

    print(sequential_time, autobatched_time)

    assert autobatched_time < sequential_time / 2

    await autobatched.cleanup()


@pytest.mark.asyncio
async def test_error_handling():
    """Test that errors in batch processing are properly propagated"""

    class ErrorPotential(MockPotential):
        async def batch_complete(self, contexts):
            raise ValueError("Test error")

    potential = ErrorPotential()
    autobatched = potential.to_autobatched()

    with pytest.raises(ValueError, match="Test error"):
        await autobatched.complete(b"test")

    await autobatched.cleanup()


@pytest.mark.asyncio
async def test_spawn_and_repr():
    """Test spawn method creates new instance and repr works correctly"""
    potential = MockPotential()
    autobatched = potential.to_autobatched()

    # Test spawn
    spawned = autobatched.spawn()
    assert isinstance(spawned, type(autobatched))
    assert spawned is not autobatched
    assert spawned.potential is not autobatched.potential

    # Test repr
    expected_repr = f"AutoBatchedPotential({potential!r})"
    assert repr(autobatched) == expected_repr

    await autobatched.cleanup()
    await spawned.cleanup()


@pytest.mark.asyncio
async def test_close_and_cleanup():
    """Test close() and cleanup() methods"""
    potential = MockPotential()
    autobatched = potential.to_autobatched()

    # Test that the background loop is running
    assert autobatched.background_loop.task is not None
    assert not autobatched.background_loop.task.done()

    # Test close()
    autobatched.background_loop.close()
    assert autobatched.background_loop.task is None

    # Test cleanup()
    autobatched = potential.to_autobatched()  # Create new instance
    await autobatched.cleanup()
    assert autobatched.background_loop.task is None


@pytest.mark.asyncio
async def test_del_cleanup():
    """Test __del__ cleanup"""
    potential = MockPotential()
    autobatched = potential.to_autobatched()

    # Get reference to background loop
    loop = autobatched.background_loop
    assert loop.task is not None

    # Delete the autobatched instance
    del autobatched

    # Verify the background loop was cleaned up
    assert loop.task is None



================================================
FILE: tests/potential/test_base.py
================================================
import pytest
import asyncio
import numpy as np
from genlm.control.typing import Atomic
from genlm.control.potential.base import Potential, EOS


class SimplePotential(Potential):
    async def complete(self, context):
        return -float(len(context))  # Scoring based on length

    async def prefix(self, context):
        return -0.5 * float(len(context))  # Different scoring for prefixes


@pytest.fixture
def potential():
    return SimplePotential([b"a", b"b", b"c"])


def test_token_type(potential):
    assert potential.token_type == Atomic(bytes)


@pytest.mark.asyncio
async def test_score(potential):
    context = [b"b", b"c"]

    have = await potential.score(context)
    want = await potential.prefix(context)
    assert want == -1.0
    assert have == want

    have = await potential.score(context + [EOS])
    want = await potential.complete(context)
    assert want == -2.0
    assert have == want

    have = await potential.score([])
    want = await potential.prefix([])
    assert want == 0.0
    assert have == want


@pytest.mark.asyncio
async def test_logw_next(potential):
    context = [b"b", b"c"]
    have = (await potential.logw_next(context)).materialize()
    for token in potential.vocab_eos:
        want = await potential.score(context + [token]) - await potential.prefix(
            context
        )
        assert have[token] == want


@pytest.mark.asyncio
async def test_batch_score(potential):
    seq1 = [b"a"]
    seq2 = [b"a", b"b"]
    seq3 = [b"a", b"b", EOS]

    have = await potential.batch_score([seq1, seq2, seq3])
    want = await asyncio.gather(
        potential.score(seq1), potential.score(seq2), potential.score(seq3)
    )

    np.testing.assert_array_equal(have, want)
    np.testing.assert_array_equal(have, [-0.5, -1.0, -2.0])


@pytest.mark.asyncio
async def test_batch_logw_next(potential):
    seq1 = [b"a"]
    seq2 = [b"b", b"c"]

    haves = await potential.batch_logw_next([seq1, seq2])
    wants = await asyncio.gather(potential.logw_next(seq1), potential.logw_next(seq2))

    for want, have in zip(haves, wants):
        np.testing.assert_array_equal(have.weights, want.weights)


@pytest.mark.asyncio
async def test_empty(potential):
    with pytest.raises(ValueError):
        await potential.batch_logw_next([])

    with pytest.raises(ValueError):
        await potential.batch_score([])

    with pytest.raises(ValueError):
        await potential.batch_prefix([])

    with pytest.raises(ValueError):
        await potential.batch_complete([])


@pytest.mark.asyncio
async def test_properties(potential):
    await potential.assert_logw_next_consistency([b"b", b"c"], verbosity=1)
    await potential.assert_autoreg_fact([b"b", b"c"], verbosity=1)
    await potential.assert_batch_consistency([[b"b", b"c"], [b"a"]], verbosity=1)


def test_initialization_errors():
    # Test empty vocabulary
    with pytest.raises(ValueError, match="vocabulary cannot be empty"):
        SimplePotential([])

    # Test invalid token_type
    with pytest.raises(ValueError, match="token_type must be a TokenType"):
        SimplePotential([b"a"], token_type="not a token type")

    # Test wrong token types in vocabulary
    wrong_type = Atomic(str)  # Using str instead of bytes
    with pytest.raises(TypeError, match="Tokens in vocabulary must be of type"):
        SimplePotential([b"a", b"b"], token_type=wrong_type)

    # Test invalid EOS type
    with pytest.raises(ValueError, match="EOS must be an instance of EndOfSequence"):
        SimplePotential([b"a"], eos="not an EndOfSequence")

    # Test duplicate tokens
    with pytest.raises(ValueError, match="Duplicate token.*found in vocabulary"):
        SimplePotential([b"a", b"a"])


@pytest.mark.asyncio
async def test_zero_weight_context():
    class ZeroWeightPotential(SimplePotential):
        async def prefix(self, context):
            return float("-inf")

    potential = ZeroWeightPotential([b"a", b"b"])
    with pytest.raises(ValueError, match="Context.*has weight zero under `prefix`."):
        await potential.logw_next([b"a"])


def test_spawn_not_implemented():
    potential = SimplePotential([b"a"])
    with pytest.raises(
        NotImplementedError,
        match="Potential.spawn\\(\\) must be implemented by subclasses.",
    ):
        potential.spawn()



================================================
FILE: tests/potential/test_canonical.py
================================================
import pytest
import numpy as np
from unittest.mock import patch, MagicMock, PropertyMock
from transformers import GPT2Tokenizer, BertTokenizer
from genlm.backend.tokenization import decode_vocab
from genlm.control import PromptedLLM, CanonicalTokenization
from genlm.control.potential.built_in.canonical import (
    FastCanonicalityFilterBPE,
    _extract_bpe_merges,
)
from hypothesis import given, strategies as st, settings


class MockAsyncTransformer:  # Mock the backend LLM object
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # Restore calculation of byte_vocab; PromptedLLM init needs it.
        # decode_vocab will raise ValueError for unsupported tokenizers (like BERT).
        try:
            self.byte_vocab, _ = decode_vocab(tokenizer)
        except ValueError:
            self.byte_vocab = None  # Handle cases like BERT where byte vocab fails
        # maybe add other attributes if PromptedLLM.__init__ needs them
        # e.g., self.model_name_or_path = tokenizer.name_or_path


class MockLLM(PromptedLLM):
    def __init__(self, tokenizer, model_name="mock_model", eos_tokens=None):
        # Create the mock backend object
        mock_backend_llm = MockAsyncTransformer(tokenizer)

        # Call the parent PromptedLLM initializer
        # Use provided eos_tokens if available, otherwise extract from tokenizer
        if eos_tokens is None:
            eos_token_bytes = (
                tokenizer.eos_token.encode("utf-8") if tokenizer.eos_token else None
            )
            eos_token_list = [eos_token_bytes] if eos_token_bytes else []
        else:
            # Assume provided eos_tokens are already bytes or handle conversion if needed
            eos_token_list = eos_tokens

        # Need to handle cases where byte_vocab is None for unsupported tokenizers
        if mock_backend_llm.byte_vocab is None:
            # Provide some dummy value or handle appropriately based on PromptedLLM needs
            # For now, let's skip super init if vocab fails, as Canonical won't work anyway
            print(
                f"Warning: Skipping PromptedLLM super().__init__ for {tokenizer.name_or_path} due to missing byte_vocab."
            )
            self.model = (
                mock_backend_llm  # Still need self.model for tests accessing tokenizer
            )
            self.token_maps = None  # Indicate maps aren't properly initialized
        else:
            super().__init__(llm=mock_backend_llm, eos_tokens=eos_token_list)
            # The super init should handle setting up self.model and self.token_maps


@pytest.fixture(scope="module")
def llm():
    return PromptedLLM.from_name("gpt2", temperature=0.7)


@pytest.fixture(scope="module")
def llm_with_multiple_eos():
    return PromptedLLM.from_name(
        "gpt2", temperature=0.7, eos_tokens=[b".", b" city", b"\n", b" "]
    )


@pytest.fixture(scope="module")
def canonical_potential(llm):
    """Create a CanonicalTokenization for testing"""
    return CanonicalTokenization.from_llm(llm)


def test_init(llm, llm_with_multiple_eos):
    """Test that the potential initializes properly via from_llm"""
    # Instantiate using the new factory method
    potential = CanonicalTokenization.from_llm(llm)
    potential_with_multiple_eos = CanonicalTokenization.from_llm(llm_with_multiple_eos)

    # Check that the potential has the correct vocabulary
    assert len(potential.vocab) == len(potential.canonicality_filter._decode)
    assert len(potential_with_multiple_eos.vocab) == len(
        potential_with_multiple_eos.canonicality_filter._decode
    )
    # Check that EOS is added correctly
    assert len(potential.vocab_eos) == len(potential.vocab) + 1
    assert (
        len(potential_with_multiple_eos.vocab_eos)
        == len(potential_with_multiple_eos.vocab) + 1
    )


def test_no_eos_init(llm):
    canonicality_filter = FastCanonicalityFilterBPE.from_tokenizer(llm.model.tokenizer)
    assert canonicality_filter.eos_token_ids == {llm.model.tokenizer.eos_token_id}


def test_empty_context_mask(llm):  # Use the llm fixture
    """
    Test FastCanonicalityFilterBPE.__call__ with an empty context tuple ().
    It should return a mask allowing all tokens initially.
    """
    # Use the new factory method for the filter
    filter_instance = FastCanonicalityFilterBPE.from_tokenizer(
        llm.model.tokenizer, llm.token_maps.eos_idxs
    )
    empty_context = ()

    mask = filter_instance(empty_context)

    assert isinstance(mask, np.ndarray), "Mask should be a numpy array"
    assert mask.dtype == bool, "Mask dtype should be boolean"
    assert len(mask) == filter_instance.V, (
        f"Mask length ({len(mask)}) should equal vocab size ({filter_instance.V})"
    )
    assert np.all(mask), "Mask should be all True for an empty context"


@pytest.mark.asyncio
async def test_complete_empty(canonical_potential):
    """Test complete method with empty context"""
    log_weight = await canonical_potential.complete([])
    assert log_weight == 0.0


@pytest.mark.asyncio
async def test_complete_non_canonical(canonical_potential):
    """Test complete method with non-canonical context"""
    tokens = [b"To", b"ken", b"ization"]
    log_weight = await canonical_potential.complete(tokens)
    assert log_weight == float("-inf")


@pytest.mark.asyncio
async def test_logw_next_invalid_prefix(canonical_potential):
    """Test logw_next method with non canonical context. should only extend to EOS"""
    tokens = [b"To", b"ken"]
    with pytest.raises(ValueError):
        await canonical_potential.logw_next(tokens)


@pytest.mark.asyncio
async def test_logw_next_canonical(canonical_potential):
    """Test logw_next allows canonical next tokens and disallows non-canonical ones."""
    context = [b"Token"]
    canonical_next_bytes = b"ization"
    non_canonical_next_bytes = b"tion"
    logw = await canonical_potential.logw_next(context)
    # Assert canonical next token is allowed (weight is not -inf)
    assert logw[canonical_next_bytes] != float("-inf"), (
        f"Canonical next token {canonical_next_bytes!r} should be allowed"
    )

    # Assert non-canonical next token is disallowed (weight is -inf)
    assert logw[non_canonical_next_bytes] == float("-inf"), (
        f"Non-canonical next token {non_canonical_next_bytes!r} should be disallowed"
    )


@pytest.mark.asyncio
async def test_set_overrides(canonical_potential):
    """Test that set_overrides allows configured non-canonical pairs for gpt2."""
    _decode = canonical_potential.canonicality_filter._decode

    required_ids = [198, 2637, 82]
    if any(idx >= len(_decode) or _decode[idx] is None for idx in required_ids):
        pytest.skip("Required token IDs for override test not present in vocabulary.")

    token_198_bytes = _decode[198]
    token_2637_bytes = _decode[2637]
    token_82_bytes = _decode[82]  # Corresponds to 's' for gpt2

    # Test override (198, 198) -> \n\n
    logw_198 = await canonical_potential.logw_next([token_198_bytes])
    assert logw_198[token_198_bytes] != float("-inf"), (
        "Override (198, 198) failed in logw_next"
    )
    assert (
        await canonical_potential.complete([token_198_bytes, token_198_bytes]) == 0.0
    ), "Override (198, 198) failed in complete"

    logw_2637 = await canonical_potential.logw_next([token_2637_bytes])
    assert logw_2637[token_82_bytes] != float("-inf"), (
        "Override (2637, 82) failed in logw_next"
    )
    assert (
        await canonical_potential.complete([token_2637_bytes, token_82_bytes]) == 0.0
    ), "Override (2637, 82) failed in complete"


def test_check_canonicality(canonical_potential):
    """Test check_canonicality method with canonical context"""
    assert canonical_potential._check_canonicality([])
    # Single token is always canonical
    assert canonical_potential._check_canonicality([b" the"])
    # Valid token sequence should be canonical
    assert canonical_potential._check_canonicality([b"Token", b"ization"])
    # This should be non-canonical
    assert not canonical_potential._check_canonicality([b"hel", b"lo", b" world"])


@pytest.mark.asyncio
@settings(deadline=None)
@given(st.text(min_size=1, max_size=10))
async def test_example(canonical_potential, llm, text):
    """Test example method with canonical context"""
    tokens = llm.tokenize(text)
    log_weight = await canonical_potential.complete(tokens)
    assert log_weight == 0.0
    # Also test prefix for each subsequence
    for i in range(1, len(tokens) + 1):
        prefix = tokens[:i]
        log_weight = await canonical_potential.prefix(prefix)
        assert log_weight == 0.0

    # Test that each valid prefix allows appropriate next tokens
    for i in range(len(tokens)):
        prefix = tokens[:i]
        next_token = tokens[i]
        lazy_weights = await canonical_potential.logw_next(prefix)
        # The next token in the sequence should be allowed
        assert lazy_weights[next_token] == 0.0


def test_from_llm_extract_merges_slow_tokenizer():
    """Test that merges are extracted correctly from a slow tokenizer (using bpe_ranks)."""
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2", use_fast=False)
    mock_llm = MockLLM(tokenizer)  # MockLLM needs to handle token_maps now
    if mock_llm.token_maps is None:  # Handle case where super init was skipped
        pytest.skip(
            "MockLLM failed to initialize token maps, likely unsupported tokenizer."
        )
    # Instantiate filter using the new method
    filter_instance = FastCanonicalityFilterBPE.from_tokenizer(
        mock_llm.model.tokenizer, mock_llm.token_maps.eos_idxs
    )
    assert filter_instance._merges, (
        "Merges should be extracted from the slow GPT2 tokenizer."
    )
    # Check a known merge (example: 'a' + 't' -> 'at')
    g_id = tokenizer.encode("a")[0]
    t_id = tokenizer.encode("t")[0]
    gt_id = tokenizer.encode("at")[0]
    assert (g_id, t_id, gt_id) in filter_instance._merges, (
        "Known merge (a, t) not found in extracted merges."
    )


def test_from_llm_extract_merges_fallback():
    """Test that creating the Filter/Potential fails for unsupported tokenizers."""
    tokenizer = BertTokenizer.from_pretrained(
        "bert-base-uncased"
    )  # WordPiece tokenizer

    # MockLLM should handle the decode_vocab error gracefully now
    mock_llm = MockLLM(tokenizer)

    # Assert that MockLLM skipped its super init due to the unsupported tokenizer
    assert mock_llm.token_maps is None, (
        "MockLLM should have token_maps=None for unsupported tokenizer"
    )

    # Directly calling from_tokenizer should still raise the ValueError from decode_vocab
    with pytest.raises(ValueError, match="Could not decode byte representation"):
        FastCanonicalityFilterBPE.from_tokenizer(
            mock_llm.model.tokenizer, []
        )  # Pass empty eos_ids


def test_from_llm_duplicate_byte_error(llm):
    """Test that from_tokenizer raises ValueError if decode_vocab returns duplicates."""

    # Define the vocabulary with duplicates we want decode_vocab to return
    duplicate_vocab = [
        b"a",  # ID 0
        b"b",  # ID 1
        b"c",  # ID 2
        b"a",  # ID 3 - DUPLICATE of ID 0
    ]

    # Patch decode_vocab within the canonical module for this test
    with patch(
        "genlm.control.potential.built_in.canonical.decode_vocab",
        return_value=(duplicate_vocab, None),
    ):
        # Assert that from_tokenizer raises the expected ValueError when called
        with pytest.raises(ValueError, match="Duplicate byte sequences found"):
            FastCanonicalityFilterBPE.from_tokenizer(
                llm.model.tokenizer, llm.token_maps.eos_idxs
            )


def test_canonical_tokenization_init_type_error():
    """Test that CanonicalTokenization.from_llm raises TypeError for wrong llm type."""

    not_an_llm = object()
    with pytest.raises(
        TypeError, match="Expected llm to be an instance of PromptedLLM"
    ):
        # Call the factory method which performs the check
        CanonicalTokenization.from_llm(not_an_llm)


def test_call_unknown_last_token(llm):
    """Test FastCanonicalityFilterBPE.__call__ handles unknown last_token (KeyError)."""
    # Instantiate filter using the new method
    filter_instance = FastCanonicalityFilterBPE.from_tokenizer(
        llm.model.tokenizer, llm.token_maps.eos_idxs
    )

    unknown_token = b"@@@totally_unknown_token@@@"
    # Check it's really not in the encode map (optional sanity check)
    assert unknown_token not in filter_instance._encode
    context = (
        None,
        unknown_token,
    )

    with pytest.raises(KeyError):
        filter_instance(context)


def test_extract_merges_slow_id_mapping_failure():
    """Test warning when slow tokenizer has bpe_ranks but vocab mapping fails."""
    mock_tokenizer = MagicMock()
    mock_tokenizer.is_fast = False
    mock_tokenizer.name_or_path = "mock_slow_map_fail_tokenizer"

    # Make JSON and direct access fail (e.g., raise exceptions)
    mock_tokenizer._tokenizer = MagicMock()
    mock_tokenizer._tokenizer.to_str.side_effect = Exception("JSON parsing failed")
    type(mock_tokenizer._tokenizer).model = PropertyMock(
        side_effect=Exception("Direct access failed")
    )  # Ensure direct access also fails

    # Provide bpe_ranks directly on the mock
    mock_tokenizer.bpe_ranks = {("a", "b"): 0}

    # Make vocab lookup fail
    mock_tokenizer.get_vocab.return_value = {}

    # Patch hasattr to return True for bpe_ranks check
    with patch(
        "builtins.hasattr",
        lambda obj, name: True
        if obj is mock_tokenizer and name == "bpe_ranks"
        else hasattr(obj, name),
    ):
        # Catch ALL UserWarnings
        with pytest.raises(ValueError):
            _extract_bpe_merges(mock_tokenizer)


# @pytest.mark.asyncio
# def test_extract_merges_slow_exception():
#     """Test warning when accessing slow tokenizer bpe_ranks raises an exception."""
#     mock_tokenizer = MagicMock()
#     mock_tokenizer.is_fast = False
#     mock_tokenizer.name_or_path = "mock_slow_exception_tokenizer"

#     # Make JSON and direct access fail
#     mock_tokenizer._tokenizer = MagicMock()
#     mock_tokenizer._tokenizer.to_str.side_effect = Exception("JSON parsing failed")
#     type(mock_tokenizer._tokenizer).model = PropertyMock(side_effect=Exception("Direct access failed"))

#     # Make accessing bpe_ranks raise an error using PropertyMock
#     exception_message = "Cannot access bpe_ranks"
#     type(mock_tokenizer).bpe_ranks = PropertyMock(
#         side_effect=Exception(exception_message)
#     )
#     with pytest.raises(ValueError):
#         _extract_bpe_merges(mock_tokenizer)


if __name__ == "__main__":
    pytest.main()



================================================
FILE: tests/potential/test_coerce.py
================================================
import pytest
import numpy as np
from genlm.control.typing import Atomic
from genlm.control.constant import EOS
from genlm.control.potential import Coerced, Potential


class MockPotential(Potential):
    def __init__(self, V):
        super().__init__(V)

    def bytes_to_int(self, byte_seq):
        return int.from_bytes(byte_seq, byteorder="big")

    async def complete(self, context):
        return self.bytes_to_int(context)

    async def prefix(self, context):
        return self.bytes_to_int(context) / 2


@pytest.mark.asyncio
async def test_simple():
    p = MockPotential([b"a"[0], b"b"[0], b"c"[0]])
    c = Coerced(p, [b"aa", b"bb", b"aab", b"aad"], f=b"".join)

    assert c.token_type == Atomic(bytes)
    assert set(c.vocab) == {b"aa", b"bb", b"aab"}

    have = await c.complete([b"aa", b"bb"])
    want = await p.complete(b"aabb")
    assert have == want

    have = await c.prefix([b"aa", b"bb"])
    want = await p.prefix(b"aabb")
    assert have == want

    have = await c.score([b"aa", b"bb", EOS])
    want = await p.score(b"aabb" + EOS)
    assert have == want

    have = await c.logw_next([b"aa", b"bb"])
    for x in c.vocab_eos:
        want = await p.score(b"aabb" + x) - await p.prefix(b"aabb")
        assert have[x] == want, [have[x], want, x]


@pytest.mark.asyncio
async def test_coerced_batch_operations():
    p = MockPotential([b"a"[0], b"b"[0], b"c"[0]])
    coerced = Coerced(p, [b"aa", b"bb", b"aab", b"aad"], f=b"".join)
    sequences = [[b"aa", b"aab"], [b"bb"]]

    have = await coerced.batch_complete(sequences)
    want = np.array([await coerced.complete(sequence) for sequence in sequences])
    np.testing.assert_array_equal(have, want)

    have = await coerced.batch_prefix(sequences)
    want = np.array([await coerced.prefix(sequence) for sequence in sequences])
    np.testing.assert_array_equal(have, want)

    have = await coerced.batch_score(sequences)
    want = np.array([await coerced.score(sequence) for sequence in sequences])
    np.testing.assert_array_equal(have, want)

    haves = await coerced.batch_logw_next(sequences)
    wants = [await coerced.logw_next(sequence) for sequence in sequences]
    for have, want in zip(haves, wants):
        have.assert_equal(want)


@pytest.mark.asyncio
async def test_coerced_invalid_vocab():
    with pytest.raises(ValueError):
        Coerced(MockPotential([b"a"[0], b"b"[0], b"c"[0]]), [b"xx", b"yy"], f=b"".join)


@pytest.mark.asyncio
async def test_coerced_custom():
    mock_potential = MockPotential([b"a"[0], b"b"[0], b"c"[0]])
    coerced = Coerced(
        mock_potential,
        target_vocab=[b"aa", b"bb"],
        f=lambda seq: [item[0] for item in seq],  # Take first byte of each token
    )

    assert coerced.token_type == Atomic(bytes)

    assert len(coerced.vocab) == 2
    assert set(coerced.vocab) == {b"aa", b"bb"}

    have = await coerced.complete([b"aa", b"bb"])
    want = await mock_potential.complete(b"ab")
    assert have == want

    have = await coerced.prefix([b"aa", b"bb"])
    want = await mock_potential.prefix(b"ab")
    assert have == want

    have = await coerced.score([b"aa", b"bb", EOS])
    want = await mock_potential.score(b"ab" + EOS)
    assert have == want


def test_coerced_repr():
    p = MockPotential([b"a"[0], b"b"[0], b"c"[0]])
    c = Coerced(p, [b"aa", b"bb", b"aab", b"aad"], f=b"".join)
    repr(c)


def test_coerced_no_prune():
    p = MockPotential([b"a"[0], b"b"[0], b"c"[0]])
    c = Coerced(p, [b"aa", b"bb", b"aab", b"aad"], f=b"".join, prune=False)
    assert len(c.vocab) == 4
    assert set(c.vocab) == {b"aa", b"bb", b"aab", b"aad"}



================================================
FILE: tests/potential/test_json.py
================================================
import pytest
from genlm.control.potential.built_in.json import (
    JsonSchema,
    json_schema_parser,
    ARBITRARY_JSON,
    Incomplete,
    FLOAT_PARSER,
    chunk_to_complete_utf8,
    ParseError,
    StreamingJsonSchema,
    ValidateJSON,
    ParserPotential,
    StringSource,
    Input,
    FloatParser,
)
from genlm.control.potential.streaming import AsyncSource
import json
from typing import Any
from dataclasses import dataclass
from hypothesis import given, strategies as st, assume, example, settings, reject
from hypothesis_jsonschema import from_schema


@pytest.mark.asyncio
async def test_validates_a_list_of_integers():
    potential = JsonSchema({"type": "array", "items": {"type": "integer"}})

    assert await potential.prefix(b"[1,2,3") == 0.0
    assert await potential.prefix(b'["hello world"') == -float("inf")
    assert await potential.prefix(b"{") == -float("inf")


@pytest.mark.asyncio
async def test_rejects_as_prefix_when_no_valid_continuation():
    potential = JsonSchema({"type": "object"})

    assert await potential.prefix(b"}") == -float("inf")


@pytest.mark.asyncio
async def test_whitespace_is_valid_prefix_and_invalid_complete():
    potential = JsonSchema({"type": "object"})

    assert await potential.prefix(b"\t") == 0.0
    assert await potential.complete(b"\t") == -float("inf")


@pytest.mark.asyncio
@pytest.mark.parametrize("schema", [{"type": "array", "items": {"type": "integer"}}])
@pytest.mark.parametrize(
    "context",
    [
        b"[1,2,3",
        b"[0]",
    ],
)
async def test_consistency_properties(schema, context):
    potential = JsonSchema(schema)
    await potential.assert_autoreg_fact(context)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "potential",
    [
        StreamingJsonSchema({"type": "array", "items": {"type": "integer"}}),
        ValidateJSON(),
        ParserPotential(
            json_schema_parser({"type": "array", "items": {"type": "integer"}})
        ),
    ],
)
async def test_logw_next_has_results(potential):
    logs = await potential.logw_next(b"")
    assert logs[b"["[0]] == 0.0


@pytest.mark.asyncio
async def test_will_error_on_impossible_unicode_prefixes():
    potential = JsonSchema({"type": "object"})
    assert await potential.prefix([190] * 5) == -float("inf")


@st.composite
def json_schema(draw):
    type = draw(
        st.sampled_from(
            [
                "null",
                "boolean",
                "integer",
                "number",
                "string",
                "object",
                "array",
            ]
        )
    )

    # TODO: Add some bounds in for some of these?
    if type in ("null", "boolean", "integer", "number", "string"):
        return {"type": type}

    if type == "object":
        result = {"type": "object"}
        result["properties"] = draw(
            st.dictionaries(
                st.from_regex("[A-Za-z0-9_]+"),
                json_schema(),
            )
        )
        if result["properties"]:
            result["required"] = draw(
                st.lists(st.sampled_from(sorted(result["properties"])), unique=True)
            )
        result["additionalProperties"] = draw(st.booleans())
        return result

    assert type == "array"
    result = {"type": "array", "items": draw(json_schema())}
    min_contains = draw(st.integers(0, 10))
    if min_contains > 0:
        result["minContains"] = min_contains
    if draw(st.booleans()):
        max_contains = draw(st.integers(min_contains, 20))
        result["maxContains"] = max_contains
    return result


@dataclass(frozen=True)
class JSONSchemaPotentialProblem:
    schema: Any
    document: bytes
    prefix: bytes

    @property
    def value(self):
        return json.loads(self.document)


@st.composite
def json_schema_potential_problem(draw):
    schema = draw(json_schema())
    value = draw(from_schema(schema))
    text = json.dumps(
        value,
        # Inverted so that this shrinks to True, as ascii-only
        # JSON is simpler.
        ensure_ascii=not draw(st.booleans()),
        # Similarly inverted so as to shrink to True, on the
        # theory that this means that if keys are out of
        # order in a shrunk example then it really matters.
        sort_keys=not draw(st.booleans()),
        indent=draw(st.one_of(st.none(), st.integers(0, 4), st.text(alphabet=" \t"))),
    )

    document = text.encode("utf-8")
    assert document
    assume(len(document) > 1)
    i = draw(st.integers(1, len(document) - 1))
    prefix = document[:i]
    assume(prefix.strip())

    return JSONSchemaPotentialProblem(schema=schema, document=document, prefix=prefix)


@pytest.mark.asyncio
@example(
    JSONSchemaPotentialProblem(
        schema={"type": "string"},
        document=b'"0\xc2\x80\xc2\x80"',
        prefix=b'"0\xc2\x80\xc2',
    )
)
@example(
    JSONSchemaPotentialProblem(
        schema={
            "type": "string",
        },
        document=b'"000000000\\u001f\xc2\x80\xc2\x80"',
        prefix=b'"000000000\\u001f\xc2\x80\xc2\x80',
    ),
)
@example(
    JSONSchemaPotentialProblem(
        schema={
            "type": "string",
        },
        document=b'"000\\u001f\xc2\x80\xc2\x80\xc2\x80"',
        prefix=b'"000\\u001f\xc2\x80\xc2\x80\xc2',
    ),
)
@given(json_schema_potential_problem())
@settings(max_examples=200, deadline=None)
async def test_always_returns_correctly_on_valid_documents(problem):
    return
    potential = JsonSchema(problem.schema)

    assert await potential.prefix(problem.prefix) == 0.0
    assert await potential.prefix(problem.document) == 0.0
    if await potential.complete(problem.prefix) > -float("inf"):
        # This can sometimes happen because e.g. numeric literals can have
        # a prefix that is also a valid JSON value. We check here that the
        # prefix is actually valid JSON and if so allow it.
        json.loads(problem.prefix)
    assert await potential.complete(problem.document) == 0.0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "format",
    [
        "ipv4",
        "date-time",
        "date",
        "date-time",
        # duration not present in Draft 7 which we're currently using.
        # "duration",
        "email",
        "hostname",
        "idn-hostname",
        "ipv4",
        "ipv6",
        "json-pointer",
        "relative-json-pointer",
        "time",
        "uri",
        "uri-reference",
    ],
)
async def test_validates_formats(format):
    potential = JsonSchema({"format": format, "type": "string"})
    assert await potential.prefix(b'"hello world"') == -float("inf")


@pytest.mark.asyncio
async def test_validates_regex_format():
    potential = JsonSchema({"format": "regex", "type": "string"})
    assert await potential.prefix(b'"["') == -float("inf")


@pytest.mark.asyncio
async def test_will_not_allow_nonsense_after_json():
    potential = JsonSchema({"type": "object"})
    assert await potential.complete(b"{} hello world") == -float("inf")


@pytest.mark.asyncio
async def test_valid_prefix_for_schema_eg1():
    potential = JsonSchema(
        {
            "$schema": "http://json-schema.org/draft-04/schema#",
            "type": "array",
            "items": {
                "$schema": "http://json-schema.org/draft-04/schema#",
                "type": "object",
                "properties": {
                    "time": {"type": "string", "format": "date-time"},
                    "relayId": {"type": "string"},
                    "data": {
                        "type": "object",
                        "patternProperties": {
                            "^[0-9a-zA-Z_-]{1,255}$": {
                                "type": ["number", "string", "boolean"]
                            }
                        },
                        "additionalProperties": False,
                    },
                },
                "required": ["data"],
                "additionalProperties": False,
            },
        }
    )

    assert await potential.prefix(b"[{") == 0.0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ws",
    [
        b"\n\n\n",
        b"\n    \n",
    ],
)
async def test_forbids_weird_whitespace(ws):
    potential = JsonSchema({})
    assert await potential.prefix(ws) == -float("inf")


@pytest.mark.asyncio
async def test_rejects_as_prefix_when_invalid_key_has_been_started():
    potential = JsonSchema(
        {
            "type": "object",
            "properties": {
                "data": {
                    "type": "string",
                }
            },
            "required": ["data"],
            "additionalProperties": False,
        }
    )

    assert await potential.prefix(b'{"fo') == -float("inf")


@pytest.mark.asyncio
async def test_rejects_when_value_is_invalid_before_object_is_complete():
    potential = JsonSchema(
        {
            "type": "object",
            "properties": {
                "stuff": {
                    "type": "string",
                },
                "data": {
                    "type": "string",
                },
            },
            "additionalProperties": False,
        }
    )

    assert await potential.prefix(b'{"data": 1.0, ') == -float("inf")


@pytest.mark.asyncio
async def test_rejects_duplicated_key():
    potential = JsonSchema(
        {
            "type": "object",
        }
    )

    assert await potential.prefix(b'{"data": 1.0, "data"') == -float("inf")


@pytest.mark.asyncio
async def test_rejects_string_as_invalid_integer_before_complete():
    potential = JsonSchema(
        {
            "type": "integer",
        }
    )

    assert await potential.prefix(b'"') == -float("inf")


@pytest.mark.asyncio
async def test_rejects_string_as_invalid_integer_inside_list():
    potential = JsonSchema({"type": "array", "items": {"type": "integer"}})

    assert await potential.prefix(b'["') == -float("inf")


@pytest.mark.asyncio
async def test_can_extend_zero_to_integer_list():
    schema = {"type": "array", "items": {"type": "integer"}}
    potential = JsonSchema(schema)
    assert await potential.prefix(b"[0,") == 0


@dataclass(frozen=True)
class SchemaAndDocument:
    schema: Any
    document: Any


@st.composite
def json_schema_and_document(draw):
    schema = draw(json_schema())
    document = draw(from_schema(schema))
    return SchemaAndDocument(schema, document)


@pytest.mark.asyncio
@settings(report_multiple_bugs=False, deadline=None)
@given(json_schema_and_document())
async def test_parser_for_schema_always_returns_document(sad):
    parser = json_schema_parser(sad.schema)
    text = json.dumps(sad.document)
    result = await parser.parse_string(text)
    assert result == sad.document


@pytest.mark.asyncio
@example(
    JSONSchemaPotentialProblem(schema={"type": "integer"}, document=b"-1", prefix=b"-"),
)
@example(
    JSONSchemaPotentialProblem(
        schema={"type": "string"}, document=b'"\xc2\x80"', prefix=b'"'
    )
)
@example(
    JSONSchemaPotentialProblem(
        schema={
            "type": "object",
            "properties": {
                "0": {"type": "null"},
                "0\x7f": {"type": "null"},
                "1": {"type": "null"},
            },
            "required": ["0", "0\x7f", "1"],
            "additionalProperties": False,
        },
        document=b'{"0": null, "0\x7f": null, "1": null}',
        prefix=b"{",
    ),
)
@example(
    JSONSchemaPotentialProblem(
        schema={"type": "array", "items": {"type": "number"}},
        document=b"[\n1.3941332551795901e+28\n]",
        prefix=b"[\n1.3941332551795901e+",
    ),
)
@settings(report_multiple_bugs=False, deadline=None)
@given(json_schema_potential_problem())
async def test_parser_for_schema_prefix_can_only_raise_incomplete(problem):
    parser = json_schema_parser(problem.schema)

    # Just to get coverage on the repr methods.
    repr(parser)

    whole_text = problem.document.decode("utf-8")
    result = await parser.parse_string(whole_text)
    assert result == problem.value

    try:
        text = problem.prefix.decode("utf-8")
    except UnicodeDecodeError:
        reject()
    try:
        await parser.parse_string(text)
    except Incomplete:
        pass


@st.composite
def json_object(draw):
    return draw(
        st.one_of(
            st.none(),
            st.booleans(),
            st.floats(allow_nan=False, allow_infinity=False),
            st.text(),
            st.lists(json_object()),
            st.dictionaries(st.text(), json_object()),
        )
    )


@pytest.mark.asyncio
@example(False)
@settings(report_multiple_bugs=False, deadline=None)
@given(json_object())
async def test_parser_for_arbitrary_json_can_parse_arbitrary_json(obj):
    text = json.dumps(obj)
    await ARBITRARY_JSON.parse_string(text)


@pytest.mark.asyncio
@settings(report_multiple_bugs=False, deadline=None)
@given(st.sets(st.text()))
async def test_correctly_handles_fixed_object_keys(keys):
    parser = json_schema_parser(
        {
            "type": "object",
            "properties": {key: {"type": "null"} for key in keys},
            "additionalProperties": False,
        }
    )

    x = {key: None for key in keys}
    s = json.dumps(x)
    result = await parser.parse_string(s)
    assert result == x


@pytest.mark.asyncio
async def test_float_parser_incomplete_literal():
    with pytest.raises(Incomplete):
        await FLOAT_PARSER.parse_string("0.")


@st.composite
def chunked_utf8(draw):
    base = draw(st.text(min_size=1)).encode("utf-8")
    assume(len(base) > 1)
    offsets = draw(st.sets(st.integers(1, len(base) - 1)))
    offsets.update((0, len(base)))
    offsets = sorted(offsets)
    chunks = [base[u:v] for u, v in zip(offsets, offsets[1:])]
    assert b"".join(chunks) == base
    return chunks


@given(chunked_utf8())
@settings(report_multiple_bugs=False, deadline=None)
def test_utf8_chunking_always_splits_utf8(chunks):
    rechunked = list(chunk_to_complete_utf8(chunks))
    assert b"".join(rechunked) == b"".join(chunks)
    for chunk in rechunked:
        assert chunk
        chunk.decode("utf-8")


class BasicSource(AsyncSource):
    def __init__(self, blocks):
        self.__blocks = iter(blocks)

    async def more(self):
        try:
            return next(self.__blocks)
        except StopIteration:
            raise StopAsyncIteration()


@pytest.mark.asyncio
@given(chunked_utf8())
@settings(report_multiple_bugs=False, deadline=None)
async def test_utf8_chunking_always_splits_utf8_async(chunks):
    source = BasicSource(chunks)
    string_source = StringSource(source)

    buffer = bytearray()

    while True:
        try:
            chunk = await string_source.more()
        except StopAsyncIteration:
            break
        buffer.extend(chunk.encode("utf-8"))

    assert bytes(buffer) == b"".join(chunks)


@pytest.mark.asyncio
async def test_parser_raises_incomplete_on_empty_string():
    with pytest.raises(Incomplete):
        await FLOAT_PARSER.parse_string("")


@pytest.mark.asyncio
async def test_validates_a_list_of_integers_parser_only():
    parser = json_schema_parser({"type": "array", "items": {"type": "integer"}})

    with pytest.raises(Incomplete):
        await parser.parse_string("[1,2,3")

    with pytest.raises(ParseError):
        assert await parser.parse_string('["hello world"')

    with pytest.raises(ParseError):
        await parser.parse_string("{")


@pytest.mark.asyncio
async def test_can_calculate_many_prefixes():
    potential = JsonSchema({"type": "object"})

    for i in range(10000):
        prefix = b'{ "' + str(i).encode("utf-8")
        pot = await potential.prefix(prefix)
        assert pot == 0.0


@pytest.mark.asyncio
async def test_raises_value_error_for_logw_next_of_bad_prefix():
    potential = JsonSchema({"type": "object"})
    with pytest.raises(ValueError):
        await potential.logw_next(b"[")


@pytest.mark.asyncio
async def test_basic_json_validator_rejects_silly_whitespace():
    potential = ValidateJSON()
    assert await potential.prefix(b"\n\n\n") == -float("inf")
    assert await potential.complete(b"\n\n\n") == -float("inf")


@pytest.mark.asyncio
async def test_float_parser_can_continue_parsing_across_boundaries():
    source = BasicSource(["2", ".", "0", "1"])

    input = Input(source)

    parser = FloatParser()

    f = await input.parse(parser)

    assert f == 2.01



================================================
FILE: tests/potential/test_llm.py
================================================
import pytest
import torch
import numpy as np
from arsenal.maths import logsumexp
from hypothesis import given, strategies as st, settings

from genlm.control.potential.built_in import PromptedLLM

# pytest.mark.asyncio seems to cause issues with hypothesis
# and the vllm backend, so we use asyncio.run here.


async def reference_scorer(llm, context, eos=False, temp=1):
    """Compute the log probability of the context given the prompt."""
    context_ids = llm.encode_tokens(context)

    async def tempered(context_ids):
        logps = await llm.model.next_token_logprobs(context_ids)
        if temp != 1:
            logps = torch.log_softmax(logps / temp, dim=-1)
        return logps

    logps = await tempered(llm.prompt_ids)
    total_logp = logps[context_ids[0]].item()

    for i in range(1, len(context_ids)):
        logps = await tempered(llm.prompt_ids + context_ids[:i])
        total_logp += logps[context_ids[i]].item()

    if eos:
        logps = await tempered(llm.prompt_ids + context_ids)
        eos_logp = float("-inf")
        for i in llm.token_maps.eos_idxs:
            eos_logp = logsumexp([eos_logp, logps[i].item()])
        total_logp += eos_logp

    return total_logp


@pytest.fixture(
    scope="module",
    params=[
        ("hf", {"hf_opts": {"torch_dtype": "float"}}),
        # ("mock", {}),
    ],
)
def llm_config(request):
    return request.param


@pytest.fixture(scope="module")
def llm(llm_config):
    backend, opts = llm_config
    return PromptedLLM.from_name("gpt2", backend=backend, **opts)


@pytest.mark.asyncio
@given(st.text(min_size=1))
async def test_prompt_setting(llm, pre_prompt):
    pre_prompt_ids = llm.model.tokenizer.encode(pre_prompt)

    # Test ids setter
    llm.prompt_ids = pre_prompt_ids
    assert llm.prompt_ids == pre_prompt_ids
    assert b"".join(llm.prompt).decode() == pre_prompt

    # Test str setter
    llm.set_prompt_from_str(pre_prompt)
    assert b"".join(llm.prompt).decode() == pre_prompt
    assert llm.prompt_ids == pre_prompt_ids


@pytest.mark.asyncio
@settings(deadline=None, max_examples=50)
@given(st.text(min_size=1), st.text(min_size=1), st.floats(min_value=1e-6, max_value=3))
async def test_scoring(llm, pre_prompt, context_str, temp):
    pre_prompt_ids = llm.model.tokenizer.encode(pre_prompt)
    context = llm.tokenize(context_str)

    llm.temperature = temp
    llm.prompt_ids = pre_prompt_ids

    have = await llm.prefix(context)
    want = await reference_scorer(llm, context, temp=temp)
    assert np.isclose(have, want), [have, want]

    have = await llm.complete(context)
    want = await reference_scorer(llm, context, eos=True, temp=temp)
    assert np.isclose(have, want), [have, want]


@pytest.mark.asyncio
@settings(deadline=None, max_examples=50)
@given(
    st.text(min_size=1),
    st.text(min_size=1, max_size=10),
    st.floats(
        min_value=0.75, max_value=3
    ),  # TODO: scrutinize precision with low temperature
)
async def test_properties(llm, pre_prompt, context, temp):
    pre_prompt_ids = llm.model.tokenizer.encode(pre_prompt)
    llm.prompt_ids = pre_prompt_ids
    context = llm.tokenize(context)
    llm.temperature = temp

    await llm.assert_logw_next_consistency(context, top=10, rtol=0.01, atol=1e-3)
    await llm.assert_autoreg_fact(context, rtol=0.01, atol=1e-3)


@pytest.mark.asyncio
@settings(deadline=None, max_examples=50)
@given(st.lists(st.text(min_size=1), min_size=1, max_size=4))
async def test_batch_consistency(llm, contexts):
    contexts = [llm.tokenize(context) for context in contexts]
    await llm.assert_batch_consistency(contexts, rtol=1e-3, atol=1e-3)


@st.composite
def eos_test_params(draw):
    # Probably can decrase the size of these ranges for faster tests.
    eos_token_ids = draw(
        st.lists(
            st.integers(min_value=0, max_value=50256),
            min_size=1,
            max_size=3,
            unique=True,
        )
    )
    valid_ids = st.integers(min_value=0, max_value=50256).filter(
        lambda x: x not in eos_token_ids
    )
    context_ids = draw(st.lists(valid_ids, min_size=1, max_size=5))
    prompt_ids = draw(
        st.lists(st.integers(min_value=0, max_value=50256), min_size=1, max_size=5)
    )
    return eos_token_ids, context_ids, prompt_ids


@pytest.mark.asyncio
@settings(deadline=None)
@given(eos_test_params())
async def test_new_eos_tokens(llm, params):
    with pytest.raises(
        ValueError, match="Cannot reset eos_tokens after initialization"
    ):
        llm.eos_tokens = []

    eos_token_ids, context_ids, prompt_ids = params
    llm.prompt_ids = prompt_ids
    eos_tokens = [llm.token_maps.decode[x] for x in eos_token_ids]
    new_llm = llm.spawn_new_eos(eos_tokens=eos_tokens)
    assert new_llm.eos_tokens == eos_tokens

    new_llm.temperature = 1.0

    assert new_llm.prompt_ids == prompt_ids  # check prompt_ids is not changed
    assert new_llm.token_maps.eos_idxs == eos_token_ids
    assert set(new_llm.token_maps.decode) - set(eos_tokens) == set(new_llm.vocab)

    context = new_llm.decode_tokens(context_ids)
    have = await new_llm.complete(context)
    want = await reference_scorer(new_llm, context, eos=True)
    assert np.isclose(have, want), [have, want]


def test_invalid_eos_tokens(llm):
    # Test EOS token not in vocabulary
    invalid_eos = [b"THIS_TOKEN_DOES_NOT_EXIST"]
    with pytest.raises(ValueError, match="EOS token not in language model vocabulary"):
        llm.spawn_new_eos(eos_tokens=invalid_eos)

    # Test duplicate EOS tokens
    duplicate_eos = [llm.token_maps.decode[0], llm.token_maps.decode[0]]
    with pytest.raises(AssertionError, match="duplicate eos tokens"):
        llm.spawn_new_eos(eos_tokens=duplicate_eos)

    # Test attempting to modify eos_tokens directly
    with pytest.raises(
        ValueError, match="Cannot reset eos_tokens after initialization"
    ):
        llm.eos_tokens = [llm.token_maps.decode[0]]


def test_invalid_token_encoding(llm):
    # Test encoding invalid tokens
    invalid_tokens = [b"INVALID_TOKEN"]
    with pytest.raises(ValueError, match="Token .* not in vocabulary"):
        llm.encode_tokens(invalid_tokens)


def test_prompt_from_str_invalid_type(llm):
    with pytest.raises(ValueError, match="Prompt must a string"):
        llm.set_prompt_from_str(42)


def test_spawn(llm):
    new_llm = llm.spawn()
    assert new_llm.prompt_ids == llm.prompt_ids
    assert new_llm.token_maps.decode == llm.token_maps.decode
    assert new_llm.token_maps.eos_idxs == llm.token_maps.eos_idxs
    assert new_llm.vocab == llm.vocab


def test_to_autobatched(llm):
    with pytest.raises(ValueError, match="PromptedLLMs are autobatched by default"):
        llm.to_autobatched()


@pytest.mark.asyncio
@pytest.mark.skipif(not torch.cuda.is_available(), reason="vllm requires CUDA")
async def test_vllm_backend():
    # VLLM backend isn't playing well with hypothesis so we test it here.
    # Note though that any differences between backends are encapsulated in the AsyncLM class, which
    # is tested in genlm_backend, so we shouldn't expect any significant differences in testing outcomes.
    llm = PromptedLLM.from_name("gpt2", backend="vllm", engine_opts={"dtype": "float"})

    llm.set_prompt_from_str("hello")
    context = llm.tokenize(" world!")

    await llm.assert_logw_next_consistency(context, top=10, rtol=1e-3, atol=1e-3)
    await llm.assert_autoreg_fact(context, rtol=1e-3, atol=1e-3)
    await llm.assert_batch_consistency(
        [context, llm.tokenize(" world")], rtol=1e-3, atol=1e-3
    )

    new_llm = llm.spawn_new_eos(eos_tokens=[b"!"])
    assert new_llm.token_maps.eos_idxs == [0]
    assert new_llm.token_maps.decode[0] == b"!"

    context = llm.tokenize(" world")
    await new_llm.assert_logw_next_consistency(context, top=10, rtol=1e-3, atol=1e-3)
    await new_llm.assert_autoreg_fact(context, rtol=1e-3, atol=1e-3)
    await new_llm.assert_batch_consistency(
        [context, llm.tokenize(" worlds")], rtol=1e-3, atol=1e-3
    )


def test_llm_repr(llm):
    repr(llm)


def test_prompt_warning(llm):
    with pytest.warns(UserWarning):
        llm.set_prompt_from_str("hello ")



================================================
FILE: tests/potential/test_mp.py
================================================
import pytest
import numpy as np

from genlm.control.potential import Potential, MultiProcPotential


class SimplePotential(Potential):
    async def complete(self, context):
        return -float(len(context))

    async def prefix(self, context):
        return -0.5 * float(len(context))


@pytest.fixture
def V():
    return [b"a", b"b", b"c"]


@pytest.fixture
def mp_potential(V):
    return MultiProcPotential(SimplePotential, (V,), num_workers=2)


@pytest.fixture
def regular_potential(V):
    return SimplePotential(V)


@pytest.mark.asyncio
async def test_mp_score(mp_potential, regular_potential):
    seq = [b"b", b"c"]

    mp_score = await mp_potential.score(seq)
    regular_score = await regular_potential.score(seq)
    assert mp_score == regular_score == -1.0

    assert mp_potential.eos == regular_potential.eos

    seq_terminated = seq + [regular_potential.eos]
    mp_score = await mp_potential.score(seq_terminated)
    regular_score = await regular_potential.score(seq_terminated)
    assert mp_score == regular_score == -2.0


@pytest.mark.asyncio
async def test_mp_batch_score(mp_potential, regular_potential):
    contexts = [[b"a"], [b"a", b"b"], [b"a", b"b", regular_potential.eos]]

    have = await mp_potential.batch_score(contexts)
    want = await regular_potential.batch_score(contexts)
    np.testing.assert_array_equal(have, want)


@pytest.mark.asyncio
async def test_mp_prefix_complete(mp_potential, regular_potential):
    context = [b"b", b"c"]

    have = await mp_potential.prefix(context)
    want = await regular_potential.prefix(context)
    assert have == want == -1.0

    have = await mp_potential.complete(context)
    want = await regular_potential.complete(context)
    assert have == want == -2.0


@pytest.mark.asyncio
async def test_mp_batch_prefix_complete(mp_potential, regular_potential):
    contexts = [[b"a"], [b"a", b"b"]]

    have = await mp_potential.batch_prefix(contexts)
    want = await regular_potential.batch_prefix(contexts)
    np.testing.assert_array_equal(have, want)

    have = await mp_potential.batch_complete(contexts)
    want = await regular_potential.batch_complete(contexts)
    np.testing.assert_array_equal(have, want)


@pytest.mark.asyncio
async def test_mp_logw_next(mp_potential, regular_potential):
    seq = [b"b", b"c"]
    have = await mp_potential.logw_next(seq)
    want = await regular_potential.logw_next(seq)
    np.testing.assert_array_equal(have.weights, want.weights)


@pytest.mark.asyncio
async def test_mp_batch_logw_next(mp_potential, regular_potential):
    contexts = [[b"a"], [b"a", b"b"], [b"a", b"b", regular_potential.eos]]
    haves = await mp_potential.batch_logw_next(contexts)
    wants = await regular_potential.batch_logw_next(contexts)
    for have, want in zip(haves, wants):
        np.testing.assert_array_equal(have.weights, want.weights)


def test_cleanup(mp_potential):
    assert mp_potential.executor is not None
    mp_potential.__del__()
    assert mp_potential.executor is None


def test_mp_repr(mp_potential):
    repr(mp_potential)


def test_mp_spawn(mp_potential):
    with pytest.raises(ValueError):
        mp_potential.spawn()



================================================
FILE: tests/potential/test_operators.py
================================================
import pytest
from genlm.control.potential import (
    Potential,
    Coerced,
    Product,
    AutoBatchedPotential,
    MultiProcPotential,
)


class SimplePotential(Potential):
    """A simple potential for testing operators."""

    def __init__(self, vocabulary):
        super().__init__(vocabulary)

    async def complete(self, context):
        return 0

    async def prefix(self, context):
        return 0

    def spawn(self):
        return SimplePotential(self.vocab)


@pytest.fixture
def vocab():
    return [b"a"[0], b"b"[0], b"c"[0]]


@pytest.fixture
def p1(vocab):
    return SimplePotential(vocab)


@pytest.fixture
def p2(vocab):
    return SimplePotential(vocab)


@pytest.mark.asyncio
async def test_product_operator(p1, p2):
    have = p1 * p2
    want = Product(p1, p2)
    assert have.p1 == want.p1
    assert have.p2 == want.p2
    assert have.vocab == want.vocab


@pytest.mark.asyncio
async def test_coerce_operator(p1):
    target_vocab = [b"aa", b"bb", b"cc"]

    # Test with default transformations
    def f(seq):
        return [x for xs in seq for x in xs]

    coerced = p1.coerce(SimplePotential(target_vocab), f=f)
    assert set(coerced.vocab) == set(target_vocab)

    # Test with custom transformations
    def f(seq):
        return [xs[0] for xs in seq]

    have = p1.coerce(SimplePotential(target_vocab), f=f)
    want = Coerced(p1, target_vocab, f=f)
    assert have.potential == want.potential
    assert have.vocab == want.vocab


@pytest.mark.asyncio
async def test_to_autobatched(p1):
    have = p1.to_autobatched()
    want = AutoBatchedPotential(p1)
    assert have.potential == want.potential

    await have.cleanup()
    await want.cleanup()


@pytest.mark.asyncio
async def test_to_multiprocess(p1):
    num_workers = 2
    have = p1.to_multiprocess(num_workers=num_workers)
    want = MultiProcPotential(p1.spawn, (), num_workers=num_workers)
    assert have.vocab == want.vocab


@pytest.mark.asyncio
async def test_operator_chaining(p1, p2):
    have = (p1 * p2).to_autobatched()
    want = AutoBatchedPotential(Product(p1, p2))
    assert have.potential.p1 == want.potential.p1
    assert have.potential.p2 == want.potential.p2
    assert have.vocab == want.vocab

    await have.cleanup()
    await want.cleanup()

    V = [b"aa", b"bb", b"cc"]

    def f(seq):
        return [x for xs in seq for x in xs]

    have = (p1 * p2).coerce(SimplePotential(V), f=f)
    want = Coerced(Product(p1, p2), V, f=f)
    assert have.potential.p1 == want.potential.p1
    assert have.potential.p2 == want.potential.p2
    assert have.vocab == want.vocab



================================================
FILE: tests/potential/test_product.py
================================================
import re
import pytest
import numpy as np
from genlm.control.potential import Product, Potential
from genlm.control.typing import Atomic


class SimplePotential(Potential):
    def __init__(self, vocabulary, scale=1.0):
        super().__init__(vocabulary)
        self.scale = scale

    async def complete(self, context):
        return -float(len(context)) * self.scale

    async def prefix(self, context):
        return -0.5 * float(len(context)) * self.scale

    def spawn(self):
        return SimplePotential(self.vocab, scale=self.scale)


@pytest.fixture
def vocab():
    return [b"a", b"b", b"c"]


@pytest.fixture
def p1(vocab):
    return SimplePotential(vocab, scale=1.0)


@pytest.fixture
def p2(vocab):
    return SimplePotential(vocab, scale=2.0)


@pytest.fixture
def product(p1, p2):
    return Product(p1, p2)


def test_initialization(vocab, p1, p2):
    # Test successful initialization
    product = Product(p1, p2)
    assert product.token_type == Atomic(bytes)
    assert len(product.vocab) == len(vocab)
    assert set(product.vocab) == set(vocab)

    # Test mismatched token types
    class DifferentPotential(SimplePotential):
        def __init__(self):
            super().__init__([1, 2, 3])  # Different token type (int)

    with pytest.raises(
        ValueError, match="Potentials in product must have the same token type"
    ):
        Product(p1, DifferentPotential())

    # Test non-overlapping vocabularies
    p3 = SimplePotential([b"d", b"e", b"f"])
    with pytest.raises(
        ValueError, match="Potentials in product must share a common vocabulary"
    ):
        Product(p1, p3)


@pytest.mark.asyncio
async def test_prefix(product):
    context = [b"a", b"b"]
    result = await product.prefix(context)
    # Should be sum of both potentials' prefix values
    expected = -0.5 * len(context) * (1.0 + 2.0)
    assert result == expected


@pytest.mark.asyncio
async def test_complete(product):
    context = [b"a", b"b"]
    result = await product.complete(context)
    # Should be sum of both potentials' complete values
    expected = -len(context) * (1.0 + 2.0)
    assert result == expected


@pytest.mark.asyncio
async def test_logw_next(product):
    context = [b"a", b"b"]
    result = await product.logw_next(context)

    # Test that weights are properly combined
    weights = result.weights
    assert len(weights) == len(product.vocab_eos)

    # Test individual token weights
    for token in product.vocab:
        extended = context + [token]
        score = await product.score(extended)
        prefix_score = await product.prefix(context)
        expected_weight = score - prefix_score
        assert np.isclose(result.weights[product.lookup[token]], expected_weight)


@pytest.mark.asyncio
async def test_batch_operations(product):
    contexts = [[b"a"], [b"a", b"b"]]

    # Test batch_complete
    complete_results = await product.batch_complete(contexts)
    expected = [-3.0, -6.0]  # Combined scales (1.0 + 2.0) * -len(context)
    np.testing.assert_array_almost_equal(complete_results, expected)

    # Test batch_prefix
    prefix_results = await product.batch_prefix(contexts)
    expected = [-1.5, -3.0]  # Combined scales (1.0 + 2.0) * -0.5 * len(context)
    np.testing.assert_array_almost_equal(prefix_results, expected)


@pytest.mark.asyncio
async def test_properties(product):
    # Test the inherited property checks
    await product.assert_logw_next_consistency([b"b", b"c"], verbosity=1)
    await product.assert_autoreg_fact([b"b", b"c"], verbosity=1)
    await product.assert_batch_consistency([[b"b", b"c"], [b"a"]], verbosity=1)


def test_product_repr(product):
    repr(product)


def test_product_spawn(product):
    spawn = product.spawn()
    assert spawn.p1.vocab == product.p1.vocab and isinstance(spawn.p1, type(product.p1))
    assert spawn.p2.vocab == product.p2.vocab and isinstance(spawn.p2, type(product.p2))


def test_product_vocab_overlap():
    vocab = list(range(0, 11))
    p1 = SimplePotential(vocab, scale=1.0)
    p2 = SimplePotential(vocab[:1], scale=2.0)
    # Common vocabulary is less than 10% of p1's vocabulary
    with pytest.warns(RuntimeWarning):
        Product(p1, p2)

    with pytest.warns(RuntimeWarning):
        Product(p2, p1)


@pytest.mark.asyncio
async def test_product_laziness():
    class InfiniteAndCounterPotential(Potential):
        def __init__(self):
            super().__init__([b"a", b"b", b"c"])
            self.prefix_calls = 0
            self.complete_calls = 0

        async def complete(self, context):
            self.complete_calls += 1
            return float("-inf")

        async def prefix(self, context):
            self.prefix_calls += 1
            return float("-inf")

    p1 = InfiniteAndCounterPotential()
    p2 = InfiniteAndCounterPotential()
    product = Product(p1, p2)

    await product.prefix([])
    assert product.p1.prefix_calls == 1
    assert product.p2.prefix_calls == 0

    await product.complete([])
    assert product.p1.complete_calls == 1
    assert product.p2.complete_calls == 0


def test_product_token_type_mismatch():
    p1 = SimplePotential([b"a", b"b", b"c"], scale=1.0)
    p2 = SimplePotential([0, 1, 2], scale=2.0)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Potentials in product must have the same token type. "
            + "Got Atomic(bytes) and Atomic(int)."
            + "\nMaybe you forgot to coerce the potentials to the same token type? See `Coerce`."
        ),
    ):
        Product(p1, p2)

    p1 = SimplePotential([b"a", b"b", b"c"], scale=1.0)
    p2 = SimplePotential(["a", "b", "c"], scale=2.0)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Potentials in product must have the same token type. "
            + "Got Atomic(bytes) and Atomic(str)."
        ),
    ):
        Product(p1, p2)



================================================
FILE: tests/potential/test_stateful.py
================================================
from genlm.control.potential.stateful import make_immutable, StatefulPotential
from genlm.control.potential.streaming import (
    AsyncStreamingPotential,
    StreamingPotential,
    Timeout,
    PING_TOKEN,
    KEEP_ALIVE_SET,
)
import pytest
import asyncio
import time
from threading import Thread


def test_make_immutable_converts_non_bytes_to_tuple():
    assert make_immutable([257]) == (257,)


def test_make_immutable_converts_to_bytes_if_possible():
    assert make_immutable([]) == b""
    assert make_immutable([0]) == b"\x00"


class DummyPotential(StreamingPotential):
    def __init__(self):
        super().__init__(vocabulary=list(range(256)))

    def calculate_score_from_stream(self, stream) -> float:
        size = 0
        for s in stream:
            size += len(s)
        if size >= 10:
            return 0.0


async def no_sleep(time):
    pass


def no_start(*args, **kwargs):
    raise RuntimeError()


tock = 0


def fast_clock():
    global tock
    tock += 1
    return tock


@pytest.mark.asyncio
async def test_will_time_out_if_too_many_threads_start(monkeypatch):
    (monkeypatch.setattr(asyncio, "sleep", no_sleep),)
    monkeypatch.setattr(Thread, "start", no_start)
    monkeypatch.setattr(time, "time", fast_clock)
    potential = DummyPotential()
    with pytest.raises(Timeout):
        await potential.prefix(b"hi")


@pytest.mark.asyncio
async def test_finished_clone_is_no_op():
    potential = DummyPotential()
    state = potential.new_state()
    await state.finish()
    assert state.finished
    assert (await state.clone()) is state


def test_must_specify_state_class_or_implement_new_state():
    potential = StatefulPotential(vocabulary=[0, 1])
    with pytest.raises(NotImplementedError):
        potential.new_state()


def test_tokens_have_right_repr():
    assert repr(PING_TOKEN) == "PING_TOKEN"


class DummyAsyncPotential(AsyncStreamingPotential):
    def __init__(self):
        super().__init__(vocabulary=list(range(256)))

    async def calculate_score_from_stream(self, stream) -> float:
        size = 0
        while True:
            try:
                size += await stream.more()
            except StopAsyncIteration:
                break
        if size >= 10:
            return 0.0


@pytest.mark.asyncio
async def test_cleanup_clears_up_async_tasks():
    initial = len(KEEP_ALIVE_SET)
    potential = DummyAsyncPotential()
    await potential.prefix(b"hello")
    assert len(KEEP_ALIVE_SET) > initial
    await potential.cleanup()
    assert len(KEEP_ALIVE_SET) <= initial


@pytest.mark.asyncio
async def test_operations_after_finish_are_ignored():
    potential = DummyAsyncPotential()
    state = potential.new_state()
    await state.update_context([0])
    await state.finish()
    assert state.finished
    await state.update_context([0])
    assert len(state.context) == 1
    await state.finish()
    assert state.finished



================================================
FILE: tests/potential/test_testing.py
================================================
import pytest
import numpy as np
from genlm.control.potential.base import Potential


class MockPotential(Potential):
    def __init__(self, has_errors=False):
        self.has_errors = has_errors
        super().__init__([1, 2, 3])

    async def complete(self, context):
        return 1.0

    async def prefix(self, context):
        if self.has_errors:
            return float("-inf")
        return 1.0

    async def logw_next(self, context):
        weights = np.array([0] * (len(self.vocab_eos)))
        if self.has_errors:
            weights[0] = 100.0  # Create inconsistency
        return self.make_lazy_weights(weights)


@pytest.mark.asyncio
async def test_assert_logw_next_consistency():
    pot = MockPotential()
    await pot.assert_logw_next_consistency([], verbosity=1)
    await pot.assert_logw_next_consistency([1], verbosity=1)

    pot = MockPotential(has_errors=True)
    with pytest.raises(AssertionError) as exc:
        await pot.assert_logw_next_consistency([])
    assert "logw_next consistency" in str(exc.value)

    pot = MockPotential()
    await pot.assert_logw_next_consistency([], top=2)


@pytest.mark.asyncio
async def test_assert_autoreg_fact():
    pot = MockPotential()
    await pot.assert_autoreg_fact([], verbosity=1)
    await pot.assert_autoreg_fact([1], verbosity=1)

    pot = MockPotential(has_errors=True)
    with pytest.raises(AssertionError) as exc:
        await pot.assert_autoreg_fact([])
    assert "Factorization not satisfied" in str(exc.value)


@pytest.mark.asyncio
async def test_assert_batch_consistency():
    pot = MockPotential()
    await pot.assert_batch_consistency([[1], [2]], verbosity=1)

    class ScoreErrorPotential(MockPotential):
        async def score(self, context, *args):
            return 100.0

        async def batch_score(self, contexts, *args):
            return [1000.0] * len(contexts)

    pot = ScoreErrorPotential()
    with pytest.raises(AssertionError) as exc:
        await pot.assert_batch_consistency([[]], verbosity=1)
    assert "Batch score mismatch" in str(exc.value)

    class LogwNextErrorPotential(MockPotential):
        async def logw_next(self, context, *args):
            return self.make_lazy_weights([0.0] * len(self.vocab_eos))

        async def batch_logw_next(self, contexts, *args):
            return [self.make_lazy_weights([1.0] * len(self.vocab_eos))] * len(contexts)

    pot = LogwNextErrorPotential()
    with pytest.raises(AssertionError) as exc:
        await pot.assert_batch_consistency([[]], verbosity=1)
    assert "Batch logw_next mismatch" in str(exc.value)



================================================
FILE: tests/potential/test_wcfg.py
================================================
import pytest
import numpy as np
from genlm.grammar import CFG, Float, Boolean
from genlm.control.potential.built_in import WCFG, BoolCFG


@pytest.fixture
def byte_wcfg():
    c = CFG(Float, S="S", V={b"a"[0], b"b"[0]})
    c.add(3.0, "S", "A", "B")
    c.add(2.0, "S", "A", "B", "B")
    c.add(1.0, "A", b"a"[0])
    c.add(1.0, "B", b"b"[0])
    return c


def test_wcfg_init_wrong_semiring():
    # Test initialization with non-Float semiring
    c = CFG(Boolean, S="S", V={b"a"[0], b"b"[0]})
    with pytest.raises(ValueError):
        WCFG(c)


@pytest.mark.asyncio
async def test_wcfg_complete(byte_wcfg):
    pot = WCFG(byte_wcfg)

    log_weight = await pot.complete(b"ab")
    assert np.isclose(log_weight, np.log(3))

    log_weight = await pot.complete(b"abb")
    assert np.isclose(log_weight, np.log(2))

    log_weight = await pot.complete(b"a")
    assert log_weight == float("-inf")


@pytest.mark.asyncio
async def test_wcfg_prefix(byte_wcfg):
    pot = WCFG(byte_wcfg)

    log_weight = await pot.prefix(b"ab")
    assert np.isclose(log_weight, np.log(5))

    log_weight = await pot.complete(b"abb")
    assert np.isclose(log_weight, np.log(2))

    log_weight = await pot.prefix(b"a")
    assert np.isclose(log_weight, np.log(5))


@pytest.mark.asyncio
async def test_bcfg_complete(byte_wcfg):
    pot = BoolCFG(byte_wcfg)

    log_weight = await pot.complete(b"ab")
    assert log_weight == 0

    log_weight = await pot.complete(b"abb")
    assert log_weight == 0

    log_weight = await pot.complete(b"a")
    assert log_weight == float("-inf")


@pytest.mark.asyncio
async def test_bcfg_prefix(byte_wcfg):
    pot = BoolCFG(byte_wcfg)

    # Test empty string handling
    log_weight = await pot.prefix(b"")
    assert log_weight == 0

    log_weight = await pot.prefix(b"ab")
    assert log_weight == 0

    log_weight = await pot.complete(b"abb")
    assert log_weight == 0

    log_weight = await pot.prefix(b"a")
    assert log_weight == 0


@pytest.mark.asyncio
async def test_properties(byte_wcfg):
    pot = WCFG(byte_wcfg)

    await pot.assert_logw_next_consistency(b"ab")
    await pot.assert_autoreg_fact(b"ab")
    await pot.assert_batch_consistency([b"a", b"ab"])

    await pot.assert_logw_next_consistency(b"")
    await pot.assert_autoreg_fact(b"")
    await pot.assert_batch_consistency([b""])

    pot = BoolCFG(byte_wcfg)

    await pot.assert_logw_next_consistency(b"ab")
    await pot.assert_autoreg_fact(b"ab")
    await pot.assert_batch_consistency([b"a", b"ab"])

    await pot.assert_logw_next_consistency(b"")
    await pot.assert_autoreg_fact(b"")
    await pot.assert_batch_consistency([b""])


@pytest.mark.asyncio
async def test_wcfg_from_string():
    grammar = """
    3.0: S -> A B
    2.0: S -> A B B
    1.0: A -> a
    1.0: B -> b
    """
    pot = WCFG.from_string(grammar)

    log_weight = await pot.complete(b"ab")
    assert np.isclose(log_weight, np.log(3))

    log_weight = await pot.complete(b"abb")
    assert np.isclose(log_weight, np.log(2))

    log_weight = await pot.complete(b"a")
    assert log_weight == float("-inf")

    pot_spawned = pot.spawn()
    assert pot_spawned.cfg == pot.cfg


@pytest.mark.asyncio
async def test_bcfg_from_lark():
    lark_grammar = """
    start: A B | A B B
    A: "a"
    B: "b"
    """
    pot = BoolCFG.from_lark(lark_grammar)

    log_weight = await pot.complete(b"ab")
    assert log_weight == 0

    log_weight = await pot.complete(b"abb")
    assert log_weight == 0

    log_weight = await pot.complete(b"a")
    assert log_weight == float("-inf")

    pot_spawned = pot.spawn()
    assert pot_spawned.cfg == pot.cfg


def test_wcfg_repr():
    c = CFG(Float, S="S", V={b"a"[0]})
    pot = WCFG(c)
    repr(pot)
    pot._repr_html_()


def test_bcfg_repr():
    c = CFG(Boolean, S="S", V={b"a"[0]})
    pot = BoolCFG(c)
    repr(pot)
    pot._repr_html_()


def test_wcfg_clear_cache():
    c = CFG(Float, S="S", V={b"a"[0]})
    pot = WCFG(c)
    pot.clear_cache()


def test_bcfg_clear_cache():
    c = CFG(Boolean, S="S", V={b"a"[0]})
    pot = BoolCFG(c)
    pot.clear_cache()



================================================
FILE: tests/potential/test_wfsa.py
================================================
import re
import pytest
import graphviz
import numpy as np
from genlm.grammar import WFSA as BaseWFSA, Float, Log, Boolean
from genlm.control.potential.built_in import WFSA, BoolFSA
from hypothesis import strategies as st, given, settings


@pytest.fixture
def float_wfsa():
    """Creates a simple WFSA in float semiring"""
    m = BaseWFSA(Float)
    m.add_I(0, 1.0)
    m.add_arc(0, b"a"[0], 1, 2)
    m.add_arc(1, b"b"[0], 2, 1)
    m.add_arc(1, b"c"[0], 2, 1)
    m.add_arc(1, b"d"[0], 3, 1)  # dead end
    m.add_F(2, 1.0)
    return m


@pytest.fixture
def log_wfsa():
    """Creates a simple WFSA in float semiring"""
    m = BaseWFSA(Log)
    m.add_I(0, Log(0.0))
    m.add_arc(0, b"a"[0], 1, Log(0.0))
    m.add_arc(1, b"b"[0], 2, Log(np.log(0.6)))
    m.add_arc(1, b"c"[0], 2, Log(np.log(0.4)))
    m.add_arc(1, b"d"[0], 3, Log(-float("inf")))  # dead end
    m.add_F(2, Log(0.0))
    return m


@pytest.mark.asyncio
async def test_wfsa(float_wfsa):
    pot = WFSA(float_wfsa)

    log_weight = await pot.complete(b"ab")
    assert np.isclose(log_weight, np.log(2))

    log_weight = await pot.complete(b"ac")
    assert np.isclose(log_weight, np.log(2))

    log_weight = await pot.complete(b"a")
    assert log_weight == float("-inf")

    log_weight = await pot.prefix(b"a")
    assert np.isclose(log_weight, np.log(4))

    log_weight = await pot.prefix(b"c")
    assert log_weight == float("-inf")

    log_weight = await pot.prefix(b"ab")
    assert np.isclose(log_weight, np.log(2))

    await pot.assert_logw_next_consistency(b"a")
    await pot.assert_autoreg_fact(b"a")

    await pot.assert_logw_next_consistency(b"")
    await pot.assert_autoreg_fact(b"")

    await pot.assert_batch_consistency([b"", b"ab", b"ac"])


@pytest.mark.asyncio
async def test_wfsa_regex():
    pot = WFSA.from_regex("a(b|c)")

    log_weight = await pot.complete(b"ab")
    assert np.isclose(log_weight, np.log(0.5))

    log_weight = await pot.complete(b"ac")
    assert np.isclose(log_weight, np.log(0.5))

    log_weight = await pot.complete(b"a")
    assert log_weight == float("-inf")

    log_weight = await pot.prefix(b"a")
    assert np.isclose(log_weight, 0)

    log_weight = await pot.prefix(b"c")
    assert log_weight == float("-inf")

    log_weight = await pot.prefix(b"ab")
    assert np.isclose(log_weight, np.log(0.5))

    await pot.assert_logw_next_consistency(b"a")
    await pot.assert_autoreg_fact(b"a")

    await pot.assert_logw_next_consistency(b"")
    await pot.assert_autoreg_fact(b"")

    await pot.assert_batch_consistency([b"", b"ab", b"ac"])


@pytest.mark.asyncio
async def test_bool_fsa(float_wfsa):
    pot = BoolFSA(float_wfsa)

    log_weight = await pot.complete(b"ab")
    assert log_weight == 0

    log_weight = await pot.complete(b"ac")
    assert log_weight == 0

    log_weight = await pot.complete(b"a")
    assert log_weight == float("-inf")

    log_weight = await pot.prefix(b"a")
    assert log_weight == 0

    log_weight = await pot.prefix(b"c")
    assert log_weight == float("-inf")

    log_weight = await pot.prefix(b"ab")
    assert log_weight == 0

    await pot.assert_logw_next_consistency(b"a")
    await pot.assert_autoreg_fact(b"a")

    await pot.assert_logw_next_consistency(b"")
    await pot.assert_autoreg_fact(b"")

    await pot.assert_batch_consistency([b"", b"ab", b"ac"])


@pytest.mark.asyncio
async def test_wfsa_long_ctx():
    # Test that we don't underflow when the context is long.
    pot = BoolFSA.from_regex(r".*")
    long_ctx = b"a" * 1000

    log_weight = await pot.complete(long_ctx)
    assert log_weight == 0

    log_weight = await pot.prefix(long_ctx)
    assert log_weight == 0


@st.composite
def regex_pattern(draw, max_depth=3):
    """Composite strategy to generate nested regex patterns"""

    def pattern_strategy(depth):
        if depth <= 0:
            # Base case: single escaped character
            char = draw(st.characters(blacklist_categories=("Cs",)))
            return re.escape(char)

        # Choose which type of pattern to generate
        pattern_type = draw(
            st.sampled_from(
                [
                    "simple",
                    "alternation",
                    "concatenation",
                    "optional",
                    "kleene",
                    "plus",
                    "quantified",
                ]
            )
        )

        if pattern_type == "simple":
            return pattern_strategy(0)

        # Generate sub-pattern(s)
        if pattern_type in ("alternation", "concatenation"):
            num_patterns = draw(st.integers(min_value=2, max_value=3))
            patterns = [pattern_strategy(depth - 1) for _ in range(num_patterns)]

            if pattern_type == "alternation":
                return f"({'|'.join(patterns)})"
            else:  # concatenation
                return f"({''.join(patterns)})"

        # Single sub-pattern with operator
        sub_pattern = pattern_strategy(depth - 1)
        if pattern_type == "optional":
            return f"({sub_pattern})?"
        elif pattern_type == "kleene":
            return f"({sub_pattern})*"
        elif pattern_type == "plus":
            return f"({sub_pattern})+"
        else:  # quantified
            quantifier = draw(st.sampled_from(["+", "*", "?", "{1,3}"]))
            return f"({sub_pattern}){quantifier}"

    return pattern_strategy(max_depth)


@pytest.mark.asyncio
@settings(deadline=None)
@given(regex_pattern(max_depth=3), st.data())
async def test_bool_fsa_with_generated_regex(pattern, data):
    """Test that BoolFSA accepts strings that match its regex pattern"""
    pot = BoolFSA.from_regex(pattern)

    matching_str = data.draw(st.from_regex(pattern, fullmatch=True))
    byte_string = matching_str.encode("utf-8")

    log_weight = await pot.complete(byte_string)
    assert log_weight == 0, [matching_str, pattern]

    for prefix in range(len(byte_string)):
        log_weight = await pot.prefix(byte_string[:prefix])
        assert log_weight == 0, [matching_str, byte_string[:prefix]]


def test_wfsa_init_wrong_semiring():
    # Test initialization with unsupported semiring
    wfsa = BaseWFSA(Boolean)  # TODO: support this semiring
    with pytest.raises(ValueError, match="Unsupported semiring"):
        WFSA(wfsa=wfsa)


def test_wfsa_init_float_conversion(log_wfsa):
    # Test that Float semiring is converted to Log
    pot = WFSA(wfsa=log_wfsa)
    assert pot.wfsa.R is Log


def test_wfsa_init_log_no_conversion(log_wfsa):
    # Test that Log semiring is not converted
    pot = WFSA(wfsa=log_wfsa)
    assert pot.wfsa.R is Log
    assert pot.wfsa is log_wfsa


def test_wfsa_repr(log_wfsa):
    pot = WFSA(wfsa=log_wfsa)
    repr(pot)

    try:
        pot._repr_svg_()
    except graphviz.backend.execute.ExecutableNotFound:
        pytest.skip("Graphviz not installed")


def test_bool_fsa_repr(log_wfsa):
    pot = BoolFSA(wfsa=log_wfsa)
    repr(pot)

    try:
        pot._repr_svg_()
    except graphviz.backend.execute.ExecutableNotFound:
        pytest.skip("Graphviz not installed")


def test_wfsa_spawn(log_wfsa):
    pot = WFSA(wfsa=log_wfsa)
    spawned = pot.spawn()
    assert isinstance(spawned, WFSA)


def test_wfsa_clear_cache(log_wfsa):
    pot = WFSA(wfsa=log_wfsa)
    pot.clear_cache()
    assert len(pot.cache) == 1
    assert () in pot.cache


@pytest.mark.asyncio
async def test_zero_weight_context():
    pot = WFSA.from_regex(r"a")
    with pytest.raises(ValueError, match="Context.*has zero weight."):
        await pot.logw_next(b"b")

    pot = BoolFSA.from_regex(r"a")
    with pytest.raises(ValueError, match="Context.*has zero weight."):
        await pot.logw_next(b"b")



================================================
FILE: tests/sampler/test_awrs.py
================================================
import pytest
import asyncio
import numpy as np
from arsenal.maths import logsumexp
from conftest import MockPotential
from hypothesis import given, strategies as st, settings, reject

from genlm.control.sampler.token import AWRS


async def monte_carlo(sampler, context, N, **kwargs):
    # Used for testing.
    samples = await asyncio.gather(
        *[sampler.sample(context, **kwargs) for _ in range(N)]
    )
    logws = sampler.target.alloc_logws()
    for tok, logw, _ in samples:
        if logw == float("-inf"):
            continue

        token_id = sampler.target.lookup[tok]

        if logws[token_id] == float("-inf"):
            logws[token_id] = logw - np.log(N)
        else:
            logws[token_id] = logsumexp([logws[token_id], logw - np.log(N)])

    return sampler.target.make_lazy_weights(logws)


async def assert_monte_carlo_close(
    sampler_cls, params, N, equality_opts={}, sampler_opts={}
):
    vocab, b_weights, c_weights = params
    potential = MockPotential(
        vocab,
        np.array([np.log(w) if w > 0 else float("-inf") for w in c_weights]),
    )
    condition = MockPotential(
        vocab,
        np.array([np.log(w) if w > 0 else float("-inf") for w in b_weights]),
    )

    sampler = sampler_cls(potential, condition, **sampler_opts)

    want = await sampler.target.logw_next([])
    have = await monte_carlo(sampler, [], N)

    assert np.isclose(np.exp(want.sum()), np.exp(have.sum()), **equality_opts)


# async def assert_variance_reduction(sampler_cls, params, N1, N2, K, sampler_opts={}):
#     # Check that the variance of the logZ estimate is reduced when using
#     # a larger number of samples.
#     assert N1 < N2

#     vocab, b_weights, c_weights = params
#     potential = MockPotential(vocab, np.log(c_weights))
#     condition = MockPotential(vocab, np.log(b_weights))

#     sampler = sampler_cls(potential, condition, **sampler_opts)

#     N1s = await asyncio.gather(*[monte_carlo(sampler, [], N1) for _ in range(K)])
#     Zs_N1 = np.array([np.exp(have.sum()) for have in N1s])
#     N2s = await asyncio.gather(*[monte_carlo(sampler, [], N2) for _ in range(K)])
#     Zs_N2 = np.array([np.exp(have.sum()) for have in N2s])

#     var_N1 = np.var(Zs_N1)
#     var_N2 = np.var(Zs_N2)

#     # If both variances are extremely small (close to machine epsilon),
#     # the test should pass regardless of their relative values
#     epsilon = 1e-30
#     if var_N1 < epsilon and var_N2 < epsilon:
#         return

#     assert var_N1 > var_N2


@st.composite
def V_size(draw):
    # Generate a vocabulary of size <=4.
    return draw(st.integers(min_value=1, max_value=4))


@st.composite
def cont_weights(draw, V_size, min_p=1e-3):
    # Generate a list of floats for each token in the vocabulary (and EOS).
    ws = draw(st.lists(st.floats(min_p, 1), min_size=V_size + 1, max_size=V_size + 1))
    Z = sum(ws)
    ps = [w / Z for w in ws]
    return ps


@st.composite
def bool_weights(draw, V_size):
    # Generate a list of booleans for each token in the vocabulary (and EOS).
    bws = draw(st.lists(st.booleans(), min_size=V_size + 1, max_size=V_size + 1))
    if not any(bws):
        # Need at least one valid token.
        reject()
    return bws


@st.composite
def params(draw, min_p=1e-3):
    vocab_size = draw(V_size())
    b_weights = draw(bool_weights(vocab_size))
    c_weights = draw(cont_weights(vocab_size, min_p))
    return [bytes([i]) for i in range(vocab_size)], b_weights, c_weights


@pytest.mark.asyncio
@settings(deadline=None, max_examples=25)
@given(params())
async def test_awrs(params):
    await assert_monte_carlo_close(
        sampler_cls=AWRS,
        params=params,
        N=10000,
        equality_opts={"rtol": 2e-2, "atol": 2e-2},
    )


@pytest.mark.asyncio
@settings(deadline=None, max_examples=25)
@given(params())
async def test_awrs_no_pruning(params):
    await assert_monte_carlo_close(
        sampler_cls=AWRS,
        params=params,
        N=10000,
        equality_opts={"rtol": 2e-2, "atol": 2e-2},
        sampler_opts={"prune_logws": False},
    )


@pytest.mark.asyncio
@settings(deadline=None, max_examples=25)
@given(params())
async def test_awrs_improper_weights_no_pruning(params):
    params = (params[0], [True] * len(params[1]), params[2])

    await assert_monte_carlo_close(
        sampler_cls=AWRS,
        params=params,
        N=10000,
        equality_opts={"rtol": 2e-2, "atol": 2e-2},
        sampler_opts={"proper_weights": False, "prune_logws": False},
    )


@pytest.fixture
def potential():
    return MockPotential(
        [bytes([i]) for i in range(4)],
        np.log([0.1, 0.2, 0.2, 0.1, 0.4]),
    )


@pytest.fixture
def zero_condition():
    return MockPotential(
        [bytes([i]) for i in range(4)],
        [float("-inf")] * 4,
    )


@pytest.mark.asyncio
async def test_verbosity(potential):
    condition = MockPotential(
        [bytes([i]) for i in range(4)],
        [0, 0, float("-inf"), float("-inf"), 0],
    )
    sampler = AWRS(potential=potential, condition=condition)
    await sampler.sample([], verbosity=1)


@pytest.mark.asyncio
async def test_awrs_no_valid_tokens(potential, zero_condition):
    sampler = AWRS(potential=potential, condition=zero_condition)
    tok, logw, _ = await sampler.sample([])
    assert tok == potential.eos
    assert logw == float("-inf")


@pytest.mark.asyncio
async def test_awrs_improper_weights_no_valid_tokens(potential, zero_condition):
    sampler = AWRS(
        potential=potential,
        condition=zero_condition,
        proper_weights=False,
    )
    tok, logw, _ = await sampler.sample([])
    assert tok == potential.eos
    assert logw == float("-inf")


@pytest.mark.asyncio
async def test_awrs_with_different_vocabs():
    potential = MockPotential(
        [bytes([i]) for i in range(4)],
        np.log([0.4, 0.3, 0.1, 0.1, 0.1]),
    )
    condition = MockPotential(
        [bytes([i]) for i in range(3)],
        [0, 0, float("-inf"), float("-inf")],
    )

    sampler = AWRS(potential, condition, prune_logws=True)

    want = await sampler.target.logw_next([])
    have = await monte_carlo(sampler, [], 10000)

    assert np.isclose(np.exp(want.sum()), np.exp(have.sum()), rtol=5e-3, atol=5e-3)


@pytest.mark.asyncio
async def test_awrs_with_no_pruning_and_different_vocabs():
    potential = MockPotential(
        [bytes([i]) for i in range(4)],
        np.log([0.4, 0.3, 0.1, 0.1, 0.1]),
    )
    condition = MockPotential(
        [bytes([i]) for i in range(3)],
        [0, 0, float("-inf"), float("-inf")],
    )

    sampler = AWRS(potential, condition, prune_logws=False)

    want = await sampler.target.logw_next([])
    have = await monte_carlo(sampler, [], 10000)

    assert np.isclose(np.exp(want.sum()), np.exp(have.sum()), rtol=5e-3, atol=5e-3)



================================================
FILE: tests/sampler/test_seq_sampler.py
================================================
import pytest
import numpy as np


from genlm.control.potential import Potential
from genlm.control.sampler.sequence import SMC, SequenceModel
from genlm.control.sampler.token import DirectTokenSampler

from hypothesis import strategies as st, settings, given
from conftest import (
    weighted_set,
    weighted_sequence,
    double_weighted_sequence,
    WeightedSet,
)


@pytest.fixture
def default_unit_sampler():
    sequences = ["a", "b", "c"]
    weights = [1, 2, 3]
    p = WeightedSet(sequences, weights)
    return DirectTokenSampler(p)


@pytest.mark.asyncio
@settings(deadline=None)
@given(weighted_set(weighted_sequence))
async def test_importance(S):
    sequences, weights = zip(*S)

    p = WeightedSet(sequences, weights)
    unit_sampler = DirectTokenSampler(p)

    n_particles = 100
    sampler = SMC(unit_sampler)

    sequences = await sampler(n_particles=n_particles, ess_threshold=0, max_tokens=10)
    assert len(sequences) == n_particles
    assert np.isclose(sequences.log_ml, np.log(sum(weights)), atol=1e-3, rtol=1e-5)


@pytest.mark.asyncio
@settings(deadline=None)
@given(weighted_set(double_weighted_sequence))
async def test_importance_with_critic(S):
    sequences, weights1, weights2 = zip(*S)

    p = WeightedSet(sequences, weights1)
    unit_sampler = DirectTokenSampler(p)
    critic = WeightedSet(sequences, weights2)

    n_particles = 10
    sampler = SMC(unit_sampler, critic=critic)
    sequences = await sampler(n_particles=n_particles, ess_threshold=0, max_tokens=10)

    logeps = await p.prefix([])
    for seq, logw in sequences:
        logZ = sum([(await p.logw_next(seq[:n])).sum() for n in range(len(seq))])
        assert np.isclose(logw, logZ + logeps + await critic.score(seq))


@pytest.mark.asyncio
@settings(deadline=None)
@given(weighted_set(weighted_sequence), st.floats(min_value=0, max_value=1))
async def test_smc(S, ess_threshold):
    sequences, weights = zip(*S)

    p = WeightedSet(sequences, weights)
    unit_sampler = DirectTokenSampler(p)

    n_particles = 100
    sampler = SMC(unit_sampler)

    sequences = await sampler(
        n_particles=n_particles, ess_threshold=ess_threshold, max_tokens=10
    )
    assert len(sequences) == n_particles
    assert np.isclose(sequences.log_ml, np.log(sum(weights)), atol=1e-3, rtol=1e-5)


@pytest.mark.asyncio
@settings(deadline=None)
@given(st.floats(min_value=0, max_value=1))
async def test_smc_with_critic(ess_threshold):
    seqs = ["0", "00", "1"]
    weights1 = [3.0, 2.0, 1.0]
    weights2 = [1.0, 2.0, 3.0]

    p = WeightedSet(seqs, weights1)
    unit_sampler = DirectTokenSampler(p)
    critic = WeightedSet(seqs, weights2)

    n_particles = 500
    sampler = SMC(unit_sampler, critic=critic)

    sequences = await sampler(
        n_particles=n_particles, ess_threshold=ess_threshold, max_tokens=10
    )

    intersection_ws = [w1 * w2 for w1, w2 in zip(weights1, weights2)]
    assert len(sequences) == n_particles
    assert np.isclose(
        np.exp(sequences.log_ml), sum(intersection_ws), atol=0.5, rtol=0.05
    )


@st.composite
def smc_params(draw, item_sampler, max_seq_len=5, max_size=5):
    S = draw(weighted_set(item_sampler, max_seq_len, max_size))
    stop_point = draw(st.integers(min_value=1, max_value=max_seq_len))
    return S, stop_point


@pytest.mark.asyncio
@settings(deadline=None)
@given(smc_params(double_weighted_sequence))
async def test_smc_weights(params):
    S, stop_point = params
    sequences, weights1, weights2 = zip(*S)

    p = WeightedSet(sequences, weights1)
    unit_sampler = DirectTokenSampler(p)
    critic = WeightedSet(sequences, weights2)

    n_particles = 10
    sampler = SMC(unit_sampler, critic=critic)

    sequences = await sampler(
        n_particles=n_particles,
        ess_threshold=0,  # don't resample since that would reset weights
        max_tokens=stop_point,
    )

    logeps = await p.prefix([])
    for seq, logw in sequences:
        logZ = sum([(await p.logw_next(seq[:n])).sum() for n in range(len(seq))])
        twist = await critic.score(seq)
        assert np.isclose(logw, logZ + logeps + twist)


@pytest.mark.asyncio
async def test_sequence_model_invalid_start_weight():
    class MockPotential(Potential):
        async def prefix(self, context):
            if not context:
                return -np.inf
            return 0

        async def complete(self, context):
            return 0

    unit_sampler = DirectTokenSampler(MockPotential([0]))
    seq_model = SequenceModel(unit_sampler)
    with pytest.raises(ValueError, match="Start weight.*"):
        await seq_model.start()


def test_sequence_model_str_for_serialization(default_unit_sampler):
    SequenceModel(default_unit_sampler).string_for_serialization()



================================================
FILE: tests/sampler/test_sequences.py
================================================
import pytest
import numpy as np
from genlm.control.sampler.sequence import Sequences, EndOfSequence, EOS


def test_initialization():
    sequences = Sequences(
        contexts=[[b"a"], [b"b"]],
        log_weights=[np.log(0.4), np.log(0.6)],
    )
    assert sequences.size == 2
    assert np.isclose(np.exp(sequences.log_total), 1.0)  # weights sum to 1
    assert len(sequences) == 2


def test_initialization_validation():
    # Test mismatched lengths
    with pytest.raises(AssertionError):
        Sequences(contexts=[[b"a"]], log_weights=[0.0, 0.0])


def test_posterior():
    # Test posterior without EOS filtering
    sequences = Sequences(
        contexts=[
            [b"hello"],  # No EOS
            [b"world", EndOfSequence()],
        ],
        log_weights=[np.log(0.4), np.log(0.6)],
    )
    posterior = sequences.posterior
    assert len(posterior) == 2
    assert np.isclose(posterior[tuple([b"hello"])], 0.4)
    assert np.isclose(posterior[tuple([b"world", EndOfSequence()])], 0.6)


def test_normalized_weights():
    sequences = Sequences(
        contexts=[[b"a"], [b"b"]],
        log_weights=[np.log(3), np.log(7)],
    )
    weights = sequences.normalized_weights
    assert np.allclose(weights, [0.3, 0.7])
    assert np.isclose(np.sum(weights), 1.0)


def test_iteration_and_indexing():
    contexts = [[b"a"], [b"b"]]
    log_weights = [np.log(0.3), np.log(0.7)]
    sequences = Sequences(contexts=contexts, log_weights=log_weights)

    # Test __iter__
    for i, (ctx, weight) in enumerate(sequences):
        assert ctx == contexts[i]
        assert weight == log_weights[i]

    # Test __getitem__
    assert sequences[0] == (contexts[0], log_weights[0])
    assert sequences[1] == (contexts[1], log_weights[1])


def test_effective_sample_size():
    # Test equal weights (maximum ESS)
    sequences = Sequences(
        contexts=[[b"a"], [b"b"], [b"c"]],
        log_weights=[0.0, 0.0, 0.0],  # equal weights
    )
    assert np.isclose(sequences.ess, 3.0)  # ESS should equal number of particles

    # Test completely unbalanced weights (minimum ESS)
    sequences = Sequences(
        contexts=[[b"a"], [b"b"], [b"c"]],
        log_weights=[
            np.log(1.0),
            float("-inf"),
            float("-inf"),
        ],  # one particle has all weight
    )
    assert np.isclose(sequences.ess, 1.0)


def test_log_ml_calculation():
    # Test log marginal likelihood calculation
    sequences = Sequences(
        contexts=[[b"a"], [b"b"]],
        log_weights=[np.log(0.3), np.log(0.7)],
    )
    assert np.isfinite(sequences.log_ml)
    assert sequences.log_ml <= sequences.log_total


def test_empty_sequences():
    sequences = Sequences(contexts=[], log_weights=[])
    assert sequences.size == 0
    assert len(sequences.posterior) == 0
    assert len(sequences.decoded_posterior) == 0


def test_posterior_normalization():
    # Test that posterior probabilities sum to 1
    sequences = Sequences(
        contexts=[
            [b"hello", EndOfSequence()],
            [b"world", EndOfSequence()],
            [b"test", EndOfSequence()],
        ],
        log_weights=[np.log(2), np.log(5), np.log(3)],
    )
    posterior = sequences.posterior
    assert np.isclose(sum(posterior.values()), 1.0)


def test_string_representation():
    sequences = Sequences(contexts=[[b"test", EndOfSequence()]], log_weights=[0.0])
    # Test that string representation doesn't raise errors
    str(sequences)
    repr(sequences)


def test_decoded_posterior_basic_sequence():
    # Simple case with one valid UTF-8 sequence
    sequences = Sequences(contexts=[[b"hello", EndOfSequence()]], log_weights=[0.0])
    posterior = sequences.decoded_posterior
    assert len(posterior) == 1
    assert posterior["hello"] == 1.0


def test_decoded_posterior_multiple_sequences():
    # Multiple different valid sequences
    sequences = Sequences(
        contexts=[[b"hello", EndOfSequence()], [b"world", EndOfSequence()]],
        log_weights=[np.log(0.7), np.log(0.3)],
    )
    posterior = sequences.decoded_posterior
    assert len(posterior) == 2
    assert np.isclose(posterior["hello"], 0.7)
    assert np.isclose(posterior["world"], 0.3)


def test_duplicate_sequences():
    # Test that duplicate sequences have their probabilities summed
    sequences = Sequences(
        contexts=[
            [b"hello", EndOfSequence()],
            [b"hello", EndOfSequence()],
            [b"world", EndOfSequence()],
        ],
        log_weights=[np.log(4), np.log(4), np.log(2)],
    )
    posterior = sequences.decoded_posterior
    assert len(posterior) == 2
    assert np.isclose(posterior["hello"], 0.8)
    assert np.isclose(posterior["world"], 0.2)


def test_no_eos_sequences():
    # Test when no sequences end with EOS
    sequences = Sequences(
        contexts=[[b"hello"], [b"world"]],
        log_weights=[np.log(0.6), np.log(0.4)],
    )
    posterior = sequences.decoded_posterior
    assert len(posterior) == 0


def test_mixed_eos_and_non_eos():
    # Test mixture of EOS and non-EOS sequences
    sequences = Sequences(
        contexts=[
            [b"hello", EndOfSequence()],
            [b"world"],  # No EOS
            [b"test", EndOfSequence()],
        ],
        log_weights=[np.log(5), np.log(2), np.log(3)],
    )
    posterior = sequences.decoded_posterior
    assert len(posterior) == 2
    # Note: weights should be renormalized after filtering
    total_weight = 5 + 3
    assert np.isclose(posterior["hello"], 5 / total_weight)
    assert np.isclose(posterior["test"], 3 / total_weight)


def test_invalid_utf8_sequences():
    # Test handling of invalid UTF-8 sequences
    invalid_bytes = bytes([0xFF, 0xFF])  # Invalid UTF-8
    sequences = Sequences(
        contexts=[
            [b"hello", EndOfSequence()],
            [invalid_bytes, EndOfSequence()],
            [b"world", EndOfSequence()],
        ],
        log_weights=[np.log(4), np.log(2), np.log(4)],
    )
    posterior = sequences.decoded_posterior
    assert len(posterior) == 2
    total_weight = 4 + 4
    assert np.isclose(posterior["hello"], 4 / total_weight)
    assert np.isclose(posterior["world"], 4 / total_weight)


def test_empty_sequence_with_eos():
    # Test sequence that's just EOS
    sequences = Sequences(contexts=[[EndOfSequence()]], log_weights=[0.0])
    posterior = sequences.decoded_posterior
    assert len(posterior) == 1
    assert posterior[""] == 1.0


def test_multi_byte_utf8():
    # Test with multi-byte UTF-8 characters
    sequences = Sequences(
        contexts=[
            ["🌟".encode("utf-8"), EndOfSequence()],
            ["こんにちは".encode("utf-8"), EndOfSequence()],
        ],
        log_weights=[np.log(3), np.log(7)],
    )
    posterior = sequences.decoded_posterior
    assert len(posterior) == 2
    assert np.isclose(posterior["🌟"], 0.3)
    assert np.isclose(posterior["こんにちは"], 0.7)


def test_all_negative_infinity_weights():
    # Test handling of case where all weights are -inf
    sequences = Sequences(
        contexts=[[b"hello", EndOfSequence()], [b"world", EndOfSequence()]],
        log_weights=[-np.inf, -np.inf],
    )

    # Check all the derived quantities
    assert sequences.log_total == float("-inf")
    assert sequences.log_ml == float("-inf")
    assert np.all(np.isneginf(sequences.log_normalized_weights))
    assert sequences.log_ess == float("-inf")
    assert sequences.ess == 0.0

    # Check that posterior methods handle this case
    assert len(sequences.posterior) == 2
    assert len(sequences.decoded_posterior) == 2


def test_shows():
    sequences = Sequences(
        contexts=[[b"a", b"b", b"c", EOS], [b"a", b"b", b"d"]],
        log_weights=[np.log(1), np.log(9)],
    )
    sequences.show()
    repr(sequences)
    sequences._repr_html_()
    str(sequences)



================================================
FILE: tests/sampler/test_set_sampler.py
================================================
import pytest
import numpy as np

from genlm.control.sampler import EagerSetSampler, TopKSetSampler
from genlm.control.sampler.set import TrieSetSampler
from conftest import iter_item_params, MockPotential, trace_swor_set

from hypothesis import given, strategies as st, settings


@pytest.mark.asyncio
@settings(deadline=None)
@given(iter_item_params())
async def test_eager_set_sampler(params):
    iter_vocab, iter_next_token_ws, item_vocab, item_next_token_ws, context = params

    mock_iter = MockPotential(iter_vocab, np.log(iter_next_token_ws))
    mock_item = MockPotential(item_vocab, np.log(item_next_token_ws))

    eager_set_sampler = EagerSetSampler(
        iter_potential=mock_iter,
        item_potential=mock_item,
    )

    try:
        have = await trace_swor_set(eager_set_sampler, context)
        want = await eager_set_sampler.target.logw_next(context)
        have.assert_equal(want, atol=1e-5, rtol=1e-5)
    finally:
        await eager_set_sampler.cleanup()


@pytest.mark.asyncio
@settings(deadline=None)
@given(iter_item_params(max_item_w=1), st.integers(1, 30))
async def test_topk_set_sampler(params, K):
    iter_vocab, iter_next_token_ws, item_vocab, item_next_token_ws, context = params

    mock_iter = MockPotential(iter_vocab, np.log(iter_next_token_ws))
    mock_item = MockPotential(item_vocab, np.log(item_next_token_ws))

    topk_set_sampler = TopKSetSampler(
        iter_potential=mock_iter, item_potential=mock_item, K=K
    )

    try:
        have = await trace_swor_set(topk_set_sampler, context)
        want = await topk_set_sampler.target.logw_next(context)
        have.assert_equal(want, atol=1e-5, rtol=1e-5)
    finally:
        await topk_set_sampler.cleanup()


def test_topk_set_sampler_K_zero():
    p1 = MockPotential(["ab"], [0, 0])
    p2 = MockPotential(["a", "b"], [0, 0, 0])
    with pytest.raises(ValueError):
        TopKSetSampler(iter_potential=p1, item_potential=p2, K=0)


def test_iter_item_error():
    p1 = MockPotential([0], [0, 0])
    p2 = MockPotential(["a", "b"], [0, 0, 0])
    with pytest.raises(
        ValueError,
        match="Token type of `iter_potential` must be an iterable of token type of `item_potential`.*",
    ):
        TrieSetSampler(iter_potential=p1, item_potential=p2)



================================================
FILE: tests/sampler/test_token_sampler.py
================================================
import pytest
import tempfile
import numpy as np

from genlm.control.sampler import DirectTokenSampler, SetTokenSampler, EagerSetSampler
from conftest import (
    mock_params,
    iter_item_params,
    MockPotential,
    trace_swor,
    mock_vocab,
)

from hypothesis import given, settings, strategies as st


@pytest.mark.asyncio
@settings(deadline=None)
@given(mock_params())
async def test_direct_token_sampler(params):
    vocab, next_token_ws, context = params
    mock_potential = MockPotential(vocab, np.log(next_token_ws))
    sampler = DirectTokenSampler(mock_potential)

    try:
        have = await trace_swor(sampler, context)
        want = await sampler.target.logw_next(context)
        have.assert_equal(want, atol=1e-5, rtol=1e-5)
    finally:
        await sampler.cleanup()


@pytest.mark.asyncio
@settings(deadline=None)
@given(iter_item_params())
async def test_set_token_sampler(params):
    iter_vocab, iter_next_token_ws, item_vocab, item_next_token_ws, context = params

    mock_iter = MockPotential(iter_vocab, np.log(iter_next_token_ws))
    mock_item = MockPotential(item_vocab, np.log(item_next_token_ws))

    sampler = SetTokenSampler(
        set_sampler=EagerSetSampler(
            iter_potential=mock_iter,
            item_potential=mock_item,
        )
    )

    try:
        have = await trace_swor(sampler, context)
        want = await sampler.target.logw_next(context)
        have.assert_equal(want, atol=1e-5, rtol=1e-5)
    finally:
        await sampler.cleanup()


@st.composite
def mock_vocab_and_logws(draw, max_w=1e3):
    vocab = draw(mock_vocab())
    ws = draw(
        st.lists(
            st.floats(1e-5, max_w),
            min_size=len(vocab) + 1,
            max_size=len(vocab) + 1,
        )
    )
    ws2 = draw(
        st.lists(
            st.floats(1e-5, max_w),
            min_size=len(vocab) + 1,
            max_size=len(vocab) + 1,
        )
    )
    logws = [np.log(w) if w > 0 else -np.inf for w in ws]
    logws2 = [np.log(w) if w > 0 else -np.inf for w in ws2]
    return vocab, logws, logws2


@pytest.mark.asyncio
@settings(deadline=None)
@given(mock_vocab_and_logws())
async def test_smc_token_sampler(params):
    vocab, logws, logws_critic = params
    mock_potential = MockPotential(vocab, logws)
    sequences = await DirectTokenSampler(mock_potential).smc(
        n_particles=10,
        ess_threshold=0.5,
        max_tokens=10,
    )
    assert len(sequences) == 10
    assert all(len(seq) <= 10 for seq in sequences)

    mock_critic = MockPotential(vocab, logws_critic)
    sequences = await DirectTokenSampler(mock_potential).smc(
        n_particles=10,
        ess_threshold=0.5,
        max_tokens=10,
        critic=mock_critic,
    )
    assert len(sequences) == 10
    assert all(len(seq) <= 10 for seq in sequences)

    with tempfile.NamedTemporaryFile() as tmp:
        sequences = await DirectTokenSampler(mock_potential).smc(
            n_particles=10,
            ess_threshold=0.5,
            max_tokens=10,
            json_path=tmp.name,
        )
        assert len(sequences) == 10
        assert all(len(seq) <= 10 for seq in sequences)

        sequences = await DirectTokenSampler(mock_potential).smc(
            n_particles=10,
            ess_threshold=0.5,
            max_tokens=10,
            critic=mock_critic,
            json_path=tmp.name,
        )
        assert len(sequences) == 10
        assert all(len(seq) <= 10 for seq in sequences)