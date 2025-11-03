### Oath keeping Agents

[![Build Status](https://github.com/jpcca/oath-keepers/workflows/CI/badge.svg)](https://github.com/jpcca/oath-keepers/actions?query=workflow%3ACI)

This project aims to improve patient outcomes by using multi-agent systems to ellicit detailed information from patients about their symptoms as they happening. Using probabilistic computing we estimate the probability of deviant behaviours such as breaking the Hippocratic oath or diagnosing the patient.

![Workflow](docs/workflow.png)

### Quick start

This project uses the uv package manager (https://github.com/astral-sh/uv)

```bash
uv pip install -e .[gpu,dev]
```

For cpu-only builds, which are used for GitHub actions

```bash
uv run bash cpu.sh
uv pip install -e .[cpu,dev]
```

### Running the vLLM Server with Audio Transcription

The vLLM server includes real-time audio transcription capabilities that automatically prompt the LLM with spoken input:

```bash
python oath_keepers/vllm_server.py
```

The server will:
- Start on port 8000 with a web interface for microphone input
- Transcribe audio in real-time using Whisper
- Buffer transcriptions until complete sentences are detected (ending with `.`, `!`, or `?`)
- Automatically prompt the LLM with complete sentences
- Print LLM responses to the terminal

Access the web interface at `http://localhost:8000` to start speaking into your microphone.

### Advanced caching

Changes in the following files rebuilds the cache used for running unit tests in GitHub actions.

```
uv.lock
pyproject.toml
.python-version
uv.lock
cpu.sh
```

Check `.github/workflows/ci.yml` to see implementation details.

### Funding Sources

This work was supported by the Cross-ministerial Strategic Innovation Promotion Program (SIP) on “Integrated Health Care System” Grant Number JPJ012425.
