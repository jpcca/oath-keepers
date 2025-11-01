#!/bin/bash

# install system dependencies
sudo apt-get install -y --no-install-recommends libnuma-dev ffmpeg

# build vllm from source for CPU backend (https://docs.vllm.ai/en/latest/getting_started/installation/cpu.html#intelamd-x86_2)
git clone https://github.com/vllm-project/vllm.git vllm_source
cd vllm_source

uv pip install -r requirements/cpu-build.txt --torch-backend cpu --index-strategy unsafe-best-match
uv pip install -r requirements/cpu.txt --torch-backend cpu --index-strategy unsafe-best-match

VLLM_TARGET_DEVICE=cpu python setup.py install
cd ..

# clean up
sudo rm -rf vllm_source

# install whisperlivekit
uv pip install --index-strategy unsafe-best-match --no-cache-dir whisperlivekit