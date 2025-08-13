#!/bin/bash

# install system dependencies
sudo apt-get install -y --no-install-recommends libnuma-dev

# build vllm from source for CPU backend (https://docs.vllm.ai/en/latest/getting_started/installation/cpu.html#intelamd-x86_2)
git clone https://github.com/vllm-project/vllm.git vllm_source
cd vllm_source

uv pip install -r requirements/cpu-build.txt --torch-backend auto --index-url https://download.pytorch.org/whl/cpu
uv pip install -r requirements/cpu.txt --torch-backend auto --index-url https://download.pytorch.org/whl/cpu

VLLM_TARGET_DEVICE=cpu python setup.py install
cd ..