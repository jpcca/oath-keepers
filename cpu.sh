#!/bin/bash

# install system dependencies
sudo apt-get install -y --no-install-recommends gcc-12 g++-12 libnuma-dev python3-dev ffmpeg
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 10 --slave /usr/bin/g++ g++ /usr/bin/g++-12

# build vllm from source for CPU backend (https://docs.vllm.ai/en/latest/getting_started/installation/cpu.html#intelamd-x86_2)
git clone https://github.com/vllm-project/vllm.git vllm_source
cd vllm_source

uv pip install -r requirements/cpu-build.txt --torch-backend cpu --index-strategy unsafe-best-match
uv pip install -r requirements/cpu.txt --torch-backend cpu --index-strategy unsafe-best-match

VLLM_TARGET_DEVICE=cpu uv pip install . --no-build-isolation
cd ..

# clean up
rm -rf vllm_source

# install whisperlivekit
uv pip install --no-cache-dir whisperlivekit