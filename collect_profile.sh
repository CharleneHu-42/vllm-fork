#! /bin/bash
set -ex


model_path=/root/.cache/huggingface/DeepSeek-R1-BF16-w8afp8-static-no-ste-G2
cache_path=$model_path/.hpu_cache

# set to 0 to improve the available memory 
# export VLLM_MLA_DISABLE_REQUANTIZATION=0

export VLLM_DEVICE_PROFILER_ENABLED=false
export VLLM_DEVICE_PROFILER_WARMUP_STEPS=15
export VLLM_DEVICE_PROFILER_STEPS=3
export VLLM_DEVICE_PROFILER_REPEAT=1
# export HABANA_PROFILE='profile_api_with_nics'

bash scripts/benchmark_throughput.sh \
    -w $model_path \
    -s \
    -f \
    -i 1024 \
    -o 128 \
    -t 2048 \
    -l 2048 \
    -b 32 \
    -p 64 \
    -n 8 \
    -g 1 \
    -c $cache_path