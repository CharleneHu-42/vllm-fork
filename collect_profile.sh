#! /bin/bash
set -ex


model_path=/root/.cache/huggingface/DeepSeek-R1-BF16-w8afp8-static-no-ste-G2
model_path=/root/.cache/huggingface/DeepSeek-R1-BF16-w8afp8-dynamic-no-ste-G2
model_path=/root/.cache/huggingface/DeepSeek-R1-G2
model_path=/root/.cache/huggingface/Llama-3.1-70B
cache_path=$model_path/.hpu_cache

# set to 0 to improve the available memory 
export VLLM_MLA_DISABLE_REQUANTIZATION=1
export VLLM_MLA_PERFORM_MATRIX_ABSORPTION=0

# lazy mode controls
#export PT_HPU_LAZY_MODE=0
#export PT_HPU_LAZY_ACC_PAR_MODE=0
#export PT_HPU_ENABLE_LAZY_COLLECTIVES=false

export VLLM_DEVICE_PROFILER_ENABLED=false
export VLLM_DEVICE_PROFILER_WARMUP_STEPS=15
export VLLM_DEVICE_PROFILER_STEPS=3
export VLLM_DEVICE_PROFILER_REPEAT=1
# export HABANA_PROFILE='profile_api_with_nics'

export VLLM_PP_LAYER_PARTITION="32,29"

export VLLM_DELAYED_SAMPLING=false

bash scripts/benchmark_throughput.sh \
    -w $model_path \
    -s \
    -f \
    -i 1000 \
    -o 1000 \
    -t 4096 \
    -l 4096 \
    -b 32 \
    -p 64 \
    -n 4 \
    -g 2 \
    -c $cache_path