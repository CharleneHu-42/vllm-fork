#! /bin/bash

model_path=/dev/shm/DeepSeek-R1-BF16-w8afp8-static-no-ste-G2
host=127.0.0.1
port=8688
input=2048
output=2048
ratio=0.8
num_prompts=96
request_rate=inf
max_concurrency=32

model_name=$( echo $model_path | awk -F/ '{print $NF}' )
echo "Benchmarking ${model_path} for vllm server '${host}:${port}' with input_max=${input}, output_max=${output}, ratio=${ratio}, num_prompts=${num_prompts}, request_rate=${request_rate}, max_concurrency=${max_concurrency}"

log_name=benchmark_serving_${model_name}_random_in-${input}_out-${output}_ratio-${ratio}_rate-${request_rate}_prompts-${num_prompts}_${max_concurrency}_$(date +%F-%H-%M-%S)

python ../benchmarks/benchmark_serving.py \
    --backend vllm \
    --model $model_path \
    --trust-remote-code \
    --host $host \
    --port $port \
    --dataset-name random \
    --random-input-len $input \
    --random-output-len $output \
    --random-range-ratio $ratio \
    --num-prompts $num_prompts \
    --request-rate $request_rate \
    --seed 0 \
    --save-result \
    --max-concurrency $max_concurrency \
    --result-filename "${log_name}".json \
    --ignore-eos \
    |& tee "${log_name}".log 2>&1
