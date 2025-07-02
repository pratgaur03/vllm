#!/bin/bash

set -euo pipefail
IFS=$'\n\t'
kill_gpu_processes() {
  # kill all processes on GPU.
  pgrep pt_main_thread | xargs -r kill -9
  pgrep python3 | xargs -r kill -9
  for port in 8000 8100 8200; do lsof -t -i:$port | xargs -r kill -9; done
  sleep 1
}

wait_for_server() {
  # wait for vllm server to start
  # return 1 if vllm server crashes
  local port=$1
  timeout 1200 bash -c "
    until curl -s localhost:${port}/v1/completions > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}


launch_chunked_prefill() {
  model="meta-llama/Meta-Llama-3.1-8B-Instruct"
  # disagg prefill
  CUDA_VISIBLE_DEVICES=0 python3 \
    -m vllm.entrypoints.openai.api_server \
    --model $model \
    --port 8100 \
    --max-model-len 10000 \
    --enable-chunked-prefill \
    --gpu-memory-utilization 0.6 &
  CUDA_VISIBLE_DEVICES=1 python3 \
    -m vllm.entrypoints.openai.api_server \
    --model $model \
    --port 8200 \
    --max-model-len 10000 \
    --enable-chunked-prefill \
    --gpu-memory-utilization 0.6 &
  wait_for_server 8100
  wait_for_server 8200
  python3 round_robin_proxy.py &
  sleep 1
}


launch_vllm_prefill() {
  model="/workspace/llama_3.1_70B"
  # normal vllm prefill
  HIP_VISIBLE_DEVICES=6 vllm serve $model --no-enable-chunked-prefill --port 8100 --quantization fp8 --dtype float16  --max-model-len 10000 --gpu-memory-utilization 0.6  &

  HIP_VISIBLE_DEVICES=7 vllm serve $model --no-enable-chunked-prefill --port 8200 --quantization fp8 --dtype float16 --max-model-len 10000 --gpu-memory-utilization 0.6  &
  wait_for_server 8100
  wait_for_server 8200
  python3 disagg_benchmarks/round_robin_proxy.py &
  sleep 1
}


launch_disagg_prefill() {
  model="meta-llama/Meta-Llama-3.1-8B-Instruct" 
  # disagg prefill
  CUDA_VISIBLE_DEVICES=0 python3 \
    -m vllm.entrypoints.openai.api_server \
    --model $model \
    --port 8100 \
    --max-model-len 10000 \
    --gpu-memory-utilization 0.6 \
    --kv-transfer-config \
    '{"kv_connector":"PyNcclConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2,"kv_buffer_size":5e9}' &

  CUDA_VISIBLE_DEVICES=1 python3 \
    -m vllm.entrypoints.openai.api_server \
    --model $model \
    --port 8200 \
    --max-model-len 10000 \
    --gpu-memory-utilization 0.6 \
    --kv-transfer-config \
    '{"kv_connector":"PyNcclConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2,"kv_buffer_size":5e9}' &

  wait_for_server 8100
  wait_for_server 8200
  python3 disagg_prefill_proxy_server.py &
  sleep 1
}


benchmark() {
  results_folder="./results"
  model="/workspace/llama_3.1_70B"
  dataset_name="sonnet"
  dataset_path="sonnet_4x.txt"
  output_len=256
  prefix_len=0
  qps=$1
  input_len=$2
  tag=$3
  num_prompts=1024

  python3 benchmark_serving.py \
          --backend vllm \
          --model $model \
          --dataset-name $dataset_name \
          --dataset-path $dataset_path \
          --sonnet-input-len $input_len \
          --sonnet-output-len $output_len \
          --sonnet-prefix-len $prefix_len \
          --num-prompts $num_prompts \
          --port 8000 \
          --save-result \
          --result-dir $results_folder \
          --result-filename "$tag"-qps-"$qps".json \
          --request-rate "$qps"

  sleep 2
}