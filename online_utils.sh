#!/bin/bash
set -ex

kill_gpu_processes() {
  # kill all processes on GPU.
  pgrep pt_main_thread | xargs -r kill -9
  pgrep python3 | xargs -r kill -9
  for port in 8080 8100 8200; do lsof -t -i:$port | xargs -r kill -9; done
  sleep 1
}

wait_for_server() {
  local port=$1

  # 3) wait for the model to appear in /v1/models
  until curl -s http://localhost:$port/v1/models | grep -q '/workspace/llama_3.1_70B'; do
    sleep 10
  done

  # tiny cushion
  sleep 1
}




# launch_chunked_prefill() {
#   model="/workspace/llama_3.1_70B"
#   # disagg prefill
#   HIP_VISIBLE_DEVICES=6 python3 -m vllm.entrypoints.openai.api_server \
#   --model /workspace/llama_3.1_70B \
#     --port 8100 \
#     --max-model-len 10000 \
#     --enable-chunked-prefill \
#     --gpu-memory-utilization 0.6 &
#   HIP_VISIBLE_DEVICES=7 python3 -m vllm.entrypoints.openai.api_server \
#   --model /workspace/llama_3.1_70B \
#     --port 8200 \
#     --max-model-len 10000 \
#     --enable-chunked-prefill \
#     --gpu-memory-utilization 0.6 &
#   wait_for_server 8080
#   wait_for_server 8100
#   wait_for_server 8200
#   python3 disagg_benchmarks/round_robin_proxy.py &
#   sleep 1
# }
launch_chunked_prefill() {
  model="/workspace/llama_3.1_70B"

  HIP_VISIBLE_DEVICES=0,1,2,3  python3 \
    -m vllm.entrypoints.openai.api_server \
    --model $model \
    --port 8100 \
    --max-model-len 10000 \
    --enable-chunked-prefill \
    --quantization fp8  \
    --tensor-parallel-size 4 \
    --dtype float16 \
    --trust-remote-code \
    --gpu-memory-utilization 0.6 &

  HIP_VISIBLE_DEVICES=4,5,6,7 python3 \
    -m vllm.entrypoints.openai.api_server \
    --model $model \
    --port 8200 \
    --max-model-len 10000 \
    --tensor-parallel-size 4 \
    --trust-remote-code \
    --enable-chunked-prefill \
    --quantization fp8  \
    --dtype float16 \
    --gpu-memory-utilization 0.6 &

  wait_for_server 8100
  wait_for_server 8200
  python3 disagg_benchmarks/round_robin_proxy.py &
  sleep 2
  wait_for_server 8080
  sleep 2

}
launch_vllm_prefill() {
  model="/workspace/llama_3.1_70B"
  # normal vllm prefill
  HIP_VISIBLE_DEVICES=0,1,2,3 vllm serve $model --enable-chunked-prefill false --port 8100 --quantization fp8 --dtype float16 --tensor-parallel-size 4 --max-model-len 10000 --gpu-memory-utilization 0.6  &

  HIP_VISIBLE_DEVICES=4,5,6,7 vllm serve $model --enable-chunked-prefill false --port 8200 --quantization fp8 --dtype float16 --tensor-parallel-size 4 --max-model-len 10000 --gpu-memory-utilization 0.6  &
  wait_for_server 8100
  wait_for_server 8200
  python3 disagg_benchmarks/round_robin_proxy.py &
  sleep 2
  wait_for_server 8080
  sleep 1
}
launch_disagg_prefill() {
  model="/workspace/llama_3.1_70B" 
  HIP_VISIBLE_DEVICES=0  python3 \
    -m vllm.entrypoints.openai.api_server \
    --model $model \
    --port 8100 \
    --max-model-len 10000 \
    --quantization fp8  \
    --dtype float16 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    --kv-transfer-config \
    '{"kv_connector":"PyNcclConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2,"kv_buffer_size":5e9}' &

  HIP_VISIBLE_DEVICES=1  python3 \
    -m vllm.entrypoints.openai.api_server \
    --model  $model \
    --port 8200 \
    --max-model-len 10000 \
    --quantization fp8  \
    --dtype float16 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    --kv-transfer-config \
    '{"kv_connector":"PyNcclConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2,"kv_buffer_size":5e9}' &

  wait_for_server 8100
  wait_for_server 8200
  python3 disagg_benchmarks/disagg_prefill_proxy_server.py &
  sleep 2
  wait_for_server 8080
  sleep 60
}


benchmark() {
  results_folder="./results"
  model="/workspace/llama_3.1_70B"
  dataset_name="random"
  dataset_path="sonnet_4x.txt"
  output_len=1
  prefix_len=0
  qps=$1
  input_len=$2
  tag=$3
  tp=$4
  num_prompts=100

  python3 benchmark_serving.py \
          --backend openai-chat --endpoint /v1/chat/completions \
          --model '/workspace/llama_3.1_70B' \
          --dataset-name $dataset_name \
          --random-input-len $input_len \
          --random-output-len $output_len \
          --num-prompts 1024 \
          --port 8080 \
          --save-result \
          --result-dir $results_folder \
          --result-filename "$tag"-qps-"$qps"-input-len-"$input_len"-tp-"$tp".json \
          --request-rate "$qps"

  sleep 2
}