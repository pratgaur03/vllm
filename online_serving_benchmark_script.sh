#!/bin/bash

# Model: amd/Meta-Llama-3.1-70B-Instruct-FP8-KV
# export HIP_VISIBLE_DEVICES=1,2
# export VLLM_USE_V1=1
export MISCOPE_ROOT="/var/lib/jenkins/vllm/miscope"

set -ex

source "$(dirname "$0")/online_utils.sh"
export -f benchmark


main() {
  (which wget && which curl) || (apt-get update && apt-get install -y wget curl)
  (which jq) || (apt-get -y install jq)
  (which socat) || (apt-get -y install socat)
  (which lsof) || (apt-get -y install lsof)

  pip install quart httpx matplotlib aiohttp datasets
  # cd "$(dirname "$0")"

  cd benchmarks
  # create sonnet-4x.txt so that we can sample 2048 tokens for input
  echo "" > sonnet_4x.txt
  for _ in {1..4}
  do
    cat sonnet.txt >> sonnet_4x.txt
  done

  mkdir -p results

  default_output_len=256

  export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')
  echo "Launching vllm Processes"


  # launch_vllm_prefill

  # for qps in 5 10 20 30 40; do
  #   for input_len in 8192; do
  #     prefix="online_serving_results/profiling_data/in${input_len}_qps${qps}_vllm_prefill_tp8"
  #     # mkdir -p "$prefix"

  #     python ../miscope/miscope.py \
  #       --gpus=0,1,2,3,4,5,6,7 \
  #       --prefix="$prefix" \
  #       --redirect="online_serving_results/stdout_log_${input_len}_qps${qps}_vllm_prefill_tp8" \
  #       --cmd="bash -lc 'source /var/lib/jenkins/vllm/online_utils.sh; benchmark ${qps} ${input_len} vllm_prefill 8'"

  #   done
  # done
  # kill_gpu_processes
 
  
  # launch_chunked_prefill
  # export VLLM_USE_V1=1
  # for qps in 5 10 20 30 40; do # gets stuck after this
  #   for input_len in 8192; do
  #     prefix="online_serving_results/profiling_data/in${input_len}_qps${qps}_chunked_prefill_tp8"
  #     mkdir -p "$prefix"

  #     python ../miscope/miscope.py \
  #       --gpus=0,1,2,3,4,5,6,7 \
  #       --prefix="$prefix" \
  #       --redirect="online_serving_results/stdout_log_${input_len}_qps${qps}_chunked_prefill_tp8" \
  #       --cmd="bash -lc 'source /var/lib/jenkins/vllm/online_utils.sh; benchmark ${qps} ${input_len} chunked_prefill 8'"

  #   done
  # done
  # kill_gpu_processes
  for input_len in 32; do
    for qps in 1; do
      launch_disagg_prefill
    
      
      prefix="online_serving_results/profiling_data/in${input_len}_qps${qps}_disagg_prefill_tp2_testing"
      python ../miscope/miscope.py \
        --gpus=0,1 \
        --prefix="$prefix" \
        --redirect="online_serving_results/stdout_log_${input_len}_qps${qps}_disagg_prefill_tp2_testing" \
        --cmd="bash -lc 'source /var/lib/jenkins/vllm/online_utils.sh; benchmark ${qps} ${input_len} disagg_prefill 2'"
      kill_gpu_processes
      sleep 2
    done
  done
  
  
  # # for qps in 2 4 6 8; do
  # # benchmark $qps $default_output_len disagg_prefill
  # # done
  

  

  # python3 disagg_benchmarks/visualize_benchmark_results.py

}


main "$@"