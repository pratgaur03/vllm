#!/bin/bash

# Model: amd/Meta-Llama-3.1-70B-Instruct-FP8-KV
export HIP_VISIBLE_DEVICES=6,7
set -euo pipefail
IFS=$'\n\t'
source "$(dirname "$0")/online_utils.sh"



main() {
  # cd "$(dirname "$0")"

  cd benchmarks
  # create sonnet-4x.txt so that we can sample 2048 tokens for input
  echo "" > sonnet_4x.txt
  for _ in {1..4}
  do
    cat sonnet.txt >> sonnet_4x.txt
  done

  rm -rf results
  mkdir results

  default_output_len=256

  export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')
  echo "Launching vllm Processes"

  launch_vllm_prefill
  for input_len in 32 ; do
    for qps in 2; do
      prefix="online_serving_results/profiling_data/in${input_len}_qps${qps}"
      mkdir -p "$prefix"

      python ../miscope/miscope.py \
        --gpus=0,1,2,3,4,5,6,7 \
        --prefix="$prefix" \
        --redirect=f"online_serving_results/stdout_log_${input_len}_qps${qps}" \
        --cmd="bash -lc 'benchmark ${qps} ${input_len} vllm'"
    done
  done

  kill_gpu_processes

  # launch_disagg_prefill
  # for qps in 2 4 6 8; do
  # benchmark $qps $default_output_len disagg_prefill
  # done
  # kill_gpu_processes

  # python3 visualize_benchmark_results.py

}


main "$@"