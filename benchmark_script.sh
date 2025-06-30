#!/bin/bash

# Grid of configs
input_lengths=(32 64 128 256 512 1024 2048 4096 8192 16384 32768)
output_len=32
batch_sizes=(1 8 16 32 64 128 256 512 1024)
tensor_parallel_factors=(8)

# Model & paths
model_name="/workspace/llama_3.1_70B"
benchmark_script="benchmarks/benchmark_latency_prefill_decode.py"
output_dir="benchmark_results/latency"


# Run combinations
for input_len in "${input_lengths[@]}"; do
  for batch_size in "${batch_sizes[@]}"; do
    for tp_size in "${tensor_parallel_factors[@]}"; do
      output_file="${output_dir}/results_in${input_len}_out${output_len}_bs${batch_size}_tp${tp_size}.json"

      echo "Running: input_len=$input_len, batch_size=$batch_size, tp_size=$tp_size"

      python3 "$benchmark_script" \
        --model "$model_name" \
        --quantization fp8  \
        --kv-cache-dtype fp8 \
        --dtype float16 \
        --gpu-memory-utilization 0.9 \
        --input-len "$input_len" \
        --output-len "$output_len" \
        --batch-size "$batch_size" \
	--num-iters 1	\
	--num-iters-warmup 1	\
	--tensor-parallel-size "$tp_size" \
        --output-json "$output_file" \
        --disable-detokenize
    done
  done
done
