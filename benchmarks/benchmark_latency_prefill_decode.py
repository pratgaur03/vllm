# SPDX-License-Identifier: Apache-2.0
"""Benchmark the latency of processing a single batch of requests."""

import argparse
import dataclasses
import json
import os
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from benchmark_utils import convert_to_pytorch_benchmark_format, write_to_json
from tqdm import tqdm

from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.inputs import PromptType
from vllm.sampling_params import BeamSearchParams
from vllm.utils import FlexibleArgumentParser


def save_to_pytorch_benchmark_format(args: argparse.Namespace,
                                     results: dict[str, Any]) -> None:
    pt_records = convert_to_pytorch_benchmark_format(
        args=args,
        metrics={"latency": results["latencies"]},
        extra_info={k: results[k]
                    for k in ["avg_latency", "percentiles"]})
    if pt_records:
        pt_file = f"{os.path.splitext(args.output_json)[0]}.pytorch.json"
        write_to_json(pt_file, pt_records)


def main(args: argparse.Namespace):
    print(args)

    engine_args = EngineArgs.from_cli_args(args)

    # NOTE(woosuk): If the request cannot be processed in a single batch,
    # the engine will automatically process the request in multiple batches.
    llm = LLM(**dataclasses.asdict(engine_args))
    assert llm.llm_engine.model_config.max_model_len >= (
        args.input_len +
        args.output_len), ("Please ensure that max_model_len is greater than"
                           " the sum of input_len and output_len.")

    sampling_params = SamplingParams(
        n=args.n,
        temperature=1.0,
        top_p=1.0,
        ignore_eos=True,
        max_tokens=args.output_len,
        detokenize=not args.disable_detokenize,
    )
    print(sampling_params)
    dummy_prompt_token_ids = np.random.randint(10000,
                                               size=(args.batch_size,
                                                     args.input_len))
    dummy_prompts: list[PromptType] = [{
        "prompt_token_ids": batch
    } for batch in dummy_prompt_token_ids.tolist()]

    def llm_generate():
        if not args.use_beam_search:
            outputs = llm.generate(dummy_prompts, sampling_params=sampling_params, use_tqdm=False)
        else:
            outputs = llm.beam_search(
                dummy_prompts,
                BeamSearchParams(
                    beam_width=args.n,
                    max_tokens=args.output_len,
                    ignore_eos=True,
                ),
            )

        all_prefill_latencies = []
        all_decode_latencies = []
        all_decode_per_iter_latencies = []

        for output in outputs:
            m = output.metrics

            prefill_latency = m.first_token_time - m.arrival_time
            decode_latency = m.last_token_time  - m.first_token_time
            decode_iters = max(len(output.outputs[0].token_ids) - 1, 1)

            avg_decode_per_iter = decode_latency / decode_iters

            all_prefill_latencies.append(prefill_latency)
            all_decode_latencies.append(decode_latency)
            all_decode_per_iter_latencies.append(avg_decode_per_iter)

        return all_prefill_latencies, all_decode_latencies, all_decode_per_iter_latencies

    def run_to_completion(profile_dir: Optional[str] = None):
        if profile_dir:
            with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    on_trace_ready=torch.profiler.tensorboard_trace_handler(
                        str(profile_dir)),
            ) as p:
                llm_generate()
            print(p.key_averages().table(sort_by="self_cuda_time_total"))
        else:
            start_time = time.perf_counter()
            prefill_latencies, decode_latencies, decode_per_iter_latencies = llm_generate()
            end_time = time.perf_counter()
            total_latency = end_time - start_time
            return total_latency, prefill_latencies, decode_latencies, decode_per_iter_latencies


    print("Warming up...")
    for _ in tqdm(range(args.num_iters_warmup), desc="Warmup iterations"):
        run_to_completion(profile_dir=None)

    if args.profile:
        profile_dir = args.profile_result_dir
        if not profile_dir:
            profile_dir = (Path(".") / "vllm_benchmark_result" /
                           f"latency_result_{time.time()}")
        print(f"Profiling (results will be saved to '{profile_dir}')...")
        run_to_completion(profile_dir=profile_dir)
        return

    # Benchmark.
    latencies = []
    all_prefill_latencies = []
    all_decode_latencies = []
    all_decode_per_iter_latencies = []

    for _ in tqdm(range(args.num_iters), desc="Profiling iterations"):
        total_latency, prefill, decode, decode_per_iter = run_to_completion(profile_dir=None)
        latencies.append(total_latency)
        all_prefill_latencies.extend(prefill)
        all_decode_latencies.extend(decode)
        all_decode_per_iter_latencies.extend(decode_per_iter)

    latencies = np.array(latencies)
    percentages = [10, 25, 50, 75, 90, 99]
    percentiles = np.percentile(latencies, percentages)

    print(f"Avg total latency: {np.mean(latencies):.4f} seconds")
    print(f"Avg prefill latency per request: {np.mean(all_prefill_latencies):.4f} seconds")
    print(f"Avg decode latency per request: {np.mean(all_decode_latencies):.4f} seconds")
    print(f"Avg decode latency per iteration: {np.mean(all_decode_per_iter_latencies):.4f} seconds")

    for p in percentages:
        print(f"{p}% total latency: {np.percentile(latencies, p):.4f} s")
        print(f"{p}% prefill latency: {np.percentile(all_prefill_latencies, p):.4f} s")
        print(f"{p}% decode latency: {np.percentile(all_decode_latencies, p):.4f} s")
        print(f"{p}% decode/iter latency: {np.percentile(all_decode_per_iter_latencies, p):.4f} s")


    # Output JSON results if specified
    if args.output_json:
        results = {
            "avg_latency": np.mean(latencies),
            "latencies": latencies.tolist(),
            "percentiles": dict(zip(percentages, percentiles.tolist())),
            "avg_prefill_latency": np.mean(all_prefill_latencies),
            "avg_decode_latency": np.mean(all_decode_latencies),
            "avg_decode_per_iter_latency": np.mean(all_decode_per_iter_latencies),
            "prefill_latencies": all_prefill_latencies,
            "decode_latencies": all_decode_latencies,
            "decode_per_iter_latencies": all_decode_per_iter_latencies,
        }
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=4)
        save_to_pytorch_benchmark_format(args, results)



if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Benchmark the latency of processing a single batch of "
        "requests till completion.")
    parser.add_argument("--input-len", type=int, default=32)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--n",
        type=int,
        default=1,
        help="Number of generated sequences per prompt.",
    )
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument(
        "--num-iters-warmup",
        type=int,
        default=10,
        help="Number of iterations to run for warmup.",
    )
    parser.add_argument("--num-iters",
                        type=int,
                        default=30,
                        help="Number of iterations to run.")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="profile the generation process of a single batch",
    )
    parser.add_argument(
        "--profile-result-dir",
        type=str,
        default=None,
        help=("path to save the pytorch profiler output. Can be visualized "
              "with ui.perfetto.dev or Tensorboard."),
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to save the latency results in JSON format.",
    )
    parser.add_argument(
        "--disable-detokenize",
        action="store_true",
        help=("Do not detokenize responses (i.e. do not include "
              "detokenization time in the latency measurement)"),
    )

    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    main(args)
