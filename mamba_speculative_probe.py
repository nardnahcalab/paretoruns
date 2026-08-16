#!/usr/bin/env python3
"""
D4: Speculative Decoding Comparison for Nemotron 3.5 Lightning

Tests whether DSpark speculative decoding improves throughput at different
concurrency levels. Compares: speculative ON (3 tokens) vs OFF.

Usage:
    python3 mamba_speculative_probe.py [--url URL] [--port PORT] [--output OUTPUT]

Requires: pip install openai
"""

import argparse
import json
import os
import time
import sys

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: pip install openai")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CONCURRENCY_LEVELS = [1, 2, 4, 8, 16, 32]
REQUESTS_PER_LEVEL = 20
OUTPUT_TOKENS = 128
INPUT_PROMPT = (
    "You are a helpful AI assistant. Answer the following question concisely.\n\n"
    "Question: Explain the difference between Mamba and Transformer architectures "
    "in 3-4 sentences, focusing on computational complexity and sequence processing."
)
TIMEOUT = 120  # seconds per request


def run_single_request(client: OpenAI, model: str, prompt: str,
                       output_tokens: int) -> dict:
    """Send a single request and return timing metrics."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    t_start = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=output_tokens,
            temperature=0.0,
            stream=False
        )
        t_end = time.perf_counter()

        choice = response.choices[0]
        usage = response.usage

        # Handle reasoning model: content may be in reasoning field
        response_text = ""
        if choice.message and choice.message.content:
            response_text = choice.message.content[:200]
        elif choice.message and hasattr(choice.message, 'reasoning') and choice.message.reasoning:
            response_text = choice.message.reasoning[:200]

        return {
            "success": True,
            "latency_s": t_end - t_start,
            "input_tokens": usage.prompt_tokens if usage else 0,
            "output_tokens": usage.completion_tokens if usage else 0,
            "finish_reason": choice.finish_reason,
            "response_text": response_text
        }
    except Exception as e:
        t_end = time.perf_counter()
        return {
            "success": False,
            "error": str(e),
            "latency_s": t_end - t_start
        }


def run_concurrent_batch(client: OpenAI, model: str, concurrency: int,
                         num_requests: int, output_tokens: int) -> dict:
    """Run a batch of requests with simulated concurrency."""
    import concurrent.futures as cf

    results = []
    t_batch_start = time.perf_counter()

    with cf.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = []
        for i in range(num_requests):
            future = executor.submit(
                run_single_request, client, model, INPUT_PROMPT, output_tokens
            )
            futures.append(future)

        for future in cf.as_completed(futures, timeout=TIMEOUT * 2):
            try:
                result = future.result(timeout=TIMEOUT)
                results.append(result)
            except Exception as e:
                results.append({
                    "success": False,
                    "error": str(e),
                    "latency_s": 0
                })

    t_batch_end = time.perf_counter()
    batch_duration = t_batch_end - t_batch_start

    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    if successful:
        latencies = [r["latency_s"] for r in successful]
        output_tokens_list = [r["output_tokens"] for r in successful]
        total_output = sum(output_tokens_list)

        avg_latency = sum(latencies) / len(latencies)
        p50_latency = sorted(latencies)[len(latencies) // 2]
        p99_idx = min(int(len(latencies) * 0.99), len(latencies) - 1)
        p99_latency = sorted(latencies)[p99_idx]

        throughput_req = len(successful) / batch_duration
        throughput_tok = total_output / batch_duration

        return {
            "concurrency": concurrency,
            "num_requests": num_requests,
            "successful": len(successful),
            "failed": len(failed),
            "batch_duration_s": batch_duration,
            "avg_latency_s": avg_latency,
            "p50_latency_s": p50_latency,
            "p99_latency_s": p99_latency,
            "throughput_req_per_s": throughput_req,
            "throughput_tok_per_s": throughput_tok,
            "total_output_tokens": total_output
        }
    else:
        return {
            "concurrency": concurrency,
            "num_requests": num_requests,
            "successful": 0,
            "failed": len(failed),
            "batch_duration_s": batch_duration,
            "errors": [r.get("error", "unknown")[:100] for r in failed[:5]]
        }


def main():
    parser = argparse.ArgumentParser(description="D4: Speculative Decoding Comparison")
    parser.add_argument("--url", default="http://localhost", help="vLLM server URL")
    parser.add_argument("--port", type=int, default=80, help="vLLM server port")
    parser.add_argument("--output", default=None, help="Output JSON file path")
    parser.add_argument("--model", default="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4")
    parser.add_argument("--output-tokens", type=int, default=OUTPUT_TOKENS)
    parser.add_argument("--requests", type=int, default=REQUESTS_PER_LEVEL)
    parser.add_argument("--mode", choices=["speculative", "baseline", "both"], default="both",
                        help="Run speculative, baseline, or both")
    args = parser.parse_args()

    base_url = f"{args.url}:{args.port}/v1"
    print(f"Connecting to: {base_url}")
    print(f"Model: {args.model}")
    print(f"Concurrency levels: {CONCURRENCY_LEVELS}")
    print(f"Requests per level: {args.requests}")
    print(f"Output tokens: {args.output_tokens}")
    print(f"Mode: {args.mode}")
    print()

    client = OpenAI(base_url=base_url, api_key="dummy")

    # Health check
    try:
        models = client.models.list()
        print(f"Server OK. Models: {[m.id for m in models.data]}")
    except Exception as e:
        print(f"ERROR: Cannot connect: {e}")
        sys.exit(1)
    print()

    results = {
        "config": {
            "model": args.model,
            "url": base_url,
            "concurrency_levels": CONCURRENCY_LEVELS,
            "requests_per_level": args.requests,
            "output_tokens": args.output_tokens,
            "input_prompt": INPUT_PROMPT[:100] + "...",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")
        },
        "runs": []
    }

    # Determine label from server config
    label = args.mode
    if args.mode == "both":
        print("NOTE: Run this script twice — once with speculative ON, once OFF.")
        print("      The script auto-detects which mode based on --mode flag.\n")

    for concurrency in CONCURRENCY_LEVELS:
        print(f"--- Concurrency: {concurrency}, Mode: {label} ---")
        result = run_concurrent_batch(
            client, args.model, concurrency, args.requests, args.output_tokens
        )
        result["mode"] = label

        if result.get("successful", 0) > 0:
            print(f"  Throughput: {result['throughput_req_per_s']:.2f} req/s, "
                  f"{result['throughput_tok_per_s']:.0f} tok/s")
            print(f"  Latency: avg={result['avg_latency_s']:.2f}s, "
                  f"p50={result['p50_latency_s']:.2f}s, "
                  f"p99={result['p99_latency_s']:.2f}s")
            print(f"  Success: {result['successful']}/{result['num_requests']}")
        else:
            print(f"  FAILED: {result.get('errors', ['unknown'])[:2]}")

        results["runs"].append(result)
        print()

    # Save
    output_path = args.output or os.path.join(
        os.path.dirname(__file__), "..", "results", "final", "nemotron",
        f"mamba_d4_speculative_{label}.json"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_path}")

    # Summary
    print()
    print("=" * 80)
    print(f"SUMMARY: Speculative Decoding = {label}")
    print("=" * 80)
    print(f"{'Conc':>5} {'req/s':>8} {'tok/s':>8} {'Lat avg':>8} {'Lat p50':>8} {'Lat p99':>8} {'OK':>5}")
    print("-" * 80)
    for run in results["runs"]:
        if run.get("successful", 0) > 0:
            print(f"{run['concurrency']:>5} "
                  f"{run['throughput_req_per_s']:>8.2f} "
                  f"{run['throughput_tok_per_s']:>8.0f} "
                  f"{run['avg_latency_s']:>8.2f} "
                  f"{run['p50_latency_s']:>8.2f} "
                  f"{run['p99_latency_s']:>8.2f} "
                  f"{run['successful']:>5}")
        else:
            print(f"{run['concurrency']:>5} {'FAILED':>8}")


if __name__ == "__main__":
    main()
