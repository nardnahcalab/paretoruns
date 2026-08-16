#!/usr/bin/env python3
"""
D1: Long-Context Throughput Probe for Nemotron 3.5 Lightning

Measures how TTFT, decode throughput, and GPU memory scale with input length.
Tests the hypothesis that Mamba layers provide constant-time sequence processing.

Usage:
    python3 mamba_long_context_probe.py [--url URL] [--port PORT] [--output OUTPUT]

Requires: pip install openai psutil
"""

import argparse
import json
import os
import time
import sys
import subprocess

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: pip install openai")
    sys.exit(1)

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Input token lengths to test (approximate — we generate text of these lengths)
# Each token is roughly 4 characters for English text
INPUT_LENGTHS = [256, 1024, 4096, 16384, 65536, 131072]
OUTPUT_TOKENS = 64  # Fixed output length for fair comparison
NUM_TRIALS = 3      # Repeat each length for statistical robustness
TIMEOUT = 300       # seconds per request

# System prompt (fixed across all tests)
SYSTEM_PROMPT = "You are a helpful AI assistant. Answer the user's question concisely."

# Base query appended after the padded context
BASE_QUERY = "\n\nBased on the above context, what is the main topic? Answer in one sentence."


def generate_padded_context(target_tokens: int) -> str:
    """
    Generate a context of approximately target_tokens tokens.
    Uses a repeating technical paragraph to simulate realistic content.
    """
    # A realistic technical paragraph (~150 tokens when tokenized)
    paragraph = (
        "The Mamba architecture introduces a selective state space model that processes "
        "sequences with constant computational complexity per token. Unlike transformers "
        "which require O(n^2) attention computations, Mamba maintains a fixed-size hidden "
        "state that is updated incrementally. This makes it particularly efficient for "
        "long sequences where transformer attention becomes the bottleneck. The key innovation "
        "is the selective scan mechanism that allows the model to focus on relevant parts of "
        "the input without explicitly computing pairwise attention scores. Each Mamba layer "
        "contains a linear projection, a convolution, and a selective scan operation, all "
        "of which can be parallelized efficiently on modern hardware. "
    )
    # ~4 chars per token, paragraph is ~600 chars ≈ 150 tokens
    tokens_per_paragraph = 150
    num_paragraphs = max(1, target_tokens // tokens_per_paragraph)

    # Generate varied content by numbering paragraphs
    paragraphs = []
    for i in range(num_paragraphs):
        paragraphs.append(f"[Section {i+1}] {paragraph}")
    return "\n\n".join(paragraphs)


def get_gpu_memory_mb() -> dict:
    """Query GPU memory via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        parts = result.stdout.strip().split(", ")
        return {
            "used_mb": float(parts[0]),
            "total_mb": float(parts[1]),
            "utilization_pct": float(parts[2])
        }
    except Exception:
        return {"used_mb": 0, "total_mb": 0, "utilization_pct": 0}


def run_probe(client: OpenAI, model: str, context: str, query: str,
              output_tokens: int, trial: int) -> dict:
    """Send a single probe request and collect metrics."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": context + query}
    ]

    mem_before = get_gpu_memory_mb()
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

        mem_after = get_gpu_memory_mb()

        choice = response.choices[0]
        usage = response.usage

        # Compute approximate input tokens from usage
        input_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0

        # Handle reasoning model: content may be in reasoning field
        response_text = ""
        if choice.message and choice.message.content:
            response_text = choice.message.content[:200]
        elif choice.message and hasattr(choice.message, 'reasoning') and choice.message.reasoning:
            response_text = choice.message.reasoning[:200]

        return {
            "trial": trial,
            "success": True,
            "input_tokens": input_tokens,
            "output_tokens": completion_tokens,
            "latency_s": t_end - t_start,
            "content_chars": len(context + query),
            "gpu_mem_before_mb": mem_before["used_mb"],
            "gpu_mem_after_mb": mem_after["used_mb"],
            "gpu_util_pct": mem_after["utilization_pct"],
            "response_text": response_text,
            "finish_reason": choice.finish_reason
        }
    except Exception as e:
        t_end = time.perf_counter()
        mem_after = get_gpu_memory_mb()
        return {
            "trial": trial,
            "success": False,
            "error": str(e),
            "latency_s": t_end - t_start,
            "gpu_mem_after_mb": mem_after["used_mb"],
            "content_chars": len(context + query)
        }


def main():
    parser = argparse.ArgumentParser(description="D1: Long-Context Throughput Probe")
    parser.add_argument("--url", default="http://localhost", help="vLLM server URL")
    parser.add_argument("--port", type=int, default=80, help="vLLM server port")
    parser.add_argument("--output", default=None, help="Output JSON file path")
    parser.add_argument("--model", default="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
                        help="Model name")
    parser.add_argument("--input-lengths", nargs="+", type=int, default=INPUT_LENGTHS,
                        help="Input token lengths to test")
    parser.add_argument("--output-tokens", type=int, default=OUTPUT_TOKENS,
                        help="Output tokens per request")
    parser.add_argument("--trials", type=int, default=NUM_TRIALS,
                        help="Number of trials per length")
    args = parser.parse_args()

    base_url = f"{args.url}:{args.port}/v1"
    print(f"Connecting to: {base_url}")
    print(f"Model: {args.model}")
    print(f"Input lengths: {args.input_lengths}")
    print(f"Output tokens: {args.output_tokens}")
    print(f"Trials per length: {args.trials}")
    print()

    client = OpenAI(base_url=base_url, api_key="dummy")

    # Health check
    try:
        models = client.models.list()
        print(f"Server OK. Available models: {[m.id for m in models.data]}")
    except Exception as e:
        print(f"ERROR: Cannot connect to server: {e}")
        sys.exit(1)
    print()

    results = {
        "config": {
            "model": args.model,
            "url": base_url,
            "input_lengths": args.input_lengths,
            "output_tokens": args.output_tokens,
            "trials": args.trials,
            "system_prompt": SYSTEM_PROMPT,
            "base_query": BASE_QUERY,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")
        },
        "probes": []
    }

    # Baseline: GPU memory with no active requests
    mem_baseline = get_gpu_memory_mb()
    results["baseline_gpu"] = mem_baseline
    print(f"Baseline GPU: {mem_baseline['used_mb']:.0f} MB / {mem_baseline['total_mb']:.0f} MB")
    print()

    for target_len in args.input_lengths:
        print(f"--- Input length: ~{target_len} tokens ---")
        context = generate_padded_context(target_len)
        query = BASE_QUERY

        trial_results = []
        for trial in range(args.trials):
            print(f"  Trial {trial+1}/{args.trials}...", end=" ", flush=True)
            result = run_probe(client, args.model, context, query,
                             args.output_tokens, trial)
            trial_results.append(result)

            if result["success"]:
                ttft_approx = result["latency_s"]  # For non-streaming, total latency
                print(f"OK: {result['input_tokens']} in tokens, "
                      f"{result['output_tokens']} out tokens, "
                      f"{result['latency_s']:.2f}s, "
                      f"GPU: {result['gpu_mem_after_mb']:.0f}MB")
            else:
                print(f"FAIL: {result['error'][:80]}")

            # Brief cooldown between trials
            time.sleep(1)

        # Aggregate
        successful = [r for r in trial_results if r["success"]]
        if successful:
            avg_latency = sum(r["latency_s"] for r in successful) / len(successful)
            avg_input = sum(r["input_tokens"] for r in successful) / len(successful)
            avg_output = sum(r["output_tokens"] for r in successful) / len(successful)
            avg_gpu = sum(r["gpu_mem_after_mb"] for r in successful) / len(successful)
            throughput = avg_output / avg_latency if avg_latency > 0 else 0

            print(f"  Summary: avg_latency={avg_latency:.2f}s, "
                  f"avg_input={avg_input:.0f}, avg_output={avg_output:.0f}, "
                  f"throughput={throughput:.1f} tok/s, avg_gpu={avg_gpu:.0f}MB")
        else:
            print(f"  Summary: ALL FAILED")

        results["probes"].append({
            "target_tokens": target_len,
            "trials": trial_results
        })
        print()

    # Save results
    output_path = args.output or os.path.join(
        os.path.dirname(__file__), "..", "results", "final", "nemotron",
        "mamba_d1_long_context.json"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_path}")

    # Print summary table
    print()
    print("=" * 80)
    print("SUMMARY: Long-Context Throughput (Mamba + MoE Hybrid)")
    print("=" * 80)
    print(f"{'Input Tokens':>12} {'Latency (s)':>12} {'Output tok/s':>13} {'GPU (MB)':>10} {'Success':>8}")
    print("-" * 80)
    for probe in results["probes"]:
        successful = [r for r in probe["trials"] if r["success"]]
        if successful:
            avg_lat = sum(r["latency_s"] for r in successful) / len(successful)
            avg_out = sum(r["output_tokens"] for r in successful) / len(successful)
            tps = avg_out / avg_lat if avg_lat > 0 else 0
            avg_gpu = sum(r["gpu_mem_after_mb"] for r in successful) / len(successful)
            print(f"{probe['target_tokens']:>12} {avg_lat:>12.2f} {tps:>13.1f} {avg_gpu:>10.0f} {len(successful):>8}/{len(probe['trials'])}")
        else:
            print(f"{probe['target_tokens']:>12} {'FAILED':>12}")


if __name__ == "__main__":
    main()
