#!/usr/bin/env python3
"""
A1+E1: KV Cache Memory Scaling + Memory Budget Breakdown

A1: Measures how GPU memory grows with input context length (1K-256K).
E1: Measures memory breakdown at different concurrency levels (c1-c64).

Together these answer: "How many 256K sessions can we serve on one GPU?"

Usage:
    python3 memory_budget_probe.py [--url URL] [--port PORT] [--output OUTPUT]
"""

import argparse
import json
import os
import subprocess
import sys
import time
import concurrent.futures as cf

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: pip install openai")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CONTEXT_LENGTHS = [256, 1024, 4096, 16384, 65536, 131072]
CONCURRENCY_LEVELS = [1, 4, 16, 32, 64]
OUTPUT_TOKENS = 64
TRIALS = 3
REQUESTS_PER_CONCURRENCY = 10
TIMEOUT = 300

SYSTEM_PROMPT = "You are a helpful AI assistant."
BASE_QUERY = "\n\nBased on the above, provide a one-sentence summary."


def get_gpu_memory(container_name="vllm-nemotron-gb10"):
    """Get GPU memory via torch.cuda inside the Docker container."""
    try:
        result = subprocess.run(
            ["docker", "exec", container_name, "python3", "-c",
             "import torch; a=torch.cuda.memory_allocated(0); r=torch.cuda.memory_reserved(0); t=torch.cuda.get_device_properties(0).total_memory; print(f'{a/1e6:.0f},{r/1e6:.0f},{t/1e6:.0f}')"],
            capture_output=True, text=True, timeout=10
        )
        parts = result.stdout.strip().split(",")
        allocated_mb = float(parts[0])
        reserved_mb = float(parts[1])
        total_mb = float(parts[2])
        return {
            "used_mb": allocated_mb,
            "reserved_mb": reserved_mb,
            "total_mb": total_mb,
            "free_mb": total_mb - reserved_mb,
            "util_pct": (reserved_mb / total_mb * 100) if total_mb > 0 else 0
        }
    except Exception as e:
        return {"used_mb": 0, "reserved_mb": 0, "total_mb": 0, "free_mb": 0, "util_pct": 0}


def get_gpu_memory_detail(container_name="vllm-nemotron-gb10"):
    """Get detailed GPU memory breakdown including temperature and power."""
    base = get_gpu_memory(container_name)
    # Also try to get power/temp from vLLM metrics if available
    try:
        result = subprocess.run(
            ["docker", "exec", container_name, "python3", "-c",
             "import torch; print(f'{torch.cuda.memory_stats()[\"active_bytes.all.current\"]/1e6:.0f}')"],
            capture_output=True, text=True, timeout=10
        )
        active_mb = float(result.stdout.strip())
        base["active_mb"] = active_mb
    except Exception:
        base["active_mb"] = base.get("used_mb", 0)
    return base


def generate_context(target_tokens):
    """Generate a context of approximately target_tokens tokens."""
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
    tokens_per_para = 150
    n_para = max(1, target_tokens // tokens_per_para)
    return "\n\n".join(f"[Section {i+1}] {paragraph}" for i in range(n_para))


def single_request(client, model, context, query, output_tokens):
    """Send a single request and measure memory."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": context + query}
    ]
    mem_before = get_gpu_memory()
    t_start = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            model=model, messages=messages, max_tokens=output_tokens,
            temperature=0.0, stream=False
        )
        t_end = time.perf_counter()
        mem_after = get_gpu_memory()
        usage = resp.usage
        choice = resp.choices[0]
        # Handle reasoning model
        text = ""
        if choice.message and choice.message.content:
            text = choice.message.content[:100]
        elif choice.message and hasattr(choice.message, 'reasoning') and choice.message.reasoning:
            text = choice.message.reasoning[:100]
        return {
            "success": True,
            "latency_s": t_end - t_start,
            "input_tokens": usage.prompt_tokens if usage else 0,
            "output_tokens": usage.completion_tokens if usage else 0,
            "mem_before_mb": mem_before["used_mb"],
            "mem_after_mb": mem_after["used_mb"],
            "mem_delta_mb": mem_after["used_mb"] - mem_before["used_mb"],
            "content_chars": len(context + query),
            "finish_reason": choice.finish_reason
        }
    except Exception as e:
        t_end = time.perf_counter()
        mem_after = get_gpu_memory()
        return {
            "success": False,
            "error": str(e)[:100],
            "latency_s": t_end - t_start,
            "mem_after_mb": mem_after["used_mb"]
        }


def concurrent_batch(client, model, concurrency, num_requests, output_tokens):
    """Run concurrent requests and measure memory."""
    results = []
    t_batch_start = time.perf_counter()
    mem_before = get_gpu_memory()

    with cf.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = []
        for _ in range(num_requests):
            ctx = generate_context(1024)  # Fixed 1K context for concurrency test
            future = executor.submit(single_request, client, model, ctx, BASE_QUERY, output_tokens)
            futures.append(future)

        for future in cf.as_completed(futures, timeout=TIMEOUT * 2):
            try:
                results.append(future.result(timeout=TIMEOUT))
            except Exception as e:
                results.append({"success": False, "error": str(e)[:100], "latency_s": 0})

    t_batch_end = time.perf_counter()
    mem_after = get_gpu_memory_detail()

    successful = [r for r in results if r.get("success")]
    if successful:
        lats = [r["latency_s"] for r in successful]
        out_toks = [r["output_tokens"] for r in successful]
        total_out = sum(out_toks)
        batch_dur = t_batch_end - t_batch_start
        return {
            "concurrency": concurrency,
            "num_requests": num_requests,
            "successful": len(successful),
            "failed": len(results) - len(successful),
            "batch_duration_s": batch_dur,
            "avg_latency_s": sum(lats) / len(lats),
            "p50_latency_s": sorted(lats)[len(lats) // 2],
            "throughput_req_s": len(successful) / batch_dur,
            "throughput_tok_s": total_out / batch_dur,
            "mem_before_batch_mb": mem_before["used_mb"],
            "mem_after_batch_mb": mem_after["used_mb"],
            "gpu_detail": mem_after
        }
    return {"concurrency": concurrency, "successful": 0, "failed": len(results)}


def main():
    parser = argparse.ArgumentParser(description="A1+E1: Memory Budget Probe")
    parser.add_argument("--url", default="http://localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--output", default=None)
    parser.add_argument("--model", default="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4")
    parser.add_argument("--container", default="vllm-nemotron-gb10")
    args = parser.parse_args()

    base_url = f"{args.url}:{args.port}/v1"
    print(f"Connecting to: {base_url}")
    print(f"Model: {args.model}")
    print(f"Container: {args.container}")
    print()

    client = OpenAI(base_url=base_url, api_key="dummy")

    # Health check
    try:
        models = client.models.list()
        print(f"Server OK. Model: {models.data[0].id}")
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # Baseline
    mem_baseline = get_gpu_memory_detail(args.container)
    print(f"Baseline GPU: {mem_baseline['used_mb']:.0f}MB / {mem_baseline['total_mb']:.0f}MB "
          f"(free: {mem_baseline.get('free_mb', 0):.0f}MB, power: {mem_baseline.get('power_w', 0):.0f}W)")
    print()

    results = {
        "config": {
            "model": args.model,
            "url": base_url,
            "container": args.container,
            "context_lengths": CONTEXT_LENGTHS,
            "concurrency_levels": CONCURRENCY_LEVELS,
            "output_tokens": OUTPUT_TOKENS,
            "trials": TRIALS,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")
        },
        "baseline_gpu": mem_baseline,
        "a1_context_scaling": [],
        "e1_concurrency_scaling": []
    }

    # ========================================================================
    # A1: KV Cache Memory Scaling
    # ========================================================================
    print("=" * 70)
    print("A1: KV Cache Memory Scaling (Context Length)")
    print("=" * 70)
    print(f"Testing: {CONTEXT_LENGTHS}")
    print()

    for target_len in CONTEXT_LENGTHS:
        print(f"--- Context: ~{target_len} tokens ---")
        context = generate_context(target_len)
        trial_results = []

        for trial in range(TRIALS):
            print(f"  Trial {trial+1}/{TRIALS}...", end=" ", flush=True)
            result = single_request(client, args.model, context, BASE_QUERY, OUTPUT_TOKENS)
            trial_results.append(result)

            if result["success"]:
                print(f"OK: {result['input_tokens']} in, {result['output_tokens']} out, "
                      f"{result['latency_s']:.2f}s, "
                      f"mem_delta={result['mem_delta_mb']:+.0f}MB, "
                      f"gpu={result['mem_after_mb']:.0f}MB")
            else:
                print(f"FAIL: {result.get('error', '?')[:60]}")
            time.sleep(0.5)

        ok = [r for r in trial_results if r.get("success")]
        if ok:
            avg_lat = sum(r["latency_s"] for r in ok) / len(ok)
            avg_in = sum(r["input_tokens"] for r in ok) / len(ok)
            avg_out = sum(r["output_tokens"] for r in ok) / len(ok)
            avg_delta = sum(r["mem_delta_mb"] for r in ok) / len(ok)
            avg_gpu = sum(r["mem_after_mb"] for r in ok) / len(ok)
            mem_per_token = avg_delta / avg_in if avg_in > 0 else 0

            print(f"  Summary: avg_input={avg_in:.0f}, avg_latency={avg_lat:.2f}s, "
                  f"avg_mem_delta={avg_delta:+.0f}MB, mem/token={mem_per_token:.2f}MB")
        else:
            print("  Summary: ALL FAILED")

        results["a1_context_scaling"].append({
            "target_tokens": target_len,
            "trials": trial_results
        })
        print()

    # ========================================================================
    # E1: Memory Budget at Concurrency
    # ========================================================================
    print("=" * 70)
    print("E1: Memory Budget at Concurrency Levels")
    print("=" * 70)
    print(f"Testing: {CONCURRENCY_LEVELS}")
    print()

    for conc in CONCURRENCY_LEVELS:
        print(f"--- Concurrency: {conc} ---")
        result = concurrent_batch(client, args.model, conc, REQUESTS_PER_CONCURRENCY, OUTPUT_TOKENS)
        result["mode"] = "memory_budget"

        if result.get("successful", 0) > 0:
            gpu = result.get("gpu_detail", {})
            print(f"  Throughput: {result['throughput_tok_s']:.0f} tok/s, "
                  f"Latency: {result['avg_latency_s']:.2f}s")
            print(f"  GPU: {gpu.get('used_mb', 0):.0f}MB used, "
                  f"free={gpu.get('free_mb', 0):.0f}MB, "
                  f"power={gpu.get('power_w', 0):.0f}W, "
                  f"temp={gpu.get('temp_c', 0):.0f}C")
        else:
            print(f"  FAILED: {result.get('errors', ['?'])[:1]}")

        results["e1_concurrency_scaling"].append(result)
        print()

    # ========================================================================
    # Save
    # ========================================================================
    output_path = args.output or os.path.join(
        os.path.dirname(__file__), "results", "final", "nemotron",
        "mamba_experiments", "a1e1_memory_budget.json"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {output_path}")

    # ========================================================================
    # Summary
    # ========================================================================
    print()
    print("=" * 70)
    print("A1 SUMMARY: KV Cache Memory per Token")
    print("=" * 70)
    print(f"{'Context':>10} {'Input Tok':>10} {'Latency':>10} {'Mem Delta':>12} {'MB/token':>10}")
    print("-" * 70)
    for probe in results["a1_context_scaling"]:
        ok = [r for r in probe["trials"] if r.get("success")]
        if ok:
            avg_in = sum(r["input_tokens"] for r in ok) / len(ok)
            avg_lat = sum(r["latency_s"] for r in ok) / len(ok)
            avg_delta = sum(r["mem_delta_mb"] for r in ok) / len(ok)
            mb_per_tok = avg_delta / avg_in if avg_in > 0 else 0
            print(f"{probe['target_tokens']:>10} {avg_in:>10.0f} {avg_lat:>10.2f} {avg_delta:>+12.0f} {mb_per_tok:>10.3f}")

    print()
    print("=" * 70)
    print("E1 SUMMARY: Memory Budget at Concurrency")
    print("=" * 70)
    print(f"{'Conc':>5} {'tok/s':>8} {'GPU Used':>10} {'Free':>10} {'Power':>8} {'Temp':>6}")
    print("-" * 70)
    for r in results["e1_concurrency_scaling"]:
        if r.get("successful", 0) > 0:
            gpu = r.get("gpu_detail", {})
            print(f"{r['concurrency']:>5} {r['throughput_tok_s']:>8.0f} "
                  f"{gpu.get('used_mb', 0):>10.0f} {gpu.get('free_mb', 0):>10.0f} "
                  f"{gpu.get('power_w', 0):>8.0f} {gpu.get('temp_c', 0):>6.0f}")
        else:
            print(f"{r['concurrency']:>5} {'FAILED':>8}")

    # Capacity estimation
    print()
    print("=" * 70)
    print("CAPACITY ESTIMATION")
    print("=" * 70)
    if results["a1_context_scaling"]:
        # Find the 256K data point or extrapolate
        for probe in results["a1_context_scaling"]:
            ok = [r for r in probe["trials"] if r.get("success")]
            if ok and probe["target_tokens"] >= 65536:
                avg_in = sum(r["input_tokens"] for r in ok) / len(ok)
                avg_delta = sum(r["mem_delta_mb"] for r in ok) / len(ok)
                mb_per_tok = avg_delta / avg_in if avg_in > 0 else 0
                # Extrapolate to 256K
                est_256k = mb_per_tok * 256000
                total_mem = mem_baseline.get("total_mb", 128000)
                # Model weights ~22GB
                available_for_kv = total_mem - 22000
                max_sessions = available_for_kv / est_256k if est_256k > 0 else 0
                print(f"Memory per token: {mb_per_tok:.3f} MB")
                print(f"Estimated 256K session memory: {est_256k:.0f} MB")
                print(f"Total GPU memory: {total_mem:.0f} MB")
                print(f"Available for KV cache (after weights): ~{available_for_kv:.0f} MB")
                print(f"Max concurrent 256K sessions: ~{max_sessions:.0f}")
                break


if __name__ == "__main__":
    main()
