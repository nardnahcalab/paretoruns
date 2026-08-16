#!/usr/bin/env python3
"""
B1+B2+B4: Concurrency Push + Long-Output + Queue Depth

B1: Extended concurrency sweep (c256, c512)
B2: Long-output concurrency (1K, 2K, 4K tokens)
B4: Queue depth, rejection rate, error rate at each level

Usage:
    python3 concurrency_push_probe.py [--url URL] [--port PORT]
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: pip install openai")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# B1: Extended concurrency
B1_CONCURRENCIES = [128, 256, 512]
B1_OUTPUT_TOKENS = 128

# B2: Long-output concurrency
B2_CONCURRENCIES = [1, 4, 16, 32, 64]
B2_OUTPUT_TOKENS = [1024, 2048, 4096]

# B4: measured at each concurrency level above
PROMPT = "Write a detailed technical explanation of how the Mamba architecture processes sequences with constant computational complexity. Include specifics about selective scan mechanisms, hidden state updates, and how this differs from transformer attention. Aim for thorough coverage."

SYSTEM_PROMPT = "You are a technical AI assistant. Provide detailed, comprehensive answers."


def generate_request(client, model, messages, max_tokens, timeout=300):
    """Send a single request and return results."""
    t_start = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            model=model, messages=messages,
            max_tokens=max_tokens, temperature=0.7,
            stream=False, timeout=timeout
        )
        t_end = time.perf_counter()

        choice = resp.choices[0]
        usage = resp.usage
        finish = choice.finish_reason

        return {
            "success": True,
            "latency_s": t_end - t_start,
            "input_tokens": usage.prompt_tokens if usage else 0,
            "output_tokens": usage.completion_tokens if usage else 0,
            "total_tokens": usage.total_tokens if usage else 0,
            "finish_reason": finish,
            "error": None,
        }
    except Exception as e:
        t_end = time.perf_counter()
        return {
            "success": False,
            "latency_s": t_end - t_start,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "finish_reason": "error",
            "error": str(e)[:200],
        }


def run_concurrency_test(client, model, concurrency, output_tokens, label="", timeout=300):
    """Run a concurrency test with the given parameters."""
    print(f"  --- {label}: c{concurrency}, {output_tokens}-tok output ---")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": PROMPT}
    ]

    results = []
    t_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(
                generate_request, client, model, messages, output_tokens, timeout
            ): i for i in range(concurrency)
        }

        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                result["request_id"] = idx
                results.append(result)
            except Exception as e:
                results.append({
                    "success": False,
                    "request_id": idx,
                    "latency_s": 0,
                    "error": str(e)[:200],
                    "finish_reason": "exception",
                })

    t_total = time.perf_counter() - t_start

    # Compute statistics
    successful = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]
    errors_by_type = defaultdict(int)
    for r in failed:
        err = r.get("error", "unknown")
        if "timeout" in err.lower():
            errors_by_type["timeout"] += 1
        elif "503" in err or "overloaded" in err.lower():
            errors_by_type["overloaded"] += 1
        elif "502" in err or "bad gateway" in err.lower():
            errors_by_type["bad_gateway"] += 1
        elif "connection" in err.lower():
            errors_by_type["connection"] += 1
        else:
            errors_by_type["other"] += 1

    if successful:
        latencies = [r["latency_s"] for r in successful]
        latencies.sort()
        n = len(latencies)

        avg_lat = sum(latencies) / n
        p50 = latencies[int(n * 0.5)]
        p90 = latencies[int(n * 0.9)]
        p95 = latencies[int(n * 0.95)]
        p99 = latencies[min(int(n * 0.99), n - 1)]
        min_lat = latencies[0]
        max_lat = latencies[-1]

        total_output = sum(r["output_tokens"] for r in successful)
        throughput_tok_s = total_output / t_total if t_total > 0 else 0
        throughput_req_s = len(successful) / t_total if t_total > 0 else 0
        avg_output = total_output / len(successful)
    else:
        avg_lat = p50 = p90 = p95 = p99 = min_lat = max_lat = 0
        throughput_tok_s = throughput_req_s = avg_output = 0

    # Queue depth estimation: requests that waited
    # In vLLM, all requests are queued; we measure total time minus first-token time
    # But we don't have TTFT here, so we use a simpler metric:
    # queue_depth = (total_completion_time - avg_serial_time) / avg_serial_time
    # where avg_serial_time = avg latency at c1
    avg_serial_time = 1.0  # rough estimate from c1 data
    estimated_queue_depth = max(0, (avg_lat - avg_serial_time) / avg_serial_time * concurrency) if avg_serial_time > 0 else 0

    summary = {
        "concurrency": concurrency,
        "output_tokens": output_tokens,
        "label": label,
        "total_requests": concurrency,
        "successful": len(successful),
        "failed": len(failed),
        "success_rate": len(successful) / concurrency * 100,
        "error_rate": len(failed) / concurrency * 100,
        "errors_by_type": dict(errors_by_type),
        "total_time_s": t_total,
        "throughput_req_s": throughput_req_s,
        "throughput_tok_s": throughput_tok_s,
        "avg_output_tokens": avg_output,
        "latency": {
            "avg_s": avg_lat,
            "p50_s": p50,
            "p90_s": p90,
            "p95_s": p95,
            "p99_s": p99,
            "min_s": min_lat,
            "max_s": max_lat,
        },
        "estimated_queue_depth": estimated_queue_depth,
        "individual_results": results,
    }

    # Print summary
    print(f"    Success: {len(successful)}/{concurrency} ({summary['success_rate']:.0f}%)")
    if failed:
        print(f"    Failed: {len(failed)} ({summary['error_rate']:.0f}%) - {dict(errors_by_type)}")
    print(f"    Throughput: {throughput_req_s:.1f} req/s, {throughput_tok_s:.0f} tok/s")
    print(f"    Latency: avg={avg_lat:.2f}s, p50={p50:.2f}s, p90={p90:.2f}s, p95={p95:.2f}s, p99={p99:.2f}s")
    print(f"    Output: avg={avg_output:.0f} tok, total={total_output:.0f} tok")

    return summary


def main():
    parser = argparse.ArgumentParser(description="B1+B2+B4: Concurrency Push")
    parser.add_argument("--url", default="http://localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--skip-b1", action="store_true", help="Skip B1 (extended concurrency)")
    parser.add_argument("--skip-b2", action="store_true", help="Skip B2 (long-output)")
    args = parser.parse_args()

    base_url = f"{args.url}:{args.port}/v1"
    print(f"Connecting to: {base_url}")
    print(f"Model: {args.model}")

    client = OpenAI(base_url=base_url, api_key="dummy")

    # Health check
    try:
        models = client.models.list()
        print(f"Server OK. Model: {models.data[0].id}")
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    all_results = {
        "config": {
            "model": args.model,
            "url": base_url,
            "prompt": PROMPT[:100] + "...",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
        "b1_extended_concurrency": [],
        "b2_long_output": [],
        "b4_combined": [],
    }

    # =====================================================================
    # B1: Extended Concurrency (c256, c512) with 128-tok output
    # =====================================================================
    if not args.skip_b1:
        print("\n" + "=" * 70)
        print("B1: Extended Concurrency Sweep (128-tok output)")
        print("=" * 70)

        for conc in B1_CONCURRENCIES:
            result = run_concurrency_test(
                client, args.model, conc, B1_OUTPUT_TOKENS,
                label=f"B1 c{conc}"
            )
            all_results["b1_extended_concurrency"].append(result)
            time.sleep(2)  # Cool down between tests

    # =====================================================================
    # B2: Long-Output Concurrency (1K, 2K, 4K tokens)
    # =====================================================================
    if not args.skip_b2:
        print("\n" + "=" * 70)
        print("B2: Long-Output Concurrency")
        print("=" * 70)

        for out_tok in B2_OUTPUT_TOKENS:
            print(f"\n  Output: {out_tok} tokens")
            for conc in B2_CONCURRENCIES:
                # Skip high concurrency with long output (will timeout)
                if out_tok >= 2048 and conc > 32:
                    print(f"  Skipping c{conc} x {out_tok}tok (likely timeout)")
                    continue
                if out_tok >= 4096 and conc > 16:
                    print(f"  Skipping c{conc} x {out_tok}tok (likely timeout)")
                    continue

                result = run_concurrency_test(
                    client, args.model, conc, out_tok,
                    label=f"B2 c{conc}x{out_tok}"
                )
                all_results["b2_long_output"].append(result)
                time.sleep(2)

    # =====================================================================
    # B4: Queue Depth (analyzed from B1+B2 results)
    # =====================================================================
    print("\n" + "=" * 70)
    print("B4: Queue Depth & Rejection Analysis")
    print("=" * 70)

    # Combine all results for B4 analysis
    all_runs = all_results["b1_extended_concurrency"] + all_results["b2_long_output"]

    for run in all_runs:
        all_results["b4_combined"].append({
            "concurrency": run["concurrency"],
            "output_tokens": run["output_tokens"],
            "success_rate": run["success_rate"],
            "error_rate": run["error_rate"],
            "errors_by_type": run["errors_by_type"],
            "throughput_req_s": run["throughput_req_s"],
            "throughput_tok_s": run["throughput_tok_s"],
            "latency_avg_s": run["latency"]["avg_s"],
            "latency_p50_s": run["latency"]["p50_s"],
            "latency_p90_s": run["latency"]["p90_s"],
            "latency_p95_s": run["latency"]["p95_s"],
            "latency_p99_s": run["latency"]["p99_s"],
            "estimated_queue_depth": run["estimated_queue_depth"],
        })

    # Print B4 summary
    print(f"\n  {'Conc':>5} {'Out':>5} {'OK%':>5} {'Err%':>5} {'Req/s':>7} {'Tok/s':>7} {'p50':>6} {'p90':>6} {'p95':>6} {'p99':>6} {'Errors'}")
    print("  " + "-" * 90)
    for r in all_results["b4_combined"]:
        err_str = ", ".join(f"{k}:{v}" for k, v in r["errors_by_type"].items()) if r["errors_by_type"] else "-"
        print(f"  {r['concurrency']:>5} {r['output_tokens']:>5} {r['success_rate']:>5.0f} {r['error_rate']:>5.0f} "
              f"{r['throughput_req_s']:>7.1f} {r['throughput_tok_s']:>7.0f} "
              f"{r['latency_p50_s']:>6.1f} {r['latency_p90_s']:>6.1f} {r['latency_p95_s']:>6.1f} {r['latency_p99_s']:>6.1f} "
              f"{err_str}")

    # Save
    output_dir = args.output_dir or os.path.join(
        os.path.dirname(__file__), "results", "final", "nemotron", "mamba_experiments"
    )
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "b1b2b4_concurrency_push.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved: {output_path}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nB1 (Extended Concurrency):")
    for r in all_results["b1_extended_concurrency"]:
        print(f"  c{r['concurrency']}: {r['throughput_tok_s']:.0f} tok/s, p99={r['latency']['p99_s']:.1f}s, success={r['success_rate']:.0f}%")

    print("\nB2 (Long-Output):")
    for r in all_results["b2_long_output"]:
        print(f"  c{r['concurrency']}x{r['output_tokens']}tok: {r['throughput_tok_s']:.0f} tok/s, p99={r['latency']['p99_s']:.1f}s, success={r['success_rate']:.0f}%")

    print("\nB4 (Rejection):")
    failures = [r for r in all_results["b4_combined"] if r["error_rate"] > 0]
    if failures:
        for r in failures:
            print(f"  c{r['concurrency']}x{r['output_tokens']}tok: {r['error_rate']:.0f}% errors - {r['errors_by_type']}")
    else:
        print("  No failures at any concurrency level tested")


if __name__ == "__main__":
    main()
