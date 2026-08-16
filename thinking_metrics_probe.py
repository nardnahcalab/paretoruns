#!/usr/bin/env python3
"""
Thinking Metrics Probe

Captures thinking/reasoning metrics from streaming responses using agentic datasets:
- Thinking iterations (number of separate <think> blocks)
- Thinking time (time from first to last thinking token)
- Thinking tokens per iteration
- Thinking overhead ratio (thinking_time / total_time)
- Standard aiperf-compatible metrics

Usage:
    # With synthetic tasks
    python3 thinking_metrics_probe.py --url http://localhost --port 8000

    # With agentic dataset
    python3 thinking_metrics_probe.py --url http://localhost --port 8000 \
        --dataset /home/bala/public/multi-turn-chat-dataset/agentic/data/multi_turn_agentic_task.jsonl \
        --max-sessions 50

Requires: aiohttp
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import List, Optional

try:
    import aiohttp
except ImportError:
    print("ERROR: pip install aiohttp")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
DEFAULT_DATASET = "/home/bala/public/multi-turn-chat-dataset/agentic/data/multi_turn_agentic_task.jsonl"
CONCURRENCY_LEVELS = [1, 2, 4, 8]
SESSIONS_PER_LEVEL = 20
MAX_OUTPUT_TOKENS = 512
SYSTEM_PROMPT = "You are a helpful AI assistant. Think step by step when answering complex questions."


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_agentic_dataset(path: str, max_sessions: int = None) -> list:
    """
    Load agentic dataset from JSONL file.
    Returns list of {session_id, turns: [str]} dicts.
    """
    sessions = []
    with open(path) as f:
        for i, line in enumerate(f):
            if max_sessions and i >= max_sessions:
                break
            d = json.loads(line.strip())
            session = {
                "session_id": d["session_id"],
                "turns": [t["text"] for t in d["turns"]]
            }
            sessions.append(session)
    return sessions


def get_synthetic_tasks() -> list:
    """Return synthetic task definitions (fallback when no dataset provided)."""
    tasks = [
        {
            "name": "code_generation",
            "turns": [
                "Write a Python function that finds the longest palindromic substring in a given string. Include type hints and docstring.",
                "Now optimize it to run in O(n) time using Manacher's algorithm.",
                "Add comprehensive unit tests covering edge cases."
            ]
        },
        {
            "name": "debugging",
            "turns": [
                "I have this Python code that's supposed to merge two sorted lists but it's producing wrong output:\n```python\ndef merge(a, b):\n    result = []\n    i = j = 0\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n            result.append(a[i])\n            i += 1\n        else:\n            result.append(b[j])\n            i += 1  # Bug here\n    return result + a[i:] + b[j:]\n```\nWhat's wrong and how do I fix it?",
                "Can you also add a version that handles duplicates correctly?"
            ]
        },
        {
            "name": "analysis",
            "turns": [
                "Compare the time and space complexity of quicksort vs mergesort vs timsort. Which should I use for a dataset of 1 million integers?",
                "What about for nearly sorted data? And what about stability requirements?",
                "Write a benchmark script to compare them empirically."
            ]
        },
        {
            "name": "architecture",
            "turns": [
                "I'm building a real-time chat application that needs to handle 100K concurrent users. Design the architecture including message brokering, presence, and persistence.",
                "How would you handle message ordering guarantees and exactly-once delivery?",
                "What database schema would you use for the message history?"
            ]
        },
        {
            "name": "math_reasoning",
            "turns": [
                "Prove that for any positive integer n, the sum 1 + 2 + 3 + ... + n = n(n+1)/2 using mathematical induction.",
                "Now prove that the sum of squares 1² + 2² + ... + n² = n(n+1)(2n+1)/6.",
                "Generalize this to find a formula for 1^k + 2^k + ... + n^k for any positive integer k."
            ]
        },
        {
            "name": "algorithm_design",
            "turns": [
                "Design an LRU cache that supports O(1) get and put operations. What data structures would you use?",
                "Now extend it to support TTL (time-to-live) expiration for entries.",
                "How would you make it thread-safe for concurrent access?"
            ]
        }
    ]
    return tasks


# ---------------------------------------------------------------------------
# Metrics dataclass
# ---------------------------------------------------------------------------
@dataclass
class ThinkingMetrics:
    """Metrics for a single request's thinking behavior."""
    session_id: str
    turn_index: int
    turn_count: int
    task_name: str  # "agentic" for real dataset, or synthetic task name
    
    # Thinking block analysis
    thinking_iterations: int = 0
    thinking_tokens: int = 0
    thinking_chars: int = 0
    output_tokens: int = 0
    output_chars: int = 0
    total_chars: int = 0
    
    # Timing
    latency_s: float = 0
    thinking_time_s: float = 0
    output_time_s: float = 0
    ttft_s: float = 0
    
    # Per-iteration stats
    thinking_tokens_per_iteration: List[int] = field(default_factory=list)
    thinking_chars_per_iteration: List[int] = field(default_factory=list)
    
    # Computed ratios
    thinking_overhead_ratio: float = 0
    thinking_token_ratio: float = 0
    thinking_char_ratio: float = 0
    
    # Success
    success: bool = True
    error: Optional[str] = None


def estimate_tokens(text: str) -> int:
    """Rough token estimate (1 token ≈ 4 chars for English)."""
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Streaming request with thinking capture
# ---------------------------------------------------------------------------
async def send_streaming_request(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    messages: list,
    max_tokens: int,
    session_id: str,
    turn_index: int,
    turn_count: int,
    task_name: str,
) -> ThinkingMetrics:
    """Send a streaming request and capture thinking metrics."""
    metrics = ThinkingMetrics(
        session_id=session_id,
        turn_index=turn_index,
        turn_count=turn_count,
        task_name=task_name,
    )
    
    url = f"{base_url}/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True,
    }
    
    # Track reasoning and content separately
    reasoning_chunks = []
    content_chunks = []
    first_reasoning_time = None
    last_reasoning_time = None
    first_content_time = None
    first_token_time = None
    
    t_start = time.perf_counter()
    
    try:
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                metrics.success = False
                metrics.error = f"HTTP {resp.status}: {error_text[:200]}"
                metrics.latency_s = time.perf_counter() - t_start
                return metrics
            
            async for line in resp.content:
                if line:
                    line_str = line.decode('utf-8').strip()
                    if line_str.startswith('data: ') and line_str != 'data: [DONE]':
                        try:
                            chunk = json.loads(line_str[6:])
                            if 'choices' in chunk and len(chunk['choices']) > 0:
                                delta = chunk['choices'][0].get('delta', {})
                                now = time.perf_counter() - t_start
                                
                                # Capture reasoning tokens (thinking)
                                if 'reasoning' in delta and delta['reasoning']:
                                    reasoning_chunks.append((delta['reasoning'], now))
                                    if first_reasoning_time is None:
                                        first_reasoning_time = now
                                    last_reasoning_time = now
                                    if first_token_time is None:
                                        first_token_time = now
                                        metrics.ttft_s = now
                                
                                # Capture content tokens (output)
                                if 'content' in delta and delta['content']:
                                    content_chunks.append((delta['content'], now))
                                    if first_content_time is None:
                                        first_content_time = now
                                    if first_token_time is None:
                                        first_token_time = now
                                        metrics.ttft_s = now
                        except json.JSONDecodeError:
                            pass
    
    except Exception as e:
        metrics.success = False
        metrics.error = str(e)[:200]
        metrics.latency_s = time.perf_counter() - t_start
        return metrics
    
    t_end = time.perf_counter()
    metrics.latency_s = t_end - t_start
    
    # Combine text
    reasoning_text = ''.join(c[0] for c in reasoning_chunks)
    content_text = ''.join(c[0] for c in content_chunks)
    
    metrics.thinking_chars = len(reasoning_text)
    metrics.thinking_tokens = estimate_tokens(reasoning_text)
    metrics.output_chars = len(content_text)
    metrics.output_tokens = estimate_tokens(content_text)
    metrics.total_chars = metrics.thinking_chars + metrics.output_chars
    
    # Thinking iterations: count separate reasoning blocks
    # The model produces reasoning as one continuous block, but we can detect
    # sentence boundaries (periods followed by newline) as "thinking steps"
    reasoning_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', reasoning_text) if s.strip()]
    metrics.thinking_iterations = max(1, len(reasoning_sentences)) if reasoning_text else 0
    metrics.thinking_tokens_per_iteration = [estimate_tokens(s) for s in reasoning_sentences] if reasoning_sentences else []
    metrics.thinking_chars_per_iteration = [len(s) for s in reasoning_sentences] if reasoning_sentences else []
    
    # Timing
    if first_reasoning_time is not None and last_reasoning_time is not None:
        metrics.thinking_time_s = last_reasoning_time - first_reasoning_time
    if first_content_time is not None:
        metrics.output_time_s = (t_end - t_start) - first_content_time
    
    # Compute ratios
    total_tokens = metrics.thinking_tokens + metrics.output_tokens
    if total_tokens > 0:
        metrics.thinking_token_ratio = metrics.thinking_tokens / total_tokens
    if metrics.total_chars > 0:
        metrics.thinking_char_ratio = metrics.thinking_chars / metrics.total_chars
    if metrics.latency_s > 0:
        metrics.thinking_overhead_ratio = metrics.thinking_time_s / metrics.latency_s
    
    return metrics


# ---------------------------------------------------------------------------
# Concurrency test runner
# ---------------------------------------------------------------------------
async def run_concurrency_test(
    base_url: str,
    model: str,
    concurrency: int,
    tasks: list,
    max_output_tokens: int,
) -> dict:
    """Run a concurrency test and collect thinking metrics."""
    print(f"\n--- Concurrency: {concurrency} ---")
    
    all_metrics = []
    connector = aiohttp.TCPConnector(limit=concurrency)
    
    async with aiohttp.ClientSession(connector=connector) as session:
        # Create tasks for all requests
        request_tasks = []
        for task in tasks:
            session_id = task.get("session_id", task.get("name", "unknown"))
            turns = task["turns"]
            
            # Build conversation incrementally (each turn includes previous context)
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            
            for turn_idx, turn_text in enumerate(turns):
                messages.append({"role": "user", "content": turn_text})
                task_name = task.get("name", "agentic")
                
                request_tasks.append(send_streaming_request(
                    session, base_url, model, messages.copy(), max_output_tokens,
                    session_id, turn_idx, len(turns), task_name
                ))
        
        # Run all requests concurrently
        print(f"  Running {len(request_tasks)} requests from {len(tasks)} sessions...")
        results = await asyncio.gather(*request_tasks, return_exceptions=True)
        
        for r in results:
            if isinstance(r, ThinkingMetrics):
                all_metrics.append(r)
            elif isinstance(r, Exception):
                print(f"  Exception: {r}")
    
    # Aggregate metrics
    successful = [m for m in all_metrics if m.success]
    failed = [m for m in all_metrics if not m.success]
    
    if failed:
        for m in failed[:3]:
            print(f"  Failed: {m.session_id[:8]}... turn {m.turn_index}: {m.error}")
    
    if not successful:
        return {"error": "All requests failed"}
    
    # Compute aggregates
    agg = {
        "concurrency": concurrency,
        "total_requests": len(all_metrics),
        "successful": len(successful),
        "failed": len(failed),
        "success_rate": len(successful) / len(all_metrics) * 100,
        
        # Latency
        "latency_avg_s": sum(m.latency_s for m in successful) / len(successful),
        "latency_p50_s": sorted([m.latency_s for m in successful])[len(successful) // 2],
        "latency_p99_s": sorted([m.latency_s for m in successful])[int(len(successful) * 0.99)],
        
        # Thinking metrics
        "thinking_iterations_avg": sum(m.thinking_iterations for m in successful) / len(successful),
        "thinking_tokens_avg": sum(m.thinking_tokens for m in successful) / len(successful),
        "thinking_tokens_total": sum(m.thinking_tokens for m in successful),
        "output_tokens_avg": sum(m.output_tokens for m in successful) / len(successful),
        "output_tokens_total": sum(m.output_tokens for m in successful),
        
        # Ratios
        "thinking_overhead_ratio_avg": sum(m.thinking_overhead_ratio for m in successful) / len(successful),
        "thinking_token_ratio_avg": sum(m.thinking_token_ratio for m in successful) / len(successful),
        "thinking_time_avg_s": sum(m.thinking_time_s for m in successful) / len(successful),
        "output_time_avg_s": sum(m.output_time_s for m in successful) / len(successful),
        
        # Throughput
        "throughput_req_s": len(successful) / sum(m.latency_s for m in successful) * concurrency,
        "throughput_tok_s": sum(m.output_tokens for m in successful) / sum(m.latency_s for m in successful) * concurrency,
        
        # Breakdowns
        "by_task": {},
        "by_turn": {},
        "by_turn_count": {},
        
        # Per-request details
        "individual_metrics": [asdict(m) for m in successful],
    }
    
    # Breakdown by task
    by_task = defaultdict(list)
    for m in successful:
        by_task[m.task_name].append(m)
    
    for task_name, task_metrics in by_task.items():
        agg["by_task"][task_name] = {
            "count": len(task_metrics),
            "thinking_iterations_avg": sum(m.thinking_iterations for m in task_metrics) / len(task_metrics),
            "thinking_tokens_avg": sum(m.thinking_tokens for m in task_metrics) / len(task_metrics),
            "thinking_overhead_ratio_avg": sum(m.thinking_overhead_ratio for m in task_metrics) / len(task_metrics),
            "latency_avg_s": sum(m.latency_s for m in task_metrics) / len(task_metrics),
        }
    
    # Breakdown by turn index
    by_turn = defaultdict(list)
    for m in successful:
        by_turn[m.turn_index].append(m)
    
    for turn_idx, turn_metrics in by_turn.items():
        agg["by_turn"][str(turn_idx)] = {
            "count": len(turn_metrics),
            "thinking_iterations_avg": sum(m.thinking_iterations for m in turn_metrics) / len(turn_metrics),
            "thinking_tokens_avg": sum(m.thinking_tokens for m in turn_metrics) / len(turn_metrics),
            "thinking_overhead_ratio_avg": sum(m.thinking_overhead_ratio for m in turn_metrics) / len(turn_metrics),
            "latency_avg_s": sum(m.latency_s for m in turn_metrics) / len(turn_metrics),
        }
    
    # Breakdown by session turn count (1, 2, 3, ... 14)
    by_turn_count = defaultdict(list)
    for m in successful:
        by_turn_count[m.turn_count].append(m)
    
    for tc, tc_metrics in sorted(by_turn_count.items()):
        agg["by_turn_count"][str(tc)] = {
            "count": len(tc_metrics),
            "sessions": len(set(m.session_id for m in tc_metrics)),
            "thinking_iterations_avg": sum(m.thinking_iterations for m in tc_metrics) / len(tc_metrics),
            "thinking_tokens_avg": sum(m.thinking_tokens for m in tc_metrics) / len(tc_metrics),
            "thinking_overhead_ratio_avg": sum(m.thinking_overhead_ratio for m in tc_metrics) / len(tc_metrics),
            "latency_avg_s": sum(m.latency_s for m in tc_metrics) / len(tc_metrics),
        }
    
    # Print summary
    print(f"  Success: {len(successful)}/{len(all_metrics)} ({agg['success_rate']:.0f}%)")
    print(f"  Throughput: {agg['throughput_req_s']:.1f} req/s, {agg['throughput_tok_s']:.0f} tok/s")
    print(f"  Latency: avg={agg['latency_avg_s']:.2f}s, p50={agg['latency_p50_s']:.2f}s, p99={agg['latency_p99_s']:.2f}s")
    print(f"  Thinking: {agg['thinking_iterations_avg']:.1f} iterations, {agg['thinking_tokens_avg']:.0f} tokens/req")
    print(f"  Overhead: {agg['thinking_overhead_ratio_avg']:.1%} of latency is thinking")
    print(f"  Token ratio: {agg['thinking_token_ratio_avg']:.1%} of output is thinking")
    
    return agg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main():
    parser = argparse.ArgumentParser(description="Thinking Metrics Probe")
    parser.add_argument("--url", default="http://localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--concurrency", type=int, nargs="+", default=CONCURRENCY_LEVELS)
    parser.add_argument("--dataset", default=None, help="Path to agentic dataset JSONL")
    parser.add_argument("--max-sessions", type=int, default=SESSIONS_PER_LEVEL,
                        help="Max sessions per concurrency level")
    parser.add_argument("--output-tokens", type=int, default=MAX_OUTPUT_TOKENS)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    
    base_url = f"{args.url}:{args.port}/v1"
    print(f"Connecting to: {base_url}")
    print(f"Model: {args.model}")
    print(f"Concurrency levels: {args.concurrency}")
    
    # Load dataset
    if args.dataset:
        print(f"Dataset: {args.dataset}")
        print(f"Max sessions per level: {args.max_sessions}")
        all_sessions = load_agentic_dataset(args.dataset, max_sessions=None)
        print(f"Total sessions loaded: {len(all_sessions)}")
        
        # Get turn count distribution
        turn_counts = [len(s["turns"]) for s in all_sessions]
        print(f"Turn count distribution: {dict(sorted((tc, turn_counts.count(tc)) for tc in set(turn_counts)))}")
    else:
        print("Dataset: synthetic (no --dataset specified)")
        all_sessions = get_synthetic_tasks()
    
    print(f"Max output tokens: {args.output_tokens}")
    
    # Health check
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{args.url}:{args.port}/health") as resp:
            if resp.status != 200:
                print(f"ERROR: Server not ready (status={resp.status})")
                sys.exit(1)
            print("Server OK")
    
    all_results = {
        "config": {
            "model": args.model,
            "url": base_url,
            "dataset": args.dataset,
            "concurrency_levels": args.concurrency,
            "max_sessions_per_level": args.max_sessions,
            "max_output_tokens": args.output_tokens,
            "total_sessions": len(all_sessions),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
        "runs": []
    }
    
    for conc in args.concurrency:
        # Sample sessions for this concurrency level
        if args.dataset:
            # Use different random sample for each level
            import random
            random.seed(42 + conc)  # Reproducible but different per level
            sessions = random.sample(all_sessions, min(args.max_sessions, len(all_sessions)))
        else:
            sessions = all_sessions
        
        result = await run_concurrency_test(
            base_url, args.model, conc, sessions, args.output_tokens
        )
        all_results["runs"].append(result)
        await asyncio.sleep(2)  # Cool down
    
    # Save
    output_path = args.output or os.path.join(
        os.path.dirname(__file__), "results", "final", "nemotron",
        "mamba_experiments", "thinking_metrics_agentic.json"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved: {output_path}")
    
    # Print summary table
    print("\n" + "=" * 100)
    print("THINKING METRICS SUMMARY")
    print("=" * 100)
    print(f"{'Conc':>5} {'Req/s':>7} {'Tok/s':>7} {'Lat(s)':>7} {'Think#':>6} {'ThinkT':>6} {'Over%':>6} {'TokRat':>6}")
    print("-" * 100)
    for r in all_results["runs"]:
        if "error" not in r:
            print(f"{r['concurrency']:>5} {r['throughput_req_s']:>7.1f} {r['throughput_tok_s']:>7.0f} "
                  f"{r['latency_avg_s']:>7.2f} {r['thinking_iterations_avg']:>6.1f} "
                  f"{r['thinking_tokens_avg']:>6.0f} {r['thinking_overhead_ratio_avg']:>6.1%} "
                  f"{r['thinking_token_ratio_avg']:>6.1%}")


if __name__ == "__main__":
    asyncio.run(main())
