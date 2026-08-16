#!/usr/bin/env python3
"""
D2: Routing Entropy at Long Context

Tests whether routing behavior changes at extreme context lengths (64K-256K).
Sends long-context prompts and captures routed expert IDs to compute entropy.

Usage:
    python3 routing_entropy_long_context.py [--url URL] [--port PORT] [--output OUTPUT]

Note: This requires vLLM to be running with enable_return_routed_experts=true.
The Dell container enables this by default.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: pip install openai")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CONTEXT_LENGTHS = [1024, 16384, 65536, 131072]
NUM_SAMPLES = 5  # Number of prompts per context length
OUTPUT_TOKENS = 32
SYSTEM_PROMPT = "You are a helpful AI assistant. Answer concisely."

# Layer indices for MoE layers in Nemotron (23 MoE layers out of 52 total)
# Layers: 1,3,6,8,10,13,15,17,20,22,24,27,29,31,34,36,38,40,43,45,47,49,51
MOE_LAYERS = [1,3,6,8,10,13,15,17,20,22,24,27,29,31,34,36,38,40,43,45,47,49,51]
NUM_EXPERTS = 256
TOP_K = 6


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


def compute_entropy(load_vector):
    """Compute normalized entropy of a load distribution."""
    total = sum(load_vector)
    if total == 0:
        return 0.0
    probs = [x / total for x in load_vector if x > 0]
    if not probs:
        return 0.0
    H = -sum(p * math.log2(p) for p in probs)
    H_max = math.log2(len(load_vector))
    return H / H_max if H_max > 0 else 0.0


def compute_jaccard(top_k_a, top_k_b):
    """Compute Jaccard index between two top-k sets."""
    set_a = set(top_k_a)
    set_b = set(top_k_b)
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def main():
    parser = argparse.ArgumentParser(description="D2: Routing Entropy at Long Context")
    parser.add_argument("--url", default="http://localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--output", default=None)
    parser.add_argument("--model", default="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4")
    args = parser.parse_args()

    base_url = f"{args.url}:{args.port}/v1"
    print(f"Connecting to: {base_url}")
    print(f"Model: {args.model}")
    print(f"Context lengths: {CONTEXT_LENGTHS}")
    print(f"Samples per length: {NUM_SAMPLES}")
    print()

    client = OpenAI(base_url=base_url, api_key="dummy")

    # Health check
    try:
        models = client.models.list()
        print(f"Server OK. Model: {models.data[0].id}")
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    results = {
        "config": {
            "model": args.model,
            "url": base_url,
            "context_lengths": CONTEXT_LENGTHS,
            "num_samples": NUM_SAMPLES,
            "output_tokens": OUTPUT_TOKENS,
            "num_experts": NUM_EXPERTS,
            "top_k": TOP_K,
            "moe_layers": MOE_LAYERS,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")
        },
        "probes": []
    }

    # Note: The Dell container may not return routed_experts via the API.
    # We'll try, and if not available, we'll compute what we can.
    print("NOTE: This probe measures routing behavior via the vLLM API.")
    print("If routed_experts are not available, we'll measure latency and infer routing stability.")
    print()

    for target_len in CONTEXT_LENGTHS:
        print(f"--- Context: ~{target_len} tokens ---")
        probe_results = []

        for sample in range(NUM_SAMPLES):
            context = generate_context(target_len)
            query = "\n\nWhat is the main topic? Answer in one sentence."
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": context + query}
            ]

            t_start = time.perf_counter()
            try:
                resp = client.chat.completions.create(
                    model=args.model, messages=messages,
                    max_tokens=OUTPUT_TOKENS, temperature=0.0,
                    stream=False
                )
                t_end = time.perf_counter()

                choice = resp.choices[0]
                usage = resp.usage

                # Try to get routed experts (if available)
                routed_experts = None
                if hasattr(choice, 'routed_experts') and choice.routed_experts:
                    routed_experts = choice.routed_experts

                result = {
                    "sample": sample,
                    "success": True,
                    "input_tokens": usage.prompt_tokens if usage else 0,
                    "actual_tokens": len(context + query) // 4,  # rough estimate
                    "output_tokens": usage.completion_tokens if usage else 0,
                    "latency_s": t_end - t_start,
                    "has_routed_experts": routed_experts is not None,
                }

                # Compute routing metrics if available
                if routed_experts:
                    # Extract per-layer load distributions
                    layer_entropies = {}
                    layer_top6 = {}
                    for layer_idx in MOE_LAYERS:
                        if layer_idx < len(routed_experts):
                            layer_data = routed_experts[layer_idx]
                            if isinstance(layer_data, list):
                                load = [0] * NUM_EXPERTS
                                for expert_id in layer_data:
                                    if 0 <= expert_id < NUM_EXPERTS:
                                        load[expert_id] += 1
                                layer_entropies[layer_idx] = compute_entropy(load)
                                layer_top6[layer_idx] = sorted(range(NUM_EXPERTS),
                                    key=lambda x: load[x], reverse=True)[:TOP_K]

                    if layer_entropies:
                        avg_entropy = sum(layer_entropies.values()) / len(layer_entropies)
                        result["avg_entropy"] = avg_entropy
                        result["layer_entropies"] = layer_entropies
                        result["layer_top6"] = {str(k): v for k, v in layer_top6.items()}

                probe_results.append(result)
                print(f"  Sample {sample+1}: {result['input_tokens']} in, "
                      f"{result['output_tokens']} out, {result['latency_s']:.2f}s"
                      + (f", entropy={result.get('avg_entropy', 'N/A')}" if result.get('avg_entropy') else ""))

            except Exception as e:
                t_end = time.perf_counter()
                probe_results.append({
                    "sample": sample,
                    "success": False,
                    "error": str(e)[:100],
                    "latency_s": t_end - t_start
                })
                print(f"  Sample {sample+1}: FAIL - {str(e)[:60]}")

            time.sleep(1)

        # Aggregate
        ok = [r for r in probe_results if r.get("success")]
        if ok:
            avg_lat = sum(r["latency_s"] for r in ok) / len(ok)
            avg_in = sum(r["input_tokens"] for r in ok) / len(ok)
            has_routing = any(r.get("has_routed_experts") for r in ok)
            avg_entropy = None
            if has_routing:
                entropies = [r["avg_entropy"] for r in ok if r.get("avg_entropy")]
                if entropies:
                    avg_entropy = sum(entropies) / len(entropies)

            print(f"  Aggregate: avg_input={avg_in:.0f}, avg_latency={avg_lat:.2f}s"
                  + (f", avg_entropy={avg_entropy:.3f}" if avg_entropy else "")
                  + (f", routing_available={has_routing}"))
        else:
            print("  Aggregate: ALL FAILED")

        results["probes"].append({
            "target_tokens": target_len,
            "trials": probe_results
        })
        print()

    # Save
    output_path = args.output or os.path.join(
        os.path.dirname(__file__), "results", "final", "nemotron",
        "mamba_experiments", "d2_routing_long_context.json"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {output_path}")

    # Summary
    print()
    print("=" * 70)
    print("D2 SUMMARY: Routing Behavior at Long Context")
    print("=" * 70)
    print(f"{'Context':>10} {'Input Tok':>10} {'Latency':>10} {'Entropy':>10} {'Routing':>10}")
    print("-" * 70)
    for probe in results["probes"]:
        ok = [r for r in probe["trials"] if r.get("success")]
        if ok:
            avg_in = sum(r["input_tokens"] for r in ok) / len(ok)
            avg_lat = sum(r["latency_s"] for r in ok) / len(ok)
            has_routing = any(r.get("has_routed_experts") for r in ok)
            entropies = [r["avg_entropy"] for r in ok if r.get("avg_entropy")]
            avg_ent = sum(entropies) / len(entropies) if entropies else None
            ent_str = f"{avg_ent:.3f}" if avg_ent else "N/A"
            print(f"{probe['target_tokens']:>10} {avg_in:>10.0f} {avg_lat:>10.2f} "
                  f"{ent_str:>10} "
                  f"{'Yes' if has_routing else 'No':>10}")


if __name__ == "__main__":
    main()
