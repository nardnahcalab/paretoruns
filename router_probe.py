#!/usr/bin/env python3
"""Offline router-trace probe for Qwen3.5-MoE on GB10.

Uses vLLM's built-in `enable_return_routed_experts` to capture the exact per-
token top-8 expert IDs chosen by the real fused kernels, then aggregates
routing statistics per dataset (text / random / reasoning):
  - expert load histograms per layer
  - per-layer selection entropy + top-expert share
  - inter-dataset expert-load divergence (for correlated content effect)
Writes one JSON per dataset plus a combined summary.
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

os.environ.setdefault("VLLM_MARLIN_USE_ATOMIC_ADD", "1")
os.environ.setdefault("VLLM_USE_FLASHINFER_MOE_FP4", "0")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def load_prompts(path, limit):
    prompts = []
    try:
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                turns = d.get("turns")
                if isinstance(turns, list):
                    for t in turns:
                        txt = t.get("text") if isinstance(t, dict) else None
                        if txt:
                            prompts.append(txt)
                elif isinstance(d.get("messages"), list):
                    for m in d["messages"]:
                        if m.get("role") == "user":
                            prompts.append(m["content"])
                            break
                if len(prompts) >= limit:
                    break
    except Exception as e:
        print(f"warn: {path}: {e}", file=sys.stderr)
    return prompts


def analyze(routing: np.ndarray, n_experts: int):
    """routing: (n_tokens, n_layers, topk)."""
    n_tokens, n_layers, topk = routing.shape
    loads = np.zeros((n_layers, n_experts), dtype=np.int64)
    r = routing.reshape(-1, n_layers, topk)  # per-position
    for l in range(n_layers):
        layer_ids = r[:, l, :].reshape(-1)
        np.add.at(loads[l], layer_ids, 1)

    out = {
        "n_tokens": n_tokens,
        "n_layers": n_layers,
        "topk": topk,
    }
    # per-layer stats
    per_layer = []
    P = np.zeros((n_layers, n_experts), dtype=np.float64)
    for l in range(n_layers):
        hist = loads[l].astype(np.float64)
        if hist.sum() > 0:
            p = hist / hist.sum()
        else:
            p = np.ones(n_experts) / n_experts
        P[l] = p
        ent = -(p * np.log(p + 1e-12)).sum() / np.log(n_experts)  # normalized
        top_expert = int(np.argmax(hist)) if hist.sum() else -1
        top_share = float(hist.max() / hist.sum()) if hist.sum() else 0.0
        per_layer.append({
            "layer": l,
            "entropy_norm": float(ent),
            "top_expert": top_expert,
            "top_share": top_share,
            "load": [int(x) for x in hist],
        })
    out = {
        "summary": {"n_tokens": n_tokens, "n_layers": n_layers, "topk": topk},
        "per_layer": per_layer,
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text")
    ap.add_argument("--random")
    ap.add_argument("--reasoning")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu-mem", type=float, default=0.5)
    ap.add_argument("--n-experts", type=int, default=256)
    ap.add_argument("--experts-per-tok", type=int, default=8)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    model = os.environ.get("PROBE_MODEL", "nvidia/Qwen3.6-35B-A3B-NVFP4")
    llm = LLM(
        model=model,
        tensor_parallel_size=1,
        trust_remote_code=True,
        moe_backend="marlin",
        kv_cache_dtype="fp8_e4m3",
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_len,
        enforce_eager=True,
        load_format="auto",
        enable_return_routed_experts=True,
    )

    sp = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)
    for name, path in (("text", args.text), ("random", args.random),
                       ("reasoning", args.reasoning)):
        if not path:
            continue
        prompts = load_prompts(path, args.limit)
        if not prompts:
            print(f"no prompts for {name}", file=sys.stderr)
            continue
        outs = llm.generate(prompts, sp)
        found = 0
        for o in outs:
            for co in o.outputs:
                re_ = getattr(co, "routed_experts", None)
                if re_ is not None:
                    found += 1
                    break
        print(f"{name}: {len(prompts)} prompts, routed_experts present on "
              f"{found} completion outputs", file=sys.stderr)
        # gather any non-None array
        arrays = []
        for o in outs:
            for co in o.outputs:
                re_ = getattr(co, "routed_experts", None)
                if re_ is None:
                    continue
                arr = np.asarray(re_)
                if arr.size:
                    arrays.append(arr.reshape(-1, arr.shape[-2], arr.shape[-1]))
        if not arrays:
            print(f"{name}: NO routed data (check enable flag)", file=sys.stderr)
            continue
        routing = np.concatenate(arrays, axis=0)
        summary = analyze(routing, args.n_experts)
        with open(f"{args.out}.{name}.json", "w") as f:
            json.dump(summary, f)
        print(f"{name}: wrote {args.out}.{name}.json routing={routing.shape}",
              file=sys.stderr)


if __name__ == "__main__":
    main()