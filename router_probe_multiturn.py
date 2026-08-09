#!/usr/bin/env python3
"""Multi-turn, longer-conversation router probe for Qwen3.5-MoE on GB10. V2.

FIXES from V1:
  - Strips prompt tokens from routed_experts using RequestOutput.prompt_token_ids
  - All analyses operate on OUTPUT TOKENS ONLY (no prompt contamination)
  - Decode trajectory pools across sequences at each position index (not per-token)
  - Position analysis pools across sequences by position bins (not stream-split)

Captures per-token routing at scale across text/random/reasoning to study:
  1. How routing evolves during long decode sequences (token-position effect)
  2. How routing changes across turn depth (turn 1 vs turn 3 vs turn 5)
  3. Multi-turn context accumulation effects on expert selection
  4. Cross-dataset routing divergence at different conversation depths
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


def load_sessions(path, limit):
    sessions = []
    try:
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                turns = d.get("turns", [])
                if len(turns) >= 1:
                    texts = [t["text"] for t in turns if isinstance(t, dict) and t.get("text")]
                    if texts:
                        sessions.append({
                            "session_id": d.get("session_id", ""),
                            "turns": texts,
                            "n_turns": len(texts),
                        })
                if len(sessions) >= limit:
                    break
    except Exception as e:
        print(f"warn: {path}: {e}", file=sys.stderr)
    return sessions


def entropy_of(hist, n_experts):
    """Normalized entropy from a histogram."""
    total = hist.sum()
    if total == 0:
        return 0.0
    p = hist.astype(np.float64) / total
    return -(p * np.log(p + 1e-12)).sum() / np.log(n_experts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text")
    ap.add_argument("--random")
    ap.add_argument("--reasoning")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu-mem", type=float, default=0.8)
    ap.add_argument("--n-experts", type=int, default=256)
    ap.add_argument("--max-turns", type=int, default=6)
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
        swap_space=0,
        load_format="fastsafetensors",
        enable_return_routed_experts=True,
    )

    sp = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)

    for name, path in [("text", args.text), ("random", args.random),
                       ("reasoning", args.reasoning)]:
        if not path:
            continue

        print(f"\n{'='*60}", file=sys.stderr)
        print(f"Dataset: {name}", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)

        sessions = load_sessions(path, args.limit)
        print(f"Loaded {len(sessions)} sessions", file=sys.stderr)

        # Build prompts with metadata
        all_prompts = []
        prompt_meta = []  # (session_idx, turn_idx, turn_depth, n_turns_in_session)
        by_turns = defaultdict(list)
        for s in sessions:
            by_turns[s["n_turns"]].append(s)

        for turn_depth in sorted(by_turns.keys()):
            sess_list = by_turns[turn_depth]
            for si, sess in enumerate(sess_list[:30]):
                history = ""
                for ti, turn_text in enumerate(sess["turns"]):
                    if ti >= min(turn_depth, args.max_turns):
                        break
                    history = turn_text if ti == 0 else history + "\n" + turn_text
                    all_prompts.append(history)
                    prompt_meta.append((si, ti, turn_depth, len(sess["turns"])))

        print(f"Total prompts: {len(all_prompts)}", file=sys.stderr)

        # Run inference
        BATCH_SIZE = 20
        output_routing = []  # list of (routing_array, meta)
        found = 0

        for batch_start in range(0, len(all_prompts), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(all_prompts))
            batch_prompts = all_prompts[batch_start:batch_end]
            batch_meta = prompt_meta[batch_start:batch_end]

            outs = llm.generate(batch_prompts, sp)

            for oi, o in enumerate(outs):
                prompt_len = len(o.prompt_token_ids) if o.prompt_token_ids else 0
                for co in o.outputs:
                    re_ = getattr(co, "routed_experts", None)
                    if re_ is not None:
                        arr = np.asarray(re_)
                        if arr.size:
                            arr = arr.reshape(-1, arr.shape[-2], arr.shape[-1])
                            # STRIP PROMPT TOKENS - keep only output tokens
                            if prompt_len > 0 and prompt_len < arr.shape[0]:
                                arr = arr[prompt_len:]
                            elif prompt_len >= arr.shape[0]:
                                continue  # no output tokens
                            output_routing.append((arr, batch_meta[oi]))
                            found += 1

            print(f"  batch {batch_start//BATCH_SIZE + 1}: {found} outputs so far", file=sys.stderr)

        print(f"\nTotal output routing arrays: {found}", file=sys.stderr)

        # Collect all output routing
        all_arrays = [r for r, m in output_routing]
        if not all_arrays:
            print(f"  NO DATA for {name}", file=sys.stderr)
            continue

        combined = np.concatenate(all_arrays, axis=0)
        n_tokens, n_layers, topk = combined.shape

        output = {
            "dataset": name,
            "n_sessions": len(sessions),
            "n_prompts": len(all_prompts),
            "n_output_arrays": found,
            "max_tokens": args.max_tokens,
            "n_layers": n_layers,
            "n_experts": args.n_experts,
            "topk": topk,
            "total_output_tokens": int(n_tokens),
        }

        # 1. Overall output-token entropy per layer
        global_loads = np.zeros((n_layers, args.n_experts), dtype=np.int64)
        for l in range(n_layers):
            ids = combined[:, l, :].reshape(-1)
            np.add.at(global_loads[l], ids, 1)
        entropies = [float(entropy_of(global_loads[l], args.n_experts)) for l in range(n_layers)]
        output["overall"] = {
            "total_tokens": int(n_tokens),
            "shape": list(combined.shape),
            "entropy_per_layer": entropies,
            "mean_entropy": float(np.mean(entropies)),
        }

        # 2. Decode position analysis (pooled across sequences by position index)
        max_len = args.max_tokens
        position_loads = [np.zeros((n_layers, args.n_experts), dtype=np.int64) for _ in range(max_len)]
        position_token_counts = [0] * max_len
        for arr in all_arrays:
            for i in range(min(arr.shape[0], max_len)):
                for l in range(n_layers):
                    ids = arr[i, l, :]
                    np.add.at(position_loads[i][l], ids, 1)
                position_token_counts[i] += 1

        # Bin into early (0..max_len/4), mid (max_len/4..3*max_len/4), late (3*max_len/4..end)
        e = max_len // 4
        bins = {
            "early": list(range(0, e)),
            "mid": list(range(e, 3*e)),
            "late": list(range(3*e, max_len)),
        }
        position_analysis = {}
        for bin_name, positions in bins.items():
            bin_loads = np.zeros((n_layers, args.n_experts), dtype=np.int64)
            n_tok = 0
            for pos in positions:
                if pos < len(position_loads):
                    bin_loads += position_loads[pos]
                    n_tok += position_token_counts[pos]
            bin_ents = [float(entropy_of(bin_loads[l], args.n_experts)) for l in range(n_layers)]
            position_analysis[bin_name] = {
                "n_tokens": int(n_tok),
                "n_sequences": sum(1 for arr in all_arrays if arr.shape[0] > positions[0]),
                "mean_entropy": float(np.mean(bin_ents)),
                "std_entropy": float(np.std(bin_ents)),
                "entropy_per_layer": bin_ents,
            }
        output["position_analysis"] = position_analysis

        # 3. Decode trajectory: use position_loads from step 2
        traj = {}
        sampled_positions = [0, max_len//4, max_len//2, 3*max_len//4, max_len-1]
        pos_labels = ["first", "quarter", "half", "three_quarter", "last"]
        for label, pos in zip(pos_labels, sampled_positions):
            if pos >= len(position_loads) or position_token_counts[pos] == 0:
                continue
            layer_ents = []
            for l in range(n_layers):
                layer_ents.append(float(entropy_of(position_loads[pos][l], args.n_experts)))
            traj[label] = {
                "mean": float(np.mean(layer_ents)),
                "std": float(np.std(layer_ents)),
                "min": float(np.min(layer_ents)),
                "max": float(np.max(layer_ents)),
                "n_sequences": position_token_counts[pos],
            }
        output["decode_trajectory"] = traj

        # 4. By turn depth (output tokens only)
        by_depth_arrays = defaultdict(list)
        for arr, meta in output_routing:
            si, ti, turn_depth, total_turns = meta
            by_depth_arrays[f"depth{turn_depth}"].append(arr)

        depth_analysis = {}
        for dk in sorted(by_depth_arrays.keys()):
            arrs = by_depth_arrays[dk]
            depth_combined = np.concatenate(arrs, axis=0)
            d_loads = np.zeros((n_layers, args.n_experts), dtype=np.int64)
            for l in range(n_layers):
                ids = depth_combined[:, l, :].reshape(-1)
                np.add.at(d_loads[l], ids, 1)
            d_ents = [float(entropy_of(d_loads[l], args.n_experts)) for l in range(n_layers)]
            depth_analysis[dk] = {
                "n_tokens": int(depth_combined.shape[0]),
                "n_arrays": len(arrs),
                "mean_entropy": float(np.mean(d_ents)),
                "entropy_per_layer": d_ents,
            }
        output["by_turn_depth"] = depth_analysis

        # 5. Cross-depth expert overlap (top-8 experts per layer)
        depth_top8 = {}
        for dk, arrs in by_depth_arrays.items():
            depth_combined = np.concatenate(arrs, axis=0)
            top8_per_layer = []
            for l in range(n_layers):
                hist = np.zeros(args.n_experts, dtype=np.int64)
                ids = depth_combined[:, l, :].reshape(-1)
                np.add.at(hist, ids, 1)
                top8 = set(np.argsort(hist)[-8:].tolist())
                top8_per_layer.append(top8)
            depth_top8[dk] = top8_per_layer

        overlap_matrix = {}
        depths = sorted(depth_top8.keys(), key=lambda x: int(x.replace("depth", "")))
        for i, d1 in enumerate(depths):
            for d2 in depths[i:]:
                jaccards = []
                for l in range(n_layers):
                    s1 = depth_top8[d1][l]
                    s2 = depth_top8[d2][l]
                    if s1 or s2:
                        jaccards.append(len(s1 & s2) / len(s1 | s2))
                overlap_matrix[f"{d1}_vs_{d2}"] = {
                    "mean_jaccard": float(np.mean(jaccards)) if jaccards else 0,
                }
        output["cross_depth_overlap"] = overlap_matrix

        # 6. Session-level routing drift (turn 0 vs turn 2, output tokens only)
        session_turn_routing = defaultdict(lambda: defaultdict(list))
        for arr, meta in output_routing:
            si, ti, _, _ = meta
            session_turn_routing[si][ti].append(arr)

        multi_turn_compare = []
        for si in sorted(session_turn_routing.keys()):
            turns = session_turn_routing[si]
            if len(turns) >= 3:
                t0 = np.concatenate(turns[0], axis=0)
                t2 = np.concatenate(turns[2], axis=0)
                jaccards = []
                for l in range(n_layers):
                    h0 = np.zeros(args.n_experts, dtype=np.int64)
                    h2 = np.zeros(args.n_experts, dtype=np.int64)
                    np.add.at(h0, t0[:, l, :].reshape(-1), 1)
                    np.add.at(h2, t2[:, l, :].reshape(-1), 1)
                    s0 = set(np.argsort(h0)[-8:].tolist())
                    s2 = set(np.argsort(h2)[-8:].tolist())
                    jaccards.append(len(s0 & s2) / len(s0 | s2) if s0 or s2 else 0)
                multi_turn_compare.append({
                    "session_idx": si,
                    "n_turns": len(turns),
                    "turn0_tokens": int(t0.shape[0]),
                    "turn2_tokens": int(t2.shape[0]),
                    "turn0_vs_turn2_jaccard": float(np.mean(jaccards)),
                })
        output["session_turn_drift"] = {
            "n_sessions_with_3plus_turns": len(multi_turn_compare),
            "mean_jaccard_turn0_vs_turn2": float(np.mean([s["turn0_vs_turn2_jaccard"] for s in multi_turn_compare])) if multi_turn_compare else 0,
            "per_session": multi_turn_compare[:20],
        }

        # Write output
        with open(f"{args.out}.{name}.json", "w") as f:
            json.dump(output, f)
        print(f"\nWrote {args.out}.{name}.json ({n_tokens} output tokens)", file=sys.stderr)


if __name__ == "__main__":
    main()
