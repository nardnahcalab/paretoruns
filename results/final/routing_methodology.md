# Routing Probe Methodology

## Overview

This document describes the methodology, scripts, and parameters used for expert routing analysis on MoE models running on NVIDIA GB10 (DGX Spark) hardware.

## Scripts

### 1. `router_probe.py` - Single-Turn Routing Analysis

**Purpose:** Captures per-token expert selection from vLLM's `enable_return_routed_experts` flag and computes routing statistics.

**Key Features:**
- Loads prompts from JSONL datasets (text, random, reasoning)
- Runs inference with `enable_return_routed_experts=True`
- Extracts `routed_experts` from completion outputs
- Computes per-layer expert load histograms
- Calculates normalized entropy (0=focused, 1=uniform)
- Identifies top expert share per layer

**Usage:**
```bash
PROBE_MODEL=nvidia/Qwen3.6-35B-A3B-FP8 \
python3 router_probe.py \
  --text multi_turn_iso_text_chat.jsonl \
  --random multi_turn_iso_random_chat.jsonl \
  --reasoning multi_turn_iso_reasoning_chat.jsonl \
  --limit 20 \
  --max-tokens 8 \
  --max-len 512 \
  --gpu-mem 0.5 \
  --n-experts 256 \
  --out /path/to/output/probe
```

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--text` | - | Path to text dataset JSONL |
| `--random` | - | Path to random dataset JSONL |
| `--reasoning` | - | Path to reasoning dataset JSONL |
| `--limit` | 20 | Max prompts per dataset |
| `--max-tokens` | 8 | Max output tokens per prompt |
| `--max-len` | 512 | Max sequence length |
| `--gpu-mem` | 0.5 | GPU memory utilization (0-1) |
| `--n-experts` | 256 | Number of experts in MoE layer |
| `--experts-per-tok` | 8 | Experts selected per token |
| `--out` | - | Output path prefix |

**Output Files:**
- `{out}.text.json` - Text dataset routing statistics
- `{out}.random.json` - Random dataset routing statistics
- `{out}.reasoning.json` - Reasoning dataset routing statistics

**JSON Structure:**
```json
{
  "summary": {
    "n_tokens": 517,
    "n_layers": 52,
    "topk": 6
  },
  "per_layer": [
    {
      "layer": 0,
      "entropy_norm": 0.771,
      "top_expert": 108,
      "top_share": 0.071,
      "load": [3102, 0, 0, ...]
    },
    ...
  ]
}
```

---

### 2. `router_probe_multiturn.py` - Multi-Turn Routing Analysis

**Purpose:** Extended analysis of routing behavior across conversation turns and sequence positions.

**Key Features:**
- Loads multi-turn sessions (not just single prompts)
- Strips prompt tokens from routed_experts (output-only analysis)
- Position analysis (early/mid/late in sequence)
- Turn depth analysis (turn 1 vs turn 3 vs turn 5)
- Cross-depth expert overlap (Jaccard similarity)
- Session-level routing drift

**Usage:**
```bash
PROBE_MODEL=nvidia/Qwen3.6-35B-A3B-FP8 \
python3 router_probe_multiturn.py \
  --text multi_turn_iso_text_chat.jsonl \
  --random multi_turn_iso_random_chat.jsonl \
  --reasoning multi_turn_iso_reasoning_chat.jsonl \
  --limit 200 \
  --max-tokens 64 \
  --max-len 4096 \
  --gpu-mem 0.8 \
  --n-experts 256 \
  --max-turns 6 \
  --out /path/to/output/mt_probe
```

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--text` | - | Path to text dataset JSONL |
| `--random` | - | Path to random dataset JSONL |
| `--reasoning` | - | Path to reasoning dataset JSONL |
| `--limit` | 200 | Max sessions per dataset |
| `--max-tokens` | 64 | Max output tokens per prompt |
| `--max-len` | 4096 | Max sequence length |
| `--gpu-mem` | 0.8 | GPU memory utilization (0-1) |
| `--n-experts` | 256 | Number of experts in MoE layer |
| `--max-turns` | 6 | Max turns to process per session |
| `--out` | - | Output path prefix |

**Output Files:**
- `{out}.text.json` - Text dataset multi-turn statistics
- `{out}.random.json` - Random dataset multi-turn statistics
- `{out}.reasoning.json` - Reasoning dataset multi-turn statistics

**JSON Structure:**
```json
{
  "dataset": "text",
  "n_sessions": 20,
  "n_prompts": 55,
  "n_output_arrays": 55,
  "max_tokens": 64,
  "n_layers": 52,
  "n_experts": 256,
  "topk": 6,
  "total_output_tokens": 3465,
  "overall": {
    "total_tokens": 3465,
    "shape": [3465, 52, 6],
    "entropy_per_layer": [...],
    "mean_entropy": 0.358
  },
  "position_analysis": {
    "early": {"n_tokens": 880, "mean_entropy": 0.348, ...},
    "mid": {"n_tokens": 2640, "mean_entropy": 0.359, ...},
    "late": {"n_tokens": 880, "mean_entropy": 0.361, ...}
  },
  "decode_trajectory": {
    "first": {"mean": 0.34, ...},
    "quarter": {"mean": 0.35, ...},
    "half": {"mean": 0.36, ...},
    "three_quarter": {"mean": 0.36, ...},
    "last": {"mean": 0.37, ...}
  },
  "by_turn_depth": {
    "depth0": {"n_tokens": 1000, "mean_entropy": 0.35, ...},
    "depth1": {"n_tokens": 800, "mean_entropy": 0.36, ...},
    ...
  },
  "cross_depth_overlap": {
    "depth0_vs_depth0": {"mean_jaccard": 1.0},
    "depth0_vs_depth1": {"mean_jaccard": 0.85},
    ...
  },
  "session_turn_drift": {
    "n_sessions_with_3plus_turns": 15,
    "mean_jaccard_turn0_vs_turn2": 0.82,
    "per_session": [...]
  }
}
```

---

## Metrics Explained

### Entropy (Normalized)
- **Range:** 0 to 1
- **Definition:** `-(p * log(p)).sum() / log(n_experts)` where p is the probability distribution over experts
- **Interpretation:**
  - **0.0:** All tokens route to same expert (completely focused)
  - **0.5:** Moderate diversity
  - **1.0:** Uniform distribution across all experts (maximum diversity)

### Top Expert Share
- **Range:** 0% to 100%
- **Definition:** Percentage of tokens routed to the most popular expert in a layer
- **Interpretation:**
  - **High (>20%):** Potential load imbalance, hotspot expert
  - **Medium (5-20%):** Normal routing behavior
  - **Low (<5%):** Very uniform distribution

### Position Analysis
- **Early:** First 25% of output tokens
- **Mid:** Middle 50% of output tokens
- **Late:** Last 25% of output tokens

### Turn Depth
- **Depth 0:** First turn in conversation (single prompt)
- **Depth 1:** Second turn (prompt + first response + second prompt)
- **Depth N:** N+1 turn in conversation

### Cross-Depth Jaccard
- **Range:** 0 to 1
- **Definition:** Jaccard similarity of top-8 experts between turn depths
- **Interpretation:**
  - **1.0:** Identical expert selection
  - **0.8+:** High overlap (routing is consistent)
  - **<0.5:** Low overlap (routing changes significantly)

---

## Dataset Format

All datasets use JSONL format with one of two structures:

### Structure 1: Multi-turn with turns array
```json
{
  "session_id": "abc123",
  "turns": [
    {"text": "First user message"},
    {"text": "Second user message"},
    {"text": "Third user message"}
  ]
}
```

### Structure 2: Messages format
```json
{
  "messages": [
    {"role": "user", "content": "User message"},
    {"role": "assistant", "content": "Assistant response"},
    {"role": "user", "content": "Next user message"}
  ]
}
```

---

## Hardware Requirements

- **Platform:** NVIDIA GB10 (DGX Spark)
- **GPU Memory:** 50% for single-turn, 80% for multi-turn
- **Storage:** Results stored in `/home/bala/pareto/results/`

## Environment Variables

```bash
# Required for MoE models
export VLLM_MARLIN_USE_ATOMIC_ADD=1
export VLLM_USE_FLASHINFER_MOE_FP4=0
export HF_HUB_DISABLE_XET=1

# Model selection (for multi-model testing)
export PROBE_MODEL=nvidia/Qwen3.6-35B-A3B-FP8
```

---

## Known Limitations

1. **vLLM Dependency:** Requires vLLM Python module (not available in host environment)
2. **Memory Constraints:** Large models may require reduced `--limit` or `--gpu-mem`
3. **Dense Models:** Muse (Dense Transformer) does not have routing - probe returns no data
4. **Precision Effects:** NVFP4 may produce slightly different routing than FP8/bf16

---

## Reproducing Results

### For Nemotron 3.5 Lightning:
```bash
# Start server
./launch_vllm_nemotron.sh

# Single-turn probe
PROBE_MODEL=nvidia/Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
python3 router_probe.py \
  --text multi_turn_iso_text_chat.jsonl \
  --random multi_turn_iso_random_chat.jsonl \
  --reasoning multi_turn_iso_reasoning_chat.jsonl \
  --limit 20 \
  --out /home/bala/pareto/results/router_probe_nemotron/probe

# Multi-turn probe
PROBE_MODEL=nvidia/Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
python3 router_probe_multiturn.py \
  --text multi_turn_iso_text_chat.jsonl \
  --random multi_turn_iso_random_chat.jsonl \
  --reasoning multi_turn_iso_reasoning_chat.jsonl \
  --limit 200 \
  --out /home/bala/pareto/results/router_probe_nemotron/mt_probe
```

### For Qwen Models:
```bash
# FP8 variant
./launch_vllm_qwen_fp8.sh
PROBE_MODEL=nvidia/Qwen3.6-35B-A3B-FP8 \
python3 router_probe.py \
  --text multi_turn_iso_text_chat.jsonl \
  --random multi_turn_iso_random_chat.jsonl \
  --reasoning multi_turn_iso_reasoning_chat.jsonl \
  --limit 20 \
  --out /home/bala/pareto/results/final/qwen_fp8/routing/probe

# NVFP4 variant
./launch_vllm_qwen_nvfp4.sh
PROBE_MODEL=nvidia/Qwen3.6-35B-A3B-NVFP4 \
python3 router_probe.py \
  --text multi_turn_iso_text_chat.jsonl \
  --random multi_turn_iso_random_chat.jsonl \
  --reasoning multi_turn_iso_reasoning_chat.jsonl \
  --limit 20 \
  --out /home/bala/pareto/results/final/qwen_nvfp4/routing/probe
```

---

## References

- vLLM documentation: `enable_return_routed_experts` flag
- Qwen3.6 architecture: 256 experts, top-8 routing
- Nemotron 3.5 Lightning: Hybrid Mamba-MoE, 256 experts, top-6 routing
- Muse-Glimmer-30B: Dense Transformer, no MoE routing
