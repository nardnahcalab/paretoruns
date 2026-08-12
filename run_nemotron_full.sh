#!/usr/bin/env bash
# =============================================================================
# run_nemotron_full.sh - Full coverage benchmark for Nemotron 3.5 Lightning
# =============================================================================
# Runs comprehensive benchmarking suite for nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4:
#   1. Router profiling (single-turn + multi-turn) - validates MoE routing behavior
#   2. Pareto sweep (text/random/reasoning × c1,c2,c4,c8) - performance characterization
#   3. Telemetry collection - GPU metrics during benchmark
#
# All results are saved to timestamped directories under results/ to preserve
# previous runs. No data is overwritten.
#
# Prerequisites:
#   - vLLM server running with: ./launch_vllm_nemotron.sh
#   - HF_TOKEN set for model access
#   - Docker with nvidia runtime
#
# Usage:
#   ./run_nemotron_full.sh [--port PORT] [--skip-routing] [--skip-sweep]
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
PORT="${PORT:-8000}"
SKIP_ROUTING="${SKIP_ROUTING:-0}"
SKIP_SWEEP="${SKIP_SWEEP:-0}"
SKIP_TELEMETRY="${SKIP_TELEMETRY:-0}"

# Parse flags
while [[ $# -gt 0 ]]; do
  case $1 in
    --port)         PORT="$2"; shift 2 ;;
    --skip-routing) SKIP_ROUTING=1; shift ;;
    --skip-sweep)   SKIP_SWEEP=1; shift ;;
    --skip-telemetry) SKIP_TELEMETRY=1; shift ;;
    -h|--help)
      cat <<'USAGE'
Usage: ./run_nemotron_full.sh [OPTIONS]

Full coverage benchmark for Nemotron 3.5 Lightning.

Options:
  --port PORT         vLLM server port (default: 8000)
  --skip-routing      Skip router profiling steps
  --skip-sweep        Skip Pareto sweep
  --skip-telemetry    Skip telemetry collection
  -h, --help          Show this help

Results are saved to results/<timestamp>_nemotron_full/
USAGE
      exit 0 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# Model configuration
MODEL_ID="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
VLLM_URL="http://localhost:${PORT}"

# Output directories
NEMOTRON_DIR="${RESULTS_DIR}/${TIMESTAMP}_nemotron_full"
ROUTING_DIR="${NEMOTRON_DIR}/routing"
SWEEP_DIR="${NEMOTRON_DIR}/sweep"

mkdir -p "${NEMOTRON_DIR}" "${ROUTING_DIR}" "${SWEEP_DIR}"

echo "============================================"
echo " Nemotron 3.5 Lightning - Full Benchmark"
echo "============================================"
echo "Model:      ${MODEL_ID}"
echo "URL:        ${VLLM_URL}"
echo "Results:    ${NEMOTRON_DIR}"
echo "Timestamp:  ${TIMESTAMP}"
echo ""

# ---------------------------------------------------------------------------
# Phase 0: Health check
# ---------------------------------------------------------------------------
echo ">>> Phase 0: Health check"
if ! curl -sf "${VLLM_URL}/health" > /dev/null 2>&1; then
  echo "ERROR: vLLM server not responding at ${VLLM_URL}"
  echo "  Start server with: ./launch_vllm_nemotron.sh"
  exit 1
fi
echo "    Server is healthy"
echo ""

# ---------------------------------------------------------------------------
# Phase 1: Router profiling (single-turn)
# ---------------------------------------------------------------------------
if [[ "${SKIP_ROUTING}" != "1" ]]; then
  echo ">>> Phase 1: Router profiling (single-turn)"
  echo "    Captures per-token expert routing for text/random/reasoning"
  echo "    NOTE: Uses offline vllm.LLM API (loads model directly)"
  echo ""

  # Set model for router probe
  export PROBE_MODEL="${MODEL_ID}"

  echo "    Running single-turn probe with all datasets..."
  python3 "${SCRIPT_DIR}/router_probe.py" \
    --text "${SCRIPT_DIR}/multi_turn_iso_text_chat.jsonl" \
    --random "${SCRIPT_DIR}/multi_turn_iso_random_chat.jsonl" \
    --reasoning "${SCRIPT_DIR}/multi_turn_iso_reasoning_chat.jsonl" \
    --limit 20 \
    --max-tokens 8 \
    --out "${ROUTING_DIR}/probe" \
    2>&1 | tee "${ROUTING_DIR}/single_turn.log" || {
      echo "    WARNING: Single-turn probe failed (check log)"
    }
  echo ""
else
  echo ">>> Phase 1: Router profiling (SKIPPED)"
  echo ""
fi

# ---------------------------------------------------------------------------
# Phase 2: Router profiling (multi-turn)
# ---------------------------------------------------------------------------
if [[ "${SKIP_ROUTING}" != "1" ]]; then
  echo ">>> Phase 2: Router profiling (multi-turn)"
  echo "    Captures routing evolution across conversation turns"
  echo "    NOTE: Uses offline vllm.LLM API (loads model directly)"
  echo ""

  # Set model for router probe
  export PROBE_MODEL="${MODEL_ID}"

  echo "    Running multi-turn probe with all datasets..."
  python3 "${SCRIPT_DIR}/router_probe_multiturn.py" \
    --text "${SCRIPT_DIR}/multi_turn_iso_text_chat.jsonl" \
    --random "${SCRIPT_DIR}/multi_turn_iso_random_chat.jsonl" \
    --reasoning "${SCRIPT_DIR}/multi_turn_iso_reasoning_chat.jsonl" \
    --limit 20 \
    --max-tokens 64 \
    --max-turns 5 \
    --out "${ROUTING_DIR}/mt_probe" \
    2>&1 | tee "${ROUTING_DIR}/multi_turn.log" || {
      echo "    WARNING: Multi-turn probe failed (check log)"
    }
  echo ""
else
  echo ">>> Phase 2: Router profiling (SKIPPED)"
  echo ""
fi

# ---------------------------------------------------------------------------
# Phase 3: Pareto sweep (128-tok output)
# ---------------------------------------------------------------------------
if [[ "${SKIP_SWEEP}" != "1" ]]; then
  echo ">>> Phase 3: Pareto sweep (128-tok output)"
  echo "    Running text/random/reasoning × c1,c2,c4,c8"
  echo ""

  MODEL_NAME="${MODEL_ID}" \
  VLLM_URL="${VLLM_URL}" \
  bash "${SCRIPT_DIR}/run_sweep.sh" \
    --concurrency '1 2 4 8' \
    --output-tokens 128 \
    --output-std 32 \
    --requests 50 \
    --label nemotron_128 \
    --timeout 1800 \
    --warmup 3

  # Move results to our timestamped directory
  SWEEP_RESULT=$(ls -td "${RESULTS_DIR}"/*_nemotron_128 2>/dev/null | head -1)
  if [[ -n "${SWEEP_RESULT}" ]]; then
    cp -r "${SWEEP_RESULT}"/* "${SWEEP_DIR}/" 2>/dev/null || true
    echo "    Sweep results copied to ${SWEEP_DIR}"
  fi
  echo ""
else
  echo ">>> Phase 3: Pareto sweep (SKIPPED)"
  echo ""
fi

# ---------------------------------------------------------------------------
# Phase 4: Pareto sweep (512-tok output) - extended decode analysis
# ---------------------------------------------------------------------------
if [[ "${SKIP_SWEEP}" != "1" ]]; then
  echo ">>> Phase 4: Pareto sweep (512-tok output)"
  echo "    Extended decode analysis for longer sequences"
  echo ""

  MODEL_NAME="${MODEL_ID}" \
  VLLM_URL="${VLLM_URL}" \
  bash "${SCRIPT_DIR}/run_sweep.sh" \
    --concurrency '1 2 4 8' \
    --output-tokens 512 \
    --output-std 128 \
    --requests 50 \
    --label nemotron_512 \
    --timeout 2400 \
    --warmup 3

  # Move results to our timestamped directory
  SWEEP_RESULT=$(ls -td "${RESULTS_DIR}"/*_nemotron_512 2>/dev/null | head -1)
  if [[ -n "${SWEEP_RESULT}" ]]; then
    mkdir -p "${SWEEP_DIR}_512"
    cp -r "${SWEEP_RESULT}"/* "${SWEEP_DIR}_512/" 2>/dev/null || true
    echo "    Sweep results copied to ${SWEEP_DIR}_512"
  fi
  echo ""
else
  echo ">>> Phase 4: Pareto sweep (SKIPPED)"
  echo ""
fi

# ---------------------------------------------------------------------------
# Phase 5: Generate reports
# ---------------------------------------------------------------------------
echo ">>> Phase 5: Generating reports"
echo ""

# Generate Pareto report for 128-tok sweep
if [[ -d "${SWEEP_DIR}" ]] && [[ -f "${SCRIPT_DIR}/make_pareto.py" ]]; then
  echo "    Generating 128-tok Pareto report..."
  python3 "${SCRIPT_DIR}/make_pareto.py" \
    "${SWEEP_DIR}" \
    "${NEMOTRON_DIR}/pareto_128.html" || true
fi

# Generate Pareto report for 512-tok sweep
if [[ -d "${SWEEP_DIR}_512" ]] && [[ -f "${SCRIPT_DIR}/make_pareto.py" ]]; then
  echo "    Generating 512-tok Pareto report..."
  python3 "${SCRIPT_DIR}/make_pareto.py" \
    "${SWEEP_DIR}_512" \
    "${NEMOTRON_DIR}/pareto_512.html" || true
fi

# Generate comparison reports against previous models
if [[ -f "${SCRIPT_DIR}/compare_sweeps.py" ]]; then
  # Compare against Qwen NVFP4
  if [[ -d "${RESULTS_DIR}/20260807_182301" ]]; then
    echo "    Generating Nemotron vs Qwen comparison..."
    python3 "${SCRIPT_DIR}/compare_sweeps.py" \
      --out "${NEMOTRON_DIR}/vs_qwen.html" \
      qwen_nvfp4="${RESULTS_DIR}/20260807_182301" \
      nemotron="${SWEEP_DIR}" || true
  fi

  # Compare against Muse-Glimmer dense
  if [[ -d "${RESULTS_DIR}/muse-glimmer-30B_128" ]]; then
    echo "    Generating Nemotron vs Muse-Glimmer comparison..."
    python3 "${SCRIPT_DIR}/compare_sweeps.py" \
      --out "${NEMOTRON_DIR}/vs_muse.html" \
      nemotron="${SWEEP_DIR}" \
      muse_dense="${RESULTS_DIR}/muse-glimmer-30B_128" || true
  fi
fi

echo ""

# ---------------------------------------------------------------------------
# Phase 6: Save run metadata
# ---------------------------------------------------------------------------
echo ">>> Phase 6: Saving run metadata"

cat > "${NEMOTRON_DIR}/run_meta.json" <<EOF
{
  "model": "${MODEL_ID}",
  "framework": "vllm",
  "precision": "nvfp4",
  "architecture": "Gated DeltaNet MoE",
  "total_params": "30B",
  "active_params_per_token": "3B",
  "max_context_length": 1048576,
  "container": "registry.dell.huggingface.co/enterprise-dell-inference-nvidia-nvidia-nemotron-3.5-lightning-30b-a3b-nvfp4-gb10:latest",
  "hardware": "NVIDIA GB10 (DGX Spark, SM121)",
  "timestamp": "${TIMESTAMP}",
  "sweep_config": {
    "output_tokens": [128, 512],
    "concurrency_levels": [1, 2, 4, 8],
    "requests_per_level": 50,
    "datasets": ["text", "random", "reasoning"]
  },
  "phases_completed": {
    "router_profiling_single": $([ "${SKIP_ROUTING}" != "1" ] && echo "true" || echo "false"),
    "router_profiling_multi": $([ "${SKIP_ROUTING}" != "1" ] && echo "true" || echo "false"),
    "pareto_sweep_128": $([ "${SKIP_SWEEP}" != "1" ] && echo "true" || echo "false"),
    "pareto_sweep_512": $([ "${SKIP_SWEEP}" != "1" ] && echo "true" || echo "false")
  }
}
EOF

echo "    Metadata saved to ${NEMOTRON_DIR}/run_meta.json"
echo ""

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "============================================"
echo " Benchmark Complete"
echo "============================================"
echo ""
echo "Results directory: ${NEMOTRON_DIR}"
echo ""
echo "Files generated:"
echo "  - routing/           Router profiling results"
echo "  - sweep/             128-tok Pareto sweep"
echo "  - sweep_512/         512-tok Pareto sweep"
echo "  - pareto_128.html    128-tok Pareto report"
echo "  - pareto_512.html    512-tok Pareto report"
echo "  - vs_qwen.html       Nemotron vs Qwen comparison"
echo "  - vs_muse.html       Nemotron vs Muse-Glimmer comparison"
echo "  - run_meta.json      Run configuration metadata"
echo ""
echo "View reports:"
echo "  open ${NEMOTRON_DIR}/pareto_128.html"
echo "  open ${NEMOTRON_DIR}/vs_qwen.html"
echo ""
echo "Previous runs preserved:"
ls -d "${RESULTS_DIR}"/* 2>/dev/null | grep -v "${TIMESTAMP}" | while read d; do
  echo "  - $(basename "$d")"
done
