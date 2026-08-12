#!/usr/bin/env bash
# =============================================================================
# run_model_benchmark.sh - Full benchmark for any supported model
# =============================================================================
# Runs comprehensive benchmarking suite for a given model:
#   1. Router profiling (single-turn + multi-turn)
#   2. Pareto sweep (text/random/reasoning × c1,c2,c4,c8)
#   3. Comparison reports
#
# Usage:
#   ./run_model_benchmark.sh --model MODEL [--port PORT] [--skip-routing] [--skip-sweep]
#
# Supported models:
#   muse          - meta-models/Muse-Glimmer-30B
#   qwen-nvfp4    - nvidia/Qwen3.6-35B-A3B-NVFP4
#   qwen-fp8      - Qwen/Qwen3.6-35B-A3B-FP8
#   nemotron      - nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
PORT="${PORT:-8000}"
SKIP_ROUTING="${SKIP_ROUTING:-0}"
SKIP_SWEEP="${SKIP_SWEEP:-0}"
MODEL=""

# Parse flags
while [[ $# -gt 0 ]]; do
  case $1 in
    --model)        MODEL="$2"; shift 2 ;;
    --port)         PORT="$2"; shift 2 ;;
    --skip-routing) SKIP_ROUTING=1; shift ;;
    --skip-sweep)   SKIP_SWEEP=1; shift ;;
    -h|--help)
      cat <<'USAGE'
Usage: ./run_model_benchmark.sh [OPTIONS]

Full coverage benchmark for supported models.

Options:
  --model MODEL       Model to benchmark (muse, qwen-nvfp4, qwen-fp8, nemotron)
  --port PORT         vLLM server port (default: 8000)
  --skip-routing      Skip router profiling steps
  --skip-sweep        Skip Pareto sweep
  -h, --help          Show this help
USAGE
      exit 0 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

if [[ -z "${MODEL}" ]]; then
  echo "ERROR: --model required"
  echo "Supported: muse, qwen-nvfp4, qwen-fp8, nemotron"
  exit 1
fi

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------
case "${MODEL}" in
  muse)
    MODEL_ID="meta-models/Muse-Glimmer-30B"
    SHORT_NAME="muse-glimmer"
    CONTAINER_NAME="vllm-muse-gb10"
    PRECISION="bf16"
    ARCHITECTURE="Dense Transformer"
    TOTAL_PARAMS="30B"
    ACTIVE_PARAMS="30B"
    MAX_CONTEXT=131072
    CONTAINER_IMAGE="registry.dell.huggingface.co/enterprise-dell-inference-meta-models-muse-glimmer-30b-gb10:latest"
    LAUNCH_SCRIPT="launch_vllm_muse.sh"
    ;;
  qwen-nvfp4)
    MODEL_ID="nvidia/Qwen3.6-35B-A3B-NVFP4"
    SHORT_NAME="qwen-nvfp4"
    CONTAINER_NAME="vllm-qwen-nvfp4-gb10"
    PRECISION="nvfp4"
    ARCHITECTURE="MoE"
    TOTAL_PARAMS="35B"
    ACTIVE_PARAMS="3B"
    MAX_CONTEXT=131072
    CONTAINER_IMAGE="vllm/vllm-openai:v0.24.0-ubuntu2404"
    LAUNCH_SCRIPT="launch_vllm_qwen_nvfp4.sh"
    ;;
  qwen-fp8)
    MODEL_ID="Qwen/Qwen3.6-35B-A3B-FP8"
    SHORT_NAME="qwen-fp8"
    CONTAINER_NAME="vllm-qwen-fp8-gb10"
    PRECISION="fp8"
    ARCHITECTURE="MoE"
    TOTAL_PARAMS="35B"
    ACTIVE_PARAMS="3B"
    MAX_CONTEXT=131072
    CONTAINER_IMAGE="vllm/vllm-openai:v0.24.0-ubuntu2404"
    LAUNCH_SCRIPT="launch_vllm_qwen_fp8.sh"
    ;;
  nemotron)
    MODEL_ID="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
    SHORT_NAME="nemotron"
    CONTAINER_NAME="vllm-nemotron-gb10"
    PRECISION="nvfp4"
    ARCHITECTURE="Gated DeltaNet MoE"
    TOTAL_PARAMS="30B"
    ACTIVE_PARAMS="3B"
    MAX_CONTEXT=1048576
    CONTAINER_IMAGE="registry.dell.huggingface.co/enterprise-dell-inference-nvidia-nvidia-nemotron-3.5-lightning-30b-a3b-nvfp4-gb10:latest"
    LAUNCH_SCRIPT="launch_vllm_nemotron.sh"
    ;;
  *)
    echo "ERROR: Unknown model '${MODEL}'"
    echo "Supported: muse, qwen-nvfp4, qwen-fp8, nemotron"
    exit 1
    ;;
esac

VLLM_URL="http://localhost:${PORT}"

# Output directories
BENCH_DIR="${RESULTS_DIR}/${TIMESTAMP}_${SHORT_NAME}_full"
ROUTING_DIR="${BENCH_DIR}/routing"
SWEEP_DIR="${BENCH_DIR}/sweep"

mkdir -p "${BENCH_DIR}" "${ROUTING_DIR}" "${SWEEP_DIR}"

echo "============================================"
echo " Full Benchmark: ${MODEL_ID}"
echo "============================================"
echo "Model:      ${MODEL_ID}"
echo "Precision:  ${PRECISION}"
echo "URL:        ${VLLM_URL}"
echo "Results:    ${BENCH_DIR}"
echo "Timestamp:  ${TIMESTAMP}"
echo ""

# ---------------------------------------------------------------------------
# Phase 0: Health check
# ---------------------------------------------------------------------------
echo ">>> Phase 0: Health check"
if ! curl -sf "${VLLM_URL}/health" > /dev/null 2>&1; then
  echo "ERROR: vLLM server not responding at ${VLLM_URL}"
  echo "  Start server with: ./${LAUNCH_SCRIPT}"
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
  echo ""

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
  echo ""

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
    --label "${SHORT_NAME}_128" \
    --timeout 1800 \
    --warmup 3

  # Move results
  SWEEP_RESULT=$(ls -td "${RESULTS_DIR}"/*_${SHORT_NAME}_128 2>/dev/null | head -1)
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
# Phase 4: Pareto sweep (512-tok output)
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
    --label "${SHORT_NAME}_512" \
    --timeout 2400 \
    --warmup 3

  # Move results
  SWEEP_RESULT=$(ls -td "${RESULTS_DIR}"/*_${SHORT_NAME}_512 2>/dev/null | head -1)
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

if [[ -d "${SWEEP_DIR}" ]] && [[ -f "${SCRIPT_DIR}/make_pareto.py" ]]; then
  echo "    Generating 128-tok Pareto report..."
  python3 "${SCRIPT_DIR}/make_pareto.py" \
    "${SWEEP_DIR}" \
    "${BENCH_DIR}/pareto_128.html" || true
fi

if [[ -d "${SWEEP_DIR}_512" ]] && [[ -f "${SCRIPT_DIR}/make_pareto.py" ]]; then
  echo "    Generating 512-tok Pareto report..."
  python3 "${SCRIPT_DIR}/make_pareto.py" \
    "${SWEEP_DIR}_512" \
    "${BENCH_DIR}/pareto_512.html" || true
fi

echo ""

# ---------------------------------------------------------------------------
# Phase 6: Save run metadata
# ---------------------------------------------------------------------------
echo ">>> Phase 6: Saving run metadata"

cat > "${BENCH_DIR}/run_meta.json" <<EOF
{
  "model": "${MODEL_ID}",
  "short_name": "${SHORT_NAME}",
  "framework": "vllm",
  "precision": "${PRECISION}",
  "architecture": "${ARCHITECTURE}",
  "total_params": "${TOTAL_PARAMS}",
  "active_params_per_token": "${ACTIVE_PARAMS}",
  "max_context_length": ${MAX_CONTEXT},
  "container": "${CONTAINER_IMAGE}",
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

echo "    Metadata saved to ${BENCH_DIR}/run_meta.json"
echo ""

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "============================================"
echo " Benchmark Complete: ${MODEL_ID}"
echo "============================================"
echo ""
echo "Results directory: ${BENCH_DIR}"
echo ""
echo "Files generated:"
echo "  - routing/           Router profiling results"
echo "  - sweep/             128-tok Pareto sweep"
echo "  - sweep_512/         512-tok Pareto sweep"
echo "  - pareto_128.html    128-tok Pareto report"
echo "  - pareto_512.html    512-tok Pareto report"
echo "  - run_meta.json      Run configuration metadata"
echo ""
echo "View reports:"
echo "  open ${BENCH_DIR}/pareto_128.html"
