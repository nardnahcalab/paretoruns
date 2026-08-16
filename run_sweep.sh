#!/usr/bin/env bash
# =============================================================================
# run_sweep.sh - Parametrized AIPerf sweep runner (reuses run_benchmark.sh logic)
# =============================================================================
# Runs a configurable matrix of datasets x concurrency with given output-token
# and request-count settings, one output dir per (dataset, concurrency).
#
# Usage:
#   ./run_sweep.sh --concurrency '1 2 4' --output-tokens 512 --output-stddev 128 \
#                  --requests 100 --label longtok
#
# Result structure (under results/):
#   results/<timestamp>_<label>/<dataset>_c<conc>/
#   results/<timestamp>_<label>/pareto_report.html   (via make_pareto.py)
# =============================================================================

set -euo pipefail

AIPERF_IMAGE="${AIPERF_IMAGE:-nvcr.io/nvidia/ai-dynamo/aiperf:0.12.0}"
VLLM_URL="${VLLM_URL:-http://localhost:8000}"
MODEL_NAME="${MODEL_NAME:-nvidia/Qwen3.6-35B-A3B-NVFP4}"
CONCURRENCY_LEVELS="${CONCURRENCY_LEVELS:-1 2 4}"
REQUEST_COUNT="${REQUEST_COUNT:-50}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-3}"
OUTPUT_TOK_MEAN="${OUTPUT_TOK_MEAN:-512}"
OUTPUT_TOK_STDDEV="${OUTPUT_TOK_STDDEV:-128}"
RANDOM_SEED="${RANDOM_SEED:-42}"
GPU_TELEMETRY_MODE="${GPU_TELEMETRY_MODE:-pynvml}"
RUN_TIMEOUT="${RUN_TIMEOUT:-1500}"
GOODPUT_SLO="${GOODPUT_SLO:-}"  # e.g. "request_latency:2000" or "request_latency:2000 inter_token_latency:10"
LABEL="sweep"
SCRATCH_DIR="${SCRATCH_DIR:-}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"

while [[ $# -gt 0 ]]; do
  case $1 in
    --concurrency)   CONCURRENCY_LEVELS="$2"; shift 2 ;;
    --requests)      REQUEST_COUNT="$2"; shift 2 ;;
    --output-tokens) OUTPUT_TOK_MEAN="$2"; shift 2 ;;
    --output-std)    OUTPUT_TOK_STDDEV="$2"; shift 2 ;;
    --label)         LABEL="$2"; shift 2 ;;
    --timeout)       RUN_TIMEOUT="$2"; shift 2 ;;
    --warmup)        WARMUP_REQUESTS="$2"; shift 2 ;;
    --goodput)       GOODPUT_SLO="$2"; shift 2 ;;
    --url)           VLLM_URL="$2"; shift 2 ;;
    --scratch)       SCRATCH_DIR="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [--concurrency '1 2 4'] [--output-tokens 512] [--output-std 128] [--requests N] [--label X] [--timeout S] [--goodput 'request_latency:2000']"
      exit 0 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# Datasets
declare -A DATASETS=(
  [text]="${SCRIPT_DIR}/multi_turn_iso_text_chat.jsonl"
  [random]="${SCRIPT_DIR}/multi_turn_iso_random_chat.jsonl"
  [reasoning]="${SCRIPT_DIR}/multi_turn_iso_reasoning_chat.jsonl"
)
DATASET_NAMES=("text" "random" "reasoning")

echo "==== Extended AIPerf Sweep ===="
echo "Label:        ${LABEL}"
echo "Concurrency:  ${CONCURRENCY_LEVELS}"
echo "Requests:     ${REQUEST_COUNT}"
echo "Output tokens: ${OUTPUT_TOK_MEAN} +/- ${OUTPUT_TOK_STDDEV}"
echo "Goodput SLO:  ${GOODPUT_SLO:-none}"
echo "URL:          ${VLLM_URL}"
echo ""

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${RESULTS_DIR}/${TIMESTAMP}_${LABEL}"
mkdir -p "${OUT_ROOT}"

TOTAL=$(( ${#DATASET_NAMES[@]} * $(echo "$CONCURRENCY_LEVELS" | wc -w) ))
n=0
for ds in "${DATASET_NAMES[@]}"; do
  for conc in ${CONCURRENCY_LEVELS}; do
    n=$((n + 1))
    RUN_DIR="${OUT_ROOT}/${ds}_c${conc}"
    mkdir -p "${RUN_DIR}"
    echo ""
    echo ">>> [${n}/${TOTAL}] ${LABEL} ${ds} concurrency=${conc} (osl=${OUTPUT_TOK_MEAN})"

    CONV_NUM=$(( REQUEST_COUNT < 500 ? REQUEST_COUNT : 500 ))
    AIPERF_ARGS=(
      aiperf profile
        --model "${MODEL_NAME}"
        --endpoint-type chat
        --endpoint /v1/chat/completions
        --streaming
        --url "${VLLM_URL}"
        --input-file "/data/$(basename "${DATASETS[$ds]}")"
        --custom-dataset-type multi_turn
        --conversation-num "${CONV_NUM}"
        --concurrency "${conc}"
        --request-count "${REQUEST_COUNT}"
        --warmup-request-count "${WARMUP_REQUESTS}"
        --output-tokens-mean "${OUTPUT_TOK_MEAN}"
        --output-tokens-stddev "${OUTPUT_TOK_STDDEV}"
        --random-seed "${RANDOM_SEED}"
        --artifact-dir "/output"
        --ui none
        --gpu-telemetry "${GPU_TELEMETRY_MODE}"
    )

    # Add goodput SLOs if specified
    if [[ -n "${GOODPUT_SLO}" ]]; then
      for slo in ${GOODPUT_SLO}; do
        AIPERF_ARGS+=(--goodput "${slo}")
      done
    fi

    timeout "${RUN_TIMEOUT}" docker run --rm \
      --gpus all \
      --network host \
      --ipc=host \
      -v "${SCRIPT_DIR}:/data:ro" \
      -v "${RUN_DIR}:/output" \
      -e NVIDIA_VISIBLE_DEVICES=all \
      "${AIPERF_IMAGE}" \
      "${AIPERF_ARGS[*]}" 2>&1 | tee "${RUN_DIR}/aiperf_stdout.log" | \
        grep -E "Phase profiling .* complete|Error Running|JSON Export" || true

    if [[ -f "${RUN_DIR}/profile_export_aiperf.json" ]]; then
      echo "    OK: output written to ${RUN_DIR}"
    else
      echo "    WARNING: no profile_export_aiperf.json in ${RUN_DIR}"
    fi
  done
done

# Generate the Pareto report against this sweep's own result dir.
if [[ -f "${SCRIPT_DIR}/make_pareto.py" ]]; then
  python3 "${SCRIPT_DIR}/make_pareto.py" "${OUT_ROOT}" "${OUT_ROOT}/pareto_report.html" || true
fi

echo ""
echo "==== Sweep complete ===="
echo "Results: ${OUT_ROOT}"
echo "Report:  ${OUT_ROOT}/pareto_report.html"