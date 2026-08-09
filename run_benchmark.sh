#!/usr/bin/env bash
# =============================================================================
# run_benchmark.sh - AIPerf Multi-Dataset Pareto Benchmark for MOE Models
# =============================================================================
# Runs the three isomorphic multi-turn chat datasets (text, random, reasoning)
# at multiple concurrency levels to build Pareto curves showing how MOE models
# exhibit different prefill/decode performance characteristics across workload
# types. Uses GPU telemetry for GB10 hardware monitoring.
#
# Usage:
#   ./run_benchmark.sh [--url URL] [--model MODEL] [--concurrency LEVELS]
#
# Prerequisites:
#   - vLLM server running (see launch_vllm.sh)
#   - Docker with NVIDIA runtime
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration (override via env or flags)
# ---------------------------------------------------------------------------
# Note: Defaults are tuned for NVIDIA GB10 (DGX Spark), a single-logical-GPU
# edge device with ~273 GB/s unified-memory bandwidth and 140W TDP. Compared
# with datacenter GPUs (H100: ~3.35 TB/s, 700W), the GB10 cannot sustain high
# concurrency or long sequence generation without saturating memory bandwidth.
# Keep concurrency low (1-8), use modest request counts, and target short
# output lengths so the full 3-dataset sweep completes in reasonable time.
AIPERF_IMAGE="${AIPERF_IMAGE:-nvcr.io/nvidia/ai-dynamo/aiperf:0.12.0}"
VLLM_URL="${VLLM_URL:-http://localhost:8000}"
MODEL_NAME="${MODEL_NAME:-nvidia/Qwen3.6-35B-A3B-NVFP4}"
CONCURRENCY_LEVELS="${CONCURRENCY_LEVELS:-1 2 4 8}"
REQUEST_COUNT="${REQUEST_COUNT:-50}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-3}"
OUTPUT_TOK_MEAN="${OUTPUT_TOK_MEAN:-128}"
OUTPUT_TOK_STDDEV="${OUTPUT_TOK_STDDEV:-32}"
RANDOM_SEED="${RANDOM_SEED:-42}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"
GPU_TELEMETRY_MODE="${GPU_TELEMETRY_MODE:-pynvml}"

# Parse command-line flags
while [[ $# -gt 0 ]]; do
  case $1 in
    --url)        VLLM_URL="$2"; shift 2 ;;
    --model)      MODEL_NAME="$2"; shift 2 ;;
    --concurrency) CONCURRENCY_LEVELS="$2"; shift 2 ;;
    --requests)   REQUEST_COUNT="$2"; shift 2 ;;
    --output-dir) RESULTS_DIR="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [--url URL] [--model MODEL] [--concurrency '1 2 4 8'] [--requests N]"
      echo ""
      echo "Benchmarks three multi-turn datasets (text, random, reasoning) at"
      echo "multiple concurrency levels to build MOE Pareto curves."
      exit 0 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# ---------------------------------------------------------------------------
# Dataset definitions
# ---------------------------------------------------------------------------
declare -A DATASETS=(
  [text]="${SCRIPT_DIR}/multi_turn_iso_text_chat.jsonl"
  [random]="${SCRIPT_DIR}/multi_turn_iso_random_chat.jsonl"
  [reasoning]="${SCRIPT_DIR}/multi_turn_iso_reasoning_chat.jsonl"
)
DATASET_NAMES=("text" "random" "reasoning")

# Dataset descriptions (for analysis notes)
declare -A DATASET_DESC=(
  [text]="Natural conversational prompts with strong token correlations; tests how MOE routing handles coherent multi-turn dialogue with evolving context"
  [random]="Random word sequences with no semantic structure; stress-tests raw token processing and forces broad expert activation across all MoE layers"
  [reasoning]="Logic proofs and mathematical reasoning prompts; exercises deeper chain-of-thought routing and tests whether reasoning-heavy workloads concentrate or spread expert activation"
)

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
echo "============================================"
echo " AIPerf MOE Pareto Benchmark"
echo "============================================"
echo "Model:        ${MODEL_NAME}"
echo "Endpoint:     ${VLLM_URL}"
echo "Concurrency:  ${CONCURRENCY_LEVELS}"
echo "Requests:     ${REQUEST_COUNT}"
echo "Results:      ${RESULTS_DIR}"
echo "GPU Telemetry: ${GPU_TELEMETRY_MODE}"
echo ""

# Verify dataset files exist
for ds in "${DATASET_NAMES[@]}"; do
  if [[ ! -f "${DATASETS[$ds]}" ]]; then
    echo "ERROR: Dataset file not found: ${DATASETS[$ds]}"
    exit 1
  fi
done

# Verify server is reachable
echo -n "Checking server at ${VLLM_URL}... "
if curl -sf "${VLLM_URL}/health" > /dev/null 2>&1; then
  echo "OK"
else
  echo "WARNING: Server not responding. Start vLLM first (see launch_vllm.sh)."
  echo "         Continuing anyway - benchmark will fail if server is down."
fi
echo ""

# Create results directory structure
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
BENCH_DIR="${RESULTS_DIR}/${TIMESTAMP}"
mkdir -p "${BENCH_DIR}"

# Write run metadata
cat > "${BENCH_DIR}/run_meta.json" <<EOF
{
  "timestamp": "${TIMESTAMP}",
  "model": "${MODEL_NAME}",
  "url": "${VLLM_URL}",
  "concurrency_levels": [$(echo "${CONCURRENCY_LEVELS}" | tr ' ' ',')],
  "request_count": ${REQUEST_COUNT},
  "warmup_requests": ${WARMUP_REQUESTS},
  "output_tokens_mean": ${OUTPUT_TOK_MEAN},
  "output_tokens_stddev": ${OUTPUT_TOK_STDDEV},
  "random_seed": ${RANDOM_SEED},
  "gpu_telemetry_mode": "${GPU_TELEMETRY_MODE}",
  "datasets": {
    "text": "Natural conversational prompts",
    "random": "Random word sequences",
    "reasoning": "Logic and proof prompts"
  }
}
EOF

# ---------------------------------------------------------------------------
# Run benchmarks
# ---------------------------------------------------------------------------
TOTAL_RUNS=$(( ${#DATASET_NAMES[@]} * $(echo "${CONCURRENCY_LEVELS}" | wc -w) ))
RUN_NUM=0

echo "Starting ${TOTAL_RUNS} benchmark runs..."
echo ""

for ds in "${DATASET_NAMES[@]}"; do
  ds_file="${DATASETS[$ds]}"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo " Dataset: ${ds}"
  echo " ${DATASET_DESC[$ds]}"
  echo " File: ${ds_file}"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  for conc in ${CONCURRENCY_LEVELS}; do
    RUN_NUM=$((RUN_NUM + 1))
    RUN_LABEL="${ds}_c${conc}"
    RUN_DIR="${BENCH_DIR}/${RUN_LABEL}"
    mkdir -p "${RUN_DIR}"

    echo ""
    echo ">>> Run ${RUN_NUM}/${TOTAL_RUNS}: dataset=${ds}, concurrency=${conc}"
    echo "    Output: ${RUN_DIR}"

    # Build aiperf command
    # Use --input-file with --custom-dataset-type multi_turn for the JSONL files
    # --conversation-num controls how many sessions from the file to use
    # We cap at the available sessions (500) or request_count, whichever fits
    CONV_NUM=$(( REQUEST_COUNT < 500 ? REQUEST_COUNT : 500 ))

    # NOTE: The aiperf image entrypoint is /bin/bash -c, so DOCKER concatenates
    # ALL args after the image into a single bash -c command string AND treats
    # every arg after the first as positional $0/$1... params, NOT the command.
    # Therefore the entire "aiperf profile ..." invocation MUST be passed as ONE
    # quoted string argument, or only "aiperf" runs (printing top-level usage).
    AIPERF_ARGS=(
      aiperf profile
        --model "${MODEL_NAME}"
        --endpoint-type chat
        --endpoint /v1/chat/completions
        --streaming
        --url "${VLLM_URL}"
        --input-file "/data/$(basename "${ds_file}")"
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

    AIPERF_CMD=(
      docker run --rm
      --gpus all
      --network host
      --ipc=host
      -v "${SCRIPT_DIR}:/data:ro"
      -v "${RUN_DIR}:/output"
      -e NVIDIA_VISIBLE_DEVICES=all
      "${AIPERF_IMAGE}"
      "${AIPERF_ARGS[*]}"
    )

    echo "    Command: aiperf profile --model ${MODEL_NAME} --concurrency ${conc} ..."
    echo "    GPU telemetry: ${GPU_TELEMETRY_MODE}"

    # Run the benchmark
    if "${AIPERF_CMD[@]}" 2>&1 | tee "${RUN_DIR}/aiperf_stdout.log"; then
      echo "    Status: COMPLETED"
    else
      echo "    Status: FAILED (exit code $?)"
    fi

    echo ""
  done
done

# ---------------------------------------------------------------------------
# Generate comparison plots
# ---------------------------------------------------------------------------
echo ""
echo "============================================"
echo " Generating Comparison Plots"
echo "============================================"

# Build the aiperf plot command pointing at all run directories.
# BENCH_DIR is mounted at /results, so paths must be container-side.
# Same single-string requirement as profile (entrypoint=/bin/bash -c).
PLOT_PATHS=()
for ds in "${DATASET_NAMES[@]}"; do
  for conc in ${CONCURRENCY_LEVELS}; do
    RUN_DIR="${BENCH_DIR}/${ds}_c${conc}"
    if [[ -d "${RUN_DIR}" ]]; then
      PLOT_PATHS+=("/results/${ds}_c${conc}")
    fi
  done
done

if [[ ${#PLOT_PATHS[@]} -gt 0 ]]; then
  echo "Running: aiperf plot on ${#PLOT_PATHS[@]} directories..."
  echo "  Paths: ${PLOT_PATHS[*]}"

  PLOT_ARGS=(aiperf plot "${PLOT_PATHS[@]}")

  PLOT_CMD=(
    docker run --rm
    --gpus all
    --network host
    -v "${BENCH_DIR}:/results"
    -e NVIDIA_VISIBLE_DEVICES=all
    "${AIPERF_IMAGE}"
    "${PLOT_ARGS[*]}"
  )

  "${PLOT_CMD[@]}" 2>&1 | tee "${BENCH_DIR}/plot_stdout.log" || \
    echo "WARNING: aiperf plot had issues (needs Chrome in image)."
fi

# Fallback Pareto report using the host-side make_pareto.py.
# aiperf's built-in plot only produces PNGs if Google Chrome is installed in
# the container, which the standard image lacks. make_pareto.py reads the
# exported profile_export_aiperf.json files and writes a self-contained HTML
# report with inline SVG Pareto curves (no external dependencies).
PARETO_SCRIPT="${SCRIPT_DIR}/make_pareto.py"
if [[ -f "${PARETO_SCRIPT}" ]]; then
  echo "Generating Pareto report with make_pareto.py..."
  if python3 "${PARETO_SCRIPT}" "${BENCH_DIR}" "${BENCH_DIR}/pareto_report.html"; then
    echo "Pareto report: ${BENCH_DIR}/pareto_report.html"
  else
    echo "WARNING: make_pareto.py failed to generate the report."
  fi
else
  echo "WARNING: ${PARETO_SCRIPT} not found; skipping Pareto report."
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================"
echo " Benchmark Complete"
echo "============================================"
echo ""
echo "Results directory: ${BENCH_DIR}"
echo ""
echo "Directory structure:"
echo "  ${BENCH_DIR}/"
echo "  ├── run_meta.json"
echo "  ├── text_c1/        (text dataset, concurrency=1)"
echo "  ├── text_c2/        (text dataset, concurrency=2)"
echo "  ├── ..."
echo "  ├── random_c1/      (random dataset, concurrency=1)"
echo "  ├── ..."
echo "  ├── reasoning_c1/   (reasoning dataset, concurrency=1)"
echo "  └── ..."
echo ""
echo "Key files per run:"
echo "  profile_export_aiperf.json  - Full metrics export"
echo "  *.png                       - Generated plots"
echo ""
echo "MOE Analysis Notes:"
echo "  - TEXT prompts: Natural language triggers heterogeneous expert routing"
echo "    across MoE layers. Expect strong prefix-caching benefits on repeated"
echo "    conversation patterns. Prefill TTFT should scale with context length."
echo ""
echo "  - RANDOM prompts: Random words break token-level correlations, forcing"
echo "    broader expert activation. This stresses the router and may show"
echo "    higher prefill latency vs text. Decode throughput should be lower"
echo "    because the model cannot predict/exploit word co-occurrence patterns."
echo ""
echo "  - REASONING prompts: Logic/proof chains create deep attention patterns."
echo "    Expect highest decode ITL (inter-token latency) because each token"
echo "    depends on prior reasoning steps. Prefill may actually be FASTER than"
echo "    text because reasoning prompts are shorter and more formulaic."
echo ""
echo "  - PARETO CURVES: Plot throughput vs latency for each dataset. The"
echo "    frontier shape reveals the MOE model's efficiency profile:"
echo "    * Steep curve = good scalability"
echo "    * Flat curve = bottleneck (memory/compute bound)"
echo "    * Different curves per dataset = MOE routing sensitivity to content"
echo ""
echo "To analyze results:"
echo "  cat ${BENCH_DIR}/text_c1/profile_export_aiperf.json | python3 -m json.tool"
echo "  aiperf plot ${BENCH_DIR}/text_c1 ${BENCH_DIR}/random_c1 ${BENCH_DIR}/reasoning_c1"
