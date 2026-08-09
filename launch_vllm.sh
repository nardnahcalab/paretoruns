#!/usr/bin/env bash
# =============================================================================
# launch_vllm.sh - Launch vLLM with MOE Model on GB10 (DGX Spark)
# =============================================================================
# Deploys nvidia/Qwen3.6-35B-A3B-NVFP4 (35B total / 3B active MoE) on
# NVIDIA GB10 hardware with settings optimized for aiperf benchmarking.
#
# The model is a sparse MoE with 256 experts (8 routed + 1 shared), making it
# ideal for studying how text/random/reasoning workloads exercise different
# expert routing patterns.
#
# Key GB10 considerations:
#   - SM121 (not datacenter Blackwell SM100) - use Marlin MoE backend
#   - Unified 128GB LPDDR5X memory (CPU+GPU shared)
#   - FP4 weight quantization via ModelOpt, FP8 KV cache
#   - No NVSwitch/NVLink - single logical GPU
#
# Usage:
#   ./launch_vllm.sh [--model MODEL] [--port PORT] [--gpu-mem UTIL]
#
# Prerequisites:
#   - Docker with nvidia runtime
#   - HuggingFace token (HF_TOKEN env var) for gated models
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:v0.24.0-ubuntu2404}"
MODEL_ID="${MODEL_ID:-nvidia/Qwen3.6-35B-A3B-NVFP4}"
CONTAINER_NAME="${CONTAINER_NAME:-vllm-moe-gb10}"
PORT="${PORT:-8000}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
LOAD_FORMAT="${LOAD_FORMAT:-fastsafetensors}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
VLLM_CACHE="${VLLM_CACHE:-$HOME/.cache/vllm}"
FLASHINFER_CACHE="${FLASHINFER_CACHE:-$HOME/.cache/flashinfer}"

# Parse flags
while [[ $# -gt 0 ]]; do
  case $1 in
    --model)       MODEL_ID="$2"; shift 2 ;;
    --port)        PORT="$2"; shift 2 ;;
    --gpu-mem)     GPU_MEM_UTIL="$2"; shift 2 ;;
    --max-seqs)    MAX_NUM_SEQS="$2"; shift 2 ;;
    --max-len)     MAX_MODEL_LEN="$2"; shift 2 ;;
    --load-format) LOAD_FORMAT="$2"; shift 2 ;;
    --enforce-eager) ENFORCE_EAGER=1; shift ;;
    --image)       VLLM_IMAGE="$2"; shift 2 ;;
    --name)        CONTAINER_NAME="$2"; shift 2 ;;
    --stop)        STOP_ONLY=1; shift ;;
    --status)      STATUS_ONLY=1; shift ;;
    -h|--help)
      cat <<'USAGE'
Usage: ./launch_vllm.sh [OPTIONS]

Launch vLLM with Qwen3.6-35B-A3B-NVFP4 MOE model on GB10.

Options:
  --model MODEL       HuggingFace model ID (default: nvidia/Qwen3.6-35B-A3B-NVFP4)
  --port PORT         API port (default: 8000)
  --gpu-mem UTIL      GPU memory utilization fraction (default: 0.4)
  --max-seqs N        Max concurrent sequences (default: 4)
  --max-len N         Max model context length (default: 262144)
  --load-format FMT   Weight load format: fastsafetensors or safetensors (default: fastsafetensors)
  --image IMAGE       vLLM Docker image (default: vllm/vllm-openai:v0.24.0-ubuntu2404)
  --name NAME         Container name (default: vllm-moe-gb10)
  --stop              Stop and remove the running container
  --status            Show container status and health
  -h, --help          Show this help
USAGE
      exit 0 ;;
    --stop)    STOP_ONLY=1; shift ;;
    --status)  STATUS_ONLY=1; shift ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# ---------------------------------------------------------------------------
# Status / Stop handlers
# ---------------------------------------------------------------------------
if [[ "${STATUS_ONLY:-0}" == "1" ]]; then
  echo "=== Container Status ==="
  docker ps -a --filter "name=${CONTAINER_NAME}" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
  echo ""
  echo "=== Health Check ==="
  if docker exec "${CONTAINER_NAME}" curl -sf http://localhost:${PORT}/health > /dev/null 2>&1; then
    echo "Server: HEALTHY"
  else
    echo "Server: NOT RESPONDING"
  fi
  echo ""
  echo "=== Models ==="
  curl -sf "http://localhost:${PORT}/v1/models" 2>/dev/null | python3 -m json.tool 2>/dev/null || echo "Cannot reach endpoint"
  exit 0
fi

if [[ "${STOP_ONLY:-0}" == "1" ]]; then
  echo "Stopping container: ${CONTAINER_NAME}"
  docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true
  echo "Done."
  exit 0
fi

# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------
echo "============================================"
echo " vLLM MOE Launcher for GB10"
echo "============================================"
echo "Model:       ${MODEL_ID}"
echo "Image:       ${VLLM_IMAGE}"
echo "Port:        ${PORT}"
echo "GPU Mem:     ${GPU_MEM_UTIL}"
echo "Max Context: ${MAX_MODEL_LEN}"
echo "Max Seqs:    ${MAX_NUM_SEQS}"
echo "Max Batch:   ${MAX_NUM_BATCHED_TOKENS}"
echo ""

# Check HF_TOKEN
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "WARNING: HF_TOKEN not set. Model download may fail for gated models."
  echo "  Set it with: export HF_TOKEN=hf_your_token_here"
  echo ""
fi

# Check Docker
if ! command -v docker &> /dev/null; then
  echo "ERROR: docker not found"
  exit 1
fi

# Check nvidia runtime
if ! docker info 2>/dev/null | grep -qi nvidia; then
  echo "WARNING: NVIDIA Docker runtime may not be configured."
  echo "  Ensure nvidia-container-toolkit is installed."
fi

# Ensure cache directories exist
mkdir -p "${HF_CACHE}" "${VLLM_CACHE}" "${FLASHINFER_CACHE}"

# Remove stale container
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  echo "Removing existing container: ${CONTAINER_NAME}"
  docker rm -f "${CONTAINER_NAME}" > /dev/null 2>&1
fi

# ---------------------------------------------------------------------------
# Launch vLLM
# ---------------------------------------------------------------------------
echo ""
echo "Launching vLLM server..."
echo ""

# GB10-specific environment variables
# These are critical for correct operation on SM121:
#   VLLM_MARLIN_USE_ATOMIC_ADD=1    - Prevents Marlin kernel race condition on SM121
#   VLLM_USE_FLASHINFER_MOE_FP4=0   - FlashInfer FP4 MoE broken on SM121
#   VLLM_FLASHINFER_MOE_BACKEND=latency - Optimize MoE for latency over throughput
#   HF_HUB_DISABLE_XET=1            - Workaround for broken xet downloader
DOCKER_ENV=(
  -e HF_TOKEN="${HF_TOKEN:-}"
  -e VLLM_MARLIN_USE_ATOMIC_ADD=1
  -e VLLM_USE_FLASHINFER_MOE_FP4=0
  -e VLLM_FLASHINFER_MOE_BACKEND=latency
  -e HF_HUB_DISABLE_XET=1
  -e VLLM_USE_DEEP_GEMM=0
  -e NVIDIA_VISIBLE_DEVICES=all
  -e VLLM_ATTENTION_BACKEND=flashinfer
)

# Mount caches for persistent weight/compile storage
DOCKER_VOLUMES=(
  -v "${HF_CACHE}:/root/.cache/huggingface"
  -v "${VLLM_CACHE}:/root/.cache/vllm"
  -v "${FLASHINFER_CACHE}:/root/.cache/flashinfer"
)

# Model serving command
# Key flags explained:
#   --moe-backend marlin              CUTLASS FP4 broken on SM121, Marlin is correct
#   --kv-cache-dtype fp8              FP8 KV cache saves memory, works on GB10
#   --enable-prefix-caching           CRITICAL for benchmark: reuses KV cache for
#                                     repeated prompt prefixes across turns/sessions
#   --enable-chunked-prefill          Splits long prefill into chunks for better
#                                     latency under concurrent load
#   --max-num-seqs 4                  Low concurrency ceiling; GB10 has limited
#                                     memory bandwidth (273 GB/s) so fewer in-flight
#                                     sequences avoids TTFT spikes
#   --gpu-memory-utilization 0.4      Conservative for unified memory; OS and page
#                                     cache compete with CUDA allocations
#   --load-format fastsafetensors     2-3x faster weight loading vs safetensors
#   --reasoning-parser qwen3          Required for reasoning dataset prompts
#   --enable-auto-tool-choice         Enables tool calling if needed

VLLM_CMD=(
  docker run -d
  --name "${CONTAINER_NAME}"
  --gpus all
  --ipc=host
  --network host
  --restart unless-stopped
  "${DOCKER_ENV[@]}"
  "${DOCKER_VOLUMES[@]}"
  "${VLLM_IMAGE}"
  "${MODEL_ID}"
    --host 0.0.0.0
    --port "${PORT}"
    --tensor-parallel-size 1
    --trust-remote-code
    --moe-backend marlin
    --kv-cache-dtype fp8_e4m3
    --gpu-memory-utilization "${GPU_MEM_UTIL}"
    --max-model-len "${MAX_MODEL_LEN}"
    --max-num-seqs "${MAX_NUM_SEQS}"
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
    --enable-chunked-prefill
    --enable-prefix-caching
    --load-format "${LOAD_FORMAT}"
    --reasoning-parser qwen3
    --enable-auto-tool-choice
    --tool-call-parser qwen3_xml
)

# Conditionally add --enforce-eager
if [[ "${ENFORCE_EAGER}" == "1" ]]; then
  VLLM_CMD+=(--enforce-eager)
fi

echo "Docker command:"
echo "  ${VLLM_CMD[*]}" | sed 's/  /    /g'
echo ""

# Start container
"${VLLM_CMD[@]}"

echo "Container started: ${CONTAINER_NAME}"
echo ""

# ---------------------------------------------------------------------------
# Wait for readiness
# ---------------------------------------------------------------------------
echo -n "Waiting for server readiness"
MAX_WAIT=600
ELAPSED=0
HEALTHY=false

while [[ ${ELAPSED} -lt ${MAX_WAIT} ]]; do
  if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
    HEALTHY=true
    break
  fi
  echo -n "."
  sleep 5
  ELAPSED=$((ELAPSED + 5))
done

echo ""

if [[ "${HEALTHY}" == "true" ]]; then
  echo "Server HEALTHY after ${ELAPSED}s"
  echo ""

  # Print model info
  echo "=== Model Info ==="
  curl -sf "http://localhost:${PORT}/v1/models" 2>/dev/null | python3 -m json.tool 2>/dev/null || true
  echo ""

  # Quick sanity check
  echo "=== Quick Test (single request) ==="
  curl -sf "http://localhost:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
      "model": "'"${MODEL_ID}"'",
      "messages": [{"role": "user", "content": "Say hello in 5 words."}],
      "max_tokens": 20,
      "temperature": 0.0,
      "stream": false
    }' 2>/dev/null | python3 -c "
import sys, json
try:
    r = json.load(sys.stdin)
    content = r['choices'][0]['message']['content']
    usage = r.get('usage', {})
    print(f'Response: {content[:100]}')
    print(f'Tokens: prompt={usage.get(\"prompt_tokens\",\"?\")}, completion={usage.get(\"completion_tokens\",\"?\")}')
except Exception as e:
    print(f'Parse error: {e}')
" 2>/dev/null || echo "Test request failed (server may still be loading model weights)"
  echo ""
else
  echo "TIMEOUT: Server not ready after ${MAX_WAIT}s"
  echo "Check logs: docker logs ${CONTAINER_NAME}"
  exit 1
fi

# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------
echo "============================================"
echo " Server Ready"
echo "============================================"
echo ""
echo "Endpoint:    http://localhost:${PORT}"
echo "Health:      http://localhost:${PORT}/health"
echo "Models:      http://localhost:${PORT}/v1/models"
echo "Chat:        http://localhost:${PORT}/v1/chat/completions"
echo ""
echo "Container:   ${CONTAINER_NAME}"
echo "Logs:        docker logs -f ${CONTAINER_NAME}"
echo ""
echo "MOE Model Architecture (Qwen3.6-35B-A3B):"
echo "  Total parameters:  35B"
echo "  Active parameters: 3B (per token)"
echo "  Experts:           256 total, 8 routed + 1 shared"
echo "  Architecture:      Gated DeltaNet + MoE hybrid"
echo "  Layers:            40 (10 attention + 30 linear attention)"
echo ""
echo "Prefix Caching: ENABLED"
echo "  This is critical for the multi-turn benchmark. In multi-turn chat,"
echo "  each subsequent turn includes the full conversation history. With"
echo "  prefix caching, the KV cache for the shared prefix (all prior turns)"
echo "  is reused, so only the new turn's tokens need prefill computation."
echo ""
echo "  For TEXT dataset:  Natural language has strong token correlations."
echo "    Prefix caching reduces redundant prefill across turns."
echo ""
echo "  For RANDOM dataset: Random words break correlations. The model"
echo "    cannot exploit statistical patterns, so each token requires full"
echo "    expert routing. Prefix caching still helps for shared prefix."
echo ""
echo "  For REASONING dataset: Logic chains create deep attention deps."
echo "    Prefix caching is most beneficial here because the system prompt"
echo "    and prior reasoning steps form a long stable prefix."
echo ""
echo "Run benchmark: ./run_benchmark.sh --url http://localhost:${PORT}"
echo ""
echo "Stop server:  ./launch_vllm.sh --stop"
echo "Server logs:  docker logs -f ${CONTAINER_NAME}"
