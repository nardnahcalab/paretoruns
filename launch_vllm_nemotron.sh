#!/usr/bin/env bash
# =============================================================================
# launch_vllm_nemotron.sh - Launch Dell Container with Nemotron 3.5 Lightning
# =============================================================================
# Deploys nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 using the Dell
# Enterprise AI container optimized for NVIDIA GB10 hardware.
#
# This script uses environment variables (not CLI flags) as required by the
# Dell container image. Model weights are expected to be pre-downloaded.
#
# Usage:
#   ./launch_vllm_nemotron.sh [--port PORT] [--stop] [--status]
#
# Prerequisites:
#   - Docker with nvidia runtime
#   - HuggingFace token (HF_TOKEN env var)
#   - Model weights cached at /home/bala/data/cache/hub
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
VLLM_IMAGE="${VLLM_IMAGE:-registry.dell.huggingface.co/enterprise-dell-inference-nvidia-nvidia-nemotron-3.5-lightning-30b-a3b-nvfp4-gb10:latest}"
MODEL_ID="${MODEL_ID:-nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4}"
CONTAINER_NAME="${CONTAINER_NAME:-vllm-nemotron-gb10}"
PORT="${PORT:-80}"
HOST_PORT="${HOST_PORT:-80}"
HF_CACHE="${HF_CACHE:-/home/bala/data/cache}"

# Parse flags
while [[ $# -gt 0 ]]; do
  case $1 in
    --port)        PORT="$2"; HOST_PORT="$2"; shift 2 ;;
    --model)       MODEL_ID="$2"; shift 2 ;;
    --name)        CONTAINER_NAME="$2"; shift 2 ;;
    --image)       VLLM_IMAGE="$2"; shift 2 ;;
    --cache)       HF_CACHE="$2"; shift 2 ;;
    --stop)        STOP_ONLY=1; shift ;;
    --status)      STATUS_ONLY=1; shift ;;
    -h|--help)
      cat <<'USAGE'
Usage: ./launch_vllm_nemotron.sh [OPTIONS]

Launch Dell Enterprise AI container with Nemotron 3.5 Lightning on GB10.

Options:
  --port PORT         API port (default: 80)
  --model MODEL       HuggingFace model ID
  --name NAME         Container name (default: vllm-nemotron-gb10)
  --image IMAGE       Docker image
  --cache DIR         HuggingFace cache directory (default: /home/bala/data/cache)
  --stop              Stop and remove the running container
  --status            Show container status and health
  -h, --help          Show this help
USAGE
      exit 0 ;;
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
  if curl -sf "http://localhost:${HOST_PORT}/health" > /dev/null 2>&1; then
    echo "Server: HEALTHY"
  else
    echo "Server: NOT RESPONDING"
  fi
  echo ""
  echo "=== Models ==="
  curl -sf "http://localhost:${HOST_PORT}/v1/models" 2>/dev/null | python3 -m json.tool 2>/dev/null || echo "Cannot reach endpoint"
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
echo " Dell Container: Nemotron 3.5 Lightning"
echo "============================================"
echo "Model:       ${MODEL_ID}"
echo "Image:       ${VLLM_IMAGE}"
echo "Port:        ${HOST_PORT} -> 8000"
echo "Cache:       ${HF_CACHE}"
echo ""

# Check HF_TOKEN
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN not set"
  echo "  Set it with: export HF_TOKEN=hf_your_token_here"
  exit 1
fi

# Check Docker
if ! command -v docker &> /dev/null; then
  echo "ERROR: docker not found"
  exit 1
fi

# Check model cache
if [[ ! -d "${HF_CACHE}/hub/models--nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4" ]]; then
  echo "WARNING: Model weights not found at ${HF_CACHE}/hub/"
  echo "  Download first or specify correct cache with --cache DIR"
fi

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

# Dell container uses environment variables, not CLI flags
# These settings are validated against the working docker run command
DOCKER_ENV=(
  -e HF_TOKEN="${HF_TOKEN}"
  -e MODEL_ID="${MODEL_ID}"
  -e ENABLE_PREFIX_CACHING=true
  -e TENSOR_PARALLEL_SIZE=1
  -e MAX_MODEL_LEN=1048576
  -e ENABLE_AUTO_TOOL_CHOICE=true
  -e TOOL_CALL_PARSER=qwen3_coder
  -e REASONING_PARSER=nemotron_v3
  -e KV_CACHE_DTYPE=fp8
  -e MOE_BACKEND=marlin
  -e MAMBA_BACKEND=flashinfer
  -e MAMBA_CACHE_MODE=align
  -e VLLM_MARLIN_USE_ATOMIC_ADD=1
  -e VLLM_USE_FLASHINFER_MOE_FP4=0
  -e HF_HUB_DISABLE_XET=1
  -e VLLM_USE_DEEP_GEMM=0
  -e NVIDIA_VISIBLE_DEVICES=all
)

# Speculative decoding config for performance
SPECULATIVE_CONFIG='{"model":"nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark","num_speculative_tokens":3,"method":"dspark"}'
DOCKER_ENV+=(-e SPECULATIVE_CONFIG="${SPECULATIVE_CONFIG}")

# Mount model cache and necessary directories
DOCKER_VOLUMES=(
  -v "${HF_CACHE}:/root/.cache/huggingface"
  -v "${HF_CACHE}:/home/user/.cache/huggingface"
)

# Launch command matching the verified working configuration
DOCKER_CMD=(
  docker run -d
  --name "${CONTAINER_NAME}"
  --shm-size 1g
  --gpus 1
  --network host
  --restart unless-stopped
  "${DOCKER_ENV[@]}"
  "${DOCKER_VOLUMES[@]}"
  "${VLLM_IMAGE}"
)

echo "Docker command:"
echo "  ${DOCKER_CMD[*]}" | sed 's/  /    /g'
echo ""

# Start container
"${DOCKER_CMD[@]}"

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
  if curl -sf "http://localhost:${HOST_PORT}/health" > /dev/null 2>&1; then
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
  curl -sf "http://localhost:${HOST_PORT}/v1/models" 2>/dev/null | python3 -m json.tool 2>/dev/null || true
  echo ""

  # Quick sanity check
  echo "=== Quick Test (single request) ==="
  curl -sf "http://localhost:${HOST_PORT}/v1/chat/completions" \
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
echo "Endpoint:    http://localhost:${HOST_PORT}"
echo "Health:      http://localhost:${HOST_PORT}/health"
echo "Models:      http://localhost:${HOST_PORT}/v1/models"
echo "Chat:        http://localhost:${HOST_PORT}/v1/chat/completions"
echo ""
echo "Container:   ${CONTAINER_NAME}"
echo "Logs:        docker logs -f ${CONTAINER_NAME}"
echo ""
echo "MOE Model Architecture (Nemotron 3.5 Lightning):"
echo "  Total parameters:  30B"
echo "  Active parameters: 3B (per token)"
echo "  Architecture:      Gated DeltaNet MoE"
echo "  Precision:         NVFP4"
echo "  Max Context:       1,048,576 tokens"
echo "  Speculative:       DSpark (3 tokens)"
echo ""
echo "Features Enabled:"
echo "  - Prefix Caching: ON (multi-turn optimization)"
echo "  - Chunked Prefill: ON (concurrent load handling)"
echo "  - FP8 KV Cache: ON (memory efficiency)"
echo "  - MoE Backend: Marlin (SM121 correct)"
echo "  - Reasoning Parser: nemotron_v3"
echo ""
echo "Run benchmark: ./run_nemotron_full.sh --port ${HOST_PORT}"
echo ""
echo "Stop server:  ./launch_vllm_nemotron.sh --stop"
echo "Server logs:  docker logs -f ${CONTAINER_NAME}"
