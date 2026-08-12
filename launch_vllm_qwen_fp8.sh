#!/usr/bin/env bash
# =============================================================================
# launch_vllm_qwen_fp8.sh - Launch Qwen3.6-35B-A3B-FP8
# =============================================================================
# Deploys Qwen/Qwen3.6-35B-A3B-FP8 using generic vLLM container.
#
# Usage:
#   ./launch_vllm_qwen_fp8.sh [--port PORT] [--stop] [--status]
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:v0.24.0-ubuntu2404}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.6-35B-A3B-FP8}"
CONTAINER_NAME="${CONTAINER_NAME:-vllm-qwen-fp8-gb10}"
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
Usage: ./launch_vllm_qwen_fp8.sh [OPTIONS]

Launch vLLM container with Qwen3.6-35B-A3B-FP8 on GB10.

Options:
  --port PORT         API port (default: 80)
  --model MODEL       HuggingFace model ID
  --name NAME         Container name
  --image IMAGE       Docker image
  --cache DIR         HuggingFace cache directory
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
echo " vLLM Container: Qwen3.6-35B-A3B-FP8"
echo "============================================"
echo "Model:       ${MODEL_ID}"
echo "Image:       ${VLLM_IMAGE}"
echo "Port:        ${HOST_PORT} -> 8000"
echo ""

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN not set"
  exit 1
fi

if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  echo "Removing existing container: ${CONTAINER_NAME}"
  docker rm -f "${CONTAINER_NAME}" > /dev/null 2>&1
fi

# ---------------------------------------------------------------------------
# Launch vLLM (generic container uses CLI flags)
# ---------------------------------------------------------------------------
echo ""
echo "Launching vLLM server..."
echo ""

DOCKER_CMD=(
  docker run -d
  --name "${CONTAINER_NAME}"
  --shm-size 1g
  --gpus 1
  --network host
  --restart unless-stopped
  -e HF_TOKEN="${HF_TOKEN}"
  -v "${HF_CACHE}:/root/.cache/huggingface"
  "${VLLM_IMAGE}"
  --model "${MODEL_ID}"
  --served-model-name "${MODEL_ID}"
  --tensor-parallel-size 1
  --max-model-len 131072
  --kv-cache-dtype fp8_e4m3
  --enable-prefix-caching
  --trust-remote-code
  --enforce-eager
  --port 8000
)

echo "Docker command:"
printf "  %s \\\n" "${DOCKER_CMD[@]}" | head -20
echo ""

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
  echo "=== Model Info ==="
  curl -sf "http://localhost:${HOST_PORT}/v1/models" 2>/dev/null | python3 -m json.tool 2>/dev/null || true
else
  echo "TIMEOUT: Server not ready after ${MAX_WAIT}s"
  echo "Check logs: docker logs ${CONTAINER_NAME}"
  exit 1
fi

echo ""
echo "============================================"
echo " Server Ready"
echo "============================================"
echo "Endpoint:    http://localhost:${HOST_PORT}"
echo "Container:   ${CONTAINER_NAME}"
echo "Run benchmark: ./run_model_benchmark.sh --model qwen-fp8 --port ${HOST_PORT}"
echo "Stop server:  ./launch_vllm_qwen_fp8.sh --stop"
