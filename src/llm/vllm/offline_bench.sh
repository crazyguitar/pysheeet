#!/usr/bin/env bash
# Offline vLLM benchmark wrapper (no API server)
# Usage:
#   salloc -N1 bash offline_bench.sh --model meta-llama/Llama-3.1-8B --num-prompts 50
#   salloc -N2 bash offline_bench.sh --model Qwen/Qwen2-57B-A14B --tp-size 4 --enable-ep
set -euo pipefail

info() { echo "[$(date +'%H:%M:%S')] $*"; }

CONTAINER_MOUNT="${CONTAINER_MOUNT:-/fsx}"
IMAGE="${IMAGE:-${PWD}/vllm-serve-latest.tar.gz}"
NPROC="${NPROC:-}"

# Setup multi-node coordination early (before container launch)
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  NUM_NODES=${SLURM_JOB_NUM_NODES:-1}
  readarray -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
  HEAD_NODE=${NODES[0]}
  HEAD_IP=$(getent ahostsv4 "$HEAD_NODE" | head -1 | awk '{print $1}')
  MASTER_PORT=$((29500 + (SLURM_JOB_ID % 1000)))
else
  NUM_NODES=1
  HEAD_IP="127.0.0.1"
  MASTER_PORT=29500
fi

_run() {
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    srun bash -c "$*"
  else
    bash -c "$*"
  fi
}

load_or_pull_image() {
  if [[ "${IMAGE}" == *.tar.gz ]]; then
    CONTAINER_IMAGE=$(pigz -dc "${IMAGE}" | tar -xf - -O manifest.json |
      python3 -c "import sys,json; print(json.load(sys.stdin)[0]['RepoTags'][0])")
    info "Image tag: ${CONTAINER_IMAGE}"
    _run "
            if ! docker image inspect '${CONTAINER_IMAGE}' &>/dev/null; then
                echo 'Loading Docker image from tarball...'
                pigz -dc '${IMAGE}' | docker load
            fi
        "
  else
    CONTAINER_IMAGE="${IMAGE}"
    _run "
            if ! docker image inspect '${CONTAINER_IMAGE}' &>/dev/null; then
                echo 'Pulling ${CONTAINER_IMAGE}...'
                registry=\"\${CONTAINER_IMAGE%%/*}\"
                region=\$(echo \"\${registry}\" | sed -n 's/.*\.ecr\.\([^.]*\)\.amazonaws\.com/\1/p')
                region=\"\${region:-us-west-2}\"
                aws ecr get-login-password --region \"\${region}\" \
                    | docker login --username AWS --password-stdin \"\${registry}\"
                docker pull '${CONTAINER_IMAGE}'
            fi
        "
  fi
}

launch_container() {
  local cmd="$1"
  _run "
        docker run --rm --gpus all --privileged --ipc=host --net=host \
            -v '${PWD}:${PWD}' -w '${PWD}' \
            -v '${CONTAINER_MOUNT}:${CONTAINER_MOUNT}' \
            --entrypoint bash '${CONTAINER_IMAGE}' \
            -c '${cmd}'
    "
}

usage() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS] MODEL_ARGS...

Script Options:
  --nproc N         Number of processes for torchrun (auto-detect if not set)
  --image IMAGE     Docker image or tarball (default: ./vllm-serve-latest.tar.gz)
  --nsys            Enable nsys profiling
  -h, --help        Show this help

Model/Benchmark Args (passed to offline_bench.py):
  --model MODEL                             Model name or path (required)
  --num-prompts N                           Number of prompts (default: 50)
  --tensor-parallel-size, --tp-size N       Tensor parallel size (default: 1)
  --pipeline-parallel-size, --pp-size N     Pipeline parallel size (default: 1)
  --data-parallel-size, --dp-size N         Data parallel size (default: 1)
  --enable-expert-parallel, --enable-ep     Enable expert parallel
  --all2all-backend TYPE                    All-to-all backend (allgather_reducescatter, nccl)
  --enforce-eager                           Disable CUDA graph
  --max-tokens N                            Max output tokens (default: 128)
  --dataset-path PATH                       Path to ShareGPT dataset
  --viztracer FILE                          Enable VizTracer and save to file (e.g., ./vllm-trace.json)

Examples:
  # Single GPU
  $(basename "$0") --model meta-llama/Llama-3.1-8B --num-prompts 50

  # Multi-GPU with TP
  $(basename "$0") --model Qwen/Qwen2-57B-A14B --tp-size 4 --enable-ep --num-prompts 100

  # Multi-GPU with DP
  $(basename "$0") --model meta-llama/Llama-3.1-8B --tp-size 2 --dp-size 4 --num-prompts 200

  # With nsys profiling
  $(basename "$0") --nsys --model Qwen/Qwen2-57B-A14B --tp-size 4 --enable-ep --num-prompts 50
EOF
}

BENCH_ARGS=()
TP_SIZE=1
DP_SIZE=1
ENABLE_NSYS=false

while [[ $# -gt 0 ]]; do
  case "$1" in
  --nproc)
    NPROC="$2"
    shift 2
    ;;
  --image)
    IMAGE="$2"
    shift 2
    ;;
  --nsys)
    ENABLE_NSYS=true
    shift
    ;;
  -h | --help)
    usage
    exit 0
    ;;
  *)
    BENCH_ARGS+=("$1")
    shift
    ;;
  esac
done

# Parse BENCH_ARGS to extract TP/DP sizes for nproc calculation
for ((i = 0; i < ${#BENCH_ARGS[@]}; i++)); do
  case "${BENCH_ARGS[$i]}" in
  --tp-size | --tensor-parallel-size)
    TP_SIZE="${BENCH_ARGS[$((i + 1))]}"
    ;;
  --dp-size | --data-parallel-size)
    DP_SIZE="${BENCH_ARGS[$((i + 1))]}"
    ;;
  esac
done

# Auto-detect nproc if not set
if [[ -z "$NPROC" ]]; then
  # Calculate based on total GPUs across all nodes
  GPUS_PER_NODE=8
  TOTAL_GPUS=$((NUM_NODES * GPUS_PER_NODE))

  # If DP not specified, auto-calculate
  if [[ "$DP_SIZE" -eq 1 ]]; then
    DP_SIZE=$((TOTAL_GPUS / TP_SIZE))
    [[ $DP_SIZE -lt 1 ]] && DP_SIZE=1
    info "Auto-detected: DP_SIZE=${DP_SIZE} (${TOTAL_GPUS} GPUs / TP_SIZE=${TP_SIZE})"
    # Inject --data-parallel-size into BENCH_ARGS
    BENCH_ARGS+=("--data-parallel-size" "$DP_SIZE")
  fi

  # nproc-per-node = GPUs per node
  NPROC=$GPUS_PER_NODE
fi

# If python is not available or not in container, load image and run in container
if ! command -v python3 &>/dev/null || ! python3 -c "import vllm" &>/dev/null 2>&1; then
  load_or_pull_image
  SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

  # Build nsys command prefix
  NSYS_CMD=""
  NSYS_ARG=""
  if [[ "${ENABLE_NSYS}" == "true" ]]; then
    NSYS_DIR="${PWD}/nsys-offline"
    mkdir -p "${NSYS_DIR}"
    NSYS_PATH="${NSYS_DIR}/profile-node${SLURM_NODEID:-0}.nsys-rep"
    NSYS_CMD="nsys profile"
    NSYS_CMD+=" -t cuda,nvtx,osrt,cudnn,cublas"
    NSYS_CMD+=" --trace-fork-before-exec=true"
    NSYS_CMD+=" --cuda-graph-trace=node"
    NSYS_CMD+=" --capture-range=cudaProfilerApi"
    NSYS_CMD+=" --capture-range-end=repeat"
    NSYS_CMD+=" --cuda-memory-usage=true"
    NSYS_CMD+=" --cudabacktrace=true"
    NSYS_CMD+=" -o ${NSYS_PATH}"
    NSYS_CMD+=" --force-overwrite=true"
    NSYS_ARG="--nsys ${NSYS_PATH}"
  fi

  TORCHRUN_CMD="${NSYS_CMD:+${NSYS_CMD} }torchrun \
        --nnodes=${NUM_NODES} \
        --nproc-per-node=${NPROC} \
        --rdzv-backend=c10d \
        --rdzv-endpoint=${HEAD_IP}:${MASTER_PORT} \
        --rdzv-id=${SLURM_JOB_ID:-12345} \
        ${SCRIPT_DIR}/offline_bench.py ${BENCH_ARGS[*]}"

  info "========================================"
  info "Offline vLLM Benchmark"
  info "========================================"
  info "Nodes: ${NUM_NODES}, Processes per node: ${NPROC}"
  info "Rendezvous: ${HEAD_IP}:${MASTER_PORT}"
  info "Command: ${TORCHRUN_CMD}"
  info "========================================"

  launch_container "${TORCHRUN_CMD}"
  exit $?
fi

# This code only runs if already inside container with vllm available
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

info "========================================"
info "Offline vLLM Benchmark (inside container)"
info "========================================"
info "Nodes: ${NUM_NODES}, Processes per node: ${NPROC}"
info "Rendezvous: ${HEAD_IP}:${MASTER_PORT}"
info "========================================"

${NSYS_CMD} torchrun \
  --nnodes="${NUM_NODES}" \
  --nproc-per-node="${NPROC}" \
  --rdzv-backend=c10d \
  --rdzv-endpoint="${HEAD_IP}:${MASTER_PORT}" \
  --rdzv-id="${SLURM_JOB_ID:-12345}" \
  "${SCRIPT_DIR}/offline_bench.py" \
  ${NSYS_ARG} \
  "${BENCH_ARGS[@]}"
