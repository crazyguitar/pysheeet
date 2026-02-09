#!/bin/bash

set -euo pipefail

GPUS="${GPUS:-all}"

info() { echo -e "[$(date +'%Y-%m-%dT%H:%M:%S%z')][info] $*"; }
err()  { echo -e "[$(date +'%Y-%m-%dT%H:%M:%S%z')][error] $*" >&2; }

IMAGE=""
CONTAINER_MOUNT="/fsx"
WORKSPACE="$PWD"
FORCE_PULL=false
SERVE_ARGS=()

while (( "$#" )); do
  case "$1" in
    --image)            IMAGE="$2"; shift 2 ;;
    --container-mount)  CONTAINER_MOUNT="$2"; shift 2 ;;
    --workspace|-w)     WORKSPACE="$2"; shift 2 ;;
    --force|-f)         FORCE_PULL=true; shift ;;
    *)                  SERVE_ARGS+=("$1"); shift ;;
  esac
done
IMAGE="${IMAGE:-${WORKSPACE}/tensorrt-llm-serve-latest.tar.gz}"
LOGDIR="${WORKSPACE}/logs"

SERVE_ARGS_STR=$(printf '%q ' "${SERVE_ARGS[@]+"${SERVE_ARGS[@]}"}")

_peek_arg() {
  local short="$1" long="$2" default="$3"
  local i=0
  while (( i < ${#SERVE_ARGS[@]} )); do
    if [[ "${SERVE_ARGS[$i]}" == "$short" || "${SERVE_ARGS[$i]}" == "$long" ]]; then
      echo "${SERVE_ARGS[$((i+1))]}"; return
    fi
    ((i++))
  done
  echo "$default"
}

load_or_pull_image() {
  if [[ "${FORCE_PULL}" == "true" ]]; then
    info "Force pull: cleaning up existing images..."
    srun --ntasks-per-node=1 bash -c '
      docker ps -aq | xargs -r docker rm -f 2>/dev/null || true
      docker images -aq | xargs -r docker rmi -f 2>/dev/null || true
    '
  fi

  if [[ "${IMAGE}" == *.tar.gz ]]; then
    info "Loading Docker image from tarball..."
    CONTAINER_IMAGE=$(pigz -dc "${IMAGE}" | tar -xf - -O manifest.json | python3 -c "import sys,json; print(json.load(sys.stdin)[0]['RepoTags'][0])")
    info "Image tag: ${CONTAINER_IMAGE}"
    srun --ntasks-per-node=1 bash -c "
      if ! docker image inspect '${CONTAINER_IMAGE}' &>/dev/null; then
        pigz -dc '${IMAGE}' | docker load
      fi
    "
  else
    info "Pulling Docker image from registry..."
    local registry="${IMAGE%%/*}"
    local region=$(echo "${registry}" | sed -n 's/.*\.ecr\.\([^.]*\)\.amazonaws\.com/\1/p')
    region="${region:-us-west-2}"
    srun --ntasks-per-node=1 bash -c "
      if ! docker image inspect '${IMAGE}' &>/dev/null; then
        aws ecr get-login-password --region '${region}' | docker login --username AWS --password-stdin '${registry}'
        docker pull '${IMAGE}'
      fi
    "
    CONTAINER_IMAGE="${IMAGE}"
  fi
}

launch_container() {
  local name="${1}" cmd="${2}"
  local devices=("--device=/dev/gdrdrv")
  while IFS= read -r -d '' d; do
    devices+=("--device=${d}")
  done < <(find "/dev/infiniband" -name "uverbs*" -print0 2>/dev/null)

  local net_if="${GLOO_SOCKET_IFNAME:-$(ip -o -4 route show to default | awk '{print $5}' | head -1)}"

  docker run --gpus "${GPUS}" \
    --privileged -d \
    --name "${name}" \
    --uts=host --ipc=host --net=host \
    --ulimit stack=67108864 --ulimit memlock=-1 \
    --security-opt seccomp=unconfined \
    "${devices[@]}" \
    -v "${CONTAINER_MOUNT}:${CONTAINER_MOUNT}" \
    -e NCCL_SOCKET_IFNAME="${net_if}" \
    -e GLOO_SOCKET_IFNAME="${net_if}" \
    -e TP_SOCKET_IFNAME="${net_if}" \
    --entrypoint bash \
    "${CONTAINER_IMAGE:-${IMAGE}}" \
    -c "${cmd}"
}

setup_topology() {
  NUM_NODES=${SLURM_JOB_NUM_NODES:-1}
  GPUS_PER_NODE=8
  TOTAL_GPUS=$((NUM_NODES * GPUS_PER_NODE))

  readarray -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
  HEAD_NODE=${NODES[0]}
  HEAD_IP=$(getent ahostsv4 "$HEAD_NODE" | head -1 | awk '{print $1}')

  mkdir -p "${LOGDIR}"

  info "========================================"
  info "TensorRT-LLM Server"
  info "========================================"
  info "Image: ${IMAGE}"
  info "Nodes: ${NUM_NODES}, Head: ${HEAD_NODE} (${HEAD_IP}), GPUs: ${TOTAL_GPUS}"
  info "SERVE_ARGS: ${SERVE_ARGS[*]+"${SERVE_ARGS[*]}"}"
  info "========================================"
}

cleanup() {
  info "Cleaning up containers..."
  srun --ntasks-per-node=1 bash -c '
    docker ps -aq | xargs -r docker rm -f 2>/dev/null || true
  ' 2>/dev/null || true
  rm -f "${LOGDIR}/trtllm_server_${SLURM_JOB_ID}.log"
}

start_trtllm() {
  local logfile="${LOGDIR}/trtllm_server_${SLURM_JOB_ID}.log"

  # Launch container on head node
  srun --nodes=1 --nodelist="${HEAD_NODE}" bash -c "
    $(declare -f launch_container)
    CONTAINER_IMAGE='${CONTAINER_IMAGE:-${IMAGE}}' GPUS='${GPUS}' CONTAINER_MOUNT='${CONTAINER_MOUNT}' launch_container trtllm-node-0 'sleep infinity'
  "
  sleep 3

  # trtllm-serve syntax: trtllm-serve <MODEL> [OPTIONS]
  srun --nodes=1 --nodelist="${HEAD_NODE}" bash -c "
    docker exec -d trtllm-node-0 bash -c 'trtllm-serve ${SERVE_ARGS_STR} --host 0.0.0.0 --port 8000 2>&1 | tee ${logfile}'
  "
}

wait_for_server() {
  info "Waiting for TensorRT-LLM server at ${HEAD_IP}:8000..."
  for _ in {1..360}; do
    if srun --nodes=1 --nodelist="${HEAD_NODE}" bash -c "curl -sf localhost:8000/health" &>/dev/null &&
       srun --nodes=1 --nodelist="${HEAD_NODE}" bash -c "curl -sf localhost:8000/v1/models | grep -q '\"id\"'" &>/dev/null; then
      info "Server ready at ${HEAD_IP}:8000"
      return 0
    fi
    sleep 10
  done
  err "Timeout waiting for server"; return 1
}

setup_topology
trap cleanup EXIT
cleanup

LOGFILE="${LOGDIR}/trtllm_server_${SLURM_JOB_ID}.log"

load_or_pull_image
start_trtllm

tail -f "${LOGFILE}" 2>/dev/null &

wait_for_server || exit 1

info "TensorRT-LLM serving on ${HEAD_IP}:8000 — Ctrl+C or scancel to stop"
info "Logs: ${LOGFILE}"
sleep infinity
