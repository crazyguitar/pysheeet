#!/bin/bash
set -euo pipefail

info() { echo -e "[$(date +'%Y-%m-%dT%H:%M:%S%z')][info] $*"; }

CONTAINER_MOUNT="${CONTAINER_MOUNT:-/fsx}"
WORKSPACE="${WORKSPACE:-$PWD}"
GPUS="${GPUS:-all}"
IMAGE="${IMAGE:-$PWD/ray+latest.tar.gz}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image) IMAGE="$2"; shift 2 ;;
    *) break ;;
  esac
done

load_image() {
  info "Loading image on all nodes..."
  if [[ "${IMAGE}" == *.tar.gz ]]; then
    CONTAINER_IMAGE=$(pigz -dc "${IMAGE}" | tar -xf - -O manifest.json | python3 -c "import sys,json; print(json.load(sys.stdin)[0]['RepoTags'][0])")
    srun --ntasks-per-node=1 bash -c "
      if ! docker image inspect '${CONTAINER_IMAGE}' &>/dev/null; then
        pigz -dc '${IMAGE}' | docker load
      fi
    "
  else
    CONTAINER_IMAGE="${IMAGE}"
    srun --ntasks-per-node=1 bash -c "docker pull '${IMAGE}' || true"
  fi
}

setup_cluster_info() {
  NUM_NODES=${SLURM_JOB_NUM_NODES:-1}
  readarray -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
  HEAD_NODE=${NODES[0]}
  HEAD_IP=$(getent ahostsv4 "$HEAD_NODE" | head -1 | awk '{print $1}')
  RAY_PORT=$((6379 + (SLURM_JOB_ID % 1000)))

  info "Ray cluster: ${NUM_NODES} nodes, head=${HEAD_NODE} (${HEAD_IP}:${RAY_PORT})"
}

launch_container() {
  local name="${1}"; shift
  local devices=("--device=/dev/gdrdrv")
  while IFS= read -r -d '' d; do
    devices+=("--device=${d}")
  done < <(find "/dev/infiniband" -name "uverbs*" -print0 2>/dev/null)

  docker run --gpus "${GPUS}" --privileged -d --name "${name}" \
    --uts=host --ipc=host --net=host \
    --ulimit stack=67108864 --ulimit memlock=-1 \
    --security-opt seccomp=unconfined \
    "${devices[@]}" \
    -v "${CONTAINER_MOUNT}:${CONTAINER_MOUNT}" \
    -v "${WORKSPACE}:${WORKSPACE}" \
    "${CONTAINER_IMAGE}" "$@"
}

start_ray_head() {
  info "Starting Ray head on ${HEAD_NODE}..."
  srun --nodes=1 --nodelist="${HEAD_NODE}" bash -c "
    $(declare -f launch_container)
    CONTAINER_IMAGE='${CONTAINER_IMAGE}' GPUS='${GPUS}' CONTAINER_MOUNT='${CONTAINER_MOUNT}' \
      launch_container ray-head \
        ray start --head --port=${RAY_PORT} \
        --dashboard-host=0.0.0.0 \
        --num-gpus=\$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l) \
        --num-cpus=\$(nproc) \
        --disable-usage-stats --block
  "
}

start_ray_workers() {
  [[ "$NUM_NODES" -le 1 ]] && return

  local worker_nodes
  worker_nodes=$(echo "${NODES[@]:1}" | tr ' ' ',')
  info "Starting Ray workers on ${worker_nodes}..."

  srun --nodes=$((NUM_NODES - 1)) --nodelist="${worker_nodes}" --ntasks-per-node=1 bash -c "
    $(declare -f launch_container)
    CONTAINER_IMAGE='${CONTAINER_IMAGE}' GPUS='${GPUS}' CONTAINER_MOUNT='${CONTAINER_MOUNT}' \
      launch_container ray-worker \
        ray start --address=${HEAD_IP}:${RAY_PORT} \
        --num-gpus=\$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l) \
        --num-cpus=\$(nproc) \
        --disable-usage-stats --block
  "
}

wait_for_ray() {
  local max_retries=30
  local interval=5
  info "Waiting for Ray cluster (${NUM_NODES} nodes)..."
  for ((i = 1; i <= max_retries; i++)); do
    local alive
    alive=$(srun --nodes=1 --nodelist="${HEAD_NODE}" bash -c \
      "docker exec ray-head ray status 2>/dev/null | grep -c 'node_' || true")
    if [[ "$alive" -ge "$NUM_NODES" ]]; then
      info "All ${NUM_NODES} nodes registered."
      srun --nodes=1 --nodelist="${HEAD_NODE}" bash -c "docker exec ray-head ray status"
      return 0
    fi
    info "  ${alive}/${NUM_NODES} nodes ready (attempt ${i}/${max_retries})"
    sleep "$interval"
  done
  info "ERROR: Timed out waiting for Ray cluster."
  return 1
}

# Main
load_image
setup_cluster_info
start_ray_head
start_ray_workers
wait_for_ray

info "Ray cluster ready at ${HEAD_IP}:${RAY_PORT}"
