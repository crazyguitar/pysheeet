#!/bin/bash
# Usage:
#   salloc -N 2 bash nixl.sbatch [--image <path>] [nixlbench args...]
#
# Examples:
#   salloc -N 2 bash nixl.sbatch --backend UCX --initiator_seg_type VRAM --target_seg_type VRAM
#   salloc -N 2 bash nixl.sbatch --image ./nixl-test-latest.tar.gz --backend UCX

set -euo pipefail

info() { echo -e "[$(date +'%Y-%m-%dT%H:%M:%S%z')][info] $*"; }
err() { echo -e "[$(date +'%Y-%m-%dT%H:%M:%S%z')][error] $*" >&2; }

IMAGE=""
WORKSPACE="$PWD"
CONTAINER_MOUNT="/fsx"
ETCD_PORT=$((2379 + (SLURM_JOB_ID % 1000)))
BENCH_ARGS=()
DOCKER_ENVS=()

while (("$#")); do
  case "$1" in
  --image) IMAGE="$2"; shift 2 ;;
  --container-mount) CONTAINER_MOUNT="$2"; shift 2 ;;
  --env) DOCKER_ENVS+=("-e" "$2"); shift 2 ;;
  *) BENCH_ARGS+=("$1"); shift ;;
  esac
done

IMAGE="${IMAGE:-${WORKSPACE}/nixl-latest.tar.gz}"
NUM_NODES=${SLURM_JOB_NUM_NODES:-1}
readarray -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
HEAD_NODE=${NODES[0]}
HEAD_IP=$(getent ahostsv4 "$HEAD_NODE" | head -1 | awk '{print $1}')

info "========================================"
info "nixlbench"
info "========================================"
info "Image: ${IMAGE}"
info "Nodes: ${NUM_NODES}, Head: ${HEAD_NODE} (${HEAD_IP})"
info "Args: ${BENCH_ARGS[*]+"${BENCH_ARGS[*]}"}"
info "========================================"

load_image() {
  if [[ "${IMAGE}" == *.tar.gz ]]; then
    CONTAINER_IMAGE=$(pigz -dc "${IMAGE}" | tar -xf - -O manifest.json | python3 -c "import sys,json; print(json.load(sys.stdin)[0]['RepoTags'][0])")
    info "Image tag: ${CONTAINER_IMAGE}"
    srun --ntasks-per-node=1 bash -c "
      if ! docker image inspect '${CONTAINER_IMAGE}' &>/dev/null; then
        pigz -dc '${IMAGE}' | docker load
      fi
    "
  else
    CONTAINER_IMAGE="${IMAGE}"
    srun --ntasks-per-node=1 bash -c "
      docker image inspect '${CONTAINER_IMAGE}' &>/dev/null || docker pull '${CONTAINER_IMAGE}'
    "
  fi
}

launch_container() {
  local name="$1" node="$2" cmd="$3"
  local devices=("--device=/dev/gdrdrv")
  while IFS= read -r -d '' d; do
    devices+=("--device=${d}")
  done < <(find "/dev/infiniband" -name "uverbs*" -print0 2>/dev/null)

  srun --nodes=1 --nodelist="${node}" bash -c "
    docker run --gpus all \
      --privileged -d \
      --name '${name}' \
      --uts=host --ipc=host --net=host \
      --ulimit stack=67108864 --ulimit memlock=-1 \
      --security-opt seccomp=unconfined \
      ${devices[*]+"${devices[*]}"} \
      ${DOCKER_ENVS[*]+"${DOCKER_ENVS[*]}"} \
      -v '${CONTAINER_MOUNT}:${CONTAINER_MOUNT}' \
      --entrypoint bash \
      '${CONTAINER_IMAGE}' \
      -c '${cmd}'
  "
}

LOGDIR="${WORKSPACE}/logs"

cleanup() {
  info "Cleaning up..."
  kill $(jobs -p) 2>/dev/null || true
  srun --ntasks-per-node=1 bash -c '
    docker ps -aq | xargs -r docker stop -t 10 2>/dev/null || true
    docker ps -aq | xargs -r docker rm -f 2>/dev/null || true
  ' 2>/dev/null || true
  rm -f "${LOGDIR}"/nixlbench-*.log
}

trap cleanup EXIT
cleanup
load_image

# Start etcd on head node
info "Starting etcd on ${HEAD_NODE}:${ETCD_PORT}..."
launch_container "etcd" "${HEAD_NODE}" \
  "etcd --data-dir=/tmp/etcd-data \
        --listen-client-urls http://0.0.0.0:${ETCD_PORT} \
        --advertise-client-urls http://${HEAD_IP}:${ETCD_PORT} \
        --listen-peer-urls http://0.0.0.0:$((ETCD_PORT+1)) \
        --initial-advertise-peer-urls http://${HEAD_IP}:$((ETCD_PORT+1)) \
        --initial-cluster default=http://${HEAD_IP}:$((ETCD_PORT+1))"
ETCD_URL="http://${HEAD_IP}:${ETCD_PORT}"

# Wait for etcd to be ready
info "Waiting for etcd at ${ETCD_URL}..."
for i in $(seq 1 60); do
  if curl -sf "${ETCD_URL}/health" >/dev/null 2>&1; then
    info "etcd is ready"
    break
  fi
  if [ "$i" -eq 60 ]; then
    err "etcd failed to start within 60s"
    exit 1
  fi
  sleep 2
done

# Launch nixlbench containers
for i in $(seq 0 $((NUM_NODES - 1))); do
  launch_container "nixl-${i}" "${NODES[$i]}" "sleep infinity"
done
sleep 3

BENCH_ARGS_STR="${BENCH_ARGS[*]+"${BENCH_ARGS[*]}"}"
mkdir -p "${LOGDIR}"

# Start nixlbench on worker nodes
for i in $(seq 1 $((NUM_NODES - 1))); do
  info "Starting nixlbench on ${NODES[$i]}..."
  srun --nodes=1 --nodelist="${NODES[$i]}" bash -c "
    docker exec -d nixl-${i} bash -c 'nixlbench --etcd_endpoints ${ETCD_URL} ${BENCH_ARGS_STR} 2>&1 | tee ${LOGDIR}/nixlbench-${NODES[$i]}.log'
  "
  tail -f "${LOGDIR}/nixlbench-${NODES[$i]}.log" 2>/dev/null | sed "s/^/[${NODES[$i]}] /" &
  sleep 1
done

# Start nixlbench on head node (foreground)
info "Starting nixlbench on ${HEAD_NODE}..."
srun --nodes=1 --nodelist="${HEAD_NODE}" bash -c "
  docker exec nixl-0 bash -c 'nixlbench --etcd_endpoints ${ETCD_URL} ${BENCH_ARGS_STR} 2>&1 | tee ${LOGDIR}/nixlbench-${HEAD_NODE}.log'
"
