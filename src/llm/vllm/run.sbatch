#!/bin/bash

set -euo pipefail

GPUS="${GPUS:-all}"
DP_BACKEND="${DP_BACKEND:-rpc}"

info() { echo -e "[$(date +'%Y-%m-%dT%H:%M:%S%z')][info] $*"; }
err()  { echo -e "[$(date +'%Y-%m-%dT%H:%M:%S%z')][error] $*" >&2; }

IMAGE=""
CONTAINER_MOUNT="/fsx"
WORKSPACE="$PWD"
FORCE_PULL=false
ENABLE_NSYS=false
VLLM_STARTED=false
SERVE_ARGS=()

while (( "$#" )); do
  case "$1" in
    --image)            IMAGE="$2"; shift 2 ;;
    --container-mount)  CONTAINER_MOUNT="$2"; shift 2 ;;
    --workspace|-w)     WORKSPACE="$2"; shift 2 ;;
    --force|-f)         FORCE_PULL=true; shift ;;
    --nsys)             ENABLE_NSYS=true; shift ;;
    --profile)          SERVE_ARGS+=(--profiler-config "{\"profiler\": \"torch\", \"torch_profiler_dir\": \"${PWD}/vllm_profile\"}"); shift ;;
    --profiler-config)  SERVE_ARGS+=(--profiler-config "$2"); shift 2 ;;
    *)                  SERVE_ARGS+=("$1"); shift ;;
  esac
done

# Build nsys command prefix
NSYS_CMD=""
if [[ "${ENABLE_NSYS}" == "true" ]]; then
  NSYS_DIR="${WORKSPACE}/nsys-vllm"
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
fi
IMAGE="${IMAGE:-${WORKSPACE}/vllm-serve-latest.tar.gz}"
LOGDIR="${WORKSPACE}/logs"

# Build a shell-safe string from SERVE_ARGS for nested bash -c / docker exec
SERVE_ARGS_STR=$(printf '%q ' "${SERVE_ARGS[@]+"${SERVE_ARGS[@]}"}")

# Peek at SERVE_ARGS to extract values needed for topology computation
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
_has_flag() {
  for arg in "${SERVE_ARGS[@]}"; do [[ "$arg" == "$1" ]] && return 0; done
  return 1
}

TP=$(_peek_arg "-tp" "--tensor-parallel-size" "1")
PP=$(_peek_arg "-pp" "--pipeline-parallel-size" "1")
ENABLE_EP=$(_has_flag "--enable-expert-parallel" && echo "true" || echo "false")

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

  if [[ "$PP" -gt 1 && "$ENABLE_EP" == "true" ]]; then
    err "Pipeline parallel (PP=$PP) and expert parallel cannot be enabled simultaneously"
    exit 1
  fi

  [[ "$PP" -gt 1 ]] && DP_BACKEND="mp"

  if [[ "$ENABLE_EP" == "true" ]]; then
    DP=$((TOTAL_GPUS / TP))
    if [[ $((DP * TP)) -ne $TOTAL_GPUS ]]; then
      err "DP($DP) * TP($TP) = $((DP * TP)) != TOTAL_GPUS($TOTAL_GPUS)"; exit 1
    fi
  else
    DP=$((TOTAL_GPUS / (TP * PP)))
    if [[ $((DP * TP * PP)) -ne $TOTAL_GPUS ]]; then
      err "DP($DP) * TP($TP) * PP($PP) = $((DP * TP * PP)) != TOTAL_GPUS($TOTAL_GPUS)"; exit 1
    fi
  fi
  DP_LOCAL=$((GPUS_PER_NODE / TP))

  readarray -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
  HEAD_NODE=${NODES[0]}
  HEAD_IP=$(getent ahostsv4 "$HEAD_NODE" | head -1 | awk '{print $1}')

  RAY_PORT=$((6379 + (SLURM_JOB_ID % 1000)))
  RPC_PORT=$((13345 + (SLURM_JOB_ID % 1000)))

  mkdir -p "${LOGDIR}"

  info "========================================"
  info "vLLM Server"
  info "========================================"
  info "Image: ${IMAGE}"
  info "Nodes: ${NUM_NODES}, Head: ${HEAD_NODE} (${HEAD_IP}), GPUs: ${TOTAL_GPUS}"
  info "Parallelism: TP=${TP}, PP=${PP}, DP=${DP}, DP_LOCAL=${DP_LOCAL}, EP=${ENABLE_EP}"
  info "Backend: ${DP_BACKEND}"
  info "SERVE_ARGS: ${SERVE_ARGS[*]+"${SERVE_ARGS[*]}"}"
  info "========================================"
}

stop_nsys() {
  [[ "${VLLM_STARTED}" != "true" ]] && return 0
  info "Sending SIGINT to nsys processes for graceful shutdown..."
  srun --ntasks-per-node=1 bash -c '
    for cid in $(docker ps -q); do
      docker exec "$cid" pkill -INT -f "^nsys profile" 2>/dev/null || true
    done
  ' 2>/dev/null || true
}

wait_for_nsys() {
  [[ "${VLLM_STARTED}" != "true" ]] && return 0
  info "Waiting 30s for nsys to finalize profiles..."
  sleep 30
}

cleanup() {
  info "Cleaning up containers..."
  if [[ "${ENABLE_NSYS}" == "true" ]]; then
    stop_nsys
    wait_for_nsys
  fi
  srun --ntasks-per-node=1 bash -c '
    docker ps -aq | xargs -r docker stop -t 30 2>/dev/null || true
    docker ps -aq | xargs -r docker rm -f 2>/dev/null || true
  ' 2>/dev/null || true
  rm -f "${LOGDIR}/vllm_server_${SLURM_JOB_ID}.log"
}

start_ray_head() {
  info "Starting Ray head on ${HEAD_NODE}..."
  srun --nodes=1 --nodelist="${HEAD_NODE}" bash -c "
    $(declare -f launch_container)
    CONTAINER_IMAGE='${CONTAINER_IMAGE:-}' IMAGE='${IMAGE}' GPUS='${GPUS}' CONTAINER_MOUNT='${CONTAINER_MOUNT}' launch_container ray-head 'sleep infinity'
  "
  sleep 5
  srun --nodes=1 --nodelist="${HEAD_NODE}" bash -c "
    docker exec ray-head ray start --head --port=${RAY_PORT} \
      --num-gpus=${GPUS_PER_NODE} --num-cpus=96 --disable-usage-stats
  "
}

start_ray_workers() {
  [[ "$NUM_NODES" -le 1 ]] && return
  local worker_nodes=$(echo "${NODES[@]:1}" | tr ' ' ',')
  info "Starting Ray workers on ${worker_nodes}..."
  srun --nodes=$((NUM_NODES - 1)) --nodelist="${worker_nodes}" --ntasks-per-node=1 bash -c "
    $(declare -f launch_container)
    CONTAINER_IMAGE='${CONTAINER_IMAGE:-}' IMAGE='${IMAGE}' GPUS='${GPUS}' CONTAINER_MOUNT='${CONTAINER_MOUNT}' launch_container ray-worker 'sleep infinity'
  "
  sleep 5
  srun --nodes=$((NUM_NODES - 1)) --nodelist="${worker_nodes}" --ntasks-per-node=1 bash -c "
    docker exec ray-worker ray start --address=${HEAD_IP}:${RAY_PORT} \
      --num-gpus=${GPUS_PER_NODE} --num-cpus=96 --disable-usage-stats
  "
}

wait_for_gpus() {
  info "Waiting for ${TOTAL_GPUS} GPUs..."
  for _ in {1..120}; do
    local gpu_count
    gpu_count=$(srun --nodes=1 --nodelist="${HEAD_NODE}" bash -c "
      docker exec ray-head python3 -c \
        'import ray; ray.init(address=\"auto\"); print(int(ray.cluster_resources().get(\"GPU\",0))); ray.shutdown()' \
        2>/dev/null" || echo 0)
    [[ "$gpu_count" -ge "$TOTAL_GPUS" ]] && return 0
    sleep 5
  done
  err "Timeout waiting for GPUs"; return 1
}

start_vllm_ray() {
  info "Launching vllm serve (Ray)..."
  local logfile="${LOGDIR}/vllm_server_${SLURM_JOB_ID}.log"
  local extra="--host 0.0.0.0 --port 8000 --data-parallel-backend ray --data-parallel-address ${HEAD_IP} --data-parallel-size ${DP}"

  srun --nodes=1 --nodelist="${HEAD_NODE}" bash -c \
    "docker exec -d ray-head bash -c '${NSYS_CMD} vllm serve ${SERVE_ARGS_STR} ${extra} 2>&1 | tee ${logfile}'"
}

start_vllm_mp() {
  info "Starting vLLM with PP (multiprocessing)..."
  local logfile="${LOGDIR}/vllm_server_${SLURM_JOB_ID}.log"

  for i in $(seq 0 $((NUM_NODES - 1))); do
    srun --nodes=1 --nodelist="${NODES[$i]}" bash -c "
      $(declare -f launch_container)
      IMAGE='${CONTAINER_IMAGE:-${IMAGE}}' GPUS='${GPUS}' CONTAINER_MOUNT='${CONTAINER_MOUNT}' launch_container vllm-node-${i} 'sleep infinity'
    " &
  done
  wait

  local extra="--host 0.0.0.0 --port 8000 --nnodes ${NUM_NODES} --master-addr ${HEAD_IP} --master-port 29500"

  for i in $(seq 1 $((NUM_NODES - 1))); do
    srun --nodes=1 --nodelist="${NODES[$i]}" bash -c \
      "docker exec -d vllm-node-${i} bash -c '${NSYS_CMD} vllm serve ${SERVE_ARGS_STR} ${extra} --node-rank ${i} --headless 2>&1 | tee ${logfile}.node${i}'"
  done

  srun --nodes=1 --nodelist="${HEAD_NODE}" bash -c \
    "docker exec -d vllm-node-0 bash -c '${NSYS_CMD} vllm serve ${SERVE_ARGS_STR} ${extra} --node-rank 0 2>&1 | tee ${logfile}'"
}

# RPC backend
start_vllm_rpc() {
  info "Starting vLLM with RPC-based DP..."
  local logfile="${LOGDIR}/vllm_server_${SLURM_JOB_ID}.log"

  srun --nodes=1 --nodelist="${HEAD_NODE}" bash -c "
    $(declare -f launch_container)
    IMAGE='${CONTAINER_IMAGE:-${IMAGE}}' GPUS='${GPUS}' CONTAINER_MOUNT='${CONTAINER_MOUNT}' launch_container vllm-head 'sleep infinity'
  "
  for i in $(seq 1 $((NUM_NODES - 1))); do
    srun --nodes=1 --nodelist="${NODES[$i]}" bash -c "
      $(declare -f launch_container)
      IMAGE='${CONTAINER_IMAGE:-${IMAGE}}' GPUS='${GPUS}' CONTAINER_MOUNT='${CONTAINER_MOUNT}' launch_container vllm-worker 'sleep infinity'
    "
  done
  sleep 3

  local extra="--data-parallel-size ${DP} --data-parallel-size-local ${DP_LOCAL} --data-parallel-address ${HEAD_IP} --data-parallel-rpc-port ${RPC_PORT}"

  for i in $(seq 1 $((NUM_NODES - 1))); do
    local start_rank=$((i * DP_LOCAL))
    info "Starting RPC worker on ${NODES[$i]} (rank ${start_rank})..."
    srun --nodes=1 --nodelist="${NODES[$i]}" bash -c "
      docker exec -d vllm-worker bash -c '${NSYS_CMD} vllm serve ${SERVE_ARGS_STR} ${extra} \
        --data-parallel-start-rank ${start_rank} --headless \
        2>&1 | tee ${LOGDIR}/vllm_worker_${SLURM_JOB_ID}_${i}.log'
    "
  done

  srun --nodes=1 --nodelist="${HEAD_NODE}" bash -c "
    docker exec -d vllm-head bash -c '${NSYS_CMD} vllm serve ${SERVE_ARGS_STR} ${extra} \
      --host 0.0.0.0 --port 8000 \
      2>&1 | tee ${logfile}'
  "
}

wait_for_server() {
  info "Waiting for vLLM server at ${HEAD_IP}:8000..."
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

LOGFILE="${LOGDIR}/vllm_server_${SLURM_JOB_ID}.log"

load_or_pull_image

case "${DP_BACKEND}" in
  ray)
    start_ray_head
    start_ray_workers
    wait_for_gpus
    start_vllm_ray
    ;;
  mp)  start_vllm_mp ;;
  rpc) start_vllm_rpc ;;
  *)   err "Unknown backend: ${DP_BACKEND}"; exit 1 ;;
esac

tail -f "${LOGFILE}" 2>/dev/null &

wait_for_server || exit 1
VLLM_STARTED=true

info "vLLM serving on ${HEAD_IP}:8000 — Ctrl+C or scancel to stop"
info "Logs: ${LOGFILE}"
sleep infinity
