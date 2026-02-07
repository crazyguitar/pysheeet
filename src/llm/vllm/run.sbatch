#!/bin/bash

set -euo pipefail

IMAGE="${IMAGE:-vllm-serve-latest.tar.gz}"
MODEL="Qwen/Qwen2.5-7B-Instruct"
PORT="8000"
TP="1"
PP="1"
DP=""
ENABLE_EP=false
MODE=""
FORCE_CLEANUP=false
EXTRA_ARGS=()

info() { echo -e "[$(date +'%Y-%m-%dT%H:%M:%S%z')][info] $*"; }
err()  { echo -e "[$(date +'%Y-%m-%dT%H:%M:%S%z')][error] $*" >&2; }

usage() {
  cat <<EOF
Usage: salloc -N <nodes> bash $0 [OPTIONS]

Options:
  -h,--help              show this help
  -m,--model MODEL       model path or HF ID (default: ${MODEL})
  -p,--port PORT         API port (default: ${PORT})
  --image IMAGE          container image: .tar.gz or registry path
  --tp SIZE              tensor parallel size (default: 1)
  --pp SIZE              pipeline parallel size (default: 1)
  --dp SIZE              data parallel size (auto-computed if omitted)
  --ep                   enable expert parallel (uses Ray backend)
  --mode MODE            force mode: ray|mp|rpc (auto-selected otherwise)
  --force                force docker rm and docker rmi cleanup
  --                     remaining args passed to vllm serve

Examples:
  salloc -N 2 bash $0 --image /fsx/images/vllm.tar.gz -m Qwen/Qwen3-30B-A3B-FP8 --tp 8
  salloc -N 2 bash $0 --image /fsx/images/vllm.tar.gz -m Qwen/Qwen3-30B-A3B-FP8 --tp 8 --ep
  salloc -N 2 bash $0 --image registry.example.com/vllm:latest -m Qwen/Qwen3-30B-A3B-FP8 --tp 8 --pp 2
EOF
}

while (( "$#" )); do
  case "$1" in
    -h|--help)   usage; exit 0 ;;
    -m|--model)  MODEL="$2"; shift 2 ;;
    -p|--port)   PORT="$2"; shift 2 ;;
    --image)     IMAGE="$2"; shift 2 ;;
    --tp)        TP="$2"; shift 2 ;;
    --pp)        PP="$2"; shift 2 ;;
    --dp)        DP="$2"; shift 2 ;;
    --ep)        ENABLE_EP=true; shift ;;
    --mode)      MODE="$2"; shift 2 ;;
    --force)     FORCE_CLEANUP=true; shift ;;
    --)          shift; EXTRA_ARGS+=("$@"); break ;;
    *)           EXTRA_ARGS+=("$1"); shift ;;
  esac
done

NUM_NODES=${SLURM_JOB_NUM_NODES:-1}
GPUS_PER_NODE=8
TOTAL_GPUS=$((NUM_NODES * GPUS_PER_NODE))

readarray -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
HEAD_NODE=${NODES[0]}
HEAD_IP=$(getent ahostsv4 "$HEAD_NODE" | head -1 | awk '{print $1}')

if [[ -z "$DP" ]]; then
  DP=$((TOTAL_GPUS / (TP * PP)))
fi

if [[ $((DP * TP * PP)) -ne $TOTAL_GPUS ]]; then
  err "DP($DP) * TP($TP) * PP($PP) = $((DP * TP * PP)) != TOTAL_GPUS($TOTAL_GPUS)"
  exit 1
fi

if [[ -z "$MODE" ]]; then
  if [[ "$ENABLE_EP" == "true" ]]; then
    MODE="ray"
  elif [[ "$PP" -gt 1 ]]; then
    MODE="mp"
  else
    MODE="rpc"
  fi
fi

mkdir -p "/fsx/users/${USER}/logs"
LOGFILE="/fsx/users/${USER}/logs/vllm_server_${SLURM_JOB_ID:-$$}.log"
info "Nodes=${NUM_NODES} Head=${HEAD_NODE}(${HEAD_IP}) GPUs=${TOTAL_GPUS}"
info "Mode=${MODE} TP=${TP} PP=${PP} DP=${DP} EP=${ENABLE_EP}"
info "Model=${MODEL} Image=${IMAGE}"

CONTAINER_IMAGE=""

load_or_pull_image() {
  if [[ "${FORCE_CLEANUP}" == "true" ]]; then
    info "Force cleanup: removing all containers and images..."
    srun --ntasks-per-node=1 bash -c '
      docker ps -aq | xargs -r docker rm -f 2>/dev/null || true
      docker images -aq | xargs -r docker rmi -f 2>/dev/null || true
    '
  fi

  if [[ "${IMAGE}" == *.tar.gz ]]; then
    info "Loading Docker image from tarball: ${IMAGE}"
    srun --ntasks-per-node=1 bash -c "
      if ! docker image inspect vllm-serve:latest &>/dev/null; then
        pigz -dc '${IMAGE}' | docker load
      fi
    "
    CONTAINER_IMAGE="vllm-serve:latest"
  else
    info "Pulling Docker image from registry: ${IMAGE}"
    local registry="${IMAGE%%/*}"
    local region
    region=$(echo "${registry}" | sed -n 's/.*\.ecr\.\([^.]*\)\.amazonaws\.com/\1/p')
    region="${region:-us-west-2}"
    srun --ntasks-per-node=1 bash -c "
      if ! docker image inspect '${IMAGE}' &>/dev/null; then
        aws ecr get-login-password --region '${region}' | \
          docker login --username AWS --password-stdin '${registry}' 2>/dev/null || true
        docker pull '${IMAGE}'
      fi
    "
    CONTAINER_IMAGE="${IMAGE}"
  fi
}


# Launch a named Docker container with GPU support on a specific node.
# Container runs 'sleep infinity' — use docker exec to run commands inside.
# Usage: launch <node> <name>
launch() {
  local node="$1" name="$2"
  srun --nodes=1 --nodelist="${node}" bash -c "
    $(declare -f launch_container)
    CONTAINER_IMAGE='${CONTAINER_IMAGE}' launch_container '${name}'
  "
}

# Run docker locally — called by launch() on the target node via srun.
launch_container() {
  local name="$1"

  # idempotent: skip if already running
  if docker ps --format '{{.Names}}' | grep -q "^${name}$"; then
    return 0
  fi
  docker rm -f "${name}" 2>/dev/null || true

  local devices=("--device=/dev/gdrdrv")
  while IFS= read -r -d '' d; do
    devices+=("--device=${d}")
  done < <(find /dev/infiniband -name "uverbs*" -print0 2>/dev/null)

  local net_if="${GLOO_SOCKET_IFNAME:-$(ip -o -4 route show to default | awk '{print $5}' | head -1)}"

  docker run --gpus all \
    --privileged -d \
    --name "${name}" \
    --uts=host --ipc=host --net=host \
    --ulimit stack=67108864 --ulimit memlock=-1 \
    --security-opt seccomp=unconfined \
    "${devices[@]}" \
    -v /fsx:/fsx \
    -e NCCL_SOCKET_IFNAME="${net_if}" \
    -e GLOO_SOCKET_IFNAME="${net_if}" \
    -e TP_SOCKET_IFNAME="${net_if}" \
    --entrypoint bash \
    "${CONTAINER_IMAGE}" -c "sleep infinity"
}


launch_ray() {
  info "Starting Ray head on ${HEAD_NODE}..."
  launch "${HEAD_NODE}" "vllm-head"
  sleep 5
  srun --nodes=1 --nodelist="${HEAD_NODE}" bash -c "
    docker exec vllm-head ray start --head --port=6379 --num-gpus=${GPUS_PER_NODE} --disable-usage-stats
  "

  for i in $(seq 1 $((NUM_NODES - 1))); do
    info "Starting Ray worker on ${NODES[$i]}..."
    launch "${NODES[$i]}" "vllm-worker-${i}"
    sleep 5
    srun --nodes=1 --nodelist="${NODES[$i]}" bash -c "
      docker exec vllm-worker-${i} ray start --address=${HEAD_IP}:6379 --num-gpus=${GPUS_PER_NODE} --disable-usage-stats
    "
  done

  # wait for GPUs
  info "Waiting for ${TOTAL_GPUS} GPUs in Ray cluster..."
  for _ in $(seq 1 120); do
    local gpu_count
    gpu_count=$(srun --nodes=1 --nodelist="${HEAD_NODE}" bash -c "
      docker exec vllm-head python3 -c \
        'import ray; ray.init(address=\"auto\"); print(int(ray.cluster_resources().get(\"GPU\",0))); ray.shutdown()' \
        2>/dev/null" || echo 0)
    [[ "$gpu_count" -ge "$TOTAL_GPUS" ]] && break
    sleep 5
  done

  info "Launching vllm serve (Ray)..."
  local args="--host 0.0.0.0 --port ${PORT} --tensor-parallel-size ${TP} --data-parallel-size ${DP}"
  args+=" --data-parallel-backend ray --data-parallel-address ${HEAD_IP} --gpu-memory-utilization 0.9"
  [[ "$ENABLE_EP" == "true" ]] && args+=" --enable-expert-parallel"
  [[ "$PP" -gt 1 ]] && args+=" --pipeline-parallel-size ${PP}"

  srun --nodes=1 --nodelist="${HEAD_NODE}" bash -c "
    docker exec -d vllm-head bash -c 'vllm serve ${MODEL} ${args} ${EXTRA_ARGS[*]+"${EXTRA_ARGS[*]}"} 2>&1 | tee ${LOGFILE}'
  "
}

launch_mp() {
  info "Starting vllm serve (multiprocessing PP)..."
  local common="--tensor-parallel-size ${TP} --pipeline-parallel-size ${PP}"
  common+=" --nnodes ${NUM_NODES} --master-addr ${HEAD_IP} --master-port 29500 --gpu-memory-utilization 0.9"

  # Start containers on all nodes
  for i in $(seq 0 $((NUM_NODES - 1))); do
    launch "${NODES[$i]}" "vllm-node-${i}"
  done
  sleep 3

  # Workers first (headless)
  for i in $(seq 1 $((NUM_NODES - 1))); do
    info "Starting worker on ${NODES[$i]} (rank ${i})..."
    srun --nodes=1 --nodelist="${NODES[$i]}" bash -c "
      docker exec -d vllm-node-${i} bash -c 'vllm serve ${MODEL} ${common} --node-rank ${i} --headless ${EXTRA_ARGS[*]+"${EXTRA_ARGS[*]}"}'
    "
  done

  # Head node
  srun --nodes=1 --nodelist="${HEAD_NODE}" bash -c "
    docker exec -d vllm-node-0 bash -c 'vllm serve ${MODEL} ${common} --host 0.0.0.0 --port ${PORT} --node-rank 0 ${EXTRA_ARGS[*]+"${EXTRA_ARGS[*]}"} 2>&1 | tee ${LOGFILE}'
  "
}

launch_rpc() {
  info "Starting vllm serve (RPC DP)..."
  local dp_local=$((GPUS_PER_NODE / TP))
  local common="--tensor-parallel-size ${TP} --data-parallel-size ${DP}"
  common+=" --data-parallel-size-local ${dp_local} --data-parallel-address ${HEAD_IP}"
  common+=" --data-parallel-rpc-port 13345 --gpu-memory-utilization 0.9"
  [[ "$PP" -gt 1 ]] && common+=" --pipeline-parallel-size ${PP}"

  # Start containers on all nodes
  launch "${HEAD_NODE}" "vllm-head"
  for i in $(seq 1 $((NUM_NODES - 1))); do
    launch "${NODES[$i]}" "vllm-worker-${i}"
  done
  sleep 3

  # Workers first (headless)
  for i in $(seq 1 $((NUM_NODES - 1))); do
    local start_rank=$((i * dp_local))
    info "Starting RPC worker on ${NODES[$i]} (start-rank ${start_rank})..."
    srun --nodes=1 --nodelist="${NODES[$i]}" bash -c "
      docker exec -d vllm-worker-${i} bash -c 'vllm serve ${MODEL} ${common} --data-parallel-start-rank ${start_rank} --headless ${EXTRA_ARGS[*]+"${EXTRA_ARGS[*]}"}'
    "
  done

  # Head node
  srun --nodes=1 --nodelist="${HEAD_NODE}" bash -c "
    docker exec -d vllm-head bash -c 'vllm serve ${MODEL} ${common} --host 0.0.0.0 --port ${PORT} ${EXTRA_ARGS[*]+"${EXTRA_ARGS[*]}"} 2>&1 | tee ${LOGFILE}'
  "
}

wait_for_server() {
  info "Waiting for vLLM server at ${HEAD_IP}:${PORT}..."
  for _ in $(seq 1 360); do
    if srun --nodes=1 --nodelist="${HEAD_NODE}" bash -c \
      "curl -sf http://localhost:${PORT}/health" &>/dev/null &&
       srun --nodes=1 --nodelist="${HEAD_NODE}" bash -c \
      "curl -sf http://localhost:${PORT}/v1/models | grep -q '\"id\"'" &>/dev/null; then
      info "Server ready at ${HEAD_IP}:${PORT}"
      return 0
    fi
    sleep 10
  done
  err "Timeout waiting for server"
  return 1
}

cleanup() {
  info "Cleaning up..."
  srun --ntasks-per-node=1 bash -c '
    docker ps -aq --filter "name=vllm-" | xargs -r docker rm -f 2>/dev/null || true
  ' 2>/dev/null || true
}
trap cleanup EXIT

load_or_pull_image

case "$MODE" in
  ray)    launch_ray ;;
  mp)     launch_mp ;;
  rpc)    launch_rpc ;;
  *)      err "Unknown mode: $MODE (expected: ray|mp|rpc)"; exit 1 ;;
esac

# stream logs while waiting for server
tail -f "${LOGFILE}" 2>/dev/null &
TAIL_PID=$!

wait_for_server || exit 1

kill "${TAIL_PID}" 2>/dev/null || true

info "vLLM serving on ${HEAD_IP}:${PORT} — Ctrl+C or scancel to stop"
info "Logs: ${LOGFILE}"
sleep infinity
