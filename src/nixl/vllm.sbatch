#!/bin/bash
# Usage:
#   salloc -N <nodes> bash vllm.sbatch [--route R] [--prefill P] [--image <path>] [vllm serve args...]
#
# Modes:
#   --prefill 0 (default): pure DP, no disaggregation
#   --prefill P:           P prefill + (nodes_per_group - P) decode per group
#   --route R:             split nodes into R identical groups, vllm-router round-robins
#
# Examples:
#   # Exp 1: Pure DP — 4 nodes, TP=8
#   salloc -N 4 bash vllm.sbatch --model /fsx/models/deepseek-ai/DeepSeek-V2-Lite
#
#   # Exp 2: 2 groups of 2 nodes, pure DP each, router round-robins
#   salloc -N 4 bash vllm.sbatch --route 2 --model /fsx/models/deepseek-ai/DeepSeek-V2-Lite
#
#   # Exp 3: 2 groups of 2 nodes, each 1P+1D, router round-robins
#   salloc -N 4 bash vllm.sbatch --route 2 --prefill 1 --model /fsx/models/deepseek-ai/DeepSeek-V2-Lite

set -euo pipefail

info() { echo -e "[$(date +'%Y-%m-%dT%H:%M:%S%z')][info] $*"; }
err()  { echo -e "[$(date +'%Y-%m-%dT%H:%M:%S%z')][error] $*" >&2; }

IMAGE=""
WORKSPACE="$PWD"
CONTAINER_MOUNT="/fsx"
NUM_PREFILL=0
ROUTE_SIZE=1
SERVE_ARGS=()

while (("$#")); do
  case "$1" in
  --image)           IMAGE="$2"; shift 2 ;;
  --container-mount) CONTAINER_MOUNT="$2"; shift 2 ;;
  --prefill)         NUM_PREFILL="$2"; shift 2 ;;
  --route)           ROUTE_SIZE="$2"; shift 2 ;;
  *)                 SERVE_ARGS+=("$1"); shift ;;
  esac
done

IMAGE="${IMAGE:-${WORKSPACE}/nixl-latest.tar.gz}"
NUM_NODES=${SLURM_JOB_NUM_NODES:-1}
readarray -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
HEAD_NODE=${NODES[0]}
HEAD_IP=$(getent ahostsv4 "$HEAD_NODE" | head -1 | awk '{print $1}')
LOGDIR="${WORKSPACE}/logs"

_peek_arg() {
  local short="$1" long="$2" default="$3" i=0
  while ((i < ${#SERVE_ARGS[@]})); do
    if [[ "${SERVE_ARGS[$i]}" == "$short" || "${SERVE_ARGS[$i]}" == "$long" ]]; then
      echo "${SERVE_ARGS[$((i + 1))]}"
      return
    fi
    ((i++))
  done
  echo "$default"
}

GPUS_PER_NODE=8
TP=$(_peek_arg "-tp" "--tensor-parallel-size" "${GPUS_PER_NODE}")
MODEL=$(_peek_arg "" "--model" "deepseek-ai/DeepSeek-V2-Lite")
SERVE_ARGS_STR=$(printf '%q ' "${SERVE_ARGS[@]+"${SERVE_ARGS[@]}"}")
DP_LOCAL=$((GPUS_PER_NODE / TP))

# Validate
if (( NUM_NODES % ROUTE_SIZE != 0 )); then
  err "NUM_NODES(${NUM_NODES}) must be divisible by --route ${ROUTE_SIZE}"
  exit 1
fi
NODES_PER_GROUP=$((NUM_NODES / ROUTE_SIZE))
NUM_DECODE=$((NODES_PER_GROUP - NUM_PREFILL))
if (( NUM_PREFILL > 0 && NUM_DECODE < 1 )); then
  err "Each group needs at least 1 decode node. Got ${NODES_PER_GROUP} nodes/group, --prefill ${NUM_PREFILL}"
  exit 1
fi

ROUTER_PORT=$((8000 + ROUTE_SIZE))

mkdir -p "${LOGDIR}"

if (( NUM_PREFILL == 0 )); then
  MODE="pure-dp"
else
  MODE="${NUM_PREFILL}P+${NUM_DECODE}D"
fi

info "========================================"
info "vLLM Server"
info "========================================"
info "Image:    ${IMAGE}"
info "Model:    ${MODEL}"
info "Nodes:    ${NUM_NODES}, TP=${TP}, DP_LOCAL=${DP_LOCAL}"
info "Route:    ${ROUTE_SIZE} group(s), ${NODES_PER_GROUP} node(s)/group, mode=${MODE}"
info "Args:     ${SERVE_ARGS[*]+"${SERVE_ARGS[*]}"}"
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
  local name="$1" cmd="$2"
  local devices=("--device=/dev/gdrdrv")
  while IFS= read -r -d '' d; do
    devices+=("--device=${d}")
  done < <(find "/dev/infiniband" -name "uverbs*" -print0 2>/dev/null)

  local net_if="${GLOO_SOCKET_IFNAME:-$(ip -o -4 route show to default | awk '{print $5}' | head -1)}"
  local host_ip; host_ip=$(ip -o -4 addr show "${net_if}" | awk '{print $4}' | cut -d/ -f1 | head -1)

  docker run --gpus all \
    --privileged -d \
    --name "${name}" \
    --uts=host --ipc=host --net=host \
    --ulimit stack=67108864 --ulimit memlock=-1 \
    --security-opt seccomp=unconfined \
    "${devices[@]}" \
    -e NCCL_SOCKET_IFNAME="${net_if}" \
    -e GLOO_SOCKET_IFNAME="${net_if}" \
    -e VLLM_NIXL_SIDE_CHANNEL_HOST="${host_ip}" \
    -e VLLM_NIXL_SIDE_CHANNEL_PORT="${SIDE_CHANNEL_PORT:-5600}" \
    -e VLLM_RPC_TIMEOUT="${VLLM_RPC_TIMEOUT:-3600000}" \
    -e VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-3600}" \
    -v "${CONTAINER_MOUNT}:${CONTAINER_MOUNT}" \
    --entrypoint bash \
    "${CONTAINER_IMAGE}" \
    -c "${cmd}"
}

cleanup() {
  info "Cleaning up..."
  srun --ntasks-per-node=1 bash -c '
    docker ps -aq | xargs -r docker rm -f 2>/dev/null || true
  ' 2>/dev/null || true
  rm -f "${LOGDIR}"/vllm-*.log
}

wait_for_server() {
  local ip="$1" port="$2" label="$3"
  info "Waiting for ${label} at ${ip}:${port} ..."
  for _ in $(seq 1 360); do
    if curl -sf "http://${ip}:${port}/health" >/dev/null 2>&1; then
      info "${label} is ready"
      return 0
    fi
    sleep 5
  done
  err "${label} failed to start within 1800s"
  return 1
}

# --- Launch a pure-DP group on a set of nodes ---
# Args: group_id, node_list (space-separated), port
start_dp_group() {
  local gid="$1" port="$2"
  shift 2
  local gnodes=("$@")
  local ghead="${gnodes[0]}"
  local ghead_ip; ghead_ip=$(getent ahostsv4 "$ghead" | head -1 | awk '{print $1}')
  local gsize=${#gnodes[@]}
  local dp=$((gsize * DP_LOCAL))
  local rpc_port=$((13345 + (SLURM_JOB_ID % 1000) + gid))
  local logfile="${LOGDIR}/vllm-g${gid}.log"

  info "Group ${gid}: ${gsize} nodes, DP=${dp}, head=${ghead}:${port}"

  # Launch containers
  for i in $(seq 0 $((gsize - 1))); do
    srun --nodes=1 --nodelist="${gnodes[$i]}" bash -c "
      $(declare -f launch_container)
      CONTAINER_IMAGE='${CONTAINER_IMAGE}' CONTAINER_MOUNT='${CONTAINER_MOUNT}' \
      launch_container 'vllm-g${gid}-n${i}' 'sleep infinity'
    "
  done
  sleep 3

  local extra="--data-parallel-size ${dp} --data-parallel-size-local ${DP_LOCAL} --data-parallel-address ${ghead_ip} --data-parallel-rpc-port ${rpc_port}"

  # Start workers
  for i in $(seq 1 $((gsize - 1))); do
    local start_rank=$((i * DP_LOCAL))
    srun --nodes=1 --nodelist="${gnodes[$i]}" bash -c "
      docker exec -d vllm-g${gid}-n${i} bash -c 'vllm serve ${SERVE_ARGS_STR} ${extra} \
        --data-parallel-start-rank ${start_rank} --headless \
        2>&1 | tee ${logfile}.node${i}'
    "
  done

  # Start head
  srun --nodes=1 --nodelist="${ghead}" bash -c "
    docker exec -d vllm-g${gid}-n0 bash -c 'vllm serve ${SERVE_ARGS_STR} ${extra} \
      --host 0.0.0.0 --port ${port} \
      2>&1 | tee ${logfile}'
  "
  tail -f "${logfile}" 2>/dev/null | sed "s/^/[g${gid}] /" &
}

# --- Launch a PD-disaggregated group on a set of nodes ---
# Args: group_id, base_port, node_list (space-separated)
start_pd_group() {
  local gid="$1" base_port="$2"
  shift 2
  local gnodes=("$@")
  local gsize=${#gnodes[@]}

  info "Group ${gid}: ${NUM_PREFILL}P+${NUM_DECODE}D, nodes=${gnodes[*]}"

  # Launch prefill nodes
  for i in $(seq 0 $((NUM_PREFILL - 1))); do
    local node="${gnodes[$i]}"
    local port=$((base_port + i))
    srun --nodes=1 --nodelist="${node}" bash -c "
      $(declare -f launch_container)
      CONTAINER_IMAGE='${CONTAINER_IMAGE}' CONTAINER_MOUNT='${CONTAINER_MOUNT}' \
      launch_container 'vllm-g${gid}-p${i}' \
        'vllm serve ${SERVE_ARGS_STR} \
           --host 0.0.0.0 --port ${port} \
           --data-parallel-size ${DP_LOCAL} \
           --kv-transfer-config.kv_connector NixlConnector \
           --kv-transfer-config.kv_role kv_producer \
           --kv-transfer-config.kv_load_failure_policy fail \
           --kv-transfer-config.kv_connector_extra_config.backends+ LIBFABRIC \
           2>&1 | tee ${LOGDIR}/vllm-g${gid}-p${i}.log'
    "
    tail -f "${LOGDIR}/vllm-g${gid}-p${i}.log" 2>/dev/null | sed "s/^/[g${gid}-p${i}] /" &
  done

  # Launch decode nodes
  for i in $(seq 0 $((NUM_DECODE - 1))); do
    local node="${gnodes[$((NUM_PREFILL + i))]}"
    local port=$((base_port + NUM_PREFILL + i))
    srun --nodes=1 --nodelist="${node}" bash -c "
      $(declare -f launch_container)
      CONTAINER_IMAGE='${CONTAINER_IMAGE}' CONTAINER_MOUNT='${CONTAINER_MOUNT}' \
      launch_container 'vllm-g${gid}-d${i}' \
        'vllm serve ${SERVE_ARGS_STR} \
           --host 0.0.0.0 --port ${port} \
           --data-parallel-size ${DP_LOCAL} \
           --kv-transfer-config.kv_connector NixlConnector \
           --kv-transfer-config.kv_role kv_consumer \
           --kv-transfer-config.kv_load_failure_policy fail \
           --kv-transfer-config.kv_connector_extra_config.backends+ LIBFABRIC \
           2>&1 | tee ${LOGDIR}/vllm-g${gid}-d${i}.log'
    "
    tail -f "${LOGDIR}/vllm-g${gid}-d${i}.log" 2>/dev/null | sed "s/^/[g${gid}-d${i}] /" &
  done
}

start_router() {
  local router_args=""

  if (( NUM_PREFILL == 0 )); then
    # Pure DP: round-robin across group endpoints
    router_args="--worker-urls"
    for g in $(seq 0 $((ROUTE_SIZE - 1))); do
      router_args="${router_args} http://${GROUP_IPS[$g]}:${GROUP_PORTS[$g]}"
    done
    info "Starting vllm-router (DP) on ${HEAD_NODE}:${ROUTER_PORT} ..."
    srun --nodes=1 --nodelist="${HEAD_NODE}" bash -c "
      docker exec -d vllm-g0-n0 bash -c \
        'vllm-router \
           --policy round_robin \
           ${router_args} \
           --host 0.0.0.0 \
           --port ${ROUTER_PORT} \
           --intra-node-data-parallel-size ${DP_LOCAL} \
           2>&1 | tee ${LOGDIR}/vllm-router.log'
    "
  else
    # PD disaggregation: collect all prefill/decode endpoints across groups
    local prefill_args="" decode_args=""
    for g in $(seq 0 $((ROUTE_SIZE - 1))); do
      local goffset=$((g * NODES_PER_GROUP))
      local base_port=$((8000 + g * NODES_PER_GROUP))
      for i in $(seq 0 $((NUM_PREFILL - 1))); do
        local ip; ip=$(getent ahostsv4 "${NODES[$((goffset + i))]}" | head -1 | awk '{print $1}')
        prefill_args="${prefill_args} --prefill http://${ip}:$((base_port + i))"
      done
      for i in $(seq 0 $((NUM_DECODE - 1))); do
        local ip; ip=$(getent ahostsv4 "${NODES[$((goffset + NUM_PREFILL + i))]}" | head -1 | awk '{print $1}')
        decode_args="${decode_args} --decode http://${ip}:$((base_port + NUM_PREFILL + i))"
      done
    done
    info "Starting vllm-router (PD) on ${HEAD_NODE}:${ROUTER_PORT} ..."
    srun --nodes=1 --nodelist="${HEAD_NODE}" bash -c "
      docker exec -d vllm-g0-p0 bash -c \
        'vllm-router \
           --policy consistent_hash \
           --vllm-pd-disaggregation \
           ${prefill_args} \
           ${decode_args} \
           --host 0.0.0.0 \
           --port ${ROUTER_PORT} \
           --intra-node-data-parallel-size ${DP_LOCAL} \
           2>&1 | tee ${LOGDIR}/vllm-router.log'
    "
  fi
  tail -f "${LOGDIR}/vllm-router.log" 2>/dev/null | sed "s/^/[router] /" &
}

start_groups() {
  GROUP_IPS=()
  GROUP_PORTS=()

  for g in $(seq 0 $((ROUTE_SIZE - 1))); do
    local goffset=$((g * NODES_PER_GROUP))
    local gnodes=("${NODES[@]:$goffset:$NODES_PER_GROUP}")
    local ghead_ip; ghead_ip=$(getent ahostsv4 "${gnodes[0]}" | head -1 | awk '{print $1}')

    if (( NUM_PREFILL == 0 )); then
      local port=$((8000 + g))
      start_dp_group "$g" "$port" "${gnodes[@]}"
      GROUP_IPS+=("${ghead_ip}")
      GROUP_PORTS+=("${port}")
    else
      local base_port=$((8000 + g * NODES_PER_GROUP))
      start_pd_group "$g" "$base_port" "${gnodes[@]}"
    fi
  done
}

wait_all_groups() {
  if (( NUM_PREFILL == 0 )); then
    for g in $(seq 0 $((ROUTE_SIZE - 1))); do
      wait_for_server "${GROUP_IPS[$g]}" "${GROUP_PORTS[$g]}" "group-${g}" || exit 1
    done
  else
    for g in $(seq 0 $((ROUTE_SIZE - 1))); do
      local goffset=$((g * NODES_PER_GROUP))
      local base_port=$((8000 + g * NODES_PER_GROUP))
      for i in $(seq 0 $((NUM_PREFILL - 1))); do
        local ip; ip=$(getent ahostsv4 "${NODES[$((goffset + i))]}" | head -1 | awk '{print $1}')
        wait_for_server "${ip}" "$((base_port + i))" "g${g}-prefill-${i}" || exit 1
      done
      for i in $(seq 0 $((NUM_DECODE - 1))); do
        local ip; ip=$(getent ahostsv4 "${NODES[$((goffset + NUM_PREFILL + i))]}" | head -1 | awk '{print $1}')
        wait_for_server "${ip}" "$((base_port + NUM_PREFILL + i))" "g${g}-decode-${i}" || exit 1
      done
    done
  fi
}

print_summary() {
  info "========================================"
  info "All instances ready — mode=${MODE}"
  if (( ROUTE_SIZE > 1 || NUM_PREFILL > 0 )); then
    info "  Router: http://${HEAD_IP}:${ROUTER_PORT}"
    info ""
    info "  Test (via router):"
    info "    curl http://${HEAD_IP}:${ROUTER_PORT}/v1/completions -H 'Content-Type: application/json' \\"
    info "      -d '{\"model\":\"${MODEL}\",\"prompt\":\"Hello\",\"max_tokens\":64}'"
  else
    info "  Endpoint: http://${HEAD_IP}:8000"
    info ""
    info "  Test:"
    info "    curl http://${HEAD_IP}:8000/v1/completions -H 'Content-Type: application/json' \\"
    info "      -d '{\"model\":\"${MODEL}\",\"prompt\":\"Hello\",\"max_tokens\":64}'"
  fi
  info "========================================"
}

main() {
  trap cleanup EXIT
  cleanup
  load_image

  start_groups
  wait_all_groups

  if (( ROUTE_SIZE > 1 || NUM_PREFILL > 0 )); then
    start_router
  fi

  print_summary
  sleep infinity
}

main
