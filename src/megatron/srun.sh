#!/bin/bash
# Launch Megatron Bridge recipe inside enroot container via srun + pyxis
# Usage: salloc -N 2 ./srun.sh recipes/deepseek_v2_lite_pretrain.py [overrides...]
# Example: salloc -N 2 ./srun.sh recipes/deepseek_v2_lite_pretrain.py model.tensor_model_parallel_size=4

set -exo pipefail

DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SQSH="${SQSH:-${DIR}/megatron-lm+latest.sqsh}"
MOUNT="${MOUNT:-${DIR}:/workspace/megatron,/fsx:/fsx}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
ENABLE_NSYS=false

# Parse flags before recipe arg
while [[ "${1:-}" == --* ]]; do
  case "$1" in
  --nsys)
    ENABLE_NSYS=true
    shift
    ;;
  *) break ;;
  esac
done

RECIPE="${1:?Usage: srun.sh [--nsys] <recipe.py> [overrides...]}"
RECIPE_PATH="/workspace/megatron/${RECIPE}"
ENTRYPOINT="/workspace/megatron/entrypoint.py"
shift
OVERRIDES="$*"

master_host=$(scontrol show hostnames "$SLURM_JOB_NODELIST" 2>/dev/null | head -n1)
master_addr=$(getent hosts "${master_host}" 2>/dev/null | awk '{print $1}' || echo "${master_host}")
master_addr=${master_addr:-127.0.0.1}

cmd="$(
  cat <<EOF
export LD_LIBRARY_PATH=/opt/amazon/ofi-nccl/lib:/opt/amazon/efa/lib:/opt/aws-ofi-nccl/install/lib:\$LD_LIBRARY_PATH
export FI_PROVIDER=efa
export FI_EFA_USE_DEVICE_RDMA=1
export FI_EFA_FORK_SAFE=1
export NCCL_NET_PLUGIN=/opt/amazon/ofi-nccl/lib/libnccl-net-ofi.so
export NCCL_TUNER_PLUGIN=/opt/amazon/ofi-nccl/lib/libnccl-tuner-ofi.so
export NCCL_DEBUG=WARN
export NCCL_BUFFSIZE=8388608
export NCCL_P2P_NET_CHUNKSIZE=524288
export OMP_NUM_THREADS=1
export TRITON_CACHE_DIR=/tmp/triton_cache_\${SLURM_PROCID:-0}
export TORCH_EXTENSIONS_DIR=/tmp/torch_ext_\${SLURM_PROCID:-0}
export MASTER_ADDR=${master_addr}
export MASTER_PORT=29500
export DEEP_EP_BACKEND=nccl
export NCCL_GIN_TYPE=2
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_DISTRIBUTED_BACKEND=nccl
export RANK=\${SLURM_PROCID}
export LOCAL_RANK=\${SLURM_LOCALID}
export WORLD_SIZE=\${SLURM_NTASKS}
NSYS_CMD=""
HOST=\$(hostname)
if [ "${ENABLE_NSYS}" = "true" ] && [ "\${RANK}" = "0" ]; then
  NSYS_DIR=/workspace/megatron/nsys-megatron
  mkdir -p \${NSYS_DIR}
  NSYS_CMD="nsys profile"
  NSYS_CMD+=" -t cuda,nvtx"
  NSYS_CMD+=" -s none"
  NSYS_CMD+=" --cpuctxsw=none"
  NSYS_CMD+=" --capture-range=cudaProfilerApi"
  NSYS_CMD+=" --capture-range-end=stop"
  NSYS_CMD+=" --enable efa_metrics"
  NSYS_CMD+=" -o \${NSYS_DIR}/profile-\$(hostname)-rank\${RANK}.nsys-rep"
  NSYS_CMD+=" --force-overwrite=true"
fi
if [ -n "\${NSYS_CMD}" ]; then echo "NSYS_CMD=\${NSYS_CMD}"; fi
\${NSYS_CMD} python3 ${ENTRYPOINT} ${RECIPE_PATH} ${OVERRIDES}
EOF
)"

srun --container-image "${SQSH}" \
  --container-mounts "${MOUNT}" \
  --container-name megatron \
  --mpi=pmix \
  --ntasks-per-node=${GPUS_PER_NODE} \
  bash -c "${cmd}"
