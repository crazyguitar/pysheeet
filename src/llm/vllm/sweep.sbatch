#!/usr/bin/env bash
# sweep.sbatch — Self-contained sweep benchmark with server lifecycle management.
#
# Wraps `vllm bench sweep serve` inside a GPU-enabled Docker container.
# Follows the same CLI as the upstream sweep command — --serve-cmd and
# --bench-cmd are passed through directly.
#
# Usage:
#   bash sweep.sbatch -m Qwen/Qwen3-0.6B
#   bash sweep.sbatch -m Qwen/Qwen3-30B-A3B-FP8 \
#       --serve-cmd "vllm serve Qwen/Qwen3-30B-A3B-FP8 -tp 8 --enable-expert-parallel" \
#   salloc -N2 bash sweep.sbatch -m Qwen/Qwen1.5-MoE-A2.7B \
#       --serve-cmd "vllm serve Qwen/Qwen1.5-MoE-A2.7B -tp 8"
set -euo pipefail

info() { echo "[$(date +'%H:%M:%S')] $*"; }

MODEL="Qwen/Qwen3-0.6B"
IMAGE="${PWD}/vllm-serve-latest.tar.gz"
CONTAINER_MOUNT="${CONTAINER_MOUNT:-/fsx}"
GPUS="${GPUS:-all}"
SERVE_CMD=""            # --serve-cmd (auto-generated from MODEL if empty)
BENCH_CMD=""            # --bench-cmd (auto-generated from MODEL if empty)
SWEEP_ARGS=()           # everything else → vllm bench sweep serve

usage() {
    cat <<EOF
Usage: $(basename "$0") [options] [vllm-bench-sweep-args...]

Options (consumed by sweep.sh):
  -m, --model MODEL       Model name/path (default: $MODEL)
  -i, --image IMAGE       Docker image tarball or registry URI
  -h, --help              Show this help

All other args are passed through to vllm bench sweep serve:
  --serve-cmd, --bench-cmd, --serve-params, --bench-params,
  --resume, --show-stdout, --num-runs, --dry-run, -o, etc.
EOF
    exit 0
}

while (( "$#" )); do
    case "$1" in
        -m|--model)  MODEL="$2"; shift 2 ;;
        -i|--image)  IMAGE="$2"; shift 2 ;;
        -h|--help)   usage ;;
        *)           SWEEP_ARGS+=("$1"); shift ;;
    esac
done

_run() {
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        srun -N1 --ntasks-per-node=1 bash -c "$*"
    else
        bash -c "$*"
    fi
}

load_or_pull_image() {
    if [[ "${IMAGE}" == *.tar.gz ]]; then
        CONTAINER_IMAGE=$(pigz -dc "${IMAGE}" | tar -xf - -O manifest.json \
            | python3 -c "import sys,json; print(json.load(sys.stdin)[0]['RepoTags'][0])")
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

# Passes args as an array directly to docker so --serve-cmd/--bench-cmd quoting
# is preserved (no bash -c needed).
launch_container() {
    local devices=()
    [[ -e /dev/gdrdrv ]] && devices+=("--device=/dev/gdrdrv")
    while IFS= read -r -d '' d; do
        devices+=("--device=${d}")
    done < <(find "/dev/infiniband" -name "uverbs*" -print0 2>/dev/null)

    local docker_cmd=(
        docker run --rm --gpus "${GPUS}"
        --ipc=host --net=host --uts=host
        --ulimit stack=67108864 --ulimit memlock=-1
        "${devices[@]}"
        -v "${PWD}:${PWD}" -w "${PWD}"
        -v "${CONTAINER_MOUNT}:${CONTAINER_MOUNT}"
        -e VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
        "${CONTAINER_IMAGE}"
        "$@"
    )

    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        srun -N1 --ntasks-per-node=1 "${docker_cmd[@]}"
    else
        "${docker_cmd[@]}"
    fi
}

# Auto-generate --serve-cmd and --bench-cmd if not in SWEEP_ARGS
has_arg() { for a in "${SWEEP_ARGS[@]+"${SWEEP_ARGS[@]}"}"; do [[ "$a" == "$1" ]] && return 0; done; return 1; }

defaults=()
if ! has_arg "--serve-cmd"; then
    defaults+=(--serve-cmd "vllm serve ${MODEL}")
fi
if ! has_arg "--bench-cmd"; then
    defaults+=(--bench-cmd "vllm bench serve --model ${MODEL}")
fi

info "Model:  ${MODEL}"

load_or_pull_image
launch_container \
    vllm bench sweep serve \
    "${defaults[@]+"${defaults[@]}"}" \
    "${SWEEP_ARGS[@]+"${SWEEP_ARGS[@]}"}"

info "Sweep complete"
