#!/usr/bin/env bash
# SGLang serving benchmark suite
# Usage:
#   salloc -N1 bash bench.sh -H 10.0.128.193 -i /fsx/sglang-serve-latest.tar.gz
#   bash bench.sh -H 10.0.128.193 -i sglang-serve:latest
set -euo pipefail

info() { echo "[$(date +'%H:%M:%S')] $*"; }

CONTAINER_MOUNT="${CONTAINER_MOUNT:-/fsx}"

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

launch_container() {
    local cmd="$1"
    _run "
        docker run --rm --net=host \
            -v '${PWD}:${PWD}' -w '${PWD}' \
            -v '${CONTAINER_MOUNT}:${CONTAINER_MOUNT}' \
            --entrypoint bash '${CONTAINER_IMAGE}' \
            -c '${cmd}'
    "
}

# If sglang is not available, load image and re-exec inside container
if ! python3 -c "import sglang" &>/dev/null; then
    IMAGE="" _args=("$@")
    for ((i=0; i<${#_args[@]}; i++)); do
        [[ "${_args[$i]}" == "--image" || "${_args[$i]}" == "-i" ]] \
            && { IMAGE="${_args[$((i+1))]}"; break; }
    done
    IMAGE="${IMAGE:-${PWD}/sglang-serve-latest.tar.gz}"

    load_or_pull_image
    _SCRIPT="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
    launch_container "bash ${_SCRIPT} $*"
    exit $?
fi

HOST="localhost"
PORT="30000"
MODEL=""
SEED="42"
RESULT_DIR="./results"
TYPES="throughput,prefill,decode,latency,concurrency,sharegpt"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Options:
  -H, --host HOST         Server host (default: $HOST)
  -p, --port PORT         Server port (default: $PORT)
  -m, --model MODEL       Model name (auto-detected if omitted)
  -t, --type TYPES        Comma-separated tests (default: all)
                          Available: throughput,prefill,decode,latency,concurrency,sharegpt
  -o, --output DIR        Result directory (default: $RESULT_DIR)
  -i, --image IMAGE       Docker image or tarball (default: ./sglang-serve-latest.tar.gz)
  -h, --help              Show this help

Examples:
  $(basename "$0") -H 10.0.128.193
  $(basename "$0") -H 10.0.128.193 --type throughput,prefill
  $(basename "$0") -H 10.0.128.193 --type latency -m Qwen/Qwen2.5-14B-Instruct
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -H|--host)  HOST="$2"; shift 2 ;;
        -p|--port)  PORT="$2"; shift 2 ;;
        -m|--model) MODEL="$2"; shift 2 ;;
        -t|--type)  TYPES="$2"; shift 2 ;;
        -o|--output) RESULT_DIR="$2"; shift 2 ;;
        -i|--image)  shift 2 ;;
        -h|--help)   usage; exit 0 ;;
        *) echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

BASE_URL="http://${HOST}:${PORT}"
if [[ -z "$MODEL" ]]; then
    MODEL=$(curl -s "${BASE_URL}/v1/models" | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])")
fi
mkdir -p "$RESULT_DIR"

bench() {
    local label="$1"; shift
    local outfile="${RESULT_DIR}/$(echo "$label" | tr ' /' '_').json"
    echo "==> $label"
    python3 -m sglang.bench_serving \
        --backend sglang \
        --host "$HOST" \
        --port "$PORT" \
        --model "$MODEL" \
        --seed "$SEED" \
        --output-file "$outfile" \
        "$@"
    echo ""
}

bench_throughput() {
    bench "Throughput (random 512in/256out, max rate)" \
        --dataset-name random \
        --random-input 512 --random-output 256 \
        --num-prompts 100 --request-rate inf
}

bench_prefill() {
    for len in 128 512 2048 4096 16384; do
        bench "Prefill TTFT (input=${len})" \
            --dataset-name random \
            --random-input "$len" --random-output 1 \
            --num-prompts 100 --request-rate 4
    done
}

bench_decode() {
    for len in 128 256 512 1024; do
        bench "Decode ITL (output=${len})" \
            --dataset-name random \
            --random-input 128 --random-output "$len" \
            --num-prompts 100 --request-rate 4
    done
}

bench_latency() {
    bench "Latency (short 128/128, rate=1)" \
        --dataset-name random \
        --random-input 128 --random-output 128 \
        --num-prompts 100 --request-rate 1
    bench "Latency (medium 512/256, rate=1)" \
        --dataset-name random \
        --random-input 512 --random-output 256 \
        --num-prompts 100 --request-rate 1
    bench "Latency (long 4096/512, rate=1)" \
        --dataset-name random \
        --random-input 4096 --random-output 512 \
        --num-prompts 100 --request-rate 1
}

bench_concurrency() {
    for c in 1 4 16 64 256; do
        bench "Concurrency=${c} (512in/256out)" \
            --dataset-name random \
            --random-input 512 --random-output 256 \
            --num-prompts 100 --request-rate inf --max-concurrency "$c"
    done
}

SHAREGPT_PATH="${SHAREGPT_PATH:-ShareGPT_V3_unfiltered_cleaned_split.json}"
bench_sharegpt() {
    if [[ ! -f "$SHAREGPT_PATH" ]]; then
        echo "Downloading ShareGPT dataset..."
        wget -q -O "$SHAREGPT_PATH" \
            https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
    fi
    bench "ShareGPT (100 prompts, max rate)" \
        --dataset-name sharegpt \
        --dataset-path "$SHAREGPT_PATH" \
        --num-prompts 100 --request-rate inf
    bench "ShareGPT (100 prompts, rate=4)" \
        --dataset-name sharegpt \
        --dataset-path "$SHAREGPT_PATH" \
        --num-prompts 100 --request-rate 4
}

IFS=',' read -ra TESTS <<< "$TYPES"
for t in "${TESTS[@]}"; do
    t=$(echo "$t" | xargs)
    echo "========================================"
    echo "Running: $t"
    echo "========================================"
    case "$t" in
        throughput)  bench_throughput ;;
        prefill)     bench_prefill ;;
        decode)      bench_decode ;;
        latency)     bench_latency ;;
        concurrency) bench_concurrency ;;
        sharegpt)    bench_sharegpt ;;
        *) echo "Unknown test: $t"; exit 1 ;;
    esac
done
