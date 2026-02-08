#!/usr/bin/env bash
# vLLM serving benchmark suite
# Usage:
#   salloc -N1 bash bench.sh -H 10.0.128.193 -i /fsx/vllm-serve-latest.tar.gz
#   bash bench.sh -H 10.0.128.193 -i vllm-serve:latest
set -euo pipefail

info() { echo "[$(date +'%H:%M:%S')] $*"; }

# Docker image helpers (mirrors run.sbatch)
CONTAINER_MOUNT="${CONTAINER_MOUNT:-/fsx}"

# Wrap a command with srun if inside a SLURM allocation, otherwise run directly
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

# If vllm CLI is not available, load image and re-exec inside container
if ! command -v vllm &>/dev/null; then
    # Pre-parse --image/-i before full arg parsing
    IMAGE="" _args=("$@")
    for ((i=0; i<${#_args[@]}; i++)); do
        [[ "${_args[$i]}" == "--image" || "${_args[$i]}" == "-i" ]] \
            && { IMAGE="${_args[$((i+1))]}"; break; }
    done
    IMAGE="${IMAGE:-${PWD}/vllm-serve-latest.tar.gz}"

    load_or_pull_image
    _SCRIPT="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
    launch_container "bash ${_SCRIPT} $*"
    exit $?
fi

HOST="localhost"
PORT="8000"
MODEL=""
SEED="42"
RESULT_DIR="./results"
PROFILE=""
TYPES="throughput,prefill,decode,latency,concurrency,longctx,sharegpt,sonnet"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Options:
  -H, --host HOST         Server host (default: $HOST)
  -p, --port PORT         Server port (default: $PORT)
  -m, --model MODEL       Model name (auto-detected if omitted)
  -t, --type TYPES        Comma-separated tests (default: all)
                          Available: throughput,prefill,decode,latency,concurrency,longctx,sharegpt,sonnet
  -o, --output DIR        Result directory (default: $RESULT_DIR)
  -i, --image IMAGE       Docker image or tarball (default: ./vllm-serve-latest.tar.gz)
  --profile               Enable PyTorch profiler (server must have --profiler-config set)
  -h, --help              Show this help

Examples:
  $(basename "$0") -H 10.0.128.193
  $(basename "$0") -H 10.0.128.193 --type throughput,prefill
  $(basename "$0") -H 10.0.128.193 --type latency -m Qwen/Qwen3-30B-A3B-FP8
  $(basename "$0") -H 10.0.128.193 --type throughput --profile
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -H|--host)  HOST="$2"; shift 2 ;;
        -p|--port)  PORT="$2"; shift 2 ;;
        -m|--model) MODEL="$2"; shift 2 ;;
        -t|--type)  TYPES="$2"; shift 2 ;;
        -o|--output) RESULT_DIR="$2"; shift 2 ;;
        -i|--image)  shift 2 ;;  # consumed by preamble above
        --profile)   PROFILE="--profile"; shift ;;
        -h|--help)   usage; exit 0 ;;
        *) echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

BASE_URL="http://${HOST}:${PORT}"
if [[ -z "$MODEL" ]]; then
    MODEL=$(curl -s "${BASE_URL}/v1/models" | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])")
fi
mkdir -p "$RESULT_DIR"

if [[ -n "$PROFILE" ]]; then
    info "Profiling enabled (server must have --profiler-config set)"
    info "View traces at https://ui.perfetto.dev/"
fi

bench() {
    local label="$1"; shift
    echo "==> $label"
    vllm bench serve \
        --model "$MODEL" \
        --base-url "$BASE_URL" \
        --backend openai-chat \
        --endpoint /v1/chat/completions \
        --seed "$SEED" \
        --save-result \
        --result-dir "$RESULT_DIR" \
        $PROFILE \
        "$@"
    echo ""
}


# Throughput: measures peak output tokens/sec and request throughput.
# Uses request-rate=inf to saturate the server — all 1000 prompts are sent as fast as
# possible, forcing the scheduler to batch aggressively. 512in/256out is a moderate
# workload that exercises both prefill and decode phases. 1000 prompts follows the
# GPUStack methodology for statistically stable throughput numbers.
bench_throughput() {
    bench "Throughput (random 512in/256out, max rate)" \
        --dataset-name random \
        --random-input-len 512 --random-output-len 256 \
        --num-prompts 100 --request-rate inf
}

# Prefill (TTFT): measures Time to First Token, which reflects prompt processing speed.
# output-len=1 isolates prefill from decode — we only care how fast the model processes
# the input. Sweeps 128→16K tokens to show how TTFT scales with context length (should
# grow roughly linearly due to attention's O(n) compute per layer during prefill).
# rate=4 keeps the server lightly loaded so TTFT reflects compute, not queueing.
bench_prefill() {
    for len in 128 512 2048 4096 16384; do
        bench "Prefill TTFT (input=${len})" \
            --dataset-name random \
            --random-input-len "$len" --random-output-len 1 \
            --num-prompts 100 --request-rate 4
    done
}

# Decode (ITL): measures Inter-Token Latency and Time Per Output Token during generation.
# input=128 keeps prefill minimal so the benchmark focuses on autoregressive decode.
# Sweeps 128→1024 output tokens to reveal how ITL changes as KV cache grows — longer
# sequences increase memory pressure and may trigger preemption/swapping.
# rate=4 avoids batching interference so ITL reflects per-request decode speed.
bench_decode() {
    for len in 128 256 512 1024; do
        bench "Decode ITL (output=${len})" \
            --dataset-name random \
            --random-input-len 128 --random-output-len "$len" \
            --num-prompts 100 --request-rate 4
    done
}

# Latency (E2E): measures end-to-end request latency under minimal load.
# rate=1 ensures requests are mostly processed alone (no batching), giving a baseline
# for best-case latency. Tests short/medium/long to show how total latency scales.
# These numbers represent the "single user" experience (similar to ChatGPT-style usage
# where one user waits for a complete response).
bench_latency() {
    bench "Latency (short 128/128, rate=1)" \
        --dataset-name random \
        --random-input-len 128 --random-output-len 128 \
        --num-prompts 100 --request-rate 1
    bench "Latency (medium 512/256, rate=1)" \
        --dataset-name random \
        --random-input-len 512 --random-output-len 256 \
        --num-prompts 100 --request-rate 1
    bench "Latency (long 4096/512, rate=1)" \
        --dataset-name random \
        --random-input-len 4096 --random-output-len 512 \
        --num-prompts 100 --request-rate 1
}

# Concurrency: finds the server's saturation point by sweeping concurrent requests.
# request-rate=inf with max-concurrency=N caps how many requests run in parallel.
# At low concurrency (1-4), latency is good but throughput is low (GPU underutilized).
# At high concurrency (64-256), throughput plateaus and latency degrades (queueing).
# The "knee" where throughput stops improving is the optimal operating point.
# 500 prompts per level gives enough samples for stable percentile metrics.
bench_concurrency() {
    for c in 1 4 16 64 256; do
        bench "Concurrency=${c} (512in/256out)" \
            --dataset-name random \
            --random-input-len 512 --random-output-len 256 \
            --num-prompts 100 --request-rate inf --max-concurrency "$c"
    done
}

# Long context: tests behavior with very long inputs (4K→32K tokens).
# Inspired by GPUStack's "very long prompt" config (32000in/100out). Long inputs stress
# KV cache memory, attention compute, and may trigger chunked prefill. output=100 keeps
# decode short so we focus on prefill scaling. rate=1 and fewer prompts (50) because
# each request is expensive and we want to avoid OOM under memory pressure.
bench_longctx() {
    for len in 4096 16384 32000; do
        bench "Long context (input=${len})" \
            --dataset-name random \
            --random-input-len "$len" --random-output-len 100 \
            --num-prompts 50 --request-rate 1
    done
}

# ShareGPT: realistic conversational workload from real user conversations. Variable
# input/output lengths reflecting actual usage patterns. This is the standard dataset
# used by vLLM CI, GPUStack perf lab, and most published benchmarks. Unlike random
# datasets, ShareGPT captures the natural distribution of short/long prompts and
# responses, making it the best proxy for production chat traffic.
# Ref: github.com/vllm-project/vllm/blob/main/benchmarks/README.md
# Ref: GPUStack perf lab uses ShareGPT with 1000 prompts as primary benchmark.
#
# Requires: wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
SHAREGPT_PATH="${SHAREGPT_PATH:-ShareGPT_V3_unfiltered_cleaned_split.json}"
bench_sharegpt() {
    if [[ ! -f "$SHAREGPT_PATH" ]]; then
        echo "Downloading ShareGPT dataset..."
        wget -q -O "$SHAREGPT_PATH" \
            https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
    fi
    bench "ShareGPT (1000 prompts, max rate)" \
        --dataset-name sharegpt \
        --dataset-path "$SHAREGPT_PATH" \
        --num-prompts 100 --request-rate inf
    bench "ShareGPT (1000 prompts, rate=4)" \
        --dataset-name sharegpt \
        --dataset-path "$SHAREGPT_PATH" \
        --num-prompts 100 --request-rate 4
}

# Sonnet: uses Shakespeare's sonnets with a shared prefix to test prefix caching.
# All prompts share a common prefix (--sonnet-prefix-len=200 tokens of sonnet text),
# then each request gets a unique suffix. This exercises vLLM's automatic prefix
# caching — if enabled, the shared prefix KV cache is computed once and reused across
# requests, dramatically reducing TTFT. Comparing sonnet results with prefix caching
# on vs off shows the caching speedup.
# Ref: vllm/benchmarks/datasets.py SonnetDataset
# Ref: Default params: input=550, output=150, prefix=200
bench_sonnet() {
    SONNET_PATH="${SONNET_PATH:-sonnet.txt}"
    if [[ ! -f "$SONNET_PATH" ]]; then
        echo "Downloading Shakespeare sonnets..."
        wget -q -O "$SONNET_PATH" \
            https://raw.githubusercontent.com/vllm-project/vllm/main/benchmarks/sonnet.txt
    fi
    bench "Sonnet (prefix caching, max rate)" \
        --dataset-name sonnet --dataset-path "$SONNET_PATH" \
        --sonnet-input-len 550 --sonnet-output-len 150 --sonnet-prefix-len 200 \
        --num-prompts 100 --request-rate inf
    bench "Sonnet (prefix caching, rate=4)" \
        --dataset-name sonnet --dataset-path "$SONNET_PATH" \
        --sonnet-input-len 550 --sonnet-output-len 150 --sonnet-prefix-len 200 \
        --num-prompts 100 --request-rate 4
}

IFS=',' read -ra TESTS <<< "$TYPES"
for t in "${TESTS[@]}"; do
    t=$(echo "$t" | xargs)  # trim whitespace
    echo "========================================"
    echo "Running: $t"
    echo "========================================"
    case "$t" in
        throughput)  bench_throughput ;;
        prefill)     bench_prefill ;;
        decode)      bench_decode ;;
        latency)     bench_latency ;;
        concurrency) bench_concurrency ;;
        longctx)     bench_longctx ;;
        sharegpt)    bench_sharegpt ;;
        sonnet)      bench_sonnet ;;
        *) echo "Unknown test: $t"; exit 1 ;;
    esac
done
