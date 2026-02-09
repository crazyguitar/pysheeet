#!/usr/bin/env bash
# TensorRT-LLM API test script

set -uo pipefail

HOST="localhost"
PORT="8000"
MODEL=""

while (( "$#" )); do
  case "$1" in
    -h|--help)  echo "Usage: $0 [-H host] [-p port] [-m model]"; exit 0 ;;
    -H|--host)  HOST="$2"; shift 2 ;;
    -p|--port)  PORT="$2"; shift 2 ;;
    -m|--model) MODEL="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

BASE_URL="http://${HOST}:${PORT}"

# Auto-detect model from server if not specified
if [[ -z "$MODEL" ]]; then
  MODEL=$(curl -sf "${BASE_URL}/v1/models" | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null)
  if [[ -z "$MODEL" ]]; then
    echo "ERROR: Cannot detect model. Is the server running at ${BASE_URL}?"
    exit 1
  fi
fi

PASS=0
FAIL=0

echo "Testing TensorRT-LLM server at ${BASE_URL}"
echo "Model: ${MODEL}"
echo "========================================"

test_endpoint() {
  local name="$1" cmd="$2"
  echo -ne "\n${name}... "
  if eval "$cmd" 2>&1; then
    ((PASS++))
  else
    ((FAIL++))
  fi
}

test_endpoint "[1/8] Health check" \
  "curl -sf '${BASE_URL}/health' | jq -e '.'"

test_endpoint "[2/8] List models" \
  "curl -sf '${BASE_URL}/v1/models' | jq -e '.data'"

test_endpoint "[3/8] Basic completions" \
  "curl -sf -X POST '${BASE_URL}/v1/completions' \
    -H 'Content-Type: application/json' \
    -d '{\"model\": \"${MODEL}\", \"prompt\": \"Hello\", \"max_tokens\": 10}' | jq -e '.choices'"

test_endpoint "[4/8] Chat completions" \
  "curl -sf -X POST '${BASE_URL}/v1/chat/completions' \
    -H 'Content-Type: application/json' \
    -d '{\"model\": \"${MODEL}\", \"messages\": [{\"role\": \"user\", \"content\": \"Hi\"}], \"max_tokens\": 10}' | jq -e '.choices'"

test_endpoint "[5/8] Sampling parameters" \
  "curl -sf -X POST '${BASE_URL}/v1/chat/completions' \
    -H 'Content-Type: application/json' \
    -d '{\"model\": \"${MODEL}\", \"messages\": [{\"role\": \"user\", \"content\": \"Hi\"}], \"max_tokens\": 10, \"temperature\": 0.9, \"top_p\": 0.95}' | jq -e '.choices'"

test_endpoint "[6/8] Streaming" \
  "curl -sf -X POST '${BASE_URL}/v1/chat/completions' \
    -H 'Content-Type: application/json' \
    -d '{\"model\": \"${MODEL}\", \"messages\": [{\"role\": \"user\", \"content\": \"Hi\"}], \"stream\": true, \"max_tokens\": 10}' | grep -q 'data:'"

test_endpoint "[7/8] Stop sequences" \
  "curl -sf -X POST '${BASE_URL}/v1/completions' \
    -H 'Content-Type: application/json' \
    -d '{\"model\": \"${MODEL}\", \"prompt\": \"1.\", \"max_tokens\": 20, \"stop\": [\"3.\"]}' | jq -e '.choices'"

test_endpoint "[8/8] Batch completions" \
  "curl -sf -X POST '${BASE_URL}/v1/completions' \
    -H 'Content-Type: application/json' \
    -d '{\"model\": \"${MODEL}\", \"prompt\": [\"Once\", \"In\"], \"max_tokens\": 10}' | jq -e '.choices'"

echo -e "\n========================================"
echo "Results: ${PASS} passed, ${FAIL} failed"
[[ $FAIL -gt 0 ]] && exit 1
