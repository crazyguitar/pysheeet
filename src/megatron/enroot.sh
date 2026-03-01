#!/bin/bash

FILE="${PWD}/Dockerfile"
IMAGE=""
INPUT="${PWD}"
PROGRAM="$0"
OUTPUT="${PWD}"

err() {
  echo -e "[$(date +'%Y-%m-%dT%H:%M:%S%z')][error] $*" >&2
}

usage() {
  set +x
  cat <<EOF
usage:  $PROGRAM [OPTIONS] params

options:

  -h,--help             show this help
  -f,--file [path]      input docker file. [default: ${PWD}/Dockerfile]
  -i,--input [path]     input directory for docker build. [default: ${INPUT}]
  -o,--output [path]    output directory. [default: ${OUTPUT}]
  -n,--image [image]    image name.

example:

  ./enroot.sh -n "nccl-tests" -f "Dockerfile"
EOF
}

run() {
  local image="${1}"
  local output="${2}"
  local arr=()
  local url
  local tag

  if ! command -v enroot &>/dev/null; then
    err "please install 'enroot' cli"
    return 1
  fi

  # split string
  IFS=$'\n' read -d "" -ra arr <<<"${image//\//$'\n'}"

  url="${arr[0]}"
  tag="${arr[1]//\:/-}"

  if [ -z "${url}" ]; then
    err "parse ${image} fail"
    return 1
  fi

  if [ -z "${tag}" ]; then
    tag="latest"
  fi

  if ! enroot import -o "${output}/${url}+${tag}.sqsh" "dockerd://${image}"; then
    err "enroot import ${image} fail"
    return 1
  fi
}

build() {
  local file="${1}"
  local image="${2}"
  local input="${3}"
  local arr=()
  local url

  if ! command -v docker &>/dev/null; then
    err "please install 'docker' cli"
    return 1
  fi

  if [ -z "${image}" ]; then
    err "please specify a image name"
    return 1
  fi

  IFS=$'\n' read -d "" -ra arr <<<"${image//\//$'\n'}"

  url="${arr[0]}"

  if [ -z "${url}" ]; then
    err "parse ${image} url fail"
    return 1
  fi

  docker images "${url}" -q | xargs -I{} docker rmi -f {}

  if ! docker build -f "${file}" -t "${image}" "${input}"; then
    err "docker bukld -f $file -t $image $input fail"
    return 1
  fi
}

while (("$#")); do
  case "$1" in
  -h | -\? | --help)
    usage
    exit 0
    ;;
  -f | --file)
    FILE="$2"
    shift 2
    ;;
  -n | --image)
    IMAGE="${2}"
    shift 2
    ;;
  -i | --input)
    INPUT="${2}"
    shift 2
    ;;
  -o | --output)
    OUTPUT="${2}"
    shift 2
    ;;
  *) break ;;
  esac
done

set -exo pipefail

if ! build "${FILE}" "${IMAGE}" "${INPUT}"; then
  err "build image fail"
  exit 1
fi

if ! run "${IMAGE}" "${OUTPUT}"; then
  err "create enroot image fail"
  exit 1
fi
