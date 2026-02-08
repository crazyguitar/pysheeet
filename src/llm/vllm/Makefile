.PHONY: help docker sqush save load single ray mp rpc test bench clean

# Default target
.DEFAULT_GOAL := help

# Show help
help:
	@echo "vLLM Serving Makefile"
	@echo ""
	@echo "Build targets:"
	@echo "  docker              Build Docker image"
	@echo "  sqush               Build Enroot sqsh file"
	@echo "  save                Save Docker image to tar.gz"
	@echo "  load                Load Docker image from tar.gz"
	@echo ""
	@echo "Run targets:"
	@echo "  single              Single-node serving (default)"
	@echo "  ray                 Ray-based distributed serving"
	@echo "  mp                  Multiprocessing mode (TP/PP)"
	@echo "  rpc                 RPC mode for multi-node (TP/DP)"
	@echo "  test                Test vLLM API endpoints"
	@echo "  bench               Run benchmarks (HOST=ip IMAGE=path)"
	@echo "                      BENCH_TYPE=throughput,prefill,... to select"
	@echo ""
	@echo "Variables:"
	@echo "  MODEL=$(MODEL)"
	@echo "  PORT=$(PORT)"
	@echo "  TP=$(TP) PP=$(PP) DP=$(DP)"
	@echo "  HEAD_IP=$(HEAD_IP) NODE_RANK=$(NODE_RANK)"
	@echo ""
	@echo "Examples:"
	@echo "  make docker"
	@echo "  make single MODEL=Qwen/Qwen2.5-14B-Instruct"
	@echo "  make mp TP=4 PP=2"
	@echo "  make rpc TP=2 DP=4 HEAD_IP=10.0.0.1 NODE_RANK=0"
	@echo "  make bench HOST=10.0.128.193"
	@echo "  make bench HOST=10.0.128.193 BENCH_TYPE=throughput,prefill"
	@echo ""
	@echo "Cleanup:"
	@echo "  clean               Remove containers and images"

IMAGE_NAME ?= vllm-serve
IMAGE_TAG ?= latest
CONTAINER_NAME ?= vllm-server
MODEL ?= Qwen/Qwen2.5-7B-Instruct
PORT ?= 8000
TP ?= 1
PP ?= 1
DP ?= 1
HOST ?= localhost
BENCH_TYPE ?=
NODE_RANK ?= 0

# Discover EFA devices dynamically
DEVICES := --device=/dev/gdrdrv $(shell find /dev/infiniband -name "uverbs*" 2>/dev/null | sed 's/^/--device=/')

# Docker run base command
DOCKER_RUN = docker run --gpus all \
	--privileged \
	--uts=host \
	--ipc=host \
	--net=host \
	--ulimit stack=67108864 \
	--ulimit memlock=-1 \
	--security-opt seccomp=unconfined \
	$(DEVICES) \
	--rm \
	--name $(CONTAINER_NAME) \
	-v /fsx:/fsx \
	-v $(PWD):$(PWD) -w $(PWD) \
	--entrypoint bash \
	$(IMAGE_NAME):$(IMAGE_TAG)

# Build Docker image
docker:
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) -f Dockerfile .

# Build Enroot sqsh file
sqush: docker
	enroot import -o $(IMAGE_NAME)-$(IMAGE_TAG).sqsh dockerd://$(IMAGE_NAME):$(IMAGE_TAG)

# Save Docker image to tar.gz
save:
	docker save $(IMAGE_NAME):$(IMAGE_TAG) | pigz > $(IMAGE_NAME)-$(IMAGE_TAG).tar.gz

# Load Docker image from tar.gz
load:
	pigz -dc $(IMAGE_NAME)-$(IMAGE_TAG).tar.gz | docker load

# Launch single-node mode
single:
	$(DOCKER_RUN) ./run.sh single --model $(MODEL) --port $(PORT)

# Launch Ray backend
ray:
	$(DOCKER_RUN) ./run.sh ray --model $(MODEL) --port $(PORT) --tp $(TP) --pp $(PP) --dp $(DP)

# Launch multiprocessing mode
mp:
	$(DOCKER_RUN) ./run.sh mp --model $(MODEL) --port $(PORT) --tp $(TP) --pp $(PP) --dp $(DP)

# Launch RPC mode
rpc:
	$(DOCKER_RUN) ./run.sh rpc --model $(MODEL) --port $(PORT) --tp $(TP) --pp $(PP) --dp $(DP) \
		--head-ip $(HEAD_IP) --node-rank $(NODE_RANK)

# Test API endpoints
test:
	@./test.sh

# Run benchmarks inside Docker container
# Usage: make bench HOST=10.0.128.193
#        make bench HOST=10.0.128.193 BENCH_TYPE=throughput,prefill
bench:
	-@docker rm -f $(CONTAINER_NAME) 2>/dev/null || true
	$(DOCKER_RUN) bench.sh -H $(HOST) -p $(PORT) $(if $(BENCH_TYPE),--type $(BENCH_TYPE))

# Clean up containers and images
clean:
	-docker rm -f $(CONTAINER_NAME) 2>/dev/null || true
	-docker rmi $(IMAGE_NAME):$(IMAGE_TAG) 2>/dev/null || true
	-rm -f $(IMAGE_NAME)-$(IMAGE_TAG).sqsh
	-rm -f $(IMAGE_NAME)-$(IMAGE_TAG).tar.gz
