.PHONY: help docker sqush save load serve test bench clean

.DEFAULT_GOAL := help

help:
	@echo "SGLang Serving Makefile"
	@echo ""
	@echo "Build targets:"
	@echo "  docker              Build Docker image"
	@echo "  sqush               Build Enroot sqsh file"
	@echo "  save                Save Docker image to tar.gz"
	@echo "  load                Load Docker image from tar.gz"
	@echo ""
	@echo "Run targets:"
	@echo "  serve               Launch SGLang server"
	@echo "  test                Test SGLang API endpoints"
	@echo "  bench               Run benchmarks (HOST=ip)"
	@echo ""
	@echo "Variables:"
	@echo "  MODEL=$(MODEL)"
	@echo "  PORT=$(PORT)"
	@echo "  TP=$(TP)"
	@echo ""
	@echo "Examples:"
	@echo "  make docker"
	@echo "  make serve MODEL=Qwen/Qwen2.5-14B-Instruct TP=8"
	@echo "  make test HOST=10.0.128.193"
	@echo "  make bench HOST=10.0.128.193"
	@echo ""
	@echo "Cleanup:"
	@echo "  clean               Remove containers and images"

IMAGE_NAME ?= sglang-serve
IMAGE_TAG ?= latest
CONTAINER_NAME ?= sglang-server
MODEL ?= Qwen/Qwen2.5-7B-Instruct
PORT ?= 30000
TP ?= 1
HOST ?= localhost
BENCH_TYPE ?=

DEVICES := --device=/dev/gdrdrv $(shell find /dev/infiniband -name "uverbs*" 2>/dev/null | sed 's/^/--device=/')

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
	--entrypoint bash \
	$(IMAGE_NAME):$(IMAGE_TAG)

docker:
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) -f Dockerfile .

sqush: docker
	enroot import -o $(IMAGE_NAME)-$(IMAGE_TAG).sqsh dockerd://$(IMAGE_NAME):$(IMAGE_TAG)

save:
	docker save $(IMAGE_NAME):$(IMAGE_TAG) | pigz > $(IMAGE_NAME)-$(IMAGE_TAG).tar.gz

load:
	pigz -dc $(IMAGE_NAME)-$(IMAGE_TAG).tar.gz | docker load

serve:
	$(DOCKER_RUN) -c 'python3 -m sglang.launch_server --model-path $(MODEL) --host 0.0.0.0 --port $(PORT) --tp $(TP)'

test:
	@./test.sh -H $(HOST) -p $(PORT)

bench:
	@bash bench.sh -H $(HOST) -p $(PORT) -i $(IMAGE_NAME):$(IMAGE_TAG) $(if $(BENCH_TYPE),--type $(BENCH_TYPE))

clean:
	-docker rm -f $(CONTAINER_NAME) 2>/dev/null || true
	-docker rmi $(IMAGE_NAME):$(IMAGE_TAG) 2>/dev/null || true
	-rm -f $(IMAGE_NAME)-$(IMAGE_TAG).sqsh
	-rm -f $(IMAGE_NAME)-$(IMAGE_TAG).tar.gz
