.PHONY: help docker save clean format

.DEFAULT_GOAL := help

IMAGE_NAME ?= vllm-serve
IMAGE_TAG ?= latest

help:
	@echo "vLLM Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  docker    Build Docker image"
	@echo "  save      Save Docker image to tar.gz"
	@echo "  format    Format shell scripts with shfmt (indent=2)"
	@echo "  clean     Remove image and tarball"
	@echo ""
	@echo "Usage:"
	@echo "  make docker && make save"
	@echo "  make format"

docker:
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) -f Dockerfile .

save:
	docker save $(IMAGE_NAME):$(IMAGE_TAG) | pigz > $(IMAGE_NAME)-$(IMAGE_TAG).tar.gz

format:
	@command -v shfmt >/dev/null 2>&1 || { echo "shfmt not found. Install: brew install shfmt"; exit 1; }
	shfmt -i 2 -w *.sh *.sbatch

clean:
	-docker rmi $(IMAGE_NAME):$(IMAGE_TAG) 2>/dev/null || true
	-rm -f $(IMAGE_NAME)-$(IMAGE_TAG).tar.gz
