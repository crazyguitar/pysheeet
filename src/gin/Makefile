.PHONY: help docker save clean

.DEFAULT_GOAL := help

IMAGE_NAME ?= nccl
IMAGE_TAG ?= latest

help:
	@echo "NCCL GIN Test Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  docker    Build Docker image"
	@echo "  save      Save Docker image to tar.gz and sqsh"
	@echo "  clean     Remove image, tarball, and sqsh"
	@echo ""
	@echo "Usage:"
	@echo "  make docker && make save"

docker:
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) -f Dockerfile .

save:
	enroot import -o $(IMAGE_NAME)+$(IMAGE_TAG).sqsh dockerd://$(IMAGE_NAME):$(IMAGE_TAG)
	docker save $(IMAGE_NAME):$(IMAGE_TAG) | pigz > $(IMAGE_NAME)+$(IMAGE_TAG).tar.gz

clean:
	-docker rmi $(IMAGE_NAME):$(IMAGE_TAG) 2>/dev/null || true
	-rm -f $(IMAGE_NAME)+$(IMAGE_TAG).sqsh
	-rm -f $(IMAGE_NAME)-$(IMAGE_TAG).tar.gz
