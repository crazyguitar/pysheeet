.PHONY: help docker save clean

.DEFAULT_GOAL := help

IMAGE_NAME ?= nixl
IMAGE_TAG ?= latest

help:
	@echo "NIXL Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  docker    Build Docker image"
	@echo "  save      Save Docker image to tar.gz"
	@echo "  clean     Remove image and tarball"
	@echo ""
	@echo "Usage:"
	@echo "  make docker && make save"

docker:
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) -f Dockerfile .

save:
	docker save $(IMAGE_NAME):$(IMAGE_TAG) | pigz > $(IMAGE_NAME)-$(IMAGE_TAG).tar.gz

clean:
	-docker rmi $(IMAGE_NAME):$(IMAGE_TAG) 2>/dev/null || true
	-rm -f $(IMAGE_NAME)-$(IMAGE_TAG).tar.gz
