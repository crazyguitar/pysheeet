IMAGE_NAME ?= megatron-lm

.PHONY: build sqsh clean fmt

build:
	./enroot.sh -n "$(IMAGE_NAME)" -f Dockerfile

sqsh: build

clean:
	docker rmi -f "$(IMAGE_NAME)" 2>/dev/null || true
	rm -f *.sqsh

fmt:
	shfmt -i 2 -w *.sh
	black recipes/
