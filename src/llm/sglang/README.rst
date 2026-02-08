=============
SGLang Serving
=============

.. contents:: Table of Contents
    :backlinks: none

This cheat sheet provides quick-reference commands for launching an SGLang server in
both local (single-node) and SLURM (multi-node) environments. It covers building the
Docker image, running with different parallelism strategies, and testing the server.

Build Docker Image
------------------

The Dockerfile bundles SGLang with EFA drivers, NCCL, and GDRCopy for high-performance
multi-node inference on GPU clusters.

.. code-block:: bash

    # Build the Docker image
    make docker

    # Save as a compressed tarball for SLURM nodes
    # Output: sglang-serve-latest.tar.gz
    make save

Local Serving (Single Node)
---------------------------

For development or single-node deployments, SGLang can run directly on the host or
inside a Docker container. The server exposes an OpenAI-compatible API on port 30000.

**Bare metal** — run SGLang directly without Docker:

.. code-block:: bash

    # Single GPU
    python -m sglang.launch_server --model-path Qwen/Qwen2.5-7B-Instruct --host 0.0.0.0 --port 30000

    # Tensor parallel across 8 GPUs
    python -m sglang.launch_server --model-path Qwen/Qwen2.5-14B-Instruct --tp 8

    # MoE model with expert parallelism (EP subdivides TP)
    python -m sglang.launch_server --model-path Qwen/Qwen1.5-MoE-A2.7B --tp 8 --ep-size 2 --enable-ep

**Using Docker (via Makefile)**:

.. code-block:: bash

    # Single GPU with default model
    make serve MODEL=Qwen/Qwen2.5-7B-Instruct

    # Tensor parallel across 8 GPUs
    make serve MODEL=Qwen/Qwen2.5-14B-Instruct TP=8

**Using Docker directly**:

.. code-block:: bash

    # Single GPU
    docker run --gpus all --rm --net=host -v /fsx:/fsx \
      sglang-serve:latest \
      python3 -m sglang.launch_server --model-path Qwen/Qwen2.5-7B-Instruct --host 0.0.0.0 --port 30000

    # Tensor parallel across 8 GPUs
    docker run --gpus all --rm --net=host -v /fsx:/fsx \
      sglang-serve:latest \
      python3 -m sglang.launch_server --model-path Qwen/Qwen2.5-14B-Instruct --tp 8 --host 0.0.0.0 --port 30000

SLURM Serving (Multi-Node)
---------------------------

``run.sbatch`` orchestrates multi-node SGLang serving on SLURM clusters. It handles
Docker image distribution, container launch with EFA/GPU passthrough, and health
checking. The server runs until you stop it with ``Ctrl+C`` or ``scancel``.

**Script flags** — consumed by the script, not passed to SGLang:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Flag
     - Description
   * - ``--image PATH``
     - Docker image tarball or registry path (default: ``$WORKSPACE/sglang-serve-latest.tar.gz``)
   * - ``--workspace, -w PATH``
     - Base directory for default image and logs (default: ``$PWD``)
   * - ``--container-mount PATH``
     - Host path to bind-mount into containers (default: ``/fsx``)
   * - ``--force, -f``
     - Force remove existing containers and images before loading
   * - ``--profile``
     - Enable PyTorch profiler (writes to ``$PWD/sglang_profile``)

All other arguments are passed directly to ``python -m sglang.launch_server``.

**Basic usage**:

.. code-block:: bash

    # Allocate 2 nodes with 8 GPUs each
    salloc -N 2 --gpus-per-node=8 --exclusive

    # TP=16 across both nodes
    bash run.sbatch \
      --model-path Qwen/Qwen2.5-72B-Instruct \
      --tp 16

**MoE with expert parallelism**:

.. code-block:: bash

    # TP=8, EP=2 (experts split into 2 groups within TP)
    bash run.sbatch \
      --model-path Qwen/Qwen1.5-MoE-A2.7B \
      --tp 8 --ep-size 2 --enable-ep

**Data parallelism** — requires ``--enable-dp-attention`` for multi-node:

.. code-block:: bash

    # TP=8, DP=2 (2 replicas across 16 GPUs)
    bash run.sbatch \
      --model-path Qwen/Qwen2.5-14B-Instruct \
      --tp 8 --dp 2 --enable-dp-attention

**Pipeline parallelism**:

.. code-block:: bash

    # TP=8, PP=2
    bash run.sbatch \
      --model-path deepseek-ai/DeepSeek-V2-Lite \
      --tp 8 --pp 2

**Custom image**:

.. code-block:: bash

    bash run.sbatch \
      --image /fsx/images/sglang-serve-latest.tar.gz \
      --model-path Qwen/Qwen2.5-72B-Instruct \
      --tp 16

**Profiling**:

.. code-block:: bash

    # PyTorch profiler
    bash run.sbatch --profile \
      --model-path Qwen/Qwen2.5-14B-Instruct \
      --tp 8

Test the Server
---------------

SGLang serves on port 30000 by default:

.. code-block:: bash

    # Health check
    curl http://<HEAD_IP>:30000/health

    # List models
    curl http://<HEAD_IP>:30000/v1/models

    # Chat completion (OpenAI-compatible)
    curl -X POST http://<HEAD_IP>:30000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{
        "model": "Qwen/Qwen2.5-14B-Instruct",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 50
      }'

    # SGLang native generate endpoint
    curl -X POST http://<HEAD_IP>:30000/generate \
      -H "Content-Type: application/json" \
      -d '{"text": "Hello", "sampling_params": {"max_new_tokens": 50}}'

    # Run the included test script
    bash test.sh

    # Test against a remote server
    bash test.sh -H 10.0.128.193 -p 30000

Benchmark
---------

``bench.sh`` measures serving performance (throughput, TTFT, ITL, latency) by sending
requests to a running SGLang server. It handles Docker image loading and container
management automatically.

.. code-block:: bash

    # Run all benchmarks
    bash bench.sh -H 10.0.128.193 -i sglang-serve:latest

    # Run specific benchmarks
    bash bench.sh -H 10.0.128.193 -i sglang-serve:latest --type throughput,prefill

    # Via Makefile
    make bench HOST=10.0.128.193
    make bench HOST=10.0.128.193 BENCH_TYPE=throughput,prefill

Available benchmark types:

- **throughput** — peak output tokens/sec at max request rate
- **prefill** — TTFT scaling with input length (128→16K tokens)
- **decode** — ITL as output length grows (128→1024 tokens)
- **latency** — end-to-end latency under minimal load
- **concurrency** — throughput vs latency at different concurrency levels
- **sharegpt** — realistic conversational workload

Parallelism
-----------

SGLang's parallelism formula:

.. code-block:: text

    Total GPUs = TP × DP × PP

**EP is a subdivision of TP**, not a separate multiplier. When using ``--ep-size N``,
the TP GPUs are divided into N expert-parallel groups.

.. list-table::
   :widths: 20 10 10 10 10 40
   :header-rows: 1

   * - Config
     - TP
     - EP
     - DP
     - PP
     - Use case
   * - Dense, max throughput
     - 2
     - 1
     - 8
     - 1
     - 8 replicas of TP=2
   * - Dense, large model
     - 8
     - 1
     - 2
     - 1
     - 2 replicas of TP=8
   * - Dense, very large
     - 8
     - 1
     - 1
     - 2
     - Single replica, 2-stage pipeline
   * - MoE model
     - 8
     - 2
     - 1
     - 1
     - Experts split into 2 groups
   * - MoE, more EP
     - 8
     - 4
     - 1
     - 1
     - Experts split into 4 groups

**Constraints:**

- Multi-node DP requires ``--enable-dp-attention``
- EP only works with MoE models and requires ``--enable-ep``
- ``TP`` must be divisible by ``nnodes`` for multi-node
