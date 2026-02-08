============
vLLM Serving
============

.. contents:: Table of Contents
    :backlinks: none

This cheat sheet provides quick-reference commands for launching a vLLM server in both
local (single-node) and SLURM (multi-node) environments. It covers building the Docker
image, running with different parallelism strategies, and testing the server endpoints.

For detailed explanations of tensor parallelism, pipeline parallelism, data parallelism,
and expert parallelism, see the
`vLLM Serving Guide <https://github.com/crazyguitar/pysheeet/blob/master/docs/notes/llm/vllm-serving.rst>`_.

Build Docker Image
------------------

The Dockerfile bundles vLLM with EFA drivers, NCCL, NVSHMEM, and GDRCopy for
high-performance multi-node inference on GPU clusters. Build the image and save it
as a compressed tarball for distribution to SLURM nodes via a shared filesystem.

.. code-block:: bash

    # Build the Docker image with all dependencies
    make docker

    # Save as a compressed tarball for SLURM nodes
    # Output: vllm-serve-latest.tar.gz
    make save

Local Serving (Single Node)
---------------------------

For development or single-node deployments, vLLM can run directly on the host or
inside a Docker container. The server exposes an OpenAI-compatible API on port 8000.

**Bare metal** — run vLLM directly without Docker:

.. code-block:: bash

    # Single GPU — simplest way to serve a model
    vllm serve Qwen/Qwen2.5-7B-Instruct --host 0.0.0.0 --port 8000

    # Tensor parallel across 8 GPUs — for models too large for a single GPU
    vllm serve Qwen/Qwen2.5-14B-Instruct --tensor-parallel-size 8

**Using Docker (via Makefile)** — convenient targets for common configurations:

.. code-block:: bash

    # Single GPU with default model
    make single

    # Tensor parallel across 8 GPUs
    make mp TP=8

    # Pipeline parallel — split model into 2 stages, each with TP=4
    make mp TP=4 PP=2

    # Ray backend with data parallelism — 2 replicas, each using 4 GPUs
    make ray TP=4 DP=2

**Using Docker (via run.sh)** — the entrypoint script supports multiple serving modes
(single, ray, mp, rpc) with explicit control over parallelism settings:

.. code-block:: bash

    # Single GPU mode
    docker run --gpus all --rm --net=host -v /fsx:/fsx \
      vllm-serve:latest ./run.sh single --model Qwen/Qwen2.5-7B-Instruct

    # Multiprocessing mode with tensor parallelism
    docker run --gpus all --rm --net=host -v /fsx:/fsx \
      vllm-serve:latest ./run.sh mp --model Qwen/Qwen2.5-14B-Instruct --tp 8

SLURM Serving (Multi-Node)
---------------------------

``run.sbatch`` orchestrates multi-node vLLM serving on SLURM clusters. It handles
Docker image distribution, container launch with EFA/GPU passthrough, parallelism
computation, and health checking. The server runs until you stop it with ``Ctrl+C``
or ``scancel``.

**Script flags** — these are consumed by the script and not passed to vLLM:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Flag
     - Description
   * - ``--image PATH``
     - Docker image tarball or registry path (default: ``$WORKSPACE/vllm-serve-latest.tar.gz``)
   * - ``--workspace, -w PATH``
     - Base directory for default image and logs (default: ``$PWD``)
   * - ``--container-mount PATH``
     - Host path to bind-mount into containers (default: ``/fsx``)
   * - ``--force, -f``
     - Force remove existing containers and images before loading

All other arguments are passed directly to ``vllm serve`` as-is.

**Basic usage** — allocate nodes with ``salloc``, then run the script. The script
auto-detects the SLURM allocation and computes DP based on total GPUs and TP/PP:

.. code-block:: bash

    # Allocate 2 nodes with 8 GPUs each
    salloc -N 2 --gpus-per-node=8 --exclusive

    # Expert parallel for MoE models (TP=8, DP=2, EP=16 auto-computed)
    bash run.sbatch \
      Qwen/Qwen3-30B-A3B-FP8 \
      --tensor-parallel-size 8 \
      --enable-expert-parallel

**Custom image** — specify a different Docker image tarball or registry path:

.. code-block:: bash

    bash run.sbatch \
      --image /fsx/images/vllm-serve-latest.tar.gz \
      Qwen/Qwen3-30B-A3B-FP8 \
      --tensor-parallel-size 8 \
      --enable-expert-parallel

**Pipeline parallel** — for large dense models that need to be split across nodes.
The script auto-selects the multiprocessing backend when PP > 1:

.. code-block:: bash

    bash run.sbatch \
      deepseek-ai/DeepSeek-V2-Lite \
      --tensor-parallel-size 8 \
      --pipeline-parallel-size 2

**Force reload** — remove cached containers and images before loading:

.. code-block:: bash

    bash run.sbatch -f \
      Qwen/Qwen3-30B-A3B-FP8 \
      --tensor-parallel-size 8 \
      --enable-expert-parallel

**Backend selection** — the script supports three distributed backends. RPC is the
default and works best for most cases. Ray is useful when you need its scheduling
features. Multiprocessing is auto-selected for pipeline parallelism:

.. code-block:: bash

    # RPC backend (default) — lightweight, best for DP + EP
    bash run.sbatch Qwen/Qwen3-30B-A3B-FP8 --tensor-parallel-size 8

    # Ray backend — uses Ray cluster for worker management
    DP_BACKEND=ray bash run.sbatch \
      Qwen/Qwen3-30B-A3B-FP8 \
      --tensor-parallel-size 8 \
      --enable-expert-parallel

    # Multiprocessing backend — auto-selected when PP > 1
    bash run.sbatch \
      deepseek-ai/DeepSeek-V2-Lite \
      --tensor-parallel-size 8 \
      --pipeline-parallel-size 2

Test the Server
---------------

Once the server is ready, it prints the head node IP address. Use standard HTTP
requests to interact with the OpenAI-compatible API:

.. code-block:: bash

    # Health check — returns 200 when server is ready
    curl http://<HEAD_IP>:8000/health

    # List available models
    curl http://<HEAD_IP>:8000/v1/models

    # Chat completion request
    curl -X POST http://<HEAD_IP>:8000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{
        "model": "Qwen/Qwen3-30B-A3B-FP8",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 50
      }'

    # Or run the included test script
    bash test.sh

    # Test against a remote server
    bash test.sh -H 10.0.128.193

    # Test with specific port and model
    bash test.sh -H 10.0.128.193 -p 8000 -m Qwen/Qwen3-30B-A3B-FP8

Benchmark
---------

``bench.sh`` measures serving performance (throughput, TTFT, ITL, latency) by sending
requests to a running vLLM server. It handles Docker image loading and container
management automatically.

.. code-block:: bash

    # Run all benchmarks
    bash bench.sh -H 10.0.128.193 -i vllm-serve:latest

    # Run specific benchmarks
    bash bench.sh -H 10.0.128.193 -i vllm-serve:latest --type throughput,prefill

    # Via Makefile
    make bench HOST=10.0.128.193
    make bench HOST=10.0.128.193 BENCH_TYPE=throughput,prefill

Sweep
-----

``sweep.sh`` runs predefined sweep suites by calling ``sweep.sbatch`` for each
configuration. Each suite launches its own ``vllm serve``, sweeps a parameter, and
collects results. Requires GPU access.

.. code-block:: bash

    # All suites (rate, concurrency, input, output)
    bash sweep.sh -m Qwen/Qwen3-0.6B

    # Select specific suites
    bash sweep.sh -m Qwen/Qwen3-30B-A3B-FP8 \
        -i vllm-serve:latest \
        --serve-cmd "vllm serve Qwen/Qwen3-30B-A3B-FP8 -tp 8 --enable-expert-parallel" \
        --type rate,input

    # Via Makefile
    make sweep
    make sweep SWEEP_MODEL=Qwen/Qwen3-30B-A3B-FP8 SWEEP_TYPE=rate,concurrency

    # Show vllm serve stdout (model loading, request logs) — very useful for debugging
    bash sweep.sh -m Qwen/Qwen3-0.6B --show-stdout

    # Custom serve command — TP=2 with expert parallel, rate sweep only
    bash sweep.sh -m Qwen/Qwen1.5-MoE-A2.7B \
        --serve-cmd "vllm serve Qwen/Qwen1.5-MoE-A2.7B -tp 2 --enable-expert-parallel" \
        --type rate --show-stdout

Available suites:

- **rate** — Sweeps request rate (1, 2, 4, 8, 16, 32, inf) to find saturation point
- **concurrency** — Sweeps concurrent requests (1–128) to find optimal batch size
- **input** — Sweeps input length (128–16K) to measure TTFT scaling with context
- **output** — Sweeps output length (64–2048) to measure ITL as KV cache grows

For direct control over sweep parameters, use ``sweep.sbatch`` which passes all args
through to ``vllm bench sweep serve``:

.. code-block:: bash

    # Custom serve + bench commands
    bash sweep.sbatch -m Qwen/Qwen3-30B-A3B-FP8 \
        --serve-cmd "vllm serve Qwen/Qwen3-30B-A3B-FP8 -tp 8 --enable-expert-parallel" \
        --bench-cmd "vllm bench serve --model Qwen/Qwen3-30B-A3B-FP8 --dataset-name sharegpt" \
        --bench-params results/bench_params.json --show-stdout

See the `vLLM Benchmark Guide <https://github.com/crazyguitar/pysheeet/blob/master/docs/notes/llm/vllm-bench.rst>`_
for detailed explanations of each benchmark type and metric.

Notes and Limitations
---------------------

**Parallelism constraints:**

- vLLM does not support combining PP, TP, and DP simultaneously. When using PP mode,
  DP is not available (``TOTAL_GPUS = TP × PP``).
- EP and PP are mutually exclusive. Use EP for MoE models and PP for large dense models.
- EP is computed automatically: ``EP = TP × DP = world_size``. As TP increases, DP
  decreases proportionally to maintain the same total parallelism.

**Formulas:**

- EP mode: ``TOTAL_GPUS = DP × TP``, EP auto-computed by vLLM
- PP mode: ``TOTAL_GPUS = TP × PP`` (no DP)
- DP mode: ``TOTAL_GPUS = DP × TP``
