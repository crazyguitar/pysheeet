====================
TensorRT-LLM Serving
====================

.. contents:: Table of Contents
    :backlinks: none

This cheat sheet provides quick-reference commands for launching a TensorRT-LLM server
in both local (single-node) and SLURM environments. It covers building the Docker image,
running with different parallelism strategies, and testing the server. TensorRT-LLM v1.1.0
uses ``trtllm-serve`` for OpenAI-compatible online serving and ``trtllm-bench`` for
benchmarking.

For more details, see the
`TensorRT-LLM documentation <https://nvidia.github.io/TensorRT-LLM/>`_ and
`GitHub repository <https://github.com/NVIDIA/TensorRT-LLM>`_.

Build Docker Image
------------------

The Dockerfile bundles TensorRT-LLM with EFA drivers, NCCL, and GDRCopy for
high-performance inference on GPU clusters.

.. code-block:: bash

    # Build the Docker image
    make docker

    # Save as a compressed tarball for SLURM nodes
    # Output: tensorrt-llm-serve-latest.tar.gz
    make save

Local Serving (Single Node)
---------------------------

TensorRT-LLM exposes an OpenAI-compatible API on port 8000 by default via
``trtllm-serve``.

**Bare metal** — run directly (requires TensorRT-LLM installed):

.. code-block:: bash

    # Single GPU
    trtllm-serve Qwen/Qwen2.5-7B-Instruct --host 0.0.0.0 --port 8000

    # Tensor parallel across 8 GPUs
    trtllm-serve Qwen/Qwen2.5-14B-Instruct --tp_size 8

    # FP8 quantized model
    trtllm-serve nvidia/Qwen3-8B-FP8

**Using Docker (via Makefile)**:

.. code-block:: bash

    # Single GPU with default model
    make serve MODEL=Qwen/Qwen2.5-7B-Instruct

    # Tensor parallel across 8 GPUs
    make serve MODEL=Qwen/Qwen2.5-14B-Instruct TP=8

**Using the NGC container directly**:

.. code-block:: bash

    docker run --gpus all --rm --ipc host --net=host \
      --ulimit memlock=-1 --ulimit stack=67108864 \
      -v /fsx:/fsx \
      nvcr.io/nvidia/tensorrt-llm/release:1.1.0 \
      trtllm-serve Qwen/Qwen2.5-7B-Instruct --host 0.0.0.0 --port 8000

SLURM Serving
-------------

``run.sbatch`` orchestrates TensorRT-LLM serving on SLURM clusters. It handles Docker
image distribution, container launch with EFA/GPU passthrough, and health checking.
The server runs until you stop it with ``Ctrl+C`` or ``scancel``.

**Script flags** — consumed by the script, not passed to trtllm-serve:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Flag
     - Description
   * - ``--image PATH``
     - Docker image tarball or registry path (default: ``$WORKSPACE/tensorrt-llm-serve-latest.tar.gz``)
   * - ``--workspace, -w PATH``
     - Base directory for default image and logs (default: ``$PWD``)
   * - ``--container-mount PATH``
     - Host path to bind-mount into containers (default: ``/fsx``)
   * - ``--force, -f``
     - Force remove existing containers and images before loading

All other arguments are passed directly to ``trtllm-serve``.

**Basic usage**:

.. code-block:: bash

    # Allocate 1 node with 8 GPUs
    salloc -N 1 --gpus-per-node=8 --exclusive

    # Serve with TP=8
    bash run.sbatch \
      Qwen/Qwen2.5-14B-Instruct \
      --tp_size 8

**Custom image**:

.. code-block:: bash

    bash run.sbatch \
      --image /fsx/images/tensorrt-llm-serve-latest.tar.gz \
      Qwen/Qwen2.5-72B-Instruct \
      --tp_size 8

**FP8 quantized model**:

.. code-block:: bash

    bash run.sbatch nvidia/Qwen3-8B-FP8

Test the Server
---------------

TensorRT-LLM serves on port 8000 by default:

.. code-block:: bash

    # Health check
    curl http://<HEAD_IP>:8000/health

    # List models
    curl http://<HEAD_IP>:8000/v1/models

    # Chat completion (OpenAI-compatible)
    curl -X POST http://<HEAD_IP>:8000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{
        "model": "Qwen/Qwen2.5-14B-Instruct",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 50
      }'

    # Completions endpoint
    curl -X POST http://<HEAD_IP>:8000/v1/completions \
      -H "Content-Type: application/json" \
      -d '{
        "model": "Qwen/Qwen2.5-14B-Instruct",
        "prompt": "The capital of France is",
        "max_tokens": 20
      }'

    # Run the included test script
    bash test.sh

    # Test against a remote server
    bash test.sh -H 10.0.128.193 -p 8000

Benchmark
---------

``bench.sh`` measures serving performance (throughput, TTFT, ITL, latency) by sending
requests to a running TensorRT-LLM server. It handles Docker image loading and container
management automatically.

.. code-block:: bash

    # Run all benchmarks
    bash bench.sh -H 10.0.128.193 -i tensorrt-llm-serve:latest

    # Run specific benchmarks
    bash bench.sh -H 10.0.128.193 -i tensorrt-llm-serve:latest --type throughput,prefill

    # Via Makefile
    make bench HOST=10.0.128.193
    make bench HOST=10.0.128.193 BENCH_TYPE=throughput,prefill

Available benchmark types:

- **throughput** — peak output tokens/sec at max request rate
- **prefill** — TTFT scaling with input length (128→4096 tokens)
- **decode** — ITL as output length grows (128→1024 tokens)
- **latency** — end-to-end latency under minimal load
- **concurrency** — throughput vs latency at different concurrency levels
- **sharegpt** — realistic conversational workload

You can also use the built-in ``trtllm-bench`` CLI for more detailed benchmarking:

.. code-block:: bash

    # Static benchmark (no server needed)
    trtllm-bench --model Qwen/Qwen2.5-7B-Instruct \
      --dataset-type synthetic --num-requests 100

Parallelism
-----------

TensorRT-LLM supports multiple parallelism strategies, configured via a YAML file
(``parallel_config.yaml``) passed with ``--config``:

- **Tensor Parallel (TP)** — shards model weights across GPUs
- **Pipeline Parallel (PP)** — distributes layers across GPUs
- **Data Parallel (DP)** — replicates model across GPUs for different requests
- **Expert Parallel (EP)** — distributes MoE experts across GPUs
- **Context Parallel (CP)** — distributes long-context processing across GPUs
- **Wide-EP** — advanced EP with load balancing for large-scale MoE (DeepSeek-V3/R1, LLaMA4, Qwen3)

**Attention module** supports TP (small batches) or DP (large batches) via
``enable_attention_dp``. **MoE FFN** supports TP, EP, or hybrid ETP where
``moe_tensor_parallel_size × moe_expert_parallel_size = tensor_parallel_size``.

.. list-table::
   :widths: 25 55 20
   :header-rows: 1

   * - Strategy
     - Use case
     - Key config
   * - TP only
     - Dense models, small batch, memory-constrained
     - ``tensor_parallel_size: 8``
   * - PP
     - Very large models that don't fit in single-node GPU memory
     - ``pipeline_parallel_size: 2``
   * - DP (attention)
     - Large batch, high throughput
     - ``enable_attention_dp: true``
   * - EP only (MoE)
     - MoE models with high expert count
     - ``moe_expert_parallel_size: 8``
   * - Hybrid ETP (MoE)
     - Balance workload and kernel efficiency
     - ``moe_tensor_parallel_size: 4, moe_expert_parallel_size: 2``
   * - Wide-EP (MoE)
     - Large-scale MoE with load balancing (hot expert replication)
     - See ``examples/wide_ep/``

**Configuration via YAML** (recommended):

.. code-block:: yaml

    # parallel_config.yaml

    # Dense model: TP=8
    tensor_parallel_size: 8

    # Dense model: TP=8 with attention DP
    # tensor_parallel_size: 8
    # enable_attention_dp: true

    # MoE: EP only
    # tensor_parallel_size: 8
    # moe_expert_parallel_size: 8

    # MoE: Hybrid TP-4 × EP-2
    # tensor_parallel_size: 8
    # moe_tensor_parallel_size: 4
    # moe_expert_parallel_size: 2

.. code-block:: bash

    trtllm-serve Qwen/Qwen2.5-14B-Instruct --config parallel_config.yaml

**Quick examples via CLI flags**:

.. code-block:: bash

    # TP=4
    trtllm-serve Qwen/Qwen2.5-14B-Instruct --tp_size 4

    # TP=8, PP=2
    trtllm-serve Qwen/Qwen2.5-72B-Instruct --tp_size 8 --pp_size 2

    # MoE with EP
    trtllm-serve Qwen/Qwen1.5-MoE-A2.7B --tp_size 8 --ep_size 4

Key Differences from SGLang / vLLM
-----------------------------------

.. list-table::
   :widths: 25 25 25 25
   :header-rows: 1

   * - Feature
     - TensorRT-LLM
     - SGLang
     - vLLM
   * - Serve command
     - ``trtllm-serve <model>``
     - ``python -m sglang.launch_server``
     - ``vllm serve <model>``
   * - Default port
     - 8000
     - 30000
     - 8000
   * - TP flag
     - ``--tp_size N``
     - ``--tp N``
     - ``--tensor-parallel-size N``
   * - Bench tool
     - ``trtllm-bench``
     - ``sglang.bench_serving``
     - ``vllm bench``
   * - Container
     - ``nvcr.io/nvidia/tensorrt-llm/release``
     - Custom build
     - Custom build
   * - Quantization
     - FP8, FP4, INT4 AWQ, INT8 SQ
     - FP8, AWQ, GPTQ
     - FP8, AWQ, GPTQ, BitsAndBytes
