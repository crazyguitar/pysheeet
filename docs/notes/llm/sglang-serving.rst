.. meta::
    :description lang=en: SGLang serving guide — single-node, multi-node SLURM, Docker, tensor/pipeline/data/expert parallelism for LLM inference on GPU clusters.
    :keywords: SGLang, SGLang serving, SGLang tutorial, LLM inference, LLM serving, model serving, distributed inference, tensor parallelism, pipeline parallelism, data parallelism, expert parallelism, MoE serving, GPU inference, OpenAI compatible API, multi-node GPU, multi-GPU serving, SLURM, HPC, EFA, NCCL, RadixAttention, continuous batching, Docker, production deployment, Qwen, Llama, DeepSeek

==============
SGLang Serving
==============

.. contents:: Table of Contents
    :backlinks: none

SGLang is a high-performance inference engine for large language models, featuring
RadixAttention for efficient KV cache reuse across requests with shared prefixes,
continuous batching for maximizing GPU utilization, and optimized CUDA kernels. It
provides an OpenAI-compatible API alongside its native generation endpoint, supporting
distributed inference across multiple GPUs and nodes.

This guide covers deployment from single-GPU to multi-node distributed serving with
tensor parallelism, pipeline parallelism, data parallelism, and expert parallelism for
Mixture-of-Experts (MoE) models. All scripts and examples are located in the
`src/llm/sglang/ <https://github.com/crazyguitar/pysheeet/tree/master/src/llm/sglang>`_ directory.

Quick Start
-----------

Install SGLang and launch a server. The server exposes both an OpenAI-compatible API
and SGLang's native ``/generate`` endpoint.

.. code-block:: bash

    # Install SGLang
    pip install "sglang[all]"

    # Start server with Qwen 7B model (default port 30000)
    python -m sglang.launch_server \
      --model-path Qwen/Qwen2.5-7B-Instruct \
      --host 0.0.0.0 --port 30000

    # Test with OpenAI-compatible endpoint
    curl -X POST http://localhost:30000/v1/completions \
      -H "Content-Type: application/json" \
      -d '{
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "prompt": "Hello, my name is",
        "max_tokens": 50
      }'

Common API Examples
-------------------

SGLang provides both OpenAI-compatible endpoints and its native ``/generate`` endpoint.

**Chat Completions** - Most common usage:

.. code-block:: bash

    curl -X POST http://localhost:30000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": [
          {"role": "system", "content": "You are a helpful assistant."},
          {"role": "user", "content": "What is the capital of France?"}
        ],
        "max_tokens": 50
      }'

For more API examples, refer to `test.sh <https://github.com/crazyguitar/pysheeet/blob/master/src/llm/sglang/test.sh>`_.

Tensor Parallel (TP)
--------------------

Tensor parallelism splits model layers across multiple GPUs. Each GPU holds a portion
of the weight matrices and participates in every forward pass via all-reduce operations.

**Use when:** Model doesn't fit on a single GPU.

.. code-block:: bash

    # Serve 14B model across 8 GPUs
    python -m sglang.launch_server \
      --model-path Qwen/Qwen2.5-14B-Instruct \
      --tp 8

Pipeline Parallel (PP)
----------------------

Pipeline parallelism divides the model into sequential stages across GPUs. Each stage
processes different layers, reducing communication overhead compared to tensor parallelism.

**Use when:** Scaling across nodes with slower interconnects.

.. code-block:: bash

    # PP=2 splits model into 2 stages, TP=4 within each stage
    python -m sglang.launch_server \
      --model-path Qwen/Qwen2.5-14B-Instruct \
      --tp 4 --pp 2

Data Parallel (DP)
------------------

Data parallelism creates multiple independent model replicas, each processing different
requests. This maximizes throughput when you have GPUs for multiple copies.

**Use when:** You need higher request throughput.

.. code-block:: bash

    # 2 replicas, each using 8 GPUs (total 16 GPUs)
    python -m sglang.launch_server \
      --model-path Qwen/Qwen2.5-14B-Instruct \
      --tp 8 --dp 2

    # Multi-node DP requires --enable-dp-attention
    python -m sglang.launch_server \
      --model-path Qwen/Qwen2.5-14B-Instruct \
      --tp 8 --dp 2 --enable-dp-attention

Expert Parallel (EP)
--------------------

Expert parallelism is for Mixture-of-Experts (MoE) models. Unlike vLLM where EP is a
separate multiplier, **SGLang's EP is a subdivision of TP**. The TP GPUs are divided
into EP expert-parallel groups.

**Use when:** Serving MoE models.

.. code-block:: bash

    # TP=8 GPUs divided into EP=2 expert groups (4 GPUs per group)
    python -m sglang.launch_server \
      --model-path Qwen/Qwen1.5-MoE-A2.7B \
      --tp 8 --ep 2

Parallelism Formula
-------------------

SGLang's parallelism relationship:

.. code-block:: text

    Total GPUs = TP × DP × PP

**EP is a subdivision of TP**, not a separate multiplier. When using ``--ep N``,
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

Distributed Serving on SLURM
----------------------------

For production deployments on HPC clusters, SGLang can be launched across multiple SLURM
nodes using ``run.sbatch``. This script automates Docker image distribution, container
launch with EFA networking, and multi-node coordination.

**Basic usage**:

.. code-block:: bash

    # Allocate 2 nodes with 8 GPUs each
    salloc -N 2 --gpus-per-node=8 --exclusive

    # TP=16 across both nodes
    bash run.sbatch \
      --model-path Qwen/Qwen2.5-72B-Instruct \
      --tp 16

    # MoE with expert parallelism
    bash run.sbatch \
      --model-path Qwen/Qwen1.5-MoE-A2.7B \
      --tp 8 --ep 2

    # Data parallelism (requires --enable-dp-attention for multi-node)
    bash run.sbatch \
      --model-path Qwen/Qwen2.5-14B-Instruct \
      --tp 8 --dp 2 --enable-dp-attention

The server prints the head node IP when ready:

.. code-block:: bash

    curl http://<HEAD_IP>:30000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{
        "model": "Qwen/Qwen2.5-72B-Instruct",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 50
      }'

For the full list of script flags and examples, see the
`SGLang README <https://github.com/crazyguitar/pysheeet/blob/master/src/llm/sglang/README.rst>`_.
