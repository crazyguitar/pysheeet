.. meta::
    :description lang=en: vLLM serving guide — single-node, multi-node SLURM, Docker, Ray, RPC, tensor/pipeline/data/expert parallelism for LLM inference on GPU clusters.
    :keywords: vLLM, vLLM serving, vLLM tutorial, LLM inference, LLM serving, model serving, distributed inference, Ray cluster, RPC backend, tensor parallelism, pipeline parallelism, data parallelism, expert parallelism, MoE serving, GPU inference, OpenAI compatible API, multi-node GPU, multi-GPU serving, SLURM, HPC, EFA, NCCL, PagedAttention, continuous batching, Docker, production deployment, Qwen, Llama, DeepSeek

============
vLLM Serving
============

.. contents:: Table of Contents
    :backlinks: none

vLLM is a high-performance inference engine designed for production-grade serving of large
language models (LLMs). It delivers exceptional throughput and low latency through advanced
optimizations including PagedAttention for efficient memory management, continuous batching
for maximizing GPU utilization, and highly optimized CUDA kernels. vLLM provides an
OpenAI-compatible API, making it a drop-in replacement for OpenAI services while supporting
distributed inference across multiple GPUs and nodes for serving the largest models.

This guide covers everything from basic single-GPU deployment to advanced multi-node
distributed serving with tensor parallelism, pipeline parallelism, data parallelism, and
expert parallelism for Mixture-of-Experts (MoE) models. All scripts and examples are
located in the `src/llm/vllm/ <https://github.com/crazyguitar/pysheeet/tree/master/src/llm/vllm>`_ directory.

Quick Start
-----------

Get started with vLLM in minutes. Install the package and launch a server with a pre-trained
model. The server exposes an OpenAI-compatible API endpoint that you can query with standard
HTTP requests or the OpenAI Python client.

.. code-block:: bash

    # Install vLLM
    pip install vllm

    # Start server with Qwen 7B model
    # The server will automatically download the model from HuggingFace
    vllm serve Qwen/Qwen2.5-7B-Instruct --host 0.0.0.0 --port 8000

    # Test the server with a completion request
    # The API is compatible with OpenAI's completion endpoint
    curl -X POST http://localhost:8000/v1/completions \
      -H "Content-Type: application/json" \
      -d '{
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "prompt": "Hello, my name is",
        "max_tokens": 50
      }'

Common API Examples
-------------------

vLLM provides an OpenAI-compatible API with support for completions, chat completions,
and embeddings. Here are the most common usage patterns:

**Completions** - Generate text from a prompt:

.. code-block:: bash

    curl -X POST http://localhost:8000/v1/completions \
      -H "Content-Type: application/json" \
      -d '{
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "prompt": "Explain quantum computing in simple terms:",
        "max_tokens": 100,
        "temperature": 0.7
      }'

**Chat Completions** - Multi-turn conversations:

.. code-block:: bash

    curl -X POST http://localhost:8000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": [
          {"role": "system", "content": "You are a helpful assistant."},
          {"role": "user", "content": "What is the capital of France?"}
        ],
        "max_tokens": 50
      }'

**Streaming Responses** - Get tokens as they're generated:

.. code-block:: bash

    curl -X POST http://localhost:8000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": [{"role": "user", "content": "Write a short poem"}],
        "stream": true
      }'

**List Models** - Check available models:

.. code-block:: bash

    curl http://localhost:8000/v1/models

For more API examples including batch completions, sampling parameters, logprobs, stop
sequences, and other advanced features, refer to `test.sh <https://github.com/crazyguitar/pysheeet/blob/master/src/llm/vllm/test.sh>`_.

Tensor Parallel (TP)
--------------------

Tensor parallelism splits individual model layers across multiple GPUs, with each GPU
holding a portion of the weight matrices. This is essential for models that don't fit
in a single GPU's memory. All GPUs participate in every forward pass, communicating
via all-reduce operations. Best for large dense models like Llama 70B or Qwen 72B.

**Use when:** Model doesn't fit on a single GPU, or you need to reduce per-GPU memory usage.

.. code-block:: bash

    # Serve 14B model across 8 GPUs using tensor parallelism
    # Each GPU holds 1/8 of the model weights
    vllm serve Qwen/Qwen2.5-14B-Instruct --tensor-parallel-size 8

    # Formula: Each GPU holds 1/TP of model weights
    # Communication: All-reduce on every forward pass

Pipeline Parallel (PP)
----------------------

Pipeline parallelism divides the model into sequential stages, with each stage assigned
to different GPUs. Unlike tensor parallelism where all GPUs work on every layer, pipeline
parallelism processes different parts of the model on different GPUs. This reduces
communication overhead since GPUs only need to pass activations between stages rather
than synchronizing on every operation. Ideal for very deep models or when network
bandwidth is limited.

**Use when:** You want to reduce inter-GPU communication or scale across nodes with
slower interconnects.

.. code-block:: bash

    # Serve 14B model with pipeline parallelism across 8 GPUs
    # PP=2 splits model into 2 stages, TP=4 within each stage
    vllm serve Qwen/Qwen2.5-14B-Instruct \
      --tensor-parallel-size 4 \
      --pipeline-parallel-size 2

    # Formula: TOTAL_GPUS = TP × PP
    # Stage 0: Layers 0-N/2 on GPUs 0-3 (TP=4)
    # Stage 1: Layers N/2-N on GPUs 4-7 (TP=4)
    # Communication: Only between pipeline stages

Data Parallel (DP)
------------------

Data parallelism creates multiple independent replicas of the model, each processing
different requests simultaneously. This is the most effective way to increase throughput
when you have sufficient GPU memory for multiple model copies. Each replica can use
tensor parallelism internally. Perfect for high-traffic production deployments where
you need to serve many concurrent requests.

**Use when:** You need higher request throughput and have enough GPUs to replicate the model.

.. code-block:: bash

    # Create 2 model replicas, each using 8 GPUs with tensor parallelism
    # Total: 16 GPUs serving 2x the requests
    vllm serve Qwen/Qwen2.5-14B-Instruct \
      --tensor-parallel-size 8 \
      --data-parallel-size 2 \
      --data-parallel-backend ray

    # Formula: TOTAL_GPUS = DP × TP
    # Replica 0: GPUs 0-7 (serves requests independently)
    # Replica 1: GPUs 8-15 (serves requests independently)
    # Each replica can handle different requests in parallel

Expert Parallel (EP)
--------------------

Expert parallelism is specifically designed for Mixture-of-Experts (MoE) models, where
the model contains multiple expert sub-networks and a gating mechanism routes tokens to
different experts. EP shards the experts across GPUs, allowing each GPU to hold a subset
of experts. This is crucial for MoE models like Qwen2.5-MoE or DeepSeek-V3, which can
have hundreds of experts. vLLM automatically computes the expert parallelism degree based
on your DP and TP configuration.

**Use when:** Serving MoE models that have too many experts to fit on a single GPU.

.. code-block:: bash

    # Serve 14B MoE model with expert parallelism across 8 GPUs
    # Experts are automatically sharded across all available GPUs
    vllm serve Qwen/Qwen2.5-14B-MoE \
      --tensor-parallel-size 8 \
      --enable-expert-parallel \
      --all2all-backend allgather_reducescatter

    # EP is computed automatically: EP = DP × TP
    # With TP=8, EP=8, each GPU holds 1/8 of the experts
    # All-to-all communication routes tokens to the right experts

Distributed Serving on SLURM
----------------------------

For production deployments on HPC clusters, vLLM can be launched across multiple SLURM
nodes using ``run.sbatch``. This script automates the full lifecycle of multi-node
serving: distributing Docker images to all nodes, launching GPU-enabled containers with
EFA networking, starting vLLM workers with the correct parallelism configuration, and
keeping the server running until manually stopped.

Unlike single-node serving where all GPUs share memory and communicate over NVLink,
multi-node serving requires high-speed inter-node communication (EFA/RDMA) for
operations like all-reduce (TP), pipeline stage transfers (PP), and expert routing
(EP). The script handles all of this transparently — you just specify the model and
parallelism settings, and it computes DP, selects the appropriate backend (RPC, Ray,
or multiprocessing), and orchestrates the cluster.

**Basic usage** — allocate nodes, then launch:

.. code-block:: bash

    # Allocate 2 nodes with 8 GPUs each
    salloc -N 2 --gpus-per-node=8 --exclusive

    # Serve a MoE model with expert parallelism (TP=8, DP=2, EP=16)
    bash run.sbatch \
      Qwen/Qwen3-30B-A3B-FP8 \
      --tensor-parallel-size 8 \
      --enable-expert-parallel

    # Serve a dense model with pipeline parallelism (TP=8, PP=2)
    bash run.sbatch \
      deepseek-ai/DeepSeek-V2-Lite \
      --tensor-parallel-size 8 \
      --pipeline-parallel-size 2

The server prints the head node IP when ready. Query it with any OpenAI-compatible
client:

.. code-block:: bash

    curl http://<HEAD_IP>:8000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{
        "model": "Qwen/Qwen3-30B-A3B-FP8",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 50
      }'

For the full list of script flags, backend selection, Docker image options, and
additional examples, see the
`vLLM README <https://github.com/crazyguitar/pysheeet/blob/master/src/llm/vllm/README.rst>`_.
