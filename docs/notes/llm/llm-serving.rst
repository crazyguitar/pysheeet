.. meta::
    :description lang=en: LLM serving guide — vLLM, SGLang, and TensorRT-LLM for single-node, multi-node SLURM, tensor/pipeline/data/expert parallelism on GPU clusters.
    :keywords: vLLM, SGLang, TensorRT-LLM, LLM serving, LLM inference, model serving, distributed inference, tensor parallelism, pipeline parallelism, data parallelism, expert parallelism, MoE serving, GPU inference, OpenAI compatible API, multi-node GPU, SLURM, HPC, EFA, NCCL, PagedAttention, RadixAttention, continuous batching, Docker, Qwen, Llama, DeepSeek

===========
LLM Serving
===========

.. contents:: Table of Contents
    :backlinks: none

This guide covers LLM inference serving with three high-performance engines:

- **vLLM** — High-throughput inference engine with PagedAttention for efficient KV cache
  memory management, continuous batching for maximizing GPU utilization, and optimized
  CUDA kernels. Provides an OpenAI-compatible API as a drop-in replacement for OpenAI
  services.

- **SGLang** — Fast inference engine with RadixAttention for efficient prefix caching
  across requests with shared prompts. Optimized for multi-turn conversations and
  workloads with common system prompts.

- **TensorRT-LLM** — NVIDIA's inference engine with PyTorch backend, optimized CUDA
  kernels, and FP8/INT4 quantization. Uses ``trtllm-serve`` for OpenAI-compatible
  serving. Supports TP, PP, EP, and attention DP via YAML config.

All support distributed inference across multiple GPUs and nodes with tensor parallelism
(TP), pipeline parallelism (PP), data parallelism (DP), and expert parallelism (EP) for
Mixture-of-Experts (MoE) models. This guide covers everything from basic single-GPU
deployment to advanced multi-node distributed serving on SLURM clusters.

Scripts and examples:

- vLLM: `src/llm/vllm/ <https://github.com/crazyguitar/pysheeet/tree/master/src/llm/vllm>`_
- SGLang: `src/llm/sglang/ <https://github.com/crazyguitar/pysheeet/tree/master/src/llm/sglang>`_
- TensorRT-LLM: `src/llm/tensorrt-llm/ <https://github.com/crazyguitar/pysheeet/tree/master/src/llm/tensorrt-llm>`_

Quick Start
-----------

Get started in minutes. Install the package, launch a server, and query it with standard
HTTP requests. Both engines expose OpenAI-compatible ``/v1/chat/completions`` and
``/v1/completions`` endpoints.

**vLLM** (port 8000):

.. code-block:: bash

    pip install vllm
    vllm serve Qwen/Qwen2.5-7B-Instruct --host 0.0.0.0 --port 8000

    curl -X POST http://localhost:8000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{"model": "Qwen/Qwen2.5-7B-Instruct", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 50}'

**SGLang** (port 30000):

.. code-block:: bash

    pip install "sglang[all]"
    python -m sglang.launch_server --model-path Qwen/Qwen2.5-7B-Instruct --host 0.0.0.0 --port 30000

    curl -X POST http://localhost:30000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{"model": "Qwen/Qwen2.5-7B-Instruct", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 50}'

**TensorRT-LLM** (port 8000):

.. code-block:: bash

    pip install tensorrt-llm
    trtllm-serve Qwen/Qwen2.5-7B-Instruct --host 0.0.0.0 --port 8000

    curl -X POST http://localhost:8000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{"model": "Qwen/Qwen2.5-7B-Instruct", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 50}'

Tensor Parallel (TP)
--------------------

Tensor parallelism splits individual model layers across multiple GPUs, with each GPU
holding a portion of the weight matrices. All GPUs participate in every forward pass,
communicating via all-reduce operations. Essential for models that don't fit in a single
GPU's memory.

**Use when:** Model doesn't fit on a single GPU, or you need to reduce per-GPU memory.

.. code-block:: bash

    # vLLM
    vllm serve Qwen/Qwen2.5-14B-Instruct --tensor-parallel-size 8

    # SGLang
    python -m sglang.launch_server --model-path Qwen/Qwen2.5-14B-Instruct --tp 8

    # TensorRT-LLM
    trtllm-serve Qwen/Qwen2.5-14B-Instruct --tp_size 8

Pipeline Parallel (PP)
----------------------

Pipeline parallelism divides the model into sequential stages, with each stage assigned
to different GPUs. Unlike tensor parallelism where all GPUs work on every layer, pipeline
parallelism processes different parts of the model on different GPUs. This reduces
communication overhead since GPUs only pass activations between stages.

**Use when:** You want to reduce inter-GPU communication or scale across nodes with
slower interconnects.

.. code-block:: bash

    # vLLM: PP=2 splits model into 2 stages, TP=4 within each
    vllm serve Qwen/Qwen2.5-14B-Instruct --tensor-parallel-size 4 --pipeline-parallel-size 2

    # SGLang
    python -m sglang.launch_server --model-path Qwen/Qwen2.5-14B-Instruct --tp 4 --pp 2

    # TensorRT-LLM
    trtllm-serve Qwen/Qwen2.5-14B-Instruct --tp_size 4 --pp_size 2

Data Parallel (DP)
------------------

Data parallelism creates multiple independent replicas of the model, each processing
different requests simultaneously. This is the most effective way to increase throughput
when you have sufficient GPU memory for multiple model copies. Each replica can use
tensor parallelism internally.

**Use when:** You need higher request throughput and have enough GPUs to replicate the model.

.. code-block:: bash

    # vLLM: 2 replicas, each using 8 GPUs
    vllm serve Qwen/Qwen2.5-14B-Instruct --tensor-parallel-size 8 --data-parallel-size 2

    # SGLang: multi-node DP requires --enable-dp-attention
    python -m sglang.launch_server --model-path Qwen/Qwen2.5-14B-Instruct --tp 8 --dp 2 --enable-dp-attention

    # TensorRT-LLM: DP via multi-node with TP=8 per node
    trtllm-serve Qwen/Qwen2.5-14B-Instruct --tp_size 8

Expert Parallel (EP)
--------------------

Expert parallelism is specifically designed for Mixture-of-Experts (MoE) models, where
the model contains multiple expert sub-networks and a gating mechanism routes tokens to
different experts. EP shards the experts across GPUs, allowing each GPU to hold a subset
of experts.

**vLLM:** EP is computed automatically (``EP = DP × TP``).

**SGLang:** EP is a subdivision of TP. With ``--tp 8 --ep 2``, the 8 TP GPUs split into
2 expert groups of 4 GPUs each.

**TensorRT-LLM:** EP subdivides TP (same as SGLang). With ``--tp_size 8 --ep_size 2``,
experts are sharded across 2 groups of 4 GPUs.

.. code-block:: bash

    # vLLM: EP auto-computed
    vllm serve Qwen/Qwen3-30B-A3B-FP8 --tensor-parallel-size 8 --enable-expert-parallel

    # SGLang: EP subdivides TP
    python -m sglang.launch_server --model-path Qwen/Qwen1.5-MoE-A2.7B --tp 8 --ep 2

    # TensorRT-LLM: EP subdivides TP
    trtllm-serve Qwen/Qwen1.5-MoE-A2.7B --tp_size 8 --ep_size 2

Parallelism Formulas
--------------------

Both engines use the same formula for computing total GPU requirements:

.. code-block:: text

    Total GPUs = TP × PP × DP

Expert parallelism (EP) is handled differently:

- **vLLM**: EP is auto-computed (``EP = TP × DP``) when ``--enable-expert-parallel`` is set.
  All GPUs in the world participate in expert parallelism automatically.
- **SGLang**: EP explicitly subdivides TP. For example, ``--tp 8 --ep 2`` splits the 8 TP
  GPUs into 2 expert groups of 4 GPUs each. Each group handles different experts while
  all 8 GPUs still perform tensor parallelism for non-expert layers.
- **TensorRT-LLM**: EP subdivides TP (same as SGLang). ``--tp_size 8 --ep_size 2`` splits
  experts across 2 groups. Constraint: ``moe_tp × ep = tp_size``.

.. _distributed-serving-on-slurm:

Distributed Serving on SLURM
----------------------------

Some large models (e.g., DeepSeek-V3, Llama-3.1-405B) may not fit into a single node.
All three engines support serving across multiple nodes with different parallelism
strategies (TP, PP, EP, DP). Multi-node deployment can be tricky at the beginning —
the ``run.sbatch`` examples below show how to use ``salloc`` with each engine to get
started quickly on Slurm. The scripts handle Docker image distribution to all nodes,
container launch with EFA/GPU passthrough, worker coordination, and health checking.
The server runs until you stop it with ``Ctrl+C`` or ``scancel``.

**vLLM:**

.. code-block:: bash

    salloc -N 2 --gpus-per-node=8 --exclusive

    # MoE with expert parallelism
    bash run.sbatch Qwen/Qwen3-30B-A3B-FP8 --tensor-parallel-size 8 --enable-expert-parallel

    # Dense with pipeline parallelism
    bash run.sbatch deepseek-ai/DeepSeek-V2-Lite --tensor-parallel-size 8 --pipeline-parallel-size 2

**SGLang:**

.. code-block:: bash

    salloc -N 2 --gpus-per-node=8 --exclusive

    # Large model with TP=8 (uses 8 GPUs on first node)
    bash run.sbatch --model-path Qwen/Qwen2.5-72B-Instruct --tp 8

    # MoE with expert parallelism (TP=8, EP=2 across 2 nodes)
    bash run.sbatch --model-path Qwen/Qwen1.5-MoE-A2.7B --tp 8 --ep 2

**TensorRT-LLM:**

.. code-block:: bash

    salloc -N 2 --gpus-per-node=8 --exclusive

    # MoE with expert parallelism
    bash run.sbatch /path/to/Qwen1.5-MoE-A2.7B --tp_size 8 --ep_size 2

    # Dense model
    bash run.sbatch /path/to/Qwen2.5-14B-Instruct --tp_size 8

See the READMEs for full script options:

- `vLLM README <https://github.com/crazyguitar/pysheeet/blob/master/src/llm/vllm/README.rst>`_
- `SGLang README <https://github.com/crazyguitar/pysheeet/blob/master/src/llm/sglang/README.rst>`_
- `TensorRT-LLM README <https://github.com/crazyguitar/pysheeet/blob/master/src/llm/tensorrt-llm/README.rst>`_
