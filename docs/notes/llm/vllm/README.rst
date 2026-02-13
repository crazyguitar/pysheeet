.. meta::
    :description lang=en: vLLM offline benchmarking guide — measure raw inference performance without API server overhead using offline_bench.sh
    :keywords: vLLM offline benchmark, vLLM throughput, offline inference, torchrun, tensor parallelism, data parallelism, expert parallelism, nsys profiling, PyTorch profiler, multi-node inference

=======================
vLLM Offline Benchmark
=======================

.. contents:: Table of Contents
    :backlinks: none

Offline benchmarking measures raw inference performance without API server overhead.
Uses `offline_bench.sh <https://github.com/crazyguitar/pysheeet/blob/master/src/llm/vllm/offline_bench.sh>`_
which wraps the Python script with Docker and multi-node coordination via ``torchrun``.

Quick Start
-----------

**Single GPU:**

.. code-block:: bash

    bash offline_bench.sh \
      --model meta-llama/Llama-3.1-8B \
      --input-len 512 --output-len 128 \
      --num-prompts 100

**Multi-GPU with tensor parallelism:**

.. code-block:: bash

    salloc -N 1 bash offline_bench.sh \
      --model Qwen/Qwen2-57B-A14B \
      --tensor-parallel-size 4 --enable-expert-parallel \
      --input-len 1024 --output-len 256 \
      --num-prompts 100

**Multi-node with custom image:**

.. code-block:: bash

    salloc -N 4 bash offline_bench.sh \
      --image "$PWD/vllm-latest.tar.gz" \
      --model Qwen/Qwen2-57B-A14B \
      --all2all-backend allgather_reducescatter \
      --tensor-parallel-size 4 --enable-expert-parallel \
      --gpu-memory-utilization 0.8 \
      --input-len 2048 --output-len 512 \
      --num-prompts 50

Script Options
--------------

- ``--nproc N`` — Number of processes for torchrun (auto-detect if not set)
- ``--image IMAGE`` — Docker image or tarball (default: ``./vllm-serve-latest.tar.gz``)
- ``--nsys`` — Enable Nsight Systems profiling
- ``-h, --help`` — Show help

Model/Benchmark Arguments
-------------------------

All other arguments are passed to ``offline_bench.py``:

- ``--model MODEL`` — Model name or path (required)
- ``--num-prompts N`` — Number of prompts (default: 50)
- ``--tensor-parallel-size, --tp-size N`` — Tensor parallel size (default: 1)
- ``--pipeline-parallel-size, --pp-size N`` — Pipeline parallel size (default: 1)
- ``--data-parallel-size, --dp-size N`` — Data parallel size (default: 1)
- ``--enable-expert-parallel, --enable-ep`` — Enable expert parallel
- ``--all2all-backend TYPE`` — All-to-all backend (``allgather_reducescatter``, ``nccl``)
- ``--enforce-eager`` — Disable CUDA graph
- ``--max-tokens N`` — Max output tokens (default: 128)
- ``--dataset-path PATH`` — Path to ShareGPT dataset
- ``--profile`` — Enable PyTorch profiler
- ``--profile-result-dir DIR`` — Profile output directory (default: ``./profile-results``)

Data Parallelism
----------------

The script automatically detects data parallelism based on available GPUs and tensor
parallelism settings. For example:

- 32 GPUs with ``--tp-size 4`` → auto-sets ``--dp-size 8`` (32 / 4 = 8 replicas)
- 8 GPUs with ``--tp-size 2`` → auto-sets ``--dp-size 4`` (8 / 2 = 4 replicas)

You can override this by explicitly passing ``--dp-size N``.

ShareGPT Dataset
----------------

Use real conversational workloads from ShareGPT:

.. code-block:: bash

    wget -O ShareGPT_V3_unfiltered_cleaned_split.json \
      https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

    bash offline_bench.sh \
      --model meta-llama/Llama-3.1-8B \
      --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
      --num-prompts 100

Profiling
---------

**Nsight Systems profiling:**

.. code-block:: bash

    salloc -N 4 bash offline_bench.sh --nsys \
      --model Qwen/Qwen2-57B-A14B \
      --tensor-parallel-size 4 --enable-expert-parallel \
      --all2all-backend allgather_reducescatter \
      --input-len 2048 --output-len 512 \
      --num-prompts 50
    # Profile files: nsys-offline/profile-node*.nsys-rep

The ``--nsys`` flag:

1. Wraps the command with ``nsys profile --capture-range=cudaProfilerApi``
2. Sets ``VLLM_NSYS_PROFILING=1`` environment variable in the container
3. Python script calls ``torch.cuda.profiler.start()`` after warmup
4. Runs the benchmark
5. Calls ``torch.cuda.profiler.stop()`` to save the profile

Open ``.nsys-rep`` files with `Nsight Systems <https://developer.nvidia.com/nsight-systems>`_.

**PyTorch profiler:**

.. code-block:: bash

    salloc -N 2 bash offline_bench.sh \
      --model Qwen/Qwen2-57B-A14B \
      --tensor-parallel-size 4 --enable-expert-parallel \
      --profile \
      --profile-result-dir ./offline-profile-results \
      --num-prompts 50
    # Profile files: offline-profile-results/

View traces at https://ui.perfetto.dev/ (supports ``.gz`` files directly).

DeepSeek-V3 Example
-------------------

Large MoE models with eager execution:

.. code-block:: bash

    salloc -N 4 bash offline_bench.sh \
      --model deepseek-ai/DeepSeek-V3-0324 \
      --tensor-parallel-size 8 --enable-expert-parallel \
      --all2all-backend allgather_reducescatter \
      --enforce-eager \
      --gpu-memory-utilization 0.9 \
      --input-len 1024 \
      --output-len 256 \
      --num-prompts 50

Note: CUDA Graph is disabled (``--enforce-eager``) due to excessive memory consumption
with large MoE models.

Multi-Node Coordination
------------------------

When running under SLURM (``salloc -N <nodes>``), the script automatically:

1. Detects the number of nodes from ``$SLURM_JOB_NUM_NODES``
2. Gets the head node IP from ``$SLURM_JOB_NODELIST``
3. Sets up ``torchrun`` rendezvous endpoint
4. Uses ``srun`` to dispatch containers to all nodes
5. Coordinates distributed training via ``torchrun``

The script handles both single-node and multi-node execution transparently.
