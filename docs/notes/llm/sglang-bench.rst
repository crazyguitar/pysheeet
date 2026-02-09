.. meta::
    :description lang=en: SGLang benchmark suite — measure throughput, TTFT, ITL, latency, concurrency, and ShareGPT for LLM serving performance with RadixAttention.
    :keywords: SGLang benchmark, SGLang performance, LLM benchmark, serving benchmark, throughput benchmark, latency benchmark, TTFT, time to first token, ITL, inter-token latency, prefill, decode, concurrency, ShareGPT, GPU benchmark, sglang bench_serving, tokens per second, request rate, LLM inference performance, RadixAttention

================
SGLang Benchmark
================

.. contents:: Table of Contents
    :backlinks: none

``bench.sh`` is a benchmark suite for measuring SGLang serving performance. It uses
``python -m sglang.bench_serving`` to send requests to a running SGLang server and
collects metrics like throughput, TTFT (Time to First Token), ITL (Inter-Token Latency),
and end-to-end latency.

The benchmark methodology mirrors the `vLLM Benchmark <vllm-bench.rst>`_ suite, using
the same test categories and metrics for easy comparison between the two inference
engines. See the vLLM benchmark documentation for detailed explanations of each test's
rationale and interpretation.

Source: `bench.sh`_, `run.sbatch`_

.. _bench.sh: https://github.com/crazyguitar/pysheeet/blob/master/src/llm/sglang/bench.sh
.. _run.sbatch: https://github.com/crazyguitar/pysheeet/blob/master/src/llm/sglang/run.sbatch

Quick Start
-----------

Launch an SGLang server in one terminal, then run benchmarks from another.

**Terminal 1** — launch the SGLang server:

.. code-block:: bash

    # Single node
    python -m sglang.launch_server \
      --model-path Qwen/Qwen2.5-7B-Instruct \
      --host 0.0.0.0 --port 30000

    # Multi-node on SLURM
    salloc -N 2 --gpus-per-node=8 --exclusive
    bash run.sbatch \
      --model-path Qwen/Qwen1.5-MoE-A2.7B \
      --tp 8 --ep 2

**Terminal 2** — run benchmarks against the server:

.. code-block:: bash

    # Run all benchmarks
    bash bench.sh -H 10.0.128.193 -i sglang-serve:latest

    # Run specific benchmarks
    bash bench.sh -H 10.0.128.193 -i sglang-serve:latest --type throughput,prefill

    # From a SLURM allocation
    salloc -N1
    bash bench.sh -H 10.0.128.193 -i /fsx/sglang-serve-latest.tar.gz

Throughput
----------

Measures peak output tokens/sec by saturating the server with requests.

.. code-block:: bash

    python -m sglang.bench_serving ... \
        --dataset-name random \
        --random-input 512 --random-output 256 \
        --num-prompts 100 --request-rate inf

``request-rate=inf`` sends all prompts immediately, forcing the scheduler to batch
aggressively. This reveals the server's maximum throughput under full load.

Prefill (TTFT)
--------------

Measures Time to First Token — how fast the server processes the input prompt.

.. code-block:: bash

    # Sweeps input length: 128, 512, 2048, 4096, 16384
    python -m sglang.bench_serving ... \
        --dataset-name random \
        --random-input $LEN --random-output 1 \
        --num-prompts 100 --request-rate 4

``output=1`` isolates prefill from decode. Sweeping input length reveals how TTFT
scales with context size. ``rate=4`` keeps the server lightly loaded so TTFT reflects
compute time, not queueing delay.

Decode (ITL)
------------

Measures Inter-Token Latency — the time between consecutive output tokens.

.. code-block:: bash

    # Sweeps output length: 128, 256, 512, 1024
    python -m sglang.bench_serving ... \
        --dataset-name random \
        --random-input 128 --random-output $LEN \
        --num-prompts 100 --request-rate 4

``input=128`` keeps prefill minimal. Sweeping output length reveals how ITL changes
as the KV cache grows. Degradation at longer outputs indicates memory pressure.

Latency (E2E)
-------------

Measures end-to-end request latency under minimal load.

.. code-block:: bash

    # Tests short (128/128), medium (512/256), long (4096/512)
    python -m sglang.bench_serving ... \
        --dataset-name random \
        --random-input $IN --random-output $OUT \
        --num-prompts 100 --request-rate 1

``rate=1`` ensures requests are processed alone with no batching, giving baseline
best-case latency.

Concurrency
-----------

Finds the server's saturation point by sweeping concurrent requests.

.. code-block:: bash

    # Sweeps concurrency: 1, 4, 16, 64, 256
    python -m sglang.bench_serving ... \
        --dataset-name random \
        --random-input 512 --random-output 256 \
        --num-prompts 100 --request-rate inf --max-concurrency $C

The "knee" where throughput stops improving is the optimal operating point.

ShareGPT
--------

Realistic conversational workload from real user conversations.

.. code-block:: bash

    python -m sglang.bench_serving ... \
        --dataset-name sharegpt \
        --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
        --num-prompts 100 --request-rate inf   # throughput mode
        --num-prompts 100 --request-rate 4     # realistic load

ShareGPT captures the natural distribution of short and long prompts from actual
conversations, making it the best proxy for production chat traffic. The dataset
is auto-downloaded if not present locally.
