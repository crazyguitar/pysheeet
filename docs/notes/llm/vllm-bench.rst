.. meta::
    :description lang=en: vLLM benchmark suite for measuring serving performance. Covers throughput, prefill TTFT, decode ITL, latency, concurrency, long context, ShareGPT, prefix caching, and parameter sweeps.
    :keywords: vLLM, benchmark, throughput, latency, TTFT, ITL, TPOT, prefill, decode, concurrency, ShareGPT, prefix caching, sonnet, sweep, GPU, LLM inference, serving performance

==============
vLLM Benchmark
==============

.. contents:: Table of Contents
    :backlinks: none

``bench.sh`` is a single-file benchmark suite for measuring vLLM serving performance.
It uses ``vllm bench serve`` to send requests to a running vLLM server and collects
metrics like throughput, TTFT (Time to First Token), ITL (Inter-Token Latency), and
end-to-end latency. All benchmarks use the OpenAI-compatible chat completions endpoint.

The script handles Docker image loading and container management automatically. If
``vllm`` CLI is not available on the host, it loads the Docker image (from tarball or
registry) and re-executes itself inside the container. When running under a SLURM
allocation, it uses ``srun`` to dispatch to the compute node.

Source: `bench.sh`_, `run.sbatch`_

.. _bench.sh: https://github.com/crazyguitar/pysheeet/blob/master/src/llm/vllm/bench.sh
.. _run.sbatch: https://github.com/crazyguitar/pysheeet/blob/master/src/llm/vllm/run.sbatch

Quick Start
-----------

Launch a vLLM server in one terminal, then run benchmarks from another. The benchmark
script auto-detects the model from the server and handles Docker image loading if
``vllm`` CLI is not installed on the host.

**Terminal 1** — launch the vLLM server:

.. code-block:: bash

    # Single node
    vllm serve Qwen/Qwen2.5-7B-Instruct --host 0.0.0.0 --port 8000

    # Multi-node on SLURM (allocate 2 nodes, serve MoE model with EP)
    # See run.sbatch_ for full options
    salloc -N 2 --gpus-per-node=8 --exclusive
    bash run.sbatch \
      Qwen/Qwen3-30B-A3B-FP8 \
      --tensor-parallel-size 8 \
      --enable-expert-parallel

**Terminal 2** — run benchmarks against the server:

.. code-block:: bash

    # Run all benchmarks
    bash bench.sh -H 10.0.128.193 -i vllm-serve:latest

    # Run specific benchmarks
    bash bench.sh -H 10.0.128.193 -i vllm-serve:latest --type throughput,prefill

    # From a SLURM allocation (srun dispatches to compute node)
    salloc -N1
    bash bench.sh -H 10.0.128.193 -i /fsx/vllm-serve-latest.tar.gz

Throughput
----------

Measures peak output tokens/sec by saturating the server with requests.

.. code-block:: bash

    vllm bench serve ... \
        --dataset-name random \
        --random-input-len 512 --random-output-len 256 \
        --num-prompts 100 --request-rate inf

``request-rate=inf`` sends all prompts immediately, forcing the scheduler to batch
aggressively. This reveals the server's maximum throughput under full load.
``512in/256out`` is a moderate workload that exercises both the prefill phase (processing
the input) and the decode phase (generating tokens). If input is too short, prefill is
trivial and you only measure decode. If too long, prefill dominates and you miss decode
bottlenecks.

Prefill (TTFT)
--------------

Measures Time to First Token — how fast the server processes the input prompt before
generating the first output token.

.. code-block:: bash

    # Sweeps input length: 128, 512, 2048, 4096, 16384
    vllm bench serve ... \
        --dataset-name random \
        --random-input-len $LEN --random-output-len 1 \
        --num-prompts 100 --request-rate 4

``output-len=1`` isolates prefill from decode. With only 1 output token, nearly all
compute goes to processing the input prompt. This makes TTFT the dominant metric.

Sweeping input length (128→16K) reveals how TTFT scales with context size. Prefill
compute is O(n) per layer (each token attends to all previous tokens), so TTFT should
grow roughly linearly. Deviations indicate chunked prefill kicking in, memory pressure,
or scheduler interference.

``rate=4`` keeps the server lightly loaded so TTFT reflects actual compute time, not
time spent waiting in the scheduler queue. At higher rates, queueing delay would
contaminate the measurement.

Decode (ITL)
------------

Measures Inter-Token Latency — the time between consecutive output tokens during
autoregressive generation.

.. code-block:: bash

    # Sweeps output length: 128, 256, 512, 1024
    vllm bench serve ... \
        --dataset-name random \
        --random-input-len 128 --random-output-len $LEN \
        --num-prompts 100 --request-rate 4

``input-len=128`` keeps prefill minimal so the benchmark focuses on the decode phase.
A short input means the KV cache starts small and grows as tokens are generated.

Sweeping output length (128→1024) reveals how ITL changes as the KV cache grows.
Longer sequences increase memory pressure and may trigger PagedAttention block
allocation, preemption, or swapping. If ITL degrades significantly at longer outputs,
it indicates the server is running low on KV cache memory.

``rate=4`` avoids batching interference. At high request rates, the scheduler batches
multiple requests together, which improves throughput but increases per-token latency.
Low rate ensures ITL reflects single-request decode speed.

Latency (E2E)
-------------

Measures end-to-end request latency under minimal load — the "single user" experience.

.. code-block:: bash

    # Tests short (128/128), medium (512/256), long (4096/512)
    vllm bench serve ... \
        --dataset-name random \
        --random-input-len $IN --random-output-len $OUT \
        --num-prompts 100 --request-rate 1

``rate=1`` ensures requests are mostly processed alone with no batching. This gives
the baseline best-case latency — what a single user would experience (similar to
ChatGPT-style usage where one person waits for a complete response).

Three size classes (short/medium/long) show how total latency scales with request size.
E2E latency = TTFT + (output_tokens × ITL), so this validates that prefill and decode
measurements compose correctly.

Concurrency
-----------

Finds the server's saturation point by sweeping the number of concurrent requests.

.. code-block:: bash

    # Sweeps concurrency: 1, 4, 16, 64, 256
    vllm bench serve ... \
        --dataset-name random \
        --random-input-len 512 --random-output-len 256 \
        --num-prompts 500 --request-rate inf --max-concurrency $C

``request-rate=inf`` with ``max-concurrency=N`` caps how many requests run in parallel.
This decouples arrival rate from concurrency, letting you measure the effect of batch
size directly.

At low concurrency (1–4), latency is good but throughput is low because the GPU is
underutilized. At high concurrency (64–256), throughput plateaus and latency degrades
due to queueing and memory pressure. The "knee" where throughput stops improving is the
optimal operating point for production — it tells you how many concurrent users the
server can handle before quality degrades.

500 prompts per level (vs 100 for other tests) gives enough samples for stable
percentile metrics at each concurrency level.

Long Context
------------

Tests behavior with very long input prompts (4K→32K tokens).

.. code-block:: bash

    # Sweeps input length: 4096, 16384, 32000
    vllm bench serve ... \
        --dataset-name random \
        --random-input-len $LEN --random-output-len 100 \
        --num-prompts 50 --request-rate 1

Long inputs stress KV cache memory, attention compute, and may trigger chunked prefill
(where the server splits a long prompt into smaller chunks to avoid blocking other
requests). ``output=100`` keeps decode short so the benchmark focuses on prefill scaling
at extreme context lengths.

``rate=1`` and fewer prompts (50) because each request is expensive — a 32K-token
prompt consumes significant GPU memory and compute. Higher rates risk OOM or excessive
queueing that would make results unreliable.

This benchmark is inspired by GPUStack's "very long prompt" configuration (32000in/100out)
and is critical for RAG workloads where documents are stuffed into the context window.

ShareGPT
--------

Realistic conversational workload from real user conversations with variable input/output
lengths.

.. code-block:: bash

    vllm bench serve ... \
        --dataset-name sharegpt \
        --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
        --num-prompts 100 --request-rate inf   # throughput mode
        --num-prompts 100 --request-rate 4     # realistic load

Unlike random datasets with fixed lengths, ShareGPT captures the natural distribution
of short and long prompts and responses from actual ChatGPT conversations. This makes
it the best proxy for production chat traffic — some requests are quick one-liners,
others are long multi-paragraph exchanges.

ShareGPT is the standard dataset used by vLLM CI, GPUStack perf lab, and most published
benchmarks. Running it at both ``rate=inf`` (peak throughput) and ``rate=4`` (realistic
load) shows how the server performs under stress vs. normal operation.

The dataset is auto-downloaded from HuggingFace if not present locally.

Sonnet (Prefix Caching)
-----------------------

Tests vLLM's automatic prefix caching using Shakespeare's sonnets as a shared prefix.

.. code-block:: bash

    vllm bench serve ... \
        --dataset-name sonnet \
        --sonnet-input-len 550 --sonnet-output-len 150 --sonnet-prefix-len 200 \
        --num-prompts 100 --request-rate inf   # throughput mode
        --num-prompts 100 --request-rate 4     # realistic load

All prompts share a common prefix (200 tokens of sonnet text), then each request gets
a unique suffix to reach 550 total input tokens. This exercises vLLM's automatic prefix
caching — when enabled, the shared prefix KV cache is computed once and reused across
all requests, dramatically reducing TTFT for the 2nd+ requests.

This is critical for production workloads with system prompts. If your application
prepends the same system prompt to every request (common in chatbots and agents), prefix
caching can cut TTFT significantly. Comparing sonnet results with prefix caching on vs
off quantifies the speedup.

Sweep
-----

Systematically tests multiple parameter combinations using ``vllm bench sweep serve``.
Unlike other benchmarks that hit an already-running server, sweep manages its own server
lifecycle — it starts and stops ``vllm serve`` for each parameter combination.

.. code-block:: bash

    vllm bench sweep serve \
        --serve-cmd "vllm serve $MODEL" \
        --bench-cmd "vllm bench serve --model $MODEL ..." \
        --bench-params results/bench_params.json \
        --num-runs 1 -o results/

``bench_params.json`` defines the parameter combinations to sweep. A default rate sweep
is auto-generated if the file doesn't exist:

.. code-block:: json

    [
      {"--request-rate": 1},
      {"--request-rate": 4},
      {"--request-rate": 8},
      {"--request-rate": 16},
      {"--request-rate": 32},
      {"--request-rate": "inf"}
    ]

Optionally, ``serve_params.json`` sweeps server-side parameters (e.g.,
``--max-num-seqs``). When both files are provided, sweep runs the Cartesian product of
all combinations. Results are saved as a CSV summary for easy plotting of
throughput-vs-latency curves.

Ref: `vLLM Sweep Documentation <https://docs.vllm.ai/en/latest/benchmarking/sweeps/>`_

Key Metrics
-----------

All benchmarks report these metrics via ``vllm bench serve``:

- **TTFT** (Time to First Token): Time from request arrival to first generated token.
  Dominated by prefill compute. Lower is better for interactive use.
- **TPOT** (Time Per Output Token): Average time per generated token. Reflects decode
  speed.
- **ITL** (Inter-Token Latency): Time between consecutive tokens. Similar to TPOT but
  measured per-token rather than averaged. Shows decode consistency.
- **E2E Latency**: Total time from request to completion. E2E ≈ TTFT + (tokens × ITL).
- **Throughput**: Output tokens/sec across all requests. Higher is better for batch
  workloads.

Results are saved as JSON files in the ``--result-dir`` directory (default: ``./results``).
