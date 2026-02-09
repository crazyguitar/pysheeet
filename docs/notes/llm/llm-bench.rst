.. meta::
    :description lang=en: LLM benchmark suite — measure throughput, TTFT, ITL, latency for vLLM, SGLang, and TensorRT-LLM serving performance.
    :keywords: LLM benchmark, vLLM benchmark, SGLang benchmark, TensorRT-LLM benchmark, serving benchmark, throughput, latency, TTFT, time to first token, ITL, inter-token latency, prefill, decode, concurrency, ShareGPT, GPU benchmark, tokens per second

=============
LLM Benchmark
=============

.. contents:: Table of Contents
    :backlinks: none

Benchmark suites for measuring LLM serving performance with vLLM, SGLang, and
TensorRT-LLM. All use similar methodology — same test categories, workloads, and
metrics — for easy comparison between the three inference engines.

- **vLLM:** ``vllm bench serve`` via `bench.sh <https://github.com/crazyguitar/pysheeet/blob/master/src/llm/vllm/bench.sh>`_
- **SGLang:** ``python -m sglang.bench_serving`` via `bench.sh <https://github.com/crazyguitar/pysheeet/blob/master/src/llm/sglang/bench.sh>`_
- **TensorRT-LLM:** ``python -m tensorrt_llm.serve.scripts.benchmark_serving`` via `bench.sh <https://github.com/crazyguitar/pysheeet/blob/master/src/llm/tensorrt-llm/bench.sh>`_

The scripts handle Docker image loading and container management automatically. If the
CLI is not available on the host, they load the Docker image and re-execute inside the
container. When running under a SLURM allocation, they use ``srun`` to dispatch to the
compute node.

Quick Start
-----------

Launch a server in one terminal, then run benchmarks from another. The benchmark script
auto-detects the model from the server.

**vLLM:**

.. code-block:: bash

    # Terminal 1: start server
    vllm serve Qwen/Qwen2.5-7B-Instruct --host 0.0.0.0 --port 8000

    # Terminal 2: run benchmarks
    bash bench.sh -H localhost -i vllm-serve:latest
    bash bench.sh -H localhost --type throughput,prefill

**SGLang:**

.. code-block:: bash

    # Terminal 1: start server
    python -m sglang.launch_server --model-path Qwen/Qwen2.5-7B-Instruct --host 0.0.0.0 --port 30000

    # Terminal 2: run benchmarks
    bash bench.sh -H localhost -i sglang-serve:latest
    bash bench.sh -H localhost --type throughput,prefill

**TensorRT-LLM:**

.. code-block:: bash

    # Terminal 1: start server
    trtllm-serve /path/to/Qwen2.5-7B-Instruct --host 0.0.0.0 --port 8000

    # Terminal 2: run benchmarks (requires -m for tokenizer loading)
    bash bench.sh -H localhost -m /path/to/Qwen2.5-7B-Instruct -i tensorrt-llm-serve:latest
    bash bench.sh -H localhost -m /path/to/Qwen2.5-7B-Instruct --type throughput,prefill

Throughput
----------

Measures peak output tokens/sec by saturating the server with requests. Uses
``request-rate=inf`` to send all prompts immediately, forcing the scheduler to batch
aggressively. This reveals the server's maximum throughput under full load.

``512in/256out`` is a moderate workload that exercises both the prefill phase (processing
the input) and the decode phase (generating tokens).

.. code-block:: bash

    --dataset-name random --random-input-len 512 --random-output-len 256 \
    --num-prompts 100 --request-rate inf

Prefill (TTFT)
--------------

Measures Time to First Token — how fast the server processes the input prompt before
generating the first output token. ``output-len=1`` isolates prefill from decode since
nearly all compute goes to processing the input.

Sweeping input length (128→16K) reveals how TTFT scales with context size. Prefill
compute is O(n) per layer, so TTFT should grow roughly linearly. ``rate=4`` keeps the
server lightly loaded so TTFT reflects compute time, not queueing delay.

.. code-block:: bash

    # Sweeps input length: 128, 512, 2048, 4096, 16384
    --dataset-name random --random-input-len $LEN --random-output-len 1 \
    --num-prompts 100 --request-rate 4

Decode (ITL)
------------

Measures Inter-Token Latency — the time between consecutive output tokens during
autoregressive generation. ``input-len=128`` keeps prefill minimal so the benchmark
focuses on the decode phase.

Sweeping output length (128→1024) reveals how ITL changes as the KV cache grows. Longer
sequences increase memory pressure and may trigger PagedAttention block allocation or
preemption. ``rate=4`` avoids batching interference so ITL reflects single-request
decode speed.

.. code-block:: bash

    # Sweeps output length: 128, 256, 512, 1024
    --dataset-name random --random-input-len 128 --random-output-len $LEN \
    --num-prompts 100 --request-rate 4

Latency (E2E)
-------------

Measures end-to-end request latency under minimal load — the "single user" experience.
``rate=1`` ensures requests are mostly processed alone with no batching, giving the
baseline best-case latency (similar to ChatGPT-style usage where one user waits for a
complete response).

Three size classes (short/medium/long) show how total latency scales with request size.
E2E latency = TTFT + (output_tokens × ITL).

.. code-block:: bash

    # Tests short (128/128), medium (512/256), long (4096/512)
    --dataset-name random --random-input-len $IN --random-output-len $OUT \
    --num-prompts 100 --request-rate 1

Concurrency
-----------

Finds the server's saturation point by sweeping the number of concurrent requests.
``request-rate=inf`` with ``max-concurrency=N`` caps how many requests run in parallel,
decoupling arrival rate from concurrency.

At low concurrency (1–4), latency is good but throughput is low (GPU underutilized).
At high concurrency (64–256), throughput plateaus and latency degrades (queueing). The
"knee" where throughput stops improving is the optimal operating point — it tells you
how many concurrent users the server can handle before quality degrades.

.. code-block:: bash

    # Sweeps concurrency: 1, 4, 16, 64, 256
    --dataset-name random --random-input-len 512 --random-output-len 256 \
    --num-prompts 100 --request-rate inf --max-concurrency $C

ShareGPT
--------

Realistic conversational workload from real user conversations with variable input/output
lengths. Unlike random datasets with fixed lengths, ShareGPT captures the natural
distribution of short and long prompts from actual ChatGPT conversations, making it the
best proxy for production chat traffic.

ShareGPT is the standard dataset used by vLLM CI, GPUStack perf lab, and most published
benchmarks. The dataset is auto-downloaded from HuggingFace if not present locally.

.. code-block:: bash

    --dataset-name sharegpt --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 100 --request-rate inf   # throughput mode
    --num-prompts 100 --request-rate 4     # realistic load

Key Metrics
-----------

- **TTFT** (Time to First Token): Time from request arrival to first generated token.
  Dominated by prefill compute. Lower is better for interactive use.
- **ITL** (Inter-Token Latency): Time between consecutive tokens. Reflects decode speed
  and consistency.
- **TPOT** (Time Per Output Token): Average time per generated token. Similar to ITL
  but averaged across all tokens.
- **E2E Latency**: Total time from request to completion. E2E ≈ TTFT + (tokens × ITL).
- **Throughput**: Output tokens/sec across all requests. Higher is better for batch
  workloads.

CLI Differences
---------------

.. list-table::
   :widths: 20 27 27 26
   :header-rows: 1

   * - Parameter
     - vLLM
     - SGLang
     - TensorRT-LLM
   * - Input length
     - ``--random-input-len``
     - ``--random-input``
     - ``--random-input-len``
   * - Output length
     - ``--random-output-len``
     - ``--random-output``
     - ``--random-output-len``
   * - Max rate
     - ``--request-rate inf``
     - ``--request-rate inf``
     - ``--max-concurrency``
   * - Random dataset
     - (works by default)
     - (works by default)
     - ``--random-ids --random-prefix-len 0``
   * - Model flag
     - auto-detected
     - auto-detected
     - ``-m`` required (tokenizer)
   * - Results
     - ``--result-dir ./results``
     - ``--output-file ./results/out.json``
     - ``--result-dir ./results``
