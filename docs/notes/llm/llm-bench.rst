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

- **vLLM:** ``vllm bench serve`` via `bench.sh <https://github.com/crazyguitar/pysheeet/blob/master/src/llm/vllm/bench.sh>`__
- **SGLang:** ``python -m sglang.bench_serving`` via `bench.sh <https://github.com/crazyguitar/pysheeet/blob/master/src/llm/sglang/bench.sh>`__
- **TensorRT-LLM:** ``python -m tensorrt_llm.serve.scripts.benchmark_serving`` via `bench.sh <https://github.com/crazyguitar/pysheeet/blob/master/src/llm/tensorrt-llm/bench.sh>`__

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

Multi-Node with Slurm
---------------------

For benchmarking larger models (e.g., DeepSeek-V3, Llama-3.1-405B) that cannot fit on a
single node, refer to `Distributed Serving on SLURM <https://www.pythonsheets.com/notes/llm/llm-serving.html#distributed-serving-on-slurm>`__
for how to deploy multi-node serving with different parallelism strategies. Once the
server is running, benchmark using ``bench.sh`` as shown in the Quick Start above.

Throughput
----------

Measures peak output tokens/sec by saturating the server with requests. Uses
``request-rate=inf`` to send all prompts immediately, forcing the scheduler to batch
aggressively. This reveals the server's maximum throughput under full load.

``512in/256out`` is a moderate workload that exercises both the prefill phase (processing
the input) and the decode phase (generating tokens).

.. code-block:: bash

    # vLLM
    vllm bench serve --dataset-name random --random-input-len 512 --random-output-len 256 \
        --num-prompts 100 --request-rate inf

    # SGLang
    python -m sglang.bench_serving --dataset-name random --random-input 512 --random-output 256 \
        --num-prompts 100 --request-rate inf

    # TensorRT-LLM
    python -m tensorrt_llm.serve.scripts.benchmark_serving \
        --dataset-name random --random-input-len 512 --random-output-len 256 \
        --num-prompts 100 --max-concurrency 100

Prefill (TTFT)
--------------

Measures Time to First Token — how fast the server processes the input prompt before
generating the first output token. ``output-len=1`` isolates prefill from decode since
nearly all compute goes to processing the input.

Sweeping input length (128→16K) reveals how TTFT scales with context size. Prefill
compute is O(n) per layer, so TTFT should grow roughly linearly. ``rate=4`` keeps the
server lightly loaded so TTFT reflects compute time, not queueing delay.

.. code-block:: bash

    # vLLM - sweeps input length: 128, 512, 2048, 4096, 16384
    vllm bench serve --dataset-name random --random-input-len $LEN --random-output-len 1 \
        --num-prompts 100 --request-rate 4

    # SGLang
    python -m sglang.bench_serving --dataset-name random --random-input $LEN --random-output 1 \
        --num-prompts 100 --request-rate 4

    # TensorRT-LLM
    python -m tensorrt_llm.serve.scripts.benchmark_serving \
        --dataset-name random --random-input-len $LEN --random-output-len 1 \
        --num-prompts 100 --max-concurrency 4

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

    # vLLM - sweeps output length: 128, 256, 512, 1024
    vllm bench serve --dataset-name random --random-input-len 128 --random-output-len $LEN \
        --num-prompts 100 --request-rate 4

    # SGLang
    python -m sglang.bench_serving --dataset-name random --random-input 128 --random-output $LEN \
        --num-prompts 100 --request-rate 4

    # TensorRT-LLM
    python -m tensorrt_llm.serve.scripts.benchmark_serving \
        --dataset-name random --random-input-len 128 --random-output-len $LEN \
        --num-prompts 100 --max-concurrency 4

Latency (E2E)
-------------

Measures end-to-end request latency under minimal load — the "single user" experience.
``rate=1`` ensures requests are mostly processed alone with no batching, giving the
baseline best-case latency (similar to ChatGPT-style usage where one user waits for a
complete response).

Three size classes (short/medium/long) show how total latency scales with request size.
E2E latency = TTFT + (output_tokens × ITL).

.. code-block:: bash

    # vLLM - tests short (128/128), medium (512/256), long (4096/512)
    vllm bench serve --dataset-name random --random-input-len $IN --random-output-len $OUT \
        --num-prompts 100 --request-rate 1

    # SGLang
    python -m sglang.bench_serving --dataset-name random --random-input $IN --random-output $OUT \
        --num-prompts 100 --request-rate 1

    # TensorRT-LLM
    python -m tensorrt_llm.serve.scripts.benchmark_serving \
        --dataset-name random --random-input-len $IN --random-output-len $OUT \
        --num-prompts 100 --max-concurrency 1

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

    # vLLM - sweeps concurrency: 1, 4, 16, 64, 256
    vllm bench serve --dataset-name random --random-input-len 512 --random-output-len 256 \
        --num-prompts 100 --request-rate inf --max-concurrency $C

    # SGLang
    python -m sglang.bench_serving --dataset-name random --random-input 512 --random-output 256 \
        --num-prompts 100 --request-rate inf --max-concurrency $C

    # TensorRT-LLM
    python -m tensorrt_llm.serve.scripts.benchmark_serving \
        --dataset-name random --random-input-len 512 --random-output-len 256 \
        --num-prompts 100 --max-concurrency $C

ShareGPT
--------

Realistic conversational workload from real user conversations with variable input/output
lengths. Unlike random datasets with fixed lengths, ShareGPT captures the natural
distribution of short and long prompts from actual ChatGPT conversations, making it the
best proxy for production chat traffic.

ShareGPT is the standard dataset used by vLLM CI, GPUStack perf lab, and most published
benchmarks. The dataset is auto-downloaded from HuggingFace if not present locally.

.. code-block:: bash

    # vLLM - throughput mode
    vllm bench serve --dataset-name sharegpt --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
        --num-prompts 100 --request-rate inf

    # vLLM - realistic load
    vllm bench serve --dataset-name sharegpt --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
        --num-prompts 100 --request-rate 4

    # SGLang - throughput mode
    python -m sglang.bench_serving --dataset-name sharegpt \
        --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
        --num-prompts 100 --request-rate inf

    # TensorRT-LLM - throughput mode
    python -m tensorrt_llm.serve.scripts.benchmark_serving \
        --dataset-name sharegpt --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
        --num-prompts 100 --max-concurrency 100

Sonnet (Prefix Caching)
-----------------------

The sonnet dataset uses Shakespeare's sonnets with a shared prefix across all prompts.
This tests prefix caching — if enabled, the shared prefix KV cache is computed once and
reused across requests, reducing TTFT.

.. code-block:: bash

    # Download sonnet dataset
    wget -q https://raw.githubusercontent.com/vllm-project/vllm/main/benchmarks/sonnet.txt

    # vLLM - prefill-heavy (short output isolates prefill)
    vllm bench serve --dataset-name sonnet --dataset-path sonnet.txt \
        --sonnet-input-len 550 --sonnet-output-len 150 --sonnet-prefix-len 200 \
        --num-prompts 100 --request-rate inf

    # vLLM - realistic load
    vllm bench serve --dataset-name sonnet --dataset-path sonnet.txt \
        --sonnet-input-len 550 --sonnet-output-len 150 --sonnet-prefix-len 200 \
        --num-prompts 100 --request-rate 4

Both ShareGPT and sonnet are used by the vLLM team's
`v0.6.0 performance blog <https://blog.vllm.ai/2024/09/05/perf-update.html>`__ to
benchmark serving engines. To learn more about the methodology, see the
`reproduction steps <https://github.com/vllm-project/vllm/issues/8176>`__ and SGLang's
`counter-benchmark <https://github.com/sgl-project/sglang/blob/main/benchmark/benchmark_vllm_060/README.md>`__,
which uses ``sglang.bench_serving`` to compare both engines:

.. code-block:: bash

    # Launch servers
    # vLLM with multi-step scheduling
    python -m vllm.entrypoints.openai.api_server \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --disable-log-requests --num-scheduler-steps 10 --max_model_len 4096

    # SGLang
    python -m sglang.launch_server \
        --model-path meta-llama/Llama-3.1-8B-Instruct \
        --enable-torch-compile --disable-radix-cache

    # Online benchmark (realistic load)
    python3 -m sglang.bench_serving --backend sglang --dataset-name sharegpt \
        --num-prompts 1200 --request-rate 4
    python3 -m sglang.bench_serving --backend vllm --dataset-name sharegpt \
        --num-prompts 1200 --request-rate 4

    # Offline benchmark (max throughput)
    python3 -m sglang.bench_serving --backend sglang --dataset-name sharegpt \
        --num-prompts 5000
    python3 -m sglang.bench_serving --backend vllm --dataset-name sharegpt \
        --num-prompts 5000

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

Profiling
---------

Benchmark runs can be combined with profiling to correlate performance metrics with
GPU-level traces. Two profiling approaches are available for vLLM:

**PyTorch profiler** — vLLM's built-in profiler triggered via REST endpoints. Start the
server with ``--profile`` (or ``--profiler-config``), then pass ``--profile`` to the
benchmark client to call ``/start_profile`` and ``/stop_profile`` around the workload.

.. code-block:: bash

    # Server — start with profiling enabled
    bash run.sbatch --profile \
      Qwen/Qwen3-30B-A3B-FP8 \
      --tensor-parallel-size 8

    # Client — benchmark with profiling
    bash bench.sh -H <server-host> --type throughput --profile

View traces at https://ui.perfetto.dev/ (supports ``.gz`` files directly).

**Nsight Systems** — wraps ``vllm serve`` with ``nsys profile`` for CUDA kernel,
NVTX, and memory tracing. Combine with ``--profiler-config '{"profiler": "cuda"}'``
to also capture vLLM's internal CUDA profiler markers.

.. code-block:: bash

    # Server — enable nsys + CUDA profiler (terminal 0)
    bash run.sbatch --nsys \
      Qwen/Qwen3-30B-A3B-FP8 \
      --tensor-parallel-size 8 \
      --enable-expert-parallel \
      --profiler-config '{"profiler": "cuda"}'

    # Client — benchmark with profiling (terminal 1)
    bash bench.sh -H <server-host> --type throughput --profile

    # Stop server with Ctrl+C (terminal 0)
    # Nsys finalizes profiles (~30s)
    # Profile files: nsys-vllm/profile-node*.nsys-rep

Open ``.nsys-rep`` files with `Nsight Systems <https://developer.nvidia.com/nsight-systems>`_.

See the `vLLM Serving Guide <https://github.com/crazyguitar/pysheeet/blob/master/src/llm/vllm/README.rst>`_
for full ``run.sbatch`` flag reference and the
`vLLM Profiling Guide <https://docs.vllm.ai/en/latest/contributing/profiling/>`_
for more details.

Offline Benchmarking
--------------------

vLLM also supports offline benchmarking to measure raw inference performance without
API server overhead. This is useful for:

- Measuring peak throughput without network/serialization overhead
- Multi-node distributed inference testing
- Profiling with Nsight Systems or PyTorch profiler
- Testing with custom datasets (ShareGPT, random prompts)

For complete offline benchmarking documentation, see the
`vLLM Offline Benchmark Guide <https://github.com/crazyguitar/pysheeet/blob/master/src/llm/vllm/README.rst#offline-benchmarking>`_.
