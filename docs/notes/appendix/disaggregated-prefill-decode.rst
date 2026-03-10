Disaggregated Prefill/Decode with NIXL on AWS
==============================================

:Date: 2026-03-10

Abstract
--------

Disaggregated prefill/decode is an emerging serving architecture that separates
the compute-intensive prefill phase from the memory-bound decode phase onto
dedicated node groups, enabling independent scaling and improved resource
utilization. This article evaluates disaggregated prefill/decode using vLLM with
NIXL over the AWS Elastic Fabric Adapter (EFA) on a 4-node cluster. We compare
data parallelism as a baseline against disaggregated configurations and examine
the trade-offs in throughput, time-to-first-token (TTFT), and inter-token
latency (ITL) across varying input and output sequence lengths.

Introduction
------------

In standard LLM serving, each node handles both prefill and decode for incoming
requests. The prefill phase is compute-bound and processes the entire input
prompt in parallel, while the decode phase is memory-bandwidth-bound and
generates tokens autoregressively. When both phases share the same GPU pool,
long prefill requests can block decode iterations, increasing inter-token
latency for concurrent requests.

Disaggregated prefill/decode addresses this interference by assigning prefill
and decode to separate node groups. After a prefill node completes prompt
processing, the KV cache is transferred to a decode node via a high-bandwidth
interconnect. NIXL [1]_ (NVIDIA Inference Xfer Library) provides the KV cache
transfer mechanism, and on AWS, this transfer occurs over EFA using the
``LIBFABRIC`` backend.

This experiment uses vLLM [2]_ with the
``NixlConnector`` to orchestrate disaggregated serving, and ``vllm-router`` as
a reverse proxy to load-balance requests across node groups. The experiment
code is available under ``src/nixl`` in the companion repository.

Container Image
---------------

The experiment uses a custom Docker image that bundles all required components.
The ``Dockerfile`` builds on ``nvidia/cuda:12.8.1-devel-ubuntu24.04`` and
installs the following stack:

- **GDRCopy** v2.5.1 for GPU-direct memory registration
- **EFA installer** v1.47.0 for AWS Elastic Fabric Adapter support
- **UCX** v1.20.0 built with verbs, rdmacm, and EFA transport
- **NIXL** v0.10.1 with ``LIBFABRIC`` backend for KV cache transfer
- **nixlbench** for standalone NIXL bandwidth/latency microbenchmarks
- **PyTorch** 2.9.1, **flash-attn** 2.8.1, and **DeepGEMM** v2.1.1.post3
- **vLLM** 0.15.1 with ``NixlConnector`` support
- **vllm-router** for load-balancing across disaggregated node groups

The image is built and saved as a portable tarball via the ``Makefile``:

.. code-block:: bash

    make docker && make save

This produces ``nixl-latest.tar.gz``, which is distributed to all Slurm nodes
at launch time via ``pigz`` decompression and ``docker load``.

Serving Script
--------------

The ``vllm.sbatch`` script orchestrates multi-node vLLM serving on Slurm. It
accepts two key flags that control the serving topology:

- ``--route R``: splits the allocated nodes into ``R`` identical groups, each
  running an independent vLLM instance. A ``vllm-router`` process on the head
  node round-robins requests across groups.
- ``--prefill P``: within each group, assigns ``P`` nodes as prefill-only
  (``kv_producer``) and the remaining nodes as decode-only (``kv_consumer``).
  KV cache transfer between prefill and decode nodes uses ``NixlConnector``
  with the ``LIBFABRIC`` backend over EFA.

When ``--prefill 0`` (default), all nodes in a group run standard data-parallel
serving. The script computes ``DP = nodes_per_group * (8 / TP)`` and launches
vLLM with ``--data-parallel-size`` accordingly.

For disaggregated mode, each prefill and decode node runs as an independent
vLLM process with explicit KV transfer configuration:

.. code-block:: bash

    # Prefill node
    vllm serve ... \
        --kv-transfer-config.kv_connector NixlConnector \
        --kv-transfer-config.kv_role kv_producer \
        --kv-transfer-config.kv_connector_extra_config.backends+ LIBFABRIC

    # Decode node
    vllm serve ... \
        --kv-transfer-config.kv_connector NixlConnector \
        --kv-transfer-config.kv_role kv_consumer \
        --kv-transfer-config.kv_connector_extra_config.backends+ LIBFABRIC

The router uses ``round_robin`` policy for pure-DP groups and
``consistent_hash`` with ``--vllm-pd-disaggregation`` for PD groups, directing
initial requests to prefill endpoints and subsequent decode traffic to decode
endpoints.

Each container is launched with ``--privileged``, ``--net=host``, and explicit
``/dev/infiniband/uverbs*`` and ``/dev/gdrdrv`` device mounts to enable
GPU-direct RDMA over EFA.

Benchmark Script
----------------

The ``bench.sh`` script wraps ``vllm bench serve`` and handles Docker image
loading transparently. If the ``vllm`` CLI is not available on the host, the
script re-executes itself inside the container. It points the benchmark client
at the router endpoint (or the direct vLLM endpoint for single-group
configurations):

.. code-block:: bash

    bash bench.sh -H <ROUTER_IP> -p <ROUTER_PORT> -- \
        --model /fsx/models/deepseek-ai/DeepSeek-V2-Lite \
        --dataset-name random \
        --random-input-len 512 --random-output-len 256 \
        --num-prompts 1024

Experimental Setup
------------------

All experiments run on 4 nodes with 8 GPUs each (TP=8) using
DeepSeek-V2-Lite as the model. The benchmark uses random input/output data
with 1024 prompts via ``vllm bench serve``.

The configurations are:

- **Baseline (data parallelism)**: 4 nodes, TP=8, DP=4. All nodes serve both
  prefill and decode. This is the standard data-parallel serving setup.
- **Route 2**: 2 groups of 2 nodes each, TP=8, DP=2 per group. A router
  round-robins requests across groups. Each group independently handles both
  prefill and decode.
- **Route 4**: 4 groups of 1 node each, TP=8, no data parallelism. A router
  distributes requests across all 4 independent nodes.
- **PD 1P3D**: Disaggregated prefill/decode with 1 prefill node and 3 decode
  nodes. KV cache is transferred from the prefill node to decode nodes via NIXL.
- **PD 2P2D**: Disaggregated prefill/decode with 2 prefill nodes and 2 decode
  nodes.

.. code-block:: bash

    # Exp 1: Baseline — 4 nodes, TP=8, pure DP
    salloc -N 4 bash vllm.sbatch \
        --model /fsx/models/deepseek-ai/DeepSeek-V2-Lite \
        --gpu-memory-utilization 0.9

    # Exp 2: 2 groups × 2 nodes, DP=2 per group, router round-robins
    salloc -N 4 bash vllm.sbatch --route 2 \
        --model /fsx/models/deepseek-ai/DeepSeek-V2-Lite \
        --gpu-memory-utilization 0.9

    # Exp 3: 4 groups × 1 node, no DP, router round-robins
    salloc -N 4 bash vllm.sbatch --route 4 \
        --model /fsx/models/deepseek-ai/DeepSeek-V2-Lite \
        --gpu-memory-utilization 0.9

    # Exp 4: 1 prefill + 3 decode
    salloc -N 4 bash vllm.sbatch --prefill 1 \
        --model /fsx/models/deepseek-ai/DeepSeek-V2-Lite \
        --gpu-memory-utilization 0.9

    # Exp 5: 2 prefill + 2 decode
    salloc -N 4 bash vllm.sbatch --prefill 2 \
        --model /fsx/models/deepseek-ai/DeepSeek-V2-Lite \
        --gpu-memory-utilization 0.9

Results
-------

Output Token Throughput
~~~~~~~~~~~~~~~~~~~~~~~

.. image:: /_static/appendix/nixl/throughput.png
    :alt: Output token throughput comparison

The left panel varies input length with a fixed output length of 256 tokens
(prefill-dominated), while the right panel varies output length with a fixed
input length of 512 tokens (decode-dominated).

For prefill-dominated workloads, Route 4 achieves the highest throughput since
each node operates independently without the overhead of data parallelism
coordination. The disaggregated configurations (PD 1P3D and PD 2P2D) show
competitive throughput at shorter input lengths but degrade at longer inputs
where the prefill nodes become the bottleneck.

For decode-dominated workloads, Route 4 again leads, followed by PD 1P3D. The
baseline DP configuration shows the lowest throughput due to synchronization
overhead across all 4 nodes.

Request Throughput
~~~~~~~~~~~~~~~~~~

.. image:: /_static/appendix/nixl/req_throughput.png
    :alt: Request throughput comparison

Request throughput follows a similar pattern. Route 4 consistently achieves the
highest request throughput across all configurations. The disaggregated PD 1P3D
configuration maintains reasonable request throughput for short inputs but drops
significantly at longer input lengths (4096 tokens), where the single prefill
node becomes saturated.

Time to First Token (TTFT)
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: /_static/appendix/nixl/ttft.png
    :alt: TTFT comparison

TTFT is critical for user-perceived latency. The baseline DP and Route 2
configurations show moderate TTFT that scales with input length. Route 4
achieves the lowest TTFT across all input lengths due to the absence of
cross-node coordination.

The disaggregated configurations exhibit higher TTFT, particularly at longer
input lengths. PD 1P3D shows TTFT exceeding 37 seconds at 4096 input tokens,
as all prefill work funnels through a single node. PD 2P2D improves on this
but still lags behind the non-disaggregated configurations. The additional
latency from KV cache transfer over NIXL contributes to the elevated TTFT.

For decode-dominated workloads (right panel), the disaggregated configurations
show competitive TTFT since the input length is fixed at 512 tokens.

Inter-Token Latency (ITL)
~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: /_static/appendix/nixl/itl.png
    :alt: ITL comparison

ITL measures the latency between consecutive generated tokens during the decode
phase. This is where disaggregated serving shows its primary advantage.

PD 1P3D achieves the lowest ITL across nearly all configurations, with mean ITL
as low as 10 ms at longer input lengths. By isolating decode nodes from prefill
interference, the decode phase runs uninterrupted. PD 2P2D also shows reduced
ITL compared to the baseline, though the benefit is less pronounced due to
having fewer decode nodes.

The baseline DP and Route configurations show higher ITL, particularly at longer
input lengths where prefill and decode contend for the same GPU resources.

Discussion
----------

The results reveal a fundamental trade-off in disaggregated prefill/decode
serving:

- **Throughput vs. latency**: Disaggregated configurations sacrifice overall
  throughput and TTFT in exchange for significantly lower ITL. This trade-off
  is favorable for interactive applications where consistent token generation
  speed matters more than time-to-first-token.

- **Prefill bottleneck**: With a fixed cluster size, dedicating nodes to prefill
  reduces decode capacity and vice versa. PD 1P3D suffers from prefill
  saturation at long input lengths, while PD 2P2D has fewer decode nodes,
  limiting decode throughput.

- **Routing without disaggregation**: Route 4 (pure routing, no DP) achieves
  the best throughput and TTFT by eliminating cross-node synchronization
  entirely. This suggests that for workloads where ITL is not the primary
  concern, simple load-balanced independent nodes outperform both DP and
  disaggregated configurations.

- **KV cache transfer overhead**: The NIXL transfer over EFA adds latency to
  TTFT in disaggregated configurations. This overhead is amortized for longer
  decode sequences but is noticeable for short output lengths.

References
----------

.. [1] NVIDIA, "NIXL: NVIDIA Inference Xfer Library," GitHub, 2025.
   https://github.com/ai-dynamo/nixl

.. [2] vLLM Project, "vLLM: Easy, fast, and cheap LLM serving," GitHub, 2024.
   https://github.com/vllm-project/vllm
