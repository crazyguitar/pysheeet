

GPU-Initiated Networking for NCCL on AWS
========================================

Abstract
--------

GPU-Initiated Networking (GIN) has attracted significant attention as a key
enabler for kernel fusion in large language model (LLM) training and inference.
Mixture-of-Experts (MoE) architectures, such as DeepSeek-V3 and Qwen3-30B,
require efficient token dispatching and combining across MoE layers.
Conventionally, inter-GPU communication is initiated by the CPU through
collective libraries such as NCCL or Gloo, necessitating explicit GPU
synchronization barriers and additional ``cudaLaunchKernel`` calls that
introduce non-trivial overhead. GPU-Initiated Networking eliminates this
CPU-mediated round-trip by allowing data exchange to occur directly within CUDA
kernels, thereby enabling kernel fusion and efficient CUDA Graph capture for
accelerating end-to-end LLM layer computation. This article demonstrates how to
enable NCCL GIN with DeepEP on AWS HyperPod Slurm using the AWS Elastic Fabric
Adapter (EFA).

Introduction
------------

Prior to 2026, adopting DeepEP as a Mixture-of-Experts dispatch and combine
backend on AWS presented a significant challenge. The DeepEP kernel was
originally built on top of InfiniBand with a customized NVSHMEM implementation,
a transport layer unavailable on AWS infrastructure. This incompatibility
effectively prevented users from leveraging DeepEP on instances equipped with
the Elastic Fabric Adapter (EFA). Recent collaborative efforts by NVIDIA and
Amazon Annapurna Labs have addressed this gap by introducing GPU-Initiated
Networking support in NCCL and the EFA provider, enabling DeepEP to operate
over EFA without relying on InfiniBand (see `DeepEP PR #521
<https://github.com/deepseek-ai/DeepEP/pull/521>`_ and `aws-ofi-nccl PR #1069
<https://github.com/aws/aws-ofi-nccl/pull/1069>`_). The following experiment
builds upon these contributions to illustrate how to deploy DeepEP with NCCL
GIN on AWS using EFA.

Build DeepEP
------------

Before deploying DeepEP on AWS HyperPod Slurm, several components must be built
from source. First, NCCL >= v2.29.3-1 is required, as this is the minimum
version that exposes the Device API needed for GPU-Initiated Networking. The
build targets ``sm_90`` (NVIDIA H100) and ``sm_100`` (NVIDIA B200) compute
capabilities to ensure compatibility with current-generation GPU instances.

.. code-block:: bash

    NCCL_VERSION=v2.29.3-1
    git clone -b ${NCCL_VERSION} https://github.com/NVIDIA/nccl.git /opt/nccl \
        && cd /opt/nccl \
        && make -j $(nproc) src.build CUDA_HOME=/usr/local/cuda \
        NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90 -gencode=arch=compute_100,code=sm_100"

Optionally, the NCCL Device API examples can be built to verify that
GPU-initiated communication functions correctly in the target environment. In
addition, the latest release of nccl-tests (v2.17.9) ships with a GIN-enabled
microbenchmark for the ``alltoall`` collective, which is useful for validating
inter-GPU bandwidth and latency before running full-scale MoE workloads (see
`nccl-tests alltoall.cu <https://github.com/NVIDIA/nccl-tests/blob/v2.17.9/src/alltoall.cu#L214-L291>`_).

.. code-block:: bash

    ## Build NCCL Device API examples
    cd /opt/nccl/examples/06_device_api \
        && make -j $(nproc) NCCL_HOME=/opt/nccl/build CUDA_HOME=/usr/local/cuda MPI=1 MPI_HOME=/opt/amazon/openmpi

    NCCL_TESTS_VERSION=v2.17.9
    git clone -b ${NCCL_TESTS_VERSION} https://github.com/NVIDIA/nccl-tests.git /opt/nccl-tests \
        && cd /opt/nccl-tests \
        && make -j $(nproc) \
        MPI=1 \
        MPI_HOME=/opt/amazon/openmpi/ \
        CUDA_HOME=/usr/local/cuda \
        NCCL_HOME=/opt/nccl/build \
        NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90 -gencode=arch=compute_100,code=sm_100"

To test DeepEP on HyperPod Slurm, both DeepEP and aws-ofi-nccl must be pinned
to specific commits that include the NCCL GIN transport path. The DeepEP fork
by Aamir Shafi introduces an NCCL-based communication backend as an alternative
to the original NVSHMEM/InfiniBand path, while the aws-ofi-nccl plugin provides
the libfabric-to-NCCL translation layer required for EFA. Note that the NCCL
GIN implementation has since been merged into the aws-ofi-nccl main branch; the
commit hash is pinned here for reproducibility.

.. code-block:: bash

    ## Install DeepEP with NCCL GIN backend (PR #521)
    unset NVSHMEM_DIR NVSHMEM_HOME \
        && export ENABLE_NCCL=1 \
        && export NCCL_DIR=/opt/nccl/build \
        && export LD_LIBRARY_PATH=/opt/nccl/build/lib:$LD_LIBRARY_PATH \
        && export LD_PRELOAD=/opt/nccl/build/lib/libnccl.so.2 \
        && git clone -b nccl https://github.com/aamirshafi/DeepEP.git /opt/DeepEP \
        && cd /opt/DeepEP \
        && git checkout 6d29f34 \
        && python3 setup.py build_ext --inplace \
        && pip install --break-system-packages --no-build-isolation .

    AWS_OFI_NCCL_VERSION=5f4202f11db1585d878196db4430aeda0e834a0c
    git clone https://github.com/aws/aws-ofi-nccl.git /tmp/aws-ofi-nccl \
        && cd /tmp/aws-ofi-nccl \
        && git checkout ${AWS_OFI_NCCL_VERSION} \
        && ./autogen.sh \
        && ./configure --prefix=/opt/amazon/ofi-nccl \
            --with-libfabric=/opt/amazon/efa \
            --with-cuda=/usr/local/cuda \
        && make -j$(nproc) \
        && make install \
        && rm -rf /tmp/aws-ofi-nccl

For a complete build with all necessary dependencies, refer to the
`Dockerfile <https://github.com/crazyguitar/pysheeet/blob/master/src/gin/Dockerfile>`_
provided in this repository.

Test NCCL GIN
-------------

With the Docker image (or Enroot squash file) prepared in the previous section,
NCCL GIN functionality can be validated on a Slurm cluster. The following
examples demonstrate how to launch the NCCL Device API samples and nccl-tests
benchmarks. The corresponding Slurm wrapper scripts are available under the
`gin <https://github.com/crazyguitar/pysheeet/blob/master/src/gin/>`_ directory
in this repository.

.. code-block:: bash

    make docker && make save  # build a docker image and import an Enroot squash file

    # 01_allreduce_lsa (single node only)
    salloc -N 1 ./run.enroot /opt/nccl/examples/06_device_api/01_allreduce_lsa/allreduce_lsa

    # 01_allreduce_lsa (multi-node) — requires MNNVL (e.g. P6e-GB200), does NOT work over RDMA/EFA
    salloc -N 2 ./run.enroot /opt/nccl/examples/06_device_api/01_allreduce_lsa/allreduce_lsa

    # 02_alltoall_gin (multi-node)
    salloc -N 2 ./run.enroot /opt/nccl/examples/06_device_api/02_alltoall_gin/alltoall_gin

    # 03_alltoall_hybrid (multi-node)
    salloc -N 2 ./run.enroot /opt/nccl/examples/06_device_api/03_alltoall_hybrid/alltoall_hybrid

The nccl-tests ``alltoall`` benchmark exposes two critical flags for selecting
the GIN transport mode and memory registration strategy:

The ``-D`` flag selects the device-side implementation for the ``alltoall``
collective:

.. code-block:: text

    -D 0 — Host API (default)
    -D 1 — NVL simple (LSA/NVLink only)
    -D 2 — NVL optimized (LSA/NVLink only)
    -D 3 — GIN only (network)
    -D 4 — Hybrid (LSA intra-node + GIN inter-node)

The ``-R`` flag controls memory registration. Symmetric memory allocation
(``NCCL_MEM_SHARED``) is required for any device-side implementation
(``-D > 0``), as it maps GPU memory across all ranks to enable direct
remote read and write over the network:

.. code-block:: text

    -R 0 — no registration (default)
    -R 1 — register memory with ncclMemAlloc
    -R 2 — register memory with symmetric memory allocation (NCCL_MEM_SHARED)

The following examples launch the nccl-tests ``alltoall_perf`` benchmark in
GIN-only mode (``-D 3``) and hybrid mode (``-D 4``), sweeping message sizes
from 32 MB to 2048 MB. The ``--blocking 0`` flag enables non-blocking
collectives, which is representative of how MoE layers overlap communication
with computation in production workloads:

.. code-block:: bash

    # alltoall_perf with GIN (-D 3)
    salloc -N 2 ./run.enroot /opt/nccl-tests/build/alltoall_perf \
      -D 3 -R 2 -b 32M -e 2048M -f 2 -n 1000 -w 10 --blocking 0

    # alltoall_perf with Hybrid LSA+GIN (-D 4)
    salloc -N 2 ./run.enroot /opt/nccl-tests/build/alltoall_perf \
      -D 4 -R 2 -b 32M -e 2048M -f 2 -n 1000 -w 10 --blocking 0

Serving MoE Models with vLLM and DeepEP over NCCL GIN
-----------------------------------------------------

With NCCL GIN and EFA validated on AWS HyperPod Slurm, this section
demonstrates an end-to-end inference deployment using vLLM with DeepEP as the
MoE all-to-all communication backend. DeepEP's low-latency dispatch and combine
kernels, now operating over NCCL GIN rather than NVSHMEM, enable efficient
expert-parallel inference for large MoE models such as DeepSeek-V3.

The Slurm launch script ``run.sbatch`` is the same one used to launch a vLLM
server in the `vllm example directory
<https://github.com/crazyguitar/pysheeet/blob/master/src/llm/vllm/>`_. However,
to direct the DeepEP backend to use NCCL GIN, the following environment
variables must be set at launch time:

.. code-block:: bash

    DEEP_EP_BACKEND=nccl
    NCCL_GIN_TYPE=2  # proxy-based GIN

``NCCL_GIN_TYPE=2`` selects the proxy-based GIN path, in which a CPU-side proxy
thread mediates network transfers on behalf of the GPU. ``NCCL_GIN_TYPE=3``
would enable GPU Direct Async Kernel-Initiated (DAKI) networking, which
bypasses the proxy entirely; however, DAKI is not yet supported on AWS with EFA
at the time of writing.

For additional details on serving configurations and benchmarking, refer to
`llm-serving.rst
<https://github.com/crazyguitar/pysheeet/blob/master/docs/notes/llm/llm-serving.rst>`_
or the `vLLM README
<https://github.com/crazyguitar/pysheeet/blob/master/src/llm/vllm/README.rst>`_.

The following example launches a multi-node vLLM inference server for
DeepSeek-V3-0324 with expert parallelism enabled and the DeepEP low-latency
all-to-all backend:

.. code-block:: bash

   IMAGE="${PWD}/src/gin/nccl+latest.tar.gz"
   MODEL="/fsx/models/deepseek-ai/DeepSeek-V3-0324"

   salloc -N 4 bash run.sbatch "${MODEL}" \
     --image "${IMAGE}" \
     --all2all-backend deepep_low_latency \
     --tensor-parallel-size 8 \
     --enable-expert-parallel \
     --gpu-memory-utilization 0.8 \
     --enforce-eager

Upon successful launch, the vLLM server logs confirm that DeepEP is active as
the all-to-all backend and that NCCL GIN is being used for inter-GPU
communication. The key indicators are the ``DeepEPLLAll2AllManager`` manager
selection and the ``[NCCL Backend]`` initialization messages showing
communicator setup, symmetric memory allocation, and window registration across
all ranks:

.. code-block:: bash

    ...
    INFO 02-22 19:06:49 [serve.py:100] Defaulting api_server_count to data_parallel_size (4).
    INFO 02-22 19:06:49 [utils.py:325]
    INFO 02-22 19:06:49 [utils.py:325]        █     █     █▄   ▄█
    INFO 02-22 19:06:49 [utils.py:325]  ▄▄ ▄█ █     █     █ ▀▄▀ █  version 0.15.1
    INFO 02-22 19:06:49 [utils.py:325]   █▄█▀ █     █     █     █  model   /fsx/models/deepseek-ai/DeepSeek-V3-0324
    INFO 02-22 19:06:49 [utils.py:325]    ▀▀  ▀▀▀▀▀ ▀▀▀▀▀ ▀     ▀
    INFO 02-22 19:06:49 [utils.py:325]
    ...
    INFO 02-22 19:07:51 [cuda_communicator.py:124] Using DeepEPLLAll2AllManager all2all manager.
    ...
    [NCCL Backend] LOW LATENCY MODE: Rank 0 connecting to all 32 ranks
    [NCCL Backend] NCCL version: 2.29.3 (loaded library)
    [NCCL Backend] Initializing 2 communicator(s) (qps_per_rank=8) for rank 0/32
    [NCCL Backend] Rank 0 successfully initialized 2 communicator(s)
    [NCCL Backend] Rank 0 created 2 device communication(s) with 32 barrier sessions each
    [NCCL Backend] Initialized global rank 0/32 (comm rank 0/32)
    [NCCL Backend - Memory Alloc] Rank 0: Allocated ptr=0xf882000000, size=3816818816
    [NCCL Backend - Memory Register] Rank 0: Copying 2 NCCL windows to GPU
    [NCCL Backend - Memory Register] Rank 0: Successfully copied windows to GPU
    [NCCL Backend - Memory Register] Rank 0: Registered windows for ptr=0xf882000000, size=3816818816

Once the server is ready, inference requests can be issued via the
OpenAI-compatible completions API:

.. code-block:: bash

    curl -sf -X POST http://<VLLM_HOST>:8000/v1/completions \
      -H 'Content-Type: application/json' \
      -d '{
        "model": "/fsx/models/deepseek-ai/DeepSeek-V3-0324",
        "prompt": "Hello",
        "max_tokens": 10
      }'

      # output
      {"id":"cmpl-b6e9530a07561f11","object":"text_completion" ... }

Conclusion
----------

This article has demonstrated how to deploy vLLM with DeepEP and NCCL GIN on
AWS HyperPod Slurm using the Elastic Fabric Adapter. As this integration is
still under active development, certain limitations remain at the time of
writing. For instance, although DeepEP's low-latency mode supports CUDA Graph
capture, enabling it by removing ``--enforce-eager`` currently results in a
startup failure in vLLM. Additionally, performance over EFA may not yet match
that of InfiniBand-based deployments, as further optimizations are ongoing.

This article is intended as an early reference for evaluating DeepEP with NCCL
GIN on AWS. For production workloads, it is advisable to wait for official
stable releases from NVIDIA and Amazon Annapurna Labs.
