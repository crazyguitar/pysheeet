.. meta::
    :description lang=en: NVSHMEM 3.6.5 Multi-NIC Support for Libfabric with AWS EFA: Round-Robin NIC Selection for MoE and DeepEP Workloads
    :keywords: NVSHMEM, Multi-NIC, AWS, EFA, libfabric, DeepEP, MoE, round-robin, GPU, HPC, LLM, IBGDA, proxy thread

NVSHMEM Multi-NIC Support with AWS EFA
======================================

:Date: 2026-03-27

Abstract
--------

`NVSHMEM 3.6.5 <https://github.com/NVIDIA/nvshmem/releases/tag/v3.6.5-0>`_
introduces multi-NIC support for the libfabric transport with round-robin NIC
selection—a collaborative effort between NVIDIA and Amazon Annapurna Labs.
NVSHMEM has gained significant attention since
`DeepEP <https://github.com/deepseek-ai/DeepEP>`_ demonstrated that
implementing MoE layer dispatch and combine operations with NVSHMEM can
substantially improve performance for large language models (LLMs) employing
Mixture-of-Experts (MoE) architectures. However, on AWS, DeepEP compatibility
was limited because it relies on
`InfiniBand GPUDirect Async <https://developer.nvidia.com/blog/improving-network-performance-of-hpc-systems-using-nvidia-magnum-io-nvshmem-and-gpudirect-async/>`_
(IBGDA), whereas AWS GPU instances support only the proxy thread transport via
libfabric. Furthermore, prior to version 3.6.5, NVSHMEM supported only a
single NIC, meaning each GPU could utilize only one EFA NIC for data
transmission. This article reviews the new multi-NIC feature and presents
experiments exploring the potential performance improvements on AWS.

Introduction
------------

Mixture-of-Experts (MoE) architectures rely on all-to-all collective
communication to dispatch and combine tokens across expert replicas. However,
conventional all-to-all implementations often become a bottleneck during LLM
training due to limited overlap with computation and suboptimal bandwidth
utilization. NVSHMEM addresses this by exposing a device-side API that enables
developers to implement custom all-to-all kernels—such as
`pplx-kernels <https://github.com/perplexityai/pplx-kernels>`_ and
`DeepEP <https://github.com/deepseek-ai/DeepEP>`_\—with GPU-initiated
networking, eliminating costly GPU–CPU context switches.

Despite these advantages, prior NVSHMEM versions (before 3.6.5) supported only
a single NIC per GPU on the libfabric transport. On AWS instances equipped with
multiple EFA NICs, this meant that NVSHMEM-based MoE kernels could not fully
utilize the available network bandwidth, and in practice did not outperform
ordinary all-to-all collective communication.

To evaluate the impact of multi-NIC support, we use the NVSHMEM device
all-to-all performance tool to compare single-NIC and multi-NIC configurations.
We also benchmark
`pplx-kernels <https://github.com/perplexityai/pplx-kernels>`_ to assess
whether MoE dispatch and combine operations benefit from multi-NIC EFA on the
libfabric backend.

NVSHMEM Device All-to-All
-------------------------

To verify that multi-NIC round-robin is functioning correctly, we use
`rdmatop <https://github.com/crazyguitar/rdmatop>`_ to monitor RDMA traffic
across all EFA NICs during NVSHMEM benchmarks. The experiments follow the
NVSHMEM examples in the
`rdmatop <https://github.com/crazyguitar/rdmatop/tree/main/examples/nvshmem>`_
repository.

We run two benchmarks from the NVSHMEM perftest suite on a Slurm cluster. The
first measures point-to-point put bandwidth between a single GPU per node, and
the second measures device-initiated all-to-all latency across all 8 GPUs per
node.

Put Bandwidth (Inter-Node, 1 GPU per Node)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this experiment, we launch a point-to-point job with one sender and one
receiver to verify whether a single GPU can fully utilize all available EFA
NICs via round-robin NIC selection.

.. code-block:: bash

    salloc -N 2 NTASKS_PER_NODE=1 \
      bash examples/nvshmem/nvshmem.sbatch \
      /opt/nvshmem/bin/perftest/device/pt-to-pt/shmem_put_bw \
      -b 8 -e 128M -f 2 -n 1000 -w 100

All-to-All Latency (Device, All 8 GPUs per Node)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This experiment uses the NVSHMEM device all-to-all performance tool to
determine whether all-to-all communication can fully saturate all EFA NICs
when all 8 GPUs per node participate.

.. code-block:: bash

    NODES=2 # 4, 8, 16
    salloc -N ${NODES} bash examples/nvshmem/nvshmem.sbatch \
      /opt/nvshmem/bin/perftest/device/coll/alltoall_latency \
      -b 16 -e 1G -f 2 -n 1000 -s all

Result
------

The following subsections present the results for point-to-point put bandwidth
and device all-to-all latency, comparing NVSHMEM 3.5.21 (single NIC) against
3.6.5 (multi-NIC).

Put Bandwidth (Point-to-Point)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the point-to-point put bandwidth experiment, NVSHMEM 3.5.21 uses only a
single EFA NIC per GPU for both Tx and Rx, as shown by the ``rdmatop`` output
below. This confirms that, prior to multi-NIC support, each GPU was limited to
the bandwidth of one NIC regardless of how many were available on the instance.

.. image:: https://raw.githubusercontent.com/crazyguitar/pysheeet/blog/nvshmem/docs/_static/appendix/nvshmem/nvshmem-put-3.5.21.gif

With NVSHMEM 3.6.5, the same experiment shows traffic distributed across all 4
EFA NICs via round-robin selection. This allows a single GPU to aggregate
bandwidth from multiple NICs, significantly increasing the achievable
point-to-point throughput.

.. image:: https://raw.githubusercontent.com/crazyguitar/pysheeet/blog/nvshmem/docs/_static/appendix/nvshmem/nvshmem-put-3.6.5.gif

All-to-All Latency (Device)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the device all-to-all experiment, NVSHMEM 3.5.21 shows that each GPU
utilizes only a single NIC to transfer data, consistent with the point-to-point
results above.

.. image:: https://raw.githubusercontent.com/crazyguitar/pysheeet/blog/nvshmem/docs/_static/appendix/nvshmem/nvshmem-3.5.21.gif

With NVSHMEM 3.6.5, ``rdmatop`` confirms that all Tx NICs carry traffic,
demonstrating that multi-NIC round-robin is active during the all-to-all
operation. However, we observed that Rx traffic was imbalanced across NICs.

.. image:: https://raw.githubusercontent.com/crazyguitar/pysheeet/blog/nvshmem/docs/_static/appendix/nvshmem/nvshmem-3.6.5.gif
