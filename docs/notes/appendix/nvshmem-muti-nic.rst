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
