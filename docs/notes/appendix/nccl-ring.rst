NCCL Multi-Rail Broadcast with More GPU Ranks
==================================================


Abstract
--------

NCCL broadcast is used to synchronize model weights between training workers
and rollout engines in frameworks such as `Slime`_. However, having multiple
RDMA NICs per node does not guarantee that a broadcast will use them all.
In our single-sender setup, weight transfers use only the source GPU's local
NIC pair, leaving most of the node's network capacity unused. This note
compares one and eight participating GPUs on the source node, with eight
receivers in both cases, to show how rank placement affects NIC utilization
and weight-transfer throughput. On two Cambrian H100 nodes, expanding the
source node from one to eight ranks increases median broadcast throughput
from 47.86 to 290.17 GB/s, a 6.06x speedup within one process group.

Introduction
------------

When a node has multiple RDMA NICs, it is natural to expect a large broadcast
to use their combined bandwidth. Reaching that bandwidth, however, requires
traffic to flow through multiple NICs in parallel. In NCCL, those paths depend
on the participating GPU ranks, the GPU-to-NIC topology, and the communication
layout. The simplified diagram below illustrates a single sender GPU
broadcasting through its local network connection. On the Cambrian nodes
used in our experiments, that connection consists of two 200 Gb/s ports
combined by NCCL into one 400 Gb/s virtual device. The question is how to
involve the other GPUs and NICs to use more of the node's available bandwidth.

.. image:: https://raw.githubusercontent.com/crazyguitar/pysheeet/2aa75ef88604baba97a9fd193ea8820e4cb9ad02/docs/_static/appendix/nccl/nccl-broadcast-0.png
   :alt: Broadcast from a single sender GPU through one NIC to the receiving node.

NCCL's ring broadcast divides the buffer into chunks and pipelines them
through the participating ranks. Within a node, these transfers can use
fast GPU interconnects such as NVLink. Including more GPU ranks on the sender
node gives NCCL additional paths through local GPUs and their nearby NICs.
Our second configuration places all GPU ranks on both nodes in a single
process group, as illustrated below. Whether traffic spreads across
multiple NICs depends on the topology and the channel layout NCCL chooses;
adding ranks alone does not guarantee it. This configuration is distinct from
using multiple process groups to distribute transfers.

.. image:: https://raw.githubusercontent.com/crazyguitar/pysheeet/2aa75ef88604baba97a9fd193ea8820e4cb9ad02/docs/_static/appendix/nccl/nccl-broadcast-8.png
   :alt: Broadcast chunks transferred through multiple sender GPUs and NICs.


Experiment
----------

Environment
~~~~~~~~~~~

* **Hardware:** two Cambrian nodes, each with 8 × NVIDIA H100 80GB GPUs.
* **Software:** PyTorch 2.11.0+cu130, CUDA 13.0, NCCL 2.28.9,
  NVIDIA driver 570.172.08.
* **Benchmark:** 1 GiB BF16 broadcast, 300 seconds per configuration.
* **NCCL:** default settings, with ``NCCL_DEBUG=INFO``.

Single-sender baseline
~~~~~~~~~~~~~~~~~~~~~~

Run these commands from the repository root on each node. They launch one
sender process on node 0 and eight receiver processes on node 1. Set
``MASTER_ADDR`` to the sender node's address on both nodes, then run the
corresponding command on each node. Each command uses ``--seconds 300`` to set
the experiment duration.

.. code-block:: bash

    # First node (node rank 0)
    export OMP_NUM_THREADS=1
    torchrun --standalone \
      --nproc-per-node=1 src/nccl/broadcast.py \
      --node-rank 0 \
      --sender-ranks 1 \
      --master-addr ${MASTER_ADDR} \
      --seconds 300

    # Second node (node rank 1)
    export OMP_NUM_THREADS=1
    torchrun --standalone --nproc-per-node=8 \
      src/nccl/broadcast.py \
      --node-rank 1 \
      --sender-ranks 1 \
      --master-addr ${MASTER_ADDR} \
      --seconds 300

The baseline achieves a median of **47.86 GB/s**, close to the 50 GB/s raw
line rate of a 400 Gb/s port pair. All ranks pass the payload check.

The figure below is an earlier `rdmatop`_ capture illustrating traffic
concentrated near the source GPU. For the Cambrian run reported here, NCCL's
logs identify the source's virtual device as ``mlx5_0+mlx5_1``. RDMA counters
show bulk traffic split across those two 200 Gb/s ports while the remaining
source-node ports are effectively idle.

.. image:: https://raw.githubusercontent.com/crazyguitar/pysheeet/f27b131543e06feddec1f4af518ca8cb0dbfbc74/docs/_static/appendix/nccl/nccl-broadcast-0.gif
   :alt: RDMA NIC traffic on node 0 during the single-sender broadcast experiment.

Eight ranks on the source node
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For the second configuration, launch eight processes on both nodes and set
``--sender-ranks 8`` on both commands. The payload size, measurement duration,
and broadcast source remain the same.

.. code-block:: bash

    # First node (node rank 0)
    export OMP_NUM_THREADS=1
    torchrun --standalone --nproc-per-node=8 \
      src/nccl/broadcast.py \
      --node-rank 0 \
      --sender-ranks 8 \
      --master-addr ${MASTER_ADDR} \
      --seconds 300

    # Second node (node rank 1)
    export OMP_NUM_THREADS=1
    torchrun --standalone --nproc-per-node=8 \
      src/nccl/broadcast.py \
      --node-rank 1 \
      --sender-ranks 8 \
      --master-addr ${MASTER_ADDR} \
      --seconds 300

.. image:: https://raw.githubusercontent.com/crazyguitar/pysheeet/f27b131543e06feddec1f4af518ca8cb0dbfbc74/docs/_static/appendix/nccl/nccl-broadcast-8.gif
   :alt: RDMA NIC traffic with eight participating GPU ranks on the source node.

Results
~~~~~~~

Both configurations completed successfully and passed the payload check on
every rank. The table summarizes one run per configuration; each sample is
a batch of 100 broadcasts, not a separate experiment.

.. list-table:: Broadcast throughput for a 1 GiB payload
   :header-rows: 1
   :widths: 24 12 18 22 12 12

   * - Source-node ranks
     - Receiver ranks
     - Median (GB/s)
     - Range (GB/s)
     - Samples
     - Speedup
   * - 1
     - 8
     - 47.86
     - 47.86–47.87
     - 134
     - 1.00x
   * - 8
     - 8
     - 290.17
     - 282.12–290.39
     - 808
     - 6.06x

With one source-node rank, NCCL configures four collective channels and sends
bulk traffic through one GPU-local pair of 200 Gb/s ports. With eight
source-node ranks, it configures 16 collective channels; an in-run counter
snapshot shows bulk traffic distributed almost evenly across all 16 data
ports, corresponding to eight GPU-local pairs. The two 100 Gb/s ports carry
negligible traffic in that snapshot.

Both experiments use one NCCL process group:

* **1 → 8:** one group containing 9 GPU ranks.
* **8 → 8:** one group containing 16 GPU ranks.

The 6.06x improvement comes from adding participating GPUs on the source node.
This result is specific to the tested topology, NCCL version, and payload
size; adding ranks is not a general guarantee of proportional speedup.

.. _Slime: https://github.com/THUDM/slime/blob/4c193f1f37509cca70f0e88807a9305b70f63f4e/slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py#L348-L352

.. _rdmatop: https://github.com/crazyguitar/rdmatop
