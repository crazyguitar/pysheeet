

NCCL Multi-Rail Broadcast with More Process Groups
==================================================


Abstract
--------

NCCL broadcast is used to synchronize model weights between training workers
and rollout engines in frameworks such as `Slime`_. However, having multiple
RDMA NICs per node does not guarantee that a broadcast will use them all.
In our setup, weight transfers use only one NIC, leaving the others idle and
limiting throughput. This post examines that behavior through experiments,
then shows how multiple process groups can spread transfers across the
available NICs to speed up weight updates between nodes. The figure below
shows broadcast traffic on node 0, captured with `rdmatop`_. Rank 0 sends
the tensor through a single NIC, leaving the other NICs idle.

.. _Slime: https://github.com/THUDM/slime/blob/4c193f1f37509cca70f0e88807a9305b70f63f4e/slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py#L348-L352

.. _rdmatop: https://github.com/crazyguitar/rdmatop

.. image:: https://raw.githubusercontent.com/crazyguitar/pysheeet/f27b131543e06feddec1f4af518ca8cb0dbfbc74/docs/_static/appendix/nccl/nccl-broadcast-0.gif
   :alt: RDMA NIC traffic during NCCL broadcast with a single process group.

Introduction
------------

When a node has multiple RDMA NICs, it is natural to expect a large broadcast
to use their combined bandwidth. Reaching that bandwidth, however, requires
traffic to flow through multiple NICs in parallel. In NCCL, those paths depend
on the participating GPU ranks, the GPU-to-NIC topology, and the communication
layout. In our single-sender setup, broadcast traffic leaves the node through
just one NIC. The question is how to involve the other GPUs and NICs so that
the transfer can use more of the node's available bandwidth.

.. image:: https://raw.githubusercontent.com/crazyguitar/pysheeet/2aa75ef88604baba97a9fd193ea8820e4cb9ad02/docs/_static/appendix/nccl/nccl-broadcast-0.png
   :alt: Broadcast from a single sender GPU through one NIC to the receiving node.

NCCL's ring broadcast divides the buffer into chunks and pipelines them
through the participating ranks. Within a node, these transfers can use
fast GPU interconnects such as NVLink. Including more GPU ranks on the sender
node gives NCCL additional paths through local GPUs and their nearby NICs.
Our next experiment places all GPU ranks on both nodes in one process group
to examine how this changes NIC utilization and throughput. Whether traffic
spreads across multiple NICs depends on the topology and the channel layout
NCCL chooses; adding ranks alone does not guarantee it.

.. image:: https://raw.githubusercontent.com/crazyguitar/pysheeet/2aa75ef88604baba97a9fd193ea8820e4cb9ad02/docs/_static/appendix/nccl/nccl-broadcast-8.png
   :alt: Broadcast chunks transferred through multiple sender GPUs and NICs.


.. code-block:: bash

    # node 0
    torchrun --standalone \
      --nproc-per-node=1 /tmp/nccl-broadcast/broadcast.py \
      --node-rank 0 \
      --sender-ranks 1 \
      --master-addr ${MASTER_ADDR} \
      --seconds 300

    # node 1
    torchrun --standalone --nproc-per-node=8 \
      /tmp/nccl-broadcast/broadcast.py \
      --node-rank 1 \
      --sender-ranks 1 \
      --master-addr ${MASTER_ADDR} \
      --seconds 300
