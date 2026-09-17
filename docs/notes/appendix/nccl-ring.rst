

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
