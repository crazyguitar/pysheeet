

NCCL Multi-Rail Broadcast with More Process Groups
==================================================

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
