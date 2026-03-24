.. meta::
    :description lang=en: Ray cluster cheat sheet for HPC environments covering cluster setup on Slurm, Docker-based deployment, monitoring with raytop, and distributed workloads
    :keywords: Ray, Ray cluster, HPC, Slurm, distributed computing, GPU cluster, raytop, monitoring, Docker, sbatch, machine learning, distributed training

===========
Ray Cluster
===========

.. contents:: Table of Contents
    :backlinks: none

`Ray <https://docs.ray.io/>`_ is an open-source framework for scaling Python applications
across clusters. It provides a simple API for distributed computing and is widely used for
distributed machine learning training, reinforcement learning, hyperparameter tuning, and
serving. In HPC environments, Ray clusters are typically launched on top of Slurm-managed
nodes using Docker containers, combining the resource management capabilities of Slurm with
Ray's flexible distributed runtime.

This cheat sheet covers the Ray Python API, launching Ray clusters on Slurm, cluster
monitoring, and real-world deployment examples for distributed GPU workloads.

Connect to a Ray Cluster
-------------------------

Ray uses three default ports:

- ``6379`` — GCS (Global Control Store), the main cluster coordination port
- ``8265`` — Dashboard and REST API (used by ``ray job submit`` and ``raytop``)
- ``10001`` — Ray Client server (used by ``ray.init(address="ray://...")``)

.. code-block:: python

    import ray

    # auto-detect (inside a Ray container or submitted via ray job submit)
    ray.init()

    # connect via GCS directly (port 6379)
    ray.init(address="<HEAD_IP>:6379")

    # connect via Ray Client (port 10001)
    ray.init(address="ray://<HEAD_IP>:10001")

    # or set RAY_ADDRESS env var and just call ray.init()
    # export RAY_ADDRESS=<HEAD_IP>:6379

    # check available resources
    print(ray.cluster_resources())
    # {'CPU': 192.0, 'GPU': 16.0, ...}

Remote Functions
----------------

``@ray.remote`` turns a function into a distributed task. Use ``.remote()`` to
submit and ``ray.get()`` to collect results.

.. code-block:: python

    import ray

    @ray.remote
    def add(a, b):
        return a + b

    # submit tasks in parallel
    futures = [add.remote(i, i) for i in range(4)]
    print(ray.get(futures))  # [0, 2, 4, 6]

    # request specific resources
    @ray.remote(num_gpus=1)
    def train_on_gpu(data):
        import torch
        device = torch.device("cuda")
        # ...

    # request multiple GPUs
    @ray.remote(num_gpus=4, num_cpus=8)
    def train_large_model(config):
        ...

Object Store
------------

``ray.put()`` places objects in the shared object store so multiple tasks can
access them without re-serializing. This is useful for large models or datasets.

.. code-block:: python

    import ray
    import numpy as np

    # put a large array into the object store (returns an ObjectRef)
    data = np.random.rand(10000, 10000)
    data_ref = ray.put(data)

    @ray.remote
    def process(data_ref):
        data = ray.get(data_ref)  # zero-copy read on same node
        return data.sum()

    # all tasks share the same copy — no re-serialization
    futures = [process.remote(data_ref) for _ in range(4)]
    print(ray.get(futures))

``ray.wait()``
--------------

``ray.wait()`` returns completed futures without blocking on all of them.
Useful for processing results as they arrive.

.. code-block:: python

    import ray

    @ray.remote
    def slow_task(i):
        import time
        time.sleep(i)
        return i

    futures = [slow_task.remote(i) for i in range(5)]

    # wait for at least 2 results, timeout after 3 seconds
    ready, not_ready = ray.wait(futures, num_returns=2, timeout=3.0)
    print(ray.get(ready))

    # process results as they complete
    while futures:
        ready, futures = ray.wait(futures, num_returns=1)
        print(f"Got: {ray.get(ready[0])}")

Resource Management
-------------------

Ray exposes cluster resource information and supports fractional GPU allocation
and custom resource types for fine-grained scheduling control.

.. code-block:: python

    import ray

    # check cluster resources
    print(ray.cluster_resources())    # total
    print(ray.available_resources())  # currently free

    # fractional GPUs (e.g., 2 tasks share 1 GPU)
    @ray.remote(num_gpus=0.5)
    def light_inference(batch):
        ...

Fault Tolerance
---------------

For long-running distributed training, Ray can automatically restart crashed actors
and retry failed tasks without bringing down the entire job.

.. code-block:: python

    import ray

    # auto-restart actors up to 3 times on failure
    @ray.remote(num_gpus=1, max_restarts=3)
    class ResilientWorker:
        def __init__(self):
            self.setup_model()

        def setup_model(self):
            ...

        def train(self, batch):
            ...

    # retry failed tasks up to 3 times
    @ray.remote(num_gpus=1, max_retries=3)
    def train_step(data):
        ...

Runtime Environments
--------------------

Runtime environments let you specify per-task or per-actor dependencies (pip
packages, env vars, working directory) without baking them into the container.

.. code-block:: python

    import ray

    # per-task runtime env
    @ray.remote(runtime_env={"pip": ["torch==2.9.1"], "env_vars": {"NCCL_DEBUG": "INFO"}})
    def train():
        ...

    # or set at ray.init for all tasks
    ray.init(runtime_env={"working_dir": "/workspace", "pip": ["vllm==0.15.1"]})

Ray Actors
----------

Actors are stateful workers. Each actor runs in its own process and holds state
across method calls.

.. code-block:: python

    import ray

    @ray.remote(num_gpus=1)
    class Worker:
        def __init__(self, model_path):
            import torch
            self.model = torch.load(model_path)
            self.device = torch.device("cuda")

        def predict(self, batch):
            return self.model(batch.to(self.device))

    # create actor instances
    workers = [Worker.remote("/fsx/model.pt") for _ in range(4)]

    # call methods
    futures = [w.predict.remote(data) for w, data in zip(workers, batches)]
    results = ray.get(futures)

Placement Groups
----------------

Placement groups control how tasks and actors are co-located across nodes.
This is essential for distributed training where you need to pin GPU workers
to specific nodes (e.g., one placement group per node with ``STRICT_PACK``).

.. code-block:: python

    import ray
    from ray.util.placement_group import placement_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    ray.init()

    # create a placement group: 8 GPUs packed on one node
    pg = placement_group([{"GPU": 1} for _ in range(8)], strategy="STRICT_PACK")
    ray.get(pg.ready())

    # schedule an actor on a specific bundle within the group
    @ray.remote(num_gpus=1)
    class GpuWorker:
        def run(self, rank):
            return f"rank {rank} on {ray.util.get_node_ip_address()}"

    workers = []
    for i in range(8):
        w = GpuWorker.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_bundle_index=i,
            )
        ).remote()
        workers.append(w)

    futures = [w.run.remote(i) for i, w in enumerate(workers)]
    print(ray.get(futures))

Multi-node placement groups (e.g., for distributed training across 2 nodes):

.. code-block:: python

    def create_placement_groups(num_nodes, gpus_per_node):
        """One placement group per node, each with gpus_per_node GPU bundles."""
        pgs = []
        for _ in range(num_nodes):
            bundles = [{"GPU": 1} for _ in range(gpus_per_node)]
            pg = placement_group(bundles, strategy="STRICT_PACK")
            ray.get(pg.ready())
            pgs.append(pg)
        return pgs

    # 2 nodes x 8 GPUs = 16 GPU workers
    pgs = create_placement_groups(num_nodes=2, gpus_per_node=8)

Placement strategies:

- ``STRICT_PACK`` — all bundles on the same node (distributed training)
- ``PACK`` — best-effort packing on fewest nodes
- ``STRICT_SPREAD`` — one bundle per node (fault tolerance)
- ``SPREAD`` — best-effort spreading across nodes

Deploy Megatron on Ray
----------------------

Ray placement groups can replace ``torchrun`` for launching Megatron distributed
training. Each Ray actor maps to one GPU rank, and placement groups ensure workers
are pinned to the correct nodes. The following example launches DeepSeek-V2-Lite
pretraining across 2 nodes with 8 GPUs each. Source:
`raytop Megatron example <https://github.com/crazyguitar/raytop/tree/main/examples/megatron>`_.

.. code-block:: bash

    # Start Ray cluster first:
    salloc -N 2 bash examples/ray/ray.sbatch --image /fsx/megatron-lm+latest.tar.gz

    # Then from inside the head container:
    python3 main.py
    python3 main.py --hf-path /fsx/models/deepseek-ai/DeepSeek-V2-Lite
    python3 main.py --nodes 2 --gpus-per-node 8

.. code-block:: python

    #!/usr/bin/env python3
    """Launch Megatron Bridge DeepSeek-V2-Lite pretrain on a Ray cluster.

    Uses Ray placement groups to pin workers to nodes (same pattern as verl's
    Megatron backend). Each Ray actor = one GPU rank, calling
    torch.distributed.init_process_group() directly — no torchrun needed.
    """

    import argparse
    import os

    import ray
    from ray.util.placement_group import placement_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    EFA_ENV = {
        "FI_PROVIDER": "efa",
        "FI_EFA_USE_DEVICE_RDMA": "1",
        "FI_EFA_FORK_SAFE": "1",
        "NCCL_DEBUG": "WARN",
        "NCCL_BUFFSIZE": "8388608",
        "NCCL_P2P_NET_CHUNKSIZE": "524288",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "OMP_NUM_THREADS": "1",
    }

    def build_config(hf_path=None):
        from megatron.bridge.recipes.deepseek.deepseek_v2 import (
            deepseek_v2_lite_pretrain_config,
        )

        cfg = deepseek_v2_lite_pretrain_config(
            **({"hf_path": hf_path} if hf_path else {}),
            tensor_model_parallel_size=8,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=2,
            sequence_parallel=True,
            seq_length=4096,
            train_iters=500,
            global_batch_size=64,
            micro_batch_size=1,
            eval_interval=100,
            lr_warmup_iters=50,
            save_interval=0,
        )
        cfg.model.moe_permute_fusion = False
        return cfg

    @ray.remote(num_cpus=0, num_gpus=1)
    class Worker:
        def run(self, rank, local_rank, world_size, master_addr, master_port, hf_path):
            # Must reset CUDA_VISIBLE_DEVICES BEFORE any torch/CUDA import
            gpu_ids = ray.get_gpu_ids()
            cuda_device = int(gpu_ids[0]) if gpu_ids else local_rank
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)

            os.environ.update(EFA_ENV)
            os.environ.update(
                {
                    "RANK": str(rank),
                    "LOCAL_RANK": str(cuda_device),
                    "WORLD_SIZE": str(world_size),
                    "MASTER_ADDR": master_addr,
                    "MASTER_PORT": str(master_port),
                }
            )

            import megatron.core.jit as _jit

            if not hasattr(_jit, "disable_jit_fuser"):
                _jit.disable_jit_fuser = lambda: None

            from megatron.bridge.training.gpt_step import forward_step
            from megatron.bridge.training.pretrain import pretrain

            cfg = build_config(hf_path)
            pretrain(config=cfg, forward_step_func=forward_step)

    def create_placement_groups(num_nodes, gpus_per_node):
        """Create one placement group per node, each with gpus_per_node GPU bundles."""
        pgs = []
        for _ in range(num_nodes):
            bundles = [{"GPU": 1} for _ in range(gpus_per_node)]
            pg = placement_group(bundles, strategy="STRICT_PACK")
            ray.get(pg.ready())
            pgs.append(pg)
        return pgs

    def resolve_master_and_sort_pgs(pgs):
        """Resolve node IPs for each PG, sort by IP, return (sorted_pgs, master_addr)."""

        @ray.remote(num_cpus=0)
        def _get_ip():
            return ray.util.get_node_ip_address()

        ip_futures = []
        for pg in pgs:
            ref = _get_ip.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=0,
                )
            ).remote()
            ip_futures.append(ref)

        ips = ray.get(ip_futures)
        pg_ip_pairs = sorted(zip(pgs, ips), key=lambda x: x[1])
        sorted_pgs = [p for p, _ in pg_ip_pairs]
        master_addr = pg_ip_pairs[0][1]
        return sorted_pgs, master_addr

    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument("--hf-path", default="deepseek-ai/DeepSeek-V2-Lite")
        parser.add_argument("--nodes", type=int, default=2)
        parser.add_argument("--gpus-per-node", type=int, default=8)
        parser.add_argument("--master-port", type=int, default=29500)
        args = parser.parse_args()

        world_size = args.nodes * args.gpus_per_node

        ray.init()
        pgs = create_placement_groups(args.nodes, args.gpus_per_node)
        sorted_pgs, master_addr = resolve_master_and_sort_pgs(pgs)

        # Spawn one worker per GPU, pinned to placement groups
        futures = []
        rank = 0
        for pg in sorted_pgs:
            for local_rank in range(args.gpus_per_node):
                worker = Worker.options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg,
                        placement_group_bundle_index=local_rank,
                    )
                ).remote()
                futures.append(
                    worker.run.remote(
                        rank, local_rank, world_size,
                        master_addr, args.master_port, args.hf_path,
                    )
                )
                rank += 1

        ray.get(futures)


    if __name__ == "__main__":
        main()

Launch a Ray Cluster on Slurm
-----------------------------

The following sbatch script provisions a Ray cluster inside Docker containers across
Slurm-managed nodes. It starts a Ray head on the first allocated node and Ray workers
on all remaining nodes, then waits until every node has registered with the cluster.

.. code-block:: bash

    # allocate nodes and launch interactively
    salloc -N 2 bash ray.sbatch

    # submit as a batch job
    sbatch -N 2 ray.sbatch

    # use a custom container image
    sbatch -N 4 ray.sbatch --image /fsx/ray+latest.tar.gz

The script handles image loading (tarball or registry pull), head/worker startup, and
health checking via ``ray status``. See ``src/ray/ray.sbatch`` for the full script.

Ray Cluster Status
------------------

.. code-block:: bash

    # check cluster status from head node
    docker exec ray-head ray status

    # check from inside the container
    ray status

    # get cluster info via REST API
    curl http://${HEAD_IP}:8265/api/cluster_status

    # list nodes via REST API
    curl http://${HEAD_IP}:8265/api/v0/nodes

    # list running jobs
    curl http://${HEAD_IP}:8265/api/jobs/

Ray Dashboard
-------------

Ray provides a built-in web dashboard on port ``8265`` of the head node. When running
on HPC clusters behind a firewall, use SSH port forwarding to access it:

.. code-block:: bash

    # forward Ray dashboard to local machine
    ssh -L 8265:${HEAD_IP}:8265 login-node

    # then open in browser
    # http://localhost:8265

Monitoring with raytop
----------------------

`raytop <https://github.com/crazyguitar/raytop>`_ is a real-time TUI monitor for Ray
clusters — like ``htop`` for distributed GPU training. It provides a unified view of
cluster-wide resource utilization including CPU, GPU, memory, per-node breakdowns,
per-GPU utilization via Prometheus metrics, running job status, and live actor counts.

Install from `crates.io <https://crates.io/crates/raytop>`_:

.. code-block:: bash

    cargo install raytop

Point ``raytop`` at the Ray dashboard endpoint:

.. code-block:: bash

    raytop --master http://${HEAD_IP}:8265

``raytop`` retrieves data from multiple Ray APIs:

- ``/api/cluster_status`` — cluster-wide CPU/GPU/memory allocation
- ``/api/v0/nodes`` — per-node info and state
- ``/api/prometheus/sd`` — real-time per-GPU utilization and GRAM usage
- ``/api/jobs/`` — running jobs
- ``/api/v0/actors`` — actor counts per node

Use ``j``/``k`` or arrow keys to navigate nodes, ``Enter`` to open the detail panel,
``Tab`` to switch focus between jobs and nodes, ``t`` to cycle themes, and ``q`` to quit.

For monitoring long-running distributed training jobs (e.g., verl RLHF, Megatron
pretraining), ``raytop`` provides a quick way to verify that all nodes are active,
GPUs are fully utilized, and no actors have crashed — without needing to open the
Ray dashboard in a browser.

Submit a Ray Job
----------------

The Ray head address is printed by the sbatch script at the end:

.. code-block:: bash

    # the sbatch script prints this on success:
    # [info] Ray cluster ready at <HEAD_IP>:<RAY_PORT>

    # or query it from inside the head container
    docker exec ray-head ray status

    # or check the Slurm job output log
    cat slurm-<JOB_ID>.out | grep "Ray cluster ready"

Submit jobs using ``ray job submit``:

.. code-block:: bash

    RAY_HEAD=http://<HEAD_IP>:8265

    # submit a job
    ray job submit --address ${RAY_HEAD} -- python train.py

    # submit the Megatron example
    ray job submit --address ${RAY_HEAD} \
        --runtime-env-json='{"working_dir": "/workspace"}' \
        -- python main.py --nodes 2 --gpus-per-node 8

    # submit with runtime env
    ray job submit --address ${RAY_HEAD} \
        --runtime-env-json='{"working_dir": "/workspace"}' \
        -- python train.py

    # list jobs
    ray job list --address ${RAY_HEAD}

    # check job status
    ray job status --address ${RAY_HEAD} JOB_ID

    # get job logs
    ray job logs --address ${RAY_HEAD} JOB_ID

    # stop a job
    ray job stop --address ${RAY_HEAD} JOB_ID

Environment Variables
---------------------

Common environment variables for Ray on HPC clusters:

.. code-block:: bash

    # Ray settings
    export RAY_DEDUP_LOGS=0           # disable log deduplication
    export RAY_USAGE_STATS_ENABLED=0  # disable usage stats

    # NCCL settings for multi-node GPU communication
    export NCCL_DEBUG=INFO
    export NCCL_SOCKET_IFNAME=^docker,lo,veth
    export NCCL_P2P_NET_CHUNKSIZE=524288

    # EFA settings (AWS)
    export FI_PROVIDER=efa
    export FI_EFA_USE_DEVICE_RDMA=1
    export FI_EFA_FORK_SAFE=1
