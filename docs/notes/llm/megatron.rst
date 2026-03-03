.. meta::
    :description lang=en: Megatron Bridge cheat sheet — pretrain recipes, Nsys profiling, and distributed training on SLURM with EFA.
    :keywords: Megatron, Megatron-LM, Megatron Bridge, distributed training, pretrain, Nsys, profiling, EFA, SLURM, DeepSeek, MoE, NCCL, GPU, LLM

===========
Megatron-LM
===========

.. contents:: Table of Contents
    :backlinks: none

`Megatron-LM <https://github.com/NVIDIA/Megatron-LM/tree/main>`_ and
`Megatron Bridge <https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main>`_
provides a recipe-based interface for configuring and launching Megatron-LM pretrain
jobs. Instead of passing dozens of CLI flags, you define a Python recipe that returns a
config object, and an entrypoint script handles the rest. Recipes can load HuggingFace
pretrained weights directly via the ``hf_path`` parameter, so you can start from any
checkpoint on the `HuggingFace Hub <https://huggingface.co/models>`_ without manual
conversion.

The examples below use the scripts in
`src/megatron <https://github.com/crazyguitar/pysheeet/blob/master/src/megatron>`__.

How to Use Megatron Bridge
--------------------------

The container image is built with Docker and exported as an
`enroot <https://github.com/NVIDIA/enroot>`_ squashfs (``.sqsh``) file. Enroot is a
lightweight container runtime designed for HPC — it converts Docker images into
unprivileged sandboxes that integrate with SLURM via the
`pyxis <https://github.com/NVIDIA/pyxis>`_ plugin. When ``srun.sh`` runs, it passes
``--container-image`` and ``--container-mounts`` to ``srun``, and pyxis handles importing
the ``.sqsh`` and launching each task inside the container.

.. code-block:: bash

    # Build Docker image and export to enroot sqsh
    make build

    # This produces megatron-lm+latest.sqsh in the current directory.
    # srun.sh picks it up via the SQSH env var (default: ./megatron-lm+latest.sqsh).

You can override the image path and container mounts with environment variables:

.. list-table::
   :header-rows: 1

   * - Variable
     - Default
     - Description
   * - ``SQSH``
     - ``./megatron-lm+latest.sqsh``
     - Path to enroot image
   * - ``MOUNT``
     - ``.:/workspace/megatron,/fsx:/fsx``
     - Container mounts
   * - ``GPUS_PER_NODE``
     - ``8``
     - GPUs per node

A recipe is a Python file that calls a Megatron Bridge config function and returns a
``ConfigContainer``. For example,
`deepseek_v2_lite_pretrain.py <https://github.com/crazyguitar/pysheeet/blob/master/src/megatron/recipes/deepseek_v2_lite_pretrain.py>`__:

.. code-block:: python

    from megatron.bridge.recipes.deepseek.deepseek_v2 import (
        deepseek_v2_lite_pretrain_config,
    )


    def configure(hf_path=None, moe_token_dispatcher_type=None):
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
        if moe_token_dispatcher_type == "deepep":
            cfg.model.moe_token_dispatcher_type = "flex"
            cfg.model.moe_flex_dispatcher_backend = "deepep"
            cfg.model.moe_enable_deepep = True
            cfg.model.moe_shared_expert_overlap = False
        return cfg

Launch a pretrain job with ``srun.sh``, which wraps ``srun`` + pyxis to run inside the
enroot container:

.. code-block:: bash

    # 2-node DeepSeek V2 Lite pretrain
    salloc -N 2
    ./srun.sh recipes/deepseek_v2_lite_pretrain.py \
        hf_path=/fsx/models/deepseek-ai/DeepSeek-V2-Lite

    # Override config with Hydra-style args
    ./srun.sh recipes/deepseek_v2_lite_pretrain.py \
        train.train_iters=1000

The `entrypoint.py <https://github.com/crazyguitar/pysheeet/blob/master/src/megatron/entrypoint.py>`__
loads the recipe, applies CLI overrides, and calls ``pretrain()``. The ``srun.sh`` script
sets up EFA environment variables (``FI_PROVIDER=efa``, ``NCCL_NET_PLUGIN``, etc.) and
maps ``SLURM_PROCID`` / ``SLURM_LOCALID`` to ``RANK`` / ``LOCAL_RANK`` so that
Megatron's distributed init works without ``torchrun``.

How to Enable Nsys Profiling
----------------------------

Pass ``--nsys`` to ``srun.sh`` and set the profiling overrides in the recipe:

.. code-block:: bash

    ./srun.sh --nsys recipes/deepseek_v2_lite_pretrain.py \
        hf_path=/fsx/models/deepseek-ai/DeepSeek-V2-Lite \
        profiling.use_nsys_profiler=true \
        profiling.profile_step_start=10 \
        profiling.profile_step_end=15 \
        profiling.profile_ranks=[0]

When ``--nsys`` is enabled, ``srun.sh`` prepends ``nsys profile`` with
``--capture-range=cudaProfilerApi`` so that only the steps between
``profile_step_start`` and ``profile_step_end`` are captured. The output
``.nsys-rep`` files are written to ``nsys-megatron/`` inside the container mount.

On AWS, if you want to monitor EFA network traffic in the Nsys timeline, add
``--enable efa_metrics`` to the nsys command (already included in ``srun.sh``). For a
detailed walkthrough on monitoring EFA with NCCL GIN and Nsys, refer to
:doc:`/notes/appendix/megatron-efa-monitoring`.

Why ``python`` Instead of ``torchrun``
--------------------------------------

The ``srun.sh`` script launches ``python3 entrypoint.py`` directly rather than using
``torchrun``. This is intentional. ``torchrun`` spawns worker processes via
``multiprocessing``, and the spawn boundary can interfere with Nsys profiling — the
profiler sometimes fails to capture the ``cudaProfilerStart`` and ``cudaProfilerStop``
calls issued by the child process, resulting in empty or incomplete traces.

By running ``python`` directly under ``srun --ntasks-per-node=<GPUS>``, each GPU gets
its own process managed by SLURM. The ``RANK``, ``LOCAL_RANK``, and ``WORLD_SIZE``
environment variables are derived from ``SLURM_PROCID``, ``SLURM_LOCALID``, and
``SLURM_NTASKS`` respectively. This avoids the spawn layer entirely, giving Nsys (and
other profilers like VizTracer) a clean, single-process view of each rank.

Custom Profilers (VizTracer)
----------------------------

Megatron Bridge's profiling hooks can be extended to support custom profilers. The
`viztracer_plugin.py <https://github.com/crazyguitar/pysheeet/blob/master/src/megatron/viztracer_plugin.py>`__
shows how to do this by monkey-patching ``megatron.bridge.training.profiling``:

1. Add a new field (e.g. ``use_viztracer``) to ``ProfilingConfig`` via
   ``dataclasses.field`` so OmegaConf recognizes it as a valid override.
2. Patch ``handle_profiling_step`` to start the profiler at ``profile_step_start``.
3. Patch ``handle_profiling_stop`` to stop and save at ``profile_step_end``.

The plugin is loaded in
`entrypoint.py <https://github.com/crazyguitar/pysheeet/blob/master/src/megatron/entrypoint.py>`__
before any Megatron imports:

.. code-block:: python

    import viztracer_plugin
    viztracer_plugin.install()

Then pass the override on the command line:

.. code-block:: bash

    ./srun.sh recipes/deepseek_v2_lite_pretrain.py \
        hf_path=/fsx/models/deepseek-ai/DeepSeek-V2-Lite \
        profiling.use_viztracer=true \
        profiling.profile_step_start=10 \
        profiling.profile_step_end=15 \
        profiling.profile_ranks=[0]

The same pattern works for any profiler — implement ``start()`` / ``stop()`` / ``save()``
in the patched hooks and register a config field so it can be toggled via Hydra overrides.
