# Megatron

```bash
make build

# Launch a 2-node DeepSeek V2 Lite pretrain job:
salloc -N 2
./srun.sh recipes/deepseek_v2_lite_pretrain.py \
    hf_path=/fsx/models/deepseek-ai/DeepSeek-V2-Lite

# Launch DeepSeek V2 pretrain (4 nodes):
salloc -N 4
./srun.sh recipes/deepseek_v2_pretrain.py \
    hf_path=/fsx/models/deepseek-ai/DeepSeek-V2

# Override config with Hydra-style args:
./srun.sh recipes/deepseek_v2_lite_pretrain.py \
    train.train_iters=1000

# Launch DeepSeek-V2-Lite using DeepEP with NCCL GIN
./srun.sh recipes/deepseek_v2_lite_pretrain.py \
    hf_path=/fsx/models/deepseek-ai/DeepSeek-V2-Lite \
    moe_token_dispatcher_type=deepep \
    model.tensor_model_parallel_size=1 \
    model.expert_model_parallel_size=64 \
    model.sequence_parallel=false

# Nsys profile
./srun.sh --nsys recipes/deepseek_v2_lite_pretrain.py \
  hf_path=/fsx/hf_pretrained_models/deepseek-ai/DeepSeek-V2-Lite \
  moe_token_dispatcher_type=deepep \
  model.tensor_model_parallel_size=1 \
  model.expert_model_parallel_size=64 \
  model.sequence_parallel=false \
  profiling.use_nsys_profiler=true \
  profiling.profile_step_start=10 \
  profiling.profile_step_end=15 \
  profiling.profile_ranks=[0]

# Viztracer profile
./srun.sh recipes/deepseek_v2_lite_pretrain.py \
  hf_path=/fsx/models/deepseek-ai/DeepSeek-V2-Lite \
  profiling.use_viztracer=true \
  profiling.profile_step_start=10 \
  profiling.profile_step_end=15 \
  profiling.profile_ranks=[0]
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `SQSH` | `./megatron-lm+latest.sqsh` | Path to enroot image |
| `MOUNT` | `.:/workspace/megatron,/fsx:/fsx` | Container mounts |
| `GPUS_PER_NODE` | `8` | GPUs per node |
