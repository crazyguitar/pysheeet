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
