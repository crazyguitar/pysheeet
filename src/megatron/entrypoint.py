#!/usr/bin/env python3
"""Generic entrypoint for Megatron Bridge recipes.

Usage:
    ./srun.sh recipes/deepseek_v2_lite_pretrain.py hf_path=/fsx/models/deepseek-ai/DeepSeek-V2-Lite
    ./srun.sh recipes/qwen3_30b_a3b_pretrain.py hf_path=/fsx/models/Qwen/Qwen3-30B-A3B-FP8
"""

import importlib.util
import sys

import megatron.core.jit as _jit
if not hasattr(_jit, "disable_jit_fuser"):
    _jit.disable_jit_fuser = lambda: None

import viztracer_plugin
viztracer_plugin.install()

from omegaconf import OmegaConf

from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.training.utils.omegaconf_utils import (
    apply_overrides,
    create_omegaconf_dict_config,
    parse_hydra_overrides,
)


def load_recipe(path, **kwargs):
    """Load a recipe module and call its `configure()` function."""
    spec = importlib.util.spec_from_file_location("recipe", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.configure(**kwargs)


def parse_cli_overrides(cfg, args):
    """Apply Hydra-style CLI overrides (key=value) to a ConfigContainer."""
    if args:
        omega_conf, excluded = create_omegaconf_dict_config(cfg)
        omega_conf = parse_hydra_overrides(omega_conf, args)
        apply_overrides(cfg, OmegaConf.to_container(omega_conf, resolve=True), excluded)


def main() -> None:
    recipe_path = sys.argv[1]
    overrides = sys.argv[2:]

    recipe_kwargs = {}
    remaining = []
    for o in overrides:
        if o.startswith("hf_path="):
            recipe_kwargs["hf_path"] = o.split("=", 1)[1]
        elif o.startswith("moe_token_dispatcher_type="):
            recipe_kwargs["moe_token_dispatcher_type"] = o.split("=", 1)[1]
        else:
            remaining.append(o)

    cfg = load_recipe(recipe_path, **recipe_kwargs)
    parse_cli_overrides(cfg, remaining)
    pretrain(config=cfg, forward_step_func=forward_step)


if __name__ == "__main__":
    main()
