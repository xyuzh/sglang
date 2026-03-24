"""RDT weight sync utilities for SchedulerActor.

Handles receiving a flattened tensor via Ray Direct Transport (RDT),
reconstructing individual named tensors from metadata, and loading
them into the model runner.

Also defines ParamLayout / ParamShardingRecipe dataclasses that describe
how to map HF-checkpoint tensors to internal model parameters.  The trainer
applies recipes mechanically — no model-specific sharding logic needed.

Note: This module assumes inference TP=1 (tp_size=1, tp_rank=0).  All
TP-sharding narrow ops have been removed; only the stacked-params concat
logic (for fused QKV / gate_up) is retained.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

# ---------------------------------------------------------------------------
# Dataclasses for shard layout & sharding recipes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParamShardingRecipe:
    """Describes how to obtain an internal param from HF tensors.

    For stacked params (QKV, gate_up), multiple HF tensors share the same
    ``concat_group`` (the internal param name).  The trainer concatenates
    them in ``concat_order`` along dim 0.
    """

    hf_name: str
    internal_name: str
    concat_group: str | None = None
    concat_order: int = 0


@dataclass
class ParamLayout:
    """Layout returned by SchedulerActor.get_param_layout().

    ``sharding_recipes`` maps HF weight names to recipes describing how
    the trainer should assemble internal parameters.
    """

    sharding_recipes: dict[str, ParamShardingRecipe] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Recipe builder — runs inside the sglang process where the model is loaded
# ---------------------------------------------------------------------------


def build_sharding_recipes(
    model: torch.nn.Module,
) -> dict[str, ParamShardingRecipe]:
    """Walk the model and build per-HF-name sharding recipes (assumes tp=1).

    Returns a dict mapping HF checkpoint weight names to ``ParamShardingRecipe``
    objects.  For stacked params, multiple HF names map to recipes that share
    a ``concat_group``; the trainer concatenates them in order.
    """
    recipes: dict[str, ParamShardingRecipe] = {}

    # Build reverse stacked-params mapping:
    #   param_suffix -> [(weight_suffix, shard_id), ...]  (preserves order)
    stacked_params_mapping = getattr(model, "stacked_params_mapping", [])
    stacked_reverse: dict[str, list[tuple[str, str | int]]] = {}
    for param_suffix, weight_suffix, shard_id in stacked_params_mapping:
        stacked_reverse.setdefault(param_suffix, []).append((weight_suffix, shard_id))

    for internal_name, param in model.named_parameters():
        # --- Check if this param belongs to a stacked group ---
        matched_suffix: str | None = None
        for param_suffix in stacked_reverse:
            if param_suffix in internal_name:
                matched_suffix = param_suffix
                break

        if matched_suffix is not None:
            for order, (hf_suffix, shard_id) in enumerate(stacked_reverse[matched_suffix]):
                hf_name = internal_name.replace(matched_suffix, hf_suffix)
                recipes[hf_name] = ParamShardingRecipe(
                    hf_name=hf_name,
                    internal_name=internal_name,
                    concat_group=internal_name,
                    concat_order=order,
                )
        else:
            recipes[internal_name] = ParamShardingRecipe(
                hf_name=internal_name,
                internal_name=internal_name,
            )

    return recipes
