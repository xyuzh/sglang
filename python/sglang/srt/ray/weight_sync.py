"""RDT weight sync utilities for SchedulerActor.

Handles receiving a flattened tensor via Ray Direct Transport (RDT),
reconstructing individual named tensors from metadata, and loading
them into the model runner.

Also defines ParamLayout / ParamShardingRecipe dataclasses that describe
how to shard HF-checkpoint tensors for each TP rank.  The trainer applies
recipes mechanically — no model-specific sharding logic needed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List

import ray
import torch

from sglang.srt.weight_sync.tensor_bucket import (
    FlattenedTensorBucket,
    FlattenedTensorMetadata,
)

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses for shard layout & sharding recipes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParamInfo:
    """Metadata for a single (already-sharded) model parameter."""

    shape: list[int]
    dtype: torch.dtype


@dataclass(frozen=True)
class NarrowOp:
    """A single narrow (slice) operation on a tensor."""

    dim: int
    start: int
    length: int


@dataclass(frozen=True)
class ParamShardingRecipe:
    """Describes how to obtain one TP rank's shard from a full HF tensor.

    For stacked params (QKV, gate_up), multiple HF tensors share the same
    ``concat_group`` (the internal param name).  After narrowing each
    component, the trainer concatenates them in ``concat_order`` along dim 0.
    """

    hf_name: str
    internal_name: str
    narrow_ops: list[NarrowOp]
    concat_group: str | None = None
    concat_order: int = 0


@dataclass
class ParamLayout:
    """Per-TP-rank layout returned by SchedulerActor.get_param_layout().

    ``sharding_recipes`` is the primary mechanism for the trainer to shard
    HF tensors.  The remaining fields are kept for diagnostics / backward
    compat.
    """

    tp_rank: int
    tp_size: int
    param_info: dict[str, ParamInfo]
    sharding_recipes: dict[str, ParamShardingRecipe] = field(default_factory=dict)

    # Legacy fields (used by older PreShardedBucketBuilder, kept for compat)
    stacked_params_mapping: list = field(default_factory=list)
    num_attention_heads: int | None = None
    num_key_value_heads: int | None = None
    hidden_size: int | None = None


# ---------------------------------------------------------------------------
# Recipe builder — runs inside the sglang process where the model is loaded
# ---------------------------------------------------------------------------


def build_sharding_recipes(
    model: torch.nn.Module,
    tp_rank: int,
    tp_size: int,
) -> dict[str, ParamShardingRecipe]:
    """Walk the model and build per-HF-name sharding recipes for one TP rank.

    Returns a dict mapping HF checkpoint weight names to ``ParamShardingRecipe``
    objects.  The trainer uses these to ``narrow()`` full HF tensors into the
    exact slices expected by each TP rank.
    """
    from sglang.srt.layers.linear import (
        ColumnParallelLinear,
        MergedColumnParallelLinear,
        QKVParallelLinear,
        RowParallelLinear,
    )

    recipes: dict[str, ParamShardingRecipe] = {}

    # Module lookup by name
    modules_by_name = dict(model.named_modules())

    # Build reverse stacked-params mapping:
    #   param_suffix -> [(weight_suffix, shard_id), ...]  (preserves order)
    stacked_params_mapping = getattr(model, "stacked_params_mapping", [])
    stacked_reverse: dict[str, list[tuple[str, str | int]]] = {}
    for param_suffix, weight_suffix, shard_id in stacked_params_mapping:
        stacked_reverse.setdefault(param_suffix, []).append((weight_suffix, shard_id))

    for internal_name, param in model.named_parameters():
        # Resolve parent module: "a.b.c.weight" → module "a.b.c"
        parts = internal_name.rsplit(".", 1)
        if len(parts) != 2:
            continue
        module_path, _param_attr = parts
        module = modules_by_name.get(module_path)
        if module is None:
            continue

        # --- Check if this param belongs to a stacked group ---
        matched_suffix: str | None = None
        for param_suffix in stacked_reverse:
            if param_suffix in internal_name:
                matched_suffix = param_suffix
                break

        if matched_suffix is not None:
            _add_stacked_recipes(
                recipes, internal_name, param, module,
                matched_suffix, stacked_reverse[matched_suffix],
                tp_rank, tp_size,
            )
        else:
            _add_direct_recipe(
                recipes, internal_name, param, module,
                tp_rank, tp_size,
            )

    return recipes


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _add_stacked_recipes(
    recipes: dict[str, ParamShardingRecipe],
    internal_name: str,
    param: torch.nn.Parameter,
    module: torch.nn.Module,
    matched_suffix: str,
    hf_variants: list[tuple[str, str | int]],
    tp_rank: int,
    tp_size: int,
) -> None:
    """Generate one recipe per HF component for a stacked (fused) param."""
    from sglang.srt.layers.linear import (
        MergedColumnParallelLinear,
        QKVParallelLinear,
    )

    output_dim = getattr(param, "output_dim", 0)

    for order, (hf_suffix, shard_id) in enumerate(hf_variants):
        hf_name = internal_name.replace(matched_suffix, hf_suffix)

        if isinstance(module, QKVParallelLinear):
            narrow_ops = _qkv_narrow_ops(module, output_dim, shard_id, tp_rank)
        elif isinstance(module, MergedColumnParallelLinear):
            narrow_ops = _merged_column_narrow_ops(
                module, output_dim, shard_id, tp_rank, tp_size,
            )
        else:
            # Fallback: treat as plain column shard
            shard_size = param.data.shape[output_dim] // len(hf_variants)
            narrow_ops = [NarrowOp(dim=output_dim, start=tp_rank * shard_size, length=shard_size)]

        recipes[hf_name] = ParamShardingRecipe(
            hf_name=hf_name,
            internal_name=internal_name,
            narrow_ops=narrow_ops,
            concat_group=internal_name,
            concat_order=order,
        )


def _add_direct_recipe(
    recipes: dict[str, ParamShardingRecipe],
    internal_name: str,
    param: torch.nn.Parameter,
    module: torch.nn.Module,
    tp_rank: int,
    tp_size: int,
) -> None:
    """Generate a recipe for a non-stacked param (HF name == internal name)."""
    from sglang.srt.layers.linear import RowParallelLinear

    narrow_ops: list[NarrowOp] = []

    if isinstance(module, RowParallelLinear):
        input_dim = getattr(param, "input_dim", None)
        if input_dim is not None:
            shard_size = param.data.shape[input_dim]
            narrow_ops = [NarrowOp(dim=input_dim, start=tp_rank * shard_size, length=shard_size)]
    else:
        output_dim = getattr(param, "output_dim", None)
        if output_dim is not None:
            shard_size = param.data.shape[output_dim]
            narrow_ops = [NarrowOp(dim=output_dim, start=tp_rank * shard_size, length=shard_size)]

    recipes[internal_name] = ParamShardingRecipe(
        hf_name=internal_name,
        internal_name=internal_name,
        narrow_ops=narrow_ops,
    )


def _qkv_narrow_ops(
    module,  # QKVParallelLinear
    output_dim: int,
    shard_id: str,
    tp_rank: int,
) -> list[NarrowOp]:
    """Compute narrow ops for one Q/K/V component from the full HF tensor."""
    if shard_id == "q":
        shard_size = module.q_proj_shard_size
        start = tp_rank * shard_size
    elif shard_id == "k":
        shard_size = module.kv_proj_shard_size
        start = (tp_rank // module.num_kv_head_replicas) * shard_size
    elif shard_id == "v":
        shard_size = module.v_proj_shard_size
        start = (tp_rank // module.num_kv_head_replicas) * shard_size
    else:
        raise ValueError(f"Unknown QKV shard_id: {shard_id!r}")

    return [NarrowOp(dim=output_dim, start=start, length=shard_size)]


def _merged_column_narrow_ops(
    module,  # MergedColumnParallelLinear
    output_dim: int,
    shard_id: int,
    tp_rank: int,
    tp_size: int,
) -> list[NarrowOp]:
    """Compute narrow ops for one component of a MergedColumnParallelLinear."""
    shard_size = module.output_sizes[shard_id] // tp_size
    start = tp_rank * shard_size
    return [NarrowOp(dim=output_dim, start=start, length=shard_size)]


# ---------------------------------------------------------------------------
# RDT receive path (unchanged)
# ---------------------------------------------------------------------------


def receive_and_load_weights_rdt(
    model_runner: ModelRunner,
    weights_ref: ray.ObjectRef,
    metadata: List[FlattenedTensorMetadata],
) -> bool:
    """Pull flattened tensor via RDT, reconstruct, and load into model.

    Args:
        model_runner: The SGLang ModelRunner that owns the model.
        weights_ref: ObjectRef to the flattened GPU tensor (transferred via RDT/NIXL).
        metadata: List of FlattenedTensorMetadata describing tensor names, shapes,
            dtypes, and offsets in the flattened buffer.

    Returns:
        True on success.
    """
    flattened_tensor: torch.Tensor = ray.get(weights_ref)

    bucket = FlattenedTensorBucket(flattened_tensor=flattened_tensor, metadata=metadata)
    named_tensors = bucket.reconstruct_tensors()

    logger.info(
        f"RDT weight sync: received {len(named_tensors)} tensors, "
        f"flat size {flattened_tensor.numel()} bytes"
    )

    model_runner.model.load_weights(named_tensors)
    return True
