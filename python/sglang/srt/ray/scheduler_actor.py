# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Ray actor wrapper for SGLang Scheduler."""

from __future__ import annotations

import logging
import os
import threading
from typing import TYPE_CHECKING, Any, Dict, Optional

import ray
import torch
import sys

if TYPE_CHECKING:
    from sglang.srt.server_args import PortArgs, ServerArgs


logging.getLogger().addHandler(logging.StreamHandler(sys.stderr))
logger = logging.getLogger("ray.serve")


@ray.remote
class SchedulerActor:
    """Ray actor wrapper for SGLang Scheduler.

    Each actor manages one GPU and runs the Scheduler + TpModelWorker stack.
    Ray is used for process lifecycle; ZMQ handles request/response communication.

    max_concurrency=2 allows weight sync methods (pull_weights, etc.)
    to run while run_event_loop() is active.
    """

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        gpu_id: int,
        tp_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        moe_ep_rank: int,
        pp_rank: int,
        dp_rank: Optional[int],
        dist_init_addr: Optional[str] = None,
    ):
        import dataclasses

        from sglang.srt.managers.scheduler import Scheduler, configure_scheduler

        # Override dist_init_addr if provided (for multi-node)
        if dist_init_addr:
            server_args = dataclasses.replace(
                server_args, dist_init_addr=dist_init_addr
            )

        # Get actual GPU IDs from Ray runtime context
        accelerator_ids = ray.get_runtime_context().get_accelerator_ids()
        assigned_gpus = accelerator_ids.get("GPU", [])

        if assigned_gpus:
            # Ray assigned specific GPU(s), use the first one
            actual_gpu_id = int(assigned_gpus[0])
        else:
            # Fallback to passed gpu_id
            actual_gpu_id = gpu_id

        # Configure worker (logging, process title, etc.)
        dp_rank = configure_scheduler(
            server_args,
            tp_rank,
            attn_cp_rank,
            moe_dp_rank,
            moe_ep_rank,
            pp_rank,
            dp_rank,
        )

        # Create scheduler (loads model into GPU, initializes NCCL)
        self.scheduler = Scheduler(
            server_args=server_args,
            port_args=port_args,
            gpu_id=actual_gpu_id,
            tp_rank=tp_rank,
            moe_ep_rank=moe_ep_rank,
            pp_rank=pp_rank,
            attn_cp_rank=attn_cp_rank,
            moe_dp_rank=moe_dp_rank,
            dp_rank=dp_rank,
        )

        self._tp_rank = tp_rank
        self._pp_rank = pp_rank

    def get_info(self) -> Dict[str, Any]:
        """Return scheduler initialization info for handshake."""
        return self.scheduler.get_init_info()

    def run_event_loop(self) -> None:
        """Run the scheduler's event loop. Blocks until shutdown."""
        try:
            import torch

            # Need to set the GPU id for the event loop for nccl to work
            torch.cuda.set_device(self.scheduler.gpu_id)
            self.scheduler.run_event_loop()
        except Exception as e:
            logger.error("Scheduler PP%s TP%s crashed: %s", self._pp_rank, self._tp_rank, e)
            raise

    # ------------------------------------------------------------------
    # Weight sync via RDT
    # ------------------------------------------------------------------

    def get_param_layout(self):
        """Return shard config for this TP rank (fetched once at connect time)."""
        from sglang.srt.ray.weight_sync import (
            ParamInfo,
            ParamLayout,
            build_sharding_recipes,
        )

        model_runner = self.scheduler.tp_worker.model_runner
        model = model_runner.model
        tp_size = model_runner.tp_size

        param_info = {
            name: ParamInfo(shape=list(param.data.shape), dtype=param.data.dtype)
            for name, param in model.named_parameters()
        }
        sharding_recipes = build_sharding_recipes(model, self._tp_rank, tp_size)

        return ParamLayout(
            tp_rank=self._tp_rank,
            tp_size=tp_size,
            param_info=param_info,
            sharding_recipes=sharding_recipes,
            stacked_params_mapping=getattr(model, "stacked_params_mapping", []),
            num_attention_heads=getattr(model.config, "num_attention_heads", None),
            num_key_value_heads=getattr(model.config, "num_key_value_heads", None),
            hidden_size=getattr(model.config, "hidden_size", None),
        )


    def pull_weights(self, trainer_handle, param_names: list, tp_rank: int) -> bool:
        """Pull pre-sharded weight bucket from trainer via RDT zero-copy.

        Uses set_target_for_ref to RDMA directly into param.data buffers,
        eliminating intermediate receive buffers and copy operations.
        """
        torch.cuda.set_device(self.scheduler.gpu_id)  # Guard for max_concurrency=2
        ref = trainer_handle.export_weights_rdt.remote(tp_rank)
        model = self.scheduler.tp_worker.model_runner.model
        params_dict = dict(model.named_parameters())
        target_buffers = [params_dict[name].data for name in param_names]
        ray.experimental.set_target_for_ref(ref, target_buffers)
        ray.get(ref)
        return True

    def get_param(self, name: str) -> torch.Tensor:
        """Get a model parameter by name (for verification). Returns on CPU."""
        model_runner = self.scheduler.tp_worker.model_runner
        tensor = model_runner.get_weights_by_name(name, truncate_size=0)
        if tensor is not None:
            return tensor.cpu()
        for pname, param in model_runner.model.named_parameters():
            if pname == name:
                return param.data.cpu()
        raise KeyError(f"Parameter {name!r} not found")
