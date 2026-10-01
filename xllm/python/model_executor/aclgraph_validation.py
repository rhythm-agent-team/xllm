# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Opt-in scalar observations; never retain tensors or create graph pools."""

from __future__ import annotations

import json
import os
import time
from typing import Any

import torch
import torch.nn as nn

from scripts.logger import logger
from xllm.python.attention.backend import AttentionMetadata

# Read once before execution. Only the exact value 1 enables validation.
VALIDATION_ENABLED = os.environ.get("XLLM_ACLGRAPH_VALIDATION", "0") == "1"


def model_fields(model: nn.Module, device: torch.device, config: dict | None = None) -> dict[str, Any]:
    cfg = getattr(model, "cfg", None)
    values = (
        config
        if config is not None
        else {name: getattr(cfg, name, None) for name in ("model_type", "rank", "tp_rank", "dp_rank")}
    )
    model_type = values.get("model_type")
    draft = values.get("is_draft_engine", False) or (
        isinstance(model_type, str) and (model_type.endswith("_mtp") or "Draft" in model_type)
    )
    return {
        "model": type(model).__name__,
        "model_type": model_type,
        "model_role": "draft" if draft else "target" if model_type is not None else "unknown",
        "device": str(device),
        "rank": values.get("rank"),
        "tp_rank": values.get("tp_rank"),
        "dp_rank": values.get("dp_rank", 0),
    }


def execution_fields(
    metadata: AttentionMetadata,
    actual_rows: int,
    dp_rank: int,
    *,
    row_expanded: bool,
) -> dict[str, Any]:
    prepared = getattr(metadata, "prepared_attention_state", None) is not None
    expanded = getattr(metadata, "expanded_decode_metadata", None)
    expanded = expanded is not None and bool(getattr(expanded, "enabled", True))
    prefill = bool(metadata.is_prefill or metadata.is_chunked_prefill)
    verify = bool(getattr(metadata, "is_spec_verify", False))
    counts = tuple(getattr(metadata, "dp_global_sequence_nums", ()))
    logical, source = None, "unknown_row_layout"
    if counts:
        if not 0 <= dp_rank < len(counts) or any(count < 0 for count in counts):
            raise ValueError(f"invalid validation DP sequence counts: rank={dp_rank}, counts={counts}")
        logical, source = counts[dp_rank], "dp_global_sequence_nums"
    elif not prepared and (prefill or not (row_expanded or verify or expanded)):
        # Native NPU prefill/chunked lengths are per-request; prepared Slots,
        # ordinary verification and draft repair metadata can be per-token.
        lengths = getattr(metadata, "q_seq_lens", None)
        source = "q_seq_lens"
        if lengths is None:
            lengths, source = getattr(metadata, "kv_seq_lens", None), "kv_seq_lens"
        logical = None if lengths is None else lengths.numel()
    q_lengths = getattr(metadata, "q_seq_lens", None)
    kv_lengths = getattr(metadata, "kv_seq_lens", None)
    phase = "decode" if verify or expanded else "prefill" if prefill else "decode"
    return {
        "phase": "mixed" if getattr(metadata, "is_mixed", False) else phase,
        "is_prefill": bool(metadata.is_prefill),
        "is_chunked_prefill": bool(metadata.is_chunked_prefill),
        "is_spec_verify": verify,
        "prepared_metadata": prepared,
        "expanded_metadata": expanded,
        "actual_rows": actual_rows,
        "logical_sequences": logical,
        "logical_sequence_source": source,
        "dp_global_sequence_nums": counts,
        "query_length_items": None if q_lengths is None else q_lengths.numel(),
        "kv_length_items": None if kv_lengths is None else kv_lengths.numel(),
    }


def log_event(event: str, fields: dict[str, Any]) -> None:
    if VALIDATION_ENABLED:
        record = {"event": event, "wall_time_ns": time.time_ns(), **fields}
        logger.info("XLLM_ACLGRAPH_VALIDATION %s", json.dumps(record, sort_keys=True, separators=(",", ":")))


def log_capture_memory(stage: str, device: torch.device, fields: dict[str, Any]) -> None:
    if VALIDATION_ENABLED:
        memory = {
            name: getattr(torch.npu, name)(device)
            for name in ("memory_allocated", "memory_reserved", "max_memory_allocated", "max_memory_reserved")
        }
        mode = "eager" if stage.endswith("warmup") else "graph"
        log_event("capture_memory", {**fields, "stage": stage, "mode": mode, **memory})
