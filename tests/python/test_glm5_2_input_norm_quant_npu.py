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

"""Direct NPU checks for GLM shared input quantization and MLA slot reuse."""

from __future__ import annotations

import math
from typing import Any

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from scripts.logger import logger

torch_npu = pytest.importorskip("torch_npu")

_TOKENS = 4
_HIDDEN_SIZE = 6144
_Q_LORA_RANK = 1536
_KV_LORA_RANK = 512
_ROPE_DIM = 64
_NOPE_DIM = 192
_HEADS = 4
_EPS = 1e-5
_SENTINEL = -17.0
# Fusing before the BF16 normalized intermediate changes rounding before INT8
# quantization. These bounds are declared for that complete projection chain,
# not borrowed from the independent attention matmul's per-element tolerance.
_SCALE_MAX_RELATIVE_ERROR = 1e-2
_PROJECTION_MAX_RELATIVE_ERROR = 2e-2
_PROJECTION_RELATIVE_L2_ERROR = 1e-2


@pytest.fixture(scope="module")
def npu_device() -> torch.device:
    assert torch.npu.is_available(), "input norm/MLA tests require an Ascend NPU"
    return torch.device("npu")


def _dynamic_projection(in_features: int, out_features: int, device: torch.device, seed: int) -> Any:
    from xllm.python.models.deepseek_v32 import W8A8AttentionLinear

    generator = torch.Generator().manual_seed(seed)
    projection = W8A8AttentionLinear(in_features, out_features, device)
    projection._set_dynamic_activation(True)
    projection.weight.data.copy_(
        torch.randint(-4, 5, (out_features, in_features), generator=generator, dtype=torch.int8)
    )
    projection.weight_scale.fill_(0.02)
    projection.weight_offset.zero_()
    projection.process_weights_after_loading()
    return projection


def _norm_inputs(kind: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(20260930)
    hidden = torch.randn(_TOKENS, _HIDDEN_SIZE, generator=generator).to(torch.bfloat16)
    residual = torch.randn(_TOKENS, _HIDDEN_SIZE, generator=generator).to(torch.bfloat16)
    if kind == "cancellation":
        residual = (-hidden.float() + 0.01 * residual.float()).to(torch.bfloat16)
    elif kind == "magnitudes":
        magnitudes = torch.tensor([1e-3, 1e-1, 10.0, 100.0]).unsqueeze(1)
        hidden = (hidden.float() * magnitudes).to(torch.bfloat16)
        residual = (residual.float() * magnitudes).to(torch.bfloat16)
    elif kind != "random":
        raise ValueError(f"unknown norm input: {kind}")
    weight = (1.0 + 0.1 * torch.randn(_HIDDEN_SIZE, generator=generator)).to(torch.bfloat16)
    return hidden.to(device), residual.to(device), weight.to(device)


def _max_relative_error(actual: torch.Tensor, reference: torch.Tensor) -> float:
    actual, reference = actual.cpu().float(), reference.cpu().float()
    return (actual - reference).abs().max().item() / max(reference.abs().max().item(), 1e-12)


@pytest.mark.parametrize("kind", ("random", "cancellation", "magnitudes"))
def test_shared_input_norm_quant_projection(npu_device: torch.device, kind: str) -> None:
    from xllm.python import kernels

    hidden, residual, weight = _norm_inputs(kind, npu_device)
    original_hidden, original_residual = hidden.cpu(), residual.cpu()
    projection = _dynamic_projection(_HIDDEN_SIZE, _KV_LORA_RANK + _ROPE_DIM + _Q_LORA_RANK, npu_device, 11)
    normalized, legacy_residual = kernels.fused_add_rms_norm(hidden.clone(), residual.clone(), weight, _EPS)
    legacy_int8, legacy_scale = kernels.dynamic_quant(normalized)
    actual_int8, actual_scale, actual_residual = kernels.fused_add_rms_norm_dynamic_quant(
        hidden, residual, weight, _EPS
    )
    assert legacy_scale is not None
    assert actual_int8.shape == hidden.shape and actual_int8.dtype == torch.int8
    assert actual_scale.dtype == torch.float32 and actual_scale.numel() == _TOKENS
    assert actual_scale.device == hidden.device
    torch.testing.assert_close(actual_residual.cpu(), legacy_residual.cpu(), rtol=0, atol=0)

    reference = projection.forward_quantized(legacy_int8, legacy_scale.reshape(-1))
    actual = projection.forward_quantized(actual_int8, actual_scale.reshape(-1))
    assert actual.dtype == torch.bfloat16
    assert torch.isfinite(actual).all()
    scale_error = _max_relative_error(actual_scale.reshape(-1), legacy_scale.reshape(-1))
    max_error = _max_relative_error(actual, reference)
    delta = actual.cpu().float() - reference.cpu().float()
    l2_error = delta.norm().item() / max(reference.cpu().float().norm().item(), 1e-12)
    code_mismatches = (actual_int8.cpu() != legacy_int8.cpu()).sum().item()
    logger.info(
        "input=%s residual_max_abs=0 codes_mismatched=%d scale_max_relative=%.6g "
        "projection_max_relative=%.6g projection_relative_l2=%.6g",
        kind,
        code_mismatches,
        scale_error,
        max_error,
        l2_error,
    )
    assert scale_error <= _SCALE_MAX_RELATIVE_ERROR
    assert max_error <= _PROJECTION_MAX_RELATIVE_ERROR
    assert l2_error <= _PROJECTION_RELATIVE_L2_ERROR
    torch.testing.assert_close(hidden.cpu(), original_hidden, rtol=0, atol=0)
    torch.testing.assert_close(residual.cpu(), original_residual, rtol=0, atol=0)


def _new_caches(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.full((2, 128, 1, _KV_LORA_RANK), _SENTINEL, dtype=torch.bfloat16, device=device),
        torch.full((2, 128, 1, _ROPE_DIM), _SENTINEL, dtype=torch.bfloat16, device=device),
    )


def _preprocess_inputs(device: torch.device, fuse_q_norm_quant: bool) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(31)
    qkv = _dynamic_projection(_HIDDEN_SIZE, _KV_LORA_RANK + _ROPE_DIM + _Q_LORA_RANK, device, 32)
    q_b = _dynamic_projection(_Q_LORA_RANK, _HEADS * (_NOPE_DIM + _ROPE_DIM), device, 33)
    kv_cache, rope_cache = _new_caches(device)
    return {
        "hidden": torch.randn(_TOKENS, _HIDDEN_SIZE, generator=generator).to(dtype=torch.bfloat16, device=device),
        "qkv_weight": qkv.weight,
        "qkv_weight_scale": qkv.weight_scale,
        "q_norm_weight": torch.ones(_Q_LORA_RANK, dtype=torch.bfloat16, device=device),
        "q_b_weight": q_b.weight,
        "q_b_weight_scale": q_b.weight_scale,
        "w_uk": (torch.randn(_HEADS, _NOPE_DIM, _KV_LORA_RANK, generator=generator) / math.sqrt(_NOPE_DIM)).to(
            dtype=torch.bfloat16, device=device
        ),
        "kv_norm_weight": torch.ones(_KV_LORA_RANK, dtype=torch.bfloat16, device=device),
        "rope_cos": torch.ones(_TOKENS, 1, 1, _ROPE_DIM, dtype=torch.bfloat16, device=device),
        "rope_sin": torch.zeros(_TOKENS, 1, 1, _ROPE_DIM, dtype=torch.bfloat16, device=device),
        "slot_mapping": torch.tensor([0, 3, -1, 130], dtype=torch.int32, device=device),
        "kv_cache": kv_cache,
        "rope_cache": rope_cache,
        "kv_lora_rank": _KV_LORA_RANK,
        "q_lora_rank": _Q_LORA_RANK,
        "num_heads": _HEADS,
        "qk_nope_head_dim": _NOPE_DIM,
        "qk_rope_head_dim": _ROPE_DIM,
        "q_norm_epsilon": _EPS,
        "kv_norm_epsilon": _EPS,
        "fuse_q_norm_quant": fuse_q_norm_quant,
    }


@pytest.mark.parametrize("fuse_q_norm_quant", (False, True))
def test_dynamic_preprocess_accepts_prepared_input_and_slots(npu_device: torch.device, fuse_q_norm_quant: bool) -> None:
    from xllm.python import kernels
    from xllm.python.kernels_npu import mla

    inputs = _preprocess_inputs(npu_device, fuse_q_norm_quant)
    reference = mla.deepseek_mla_preprocess_decode_dynamic(**inputs)
    hidden_int8, hidden_scale = kernels.dynamic_quant(inputs["hidden"])
    assert hidden_scale is not None
    actual_inputs = dict(inputs)
    actual_inputs["hidden"] = hidden_int8
    actual_inputs["hidden_scale"] = hidden_scale.reshape(-1)
    actual_inputs["slot_mapping_int64"] = inputs["slot_mapping"].to(torch.int64)
    actual_inputs["kv_cache"], actual_inputs["rope_cache"] = _new_caches(npu_device)
    actual = mla.deepseek_mla_preprocess_decode_dynamic(**actual_inputs)
    torch.npu.synchronize()

    for output, expected in zip(actual, reference):
        torch.testing.assert_close(output.cpu(), expected.cpu(), rtol=0, atol=0)
    assert actual[0].dtype == (torch.int8 if fuse_q_norm_quant else torch.bfloat16)
    assert actual[1].dtype == torch.float32
    assert actual[2].dtype == actual[3].dtype == torch.bfloat16
    for name in ("kv_cache", "rope_cache"):
        torch.testing.assert_close(actual_inputs[name].cpu(), inputs[name].cpu(), rtol=0, atol=0)
        flattened = actual_inputs[name].cpu().flatten(0, 1)
        assert (flattened[[0, 3, 130]] != _SENTINEL).any()
        untouched = torch.ones(flattened.shape[0], dtype=torch.bool)
        untouched[[0, 3, 130]] = False
        assert (flattened[untouched] == _SENTINEL).all()
    assert inputs["slot_mapping"].dtype == torch.int32

    mode = FakeTensorMode()
    fake_inputs = {
        name: mode.from_tensor(value) if isinstance(value, torch.Tensor) else value
        for name, value in actual_inputs.items()
    }
    with mode:
        fake_outputs = mla.deepseek_mla_preprocess_decode_dynamic(**fake_inputs)
    assert [(value.shape, value.dtype) for value in fake_outputs] == [(value.shape, value.dtype) for value in actual]


@pytest.mark.parametrize("invalid", ("missing_scale", "float_with_scale", "scale_dtype", "scale_rows", "scale_device"))
def test_dynamic_preprocess_rejects_invalid_input_representation(npu_device: torch.device, invalid: str) -> None:
    from xllm.python import kernels
    from xllm.python.kernels_npu import mla

    inputs = _preprocess_inputs(npu_device, True)
    quantized, scale = kernels.dynamic_quant(inputs["hidden"])
    assert scale is not None
    inputs["hidden"] = quantized
    inputs["hidden_scale"] = scale.reshape(-1)
    if invalid == "missing_scale":
        inputs["hidden_scale"] = None
    elif invalid == "float_with_scale":
        inputs["hidden"] = quantized.to(torch.bfloat16)
    elif invalid == "scale_dtype":
        inputs["hidden_scale"] = scale.to(torch.bfloat16)
    elif invalid == "scale_rows":
        inputs["hidden_scale"] = scale.reshape(-1)[:1]
    elif invalid == "scale_device":
        inputs["hidden_scale"] = scale.cpu()
    with pytest.raises(ValueError):
        mla.deepseek_mla_preprocess_decode_dynamic(**inputs)
    for name in ("kv_cache", "rope_cache"):
        assert (inputs[name].cpu() == _SENTINEL).all()


def test_shared_int64_slots_write_only_current_destinations(npu_device: torch.device) -> None:
    from xllm.python.kernels_npu import mla

    assert mla._KV_RMSNORM_ROPE_CACHE is not None, "shared raw-PA slots require npu_kv_rmsnorm_rope_cache"
    generator = torch.Generator().manual_seed(41)
    kv = torch.randn(_TOKENS, _KV_LORA_RANK + _ROPE_DIM, generator=generator).to(
        dtype=torch.bfloat16, device=npu_device
    )
    weight = torch.ones(_KV_LORA_RANK, dtype=torch.bfloat16, device=npu_device)
    cos = torch.ones(_TOKENS, 1, 1, _ROPE_DIM, dtype=torch.bfloat16, device=npu_device)
    sin = torch.zeros_like(cos)
    slots = torch.empty(_TOKENS, dtype=torch.int32, device=npu_device)
    shared_slots = torch.empty(_TOKENS, dtype=torch.int64, device=npu_device)
    for destinations in ([0, 3, -1, 130], [2, 8, 129, -1]):
        slots.copy_(torch.tensor(destinations, dtype=torch.int32))
        shared_slots.copy_(slots)
        for factor in (1.0, 1.5):
            current_kv = kv * factor
            actual_kv, actual_rope = _new_caches(npu_device)
            reference_kv, reference_rope = _new_caches(npu_device)
            arguments = (current_kv, weight, cos, sin, slots)
            mla._write_mla_kv_cache(*arguments, reference_kv, reference_rope, _KV_LORA_RANK, _ROPE_DIM, _EPS)
            mla._write_mla_kv_cache(*arguments, actual_kv, actual_rope, _KV_LORA_RANK, _ROPE_DIM, _EPS, shared_slots)
            torch.npu.synchronize()
            torch.testing.assert_close(actual_kv.cpu(), reference_kv.cpu(), rtol=0, atol=0)
            torch.testing.assert_close(actual_rope.cpu(), reference_rope.cpu(), rtol=0, atol=0)
            latent = current_kv.cpu()[:, :_KV_LORA_RANK].float()
            normalized = latent * torch.rsqrt(latent.square().mean(dim=-1, keepdim=True) + _EPS)
            expected_kv, expected_rope = _new_caches(torch.device("cpu"))
            for row, destination in enumerate(destinations):
                if destination < 0:
                    continue
                expected_kv.flatten(0, 1)[destination, 0] = normalized[row].to(torch.bfloat16)
                expected_rope.flatten(0, 1)[destination, 0] = current_kv.cpu()[row, _KV_LORA_RANK:]
            torch.testing.assert_close(actual_kv.cpu(), expected_kv, rtol=1e-2, atol=1e-2)
            torch.testing.assert_close(actual_rope.cpu(), expected_rope, rtol=0, atol=0)
            untouched = torch.ones(256, dtype=torch.bool)
            untouched[[destination for destination in destinations if destination >= 0]] = False
            assert (actual_kv.cpu().flatten(0, 1)[untouched] == _SENTINEL).all()
            assert (actual_rope.cpu().flatten(0, 1)[untouched] == _SENTINEL).all()
        assert slots.dtype == torch.int32
        torch.testing.assert_close(slots.cpu(), torch.tensor(destinations, dtype=torch.int32), rtol=0, atol=0)
        torch.testing.assert_close(shared_slots.cpu(), torch.tensor(destinations, dtype=torch.int64), rtol=0, atol=0)
