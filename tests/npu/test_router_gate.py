# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Router BF16 operands/FP32 addmm against the replaced FP32 routing path.

Run in the NPU container with XLLM_TEST_NATIVE_LIBRARY and XLLM_TEST_NPU_DEVICE
set, separately from the CPU parallel-layout fixtures.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from scripts.logger import logger

HIDDEN_SIZE = 6144
NUM_EXPERTS = 256
TOPK = 8
TOPK_GROUP = 4
NUM_EXPERT_GROUPS = 8
ROUTED_SCALING = 2.5
# Both paths multiply BF16-exact operands and accumulate in FP32. The bound
# allows summation-order drift but rejects a BF16-rounded product.
LOGITS_MAX_RELATIVE_ERROR = 2e-5
TOP_K_WEIGHT_MAX_RELATIVE_ERROR = 1e-5


@pytest.fixture(scope="module")
def npu_device() -> torch.device:
    library = os.environ.get("XLLM_TEST_NATIVE_LIBRARY")
    device_index = os.environ.get("XLLM_TEST_NPU_DEVICE")
    if not library or device_index is None:
        pytest.skip("set XLLM_TEST_NATIVE_LIBRARY and XLLM_TEST_NPU_DEVICE for real NPU tests")
    pytest.importorskip("torch_npu")
    import xllm.python as runtime

    assert Path(library).is_file(), f"native operator library does not exist: {library}"
    torch.ops.load_library(library)
    torch.npu.set_device(int(device_index))
    runtime.initialize_runtime()
    return torch.device(f"npu:{device_index}")


def _router_weight(device: torch.device) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    return (torch.randn(NUM_EXPERTS, HIDDEN_SIZE, generator=generator) * 0.02).to(torch.bfloat16).to(device)


def _hidden(num_tokens: int, device: torch.device, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(num_tokens, HIDDEN_SIZE, generator=generator).to(torch.bfloat16).to(device)


def _logits_reference(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return F.linear(hidden.to(torch.float32), weight.to(torch.float32))


def _maximum_relative_error(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    return (candidate.float() - reference.float()).abs().max().item() / reference.abs().max().item()


def _gate_logits(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    from xllm.python.models.deepseek_v32 import RouterGate

    gate = RouterGate(HIDDEN_SIZE, NUM_EXPERTS, hidden.device)
    gate.weight.data.copy_(weight)
    return gate(hidden)


@pytest.mark.parametrize("num_tokens", [1, 2, 4, 8, 512])
def test_logits_match_the_fp32_reference_path(npu_device: torch.device, num_tokens: int) -> None:
    hidden = _hidden(num_tokens, npu_device, seed=num_tokens)
    weight = _router_weight(npu_device)
    reference = _logits_reference(hidden, weight)
    logits = _gate_logits(hidden, weight)

    assert logits.dtype is torch.float32
    relative_error = _maximum_relative_error(logits, reference)
    logger.info("tokens=%d logits max relative error vs fp32 reference: %.3e", num_tokens, relative_error)
    assert relative_error < LOGITS_MAX_RELATIVE_ERROR
    bf16_error = _maximum_relative_error(torch.matmul(hidden, weight.transpose(0, 1).contiguous()), reference)
    logger.info("tokens=%d bf16-output control max relative error: %.3e", num_tokens, bf16_error)
    assert bf16_error > LOGITS_MAX_RELATIVE_ERROR


@pytest.mark.parametrize("num_tokens", [1, 2, 4, 8, 64, 512])
def test_expert_selection_matches_the_fp32_reference_path(npu_device: torch.device, num_tokens: int) -> None:
    from xllm.python import kernels

    hidden = _hidden(num_tokens, npu_device, seed=1000 + num_tokens)
    weight = _router_weight(npu_device)
    generator = torch.Generator().manual_seed(7)
    correction_bias = (torch.randn(NUM_EXPERTS, generator=generator) * 0.1).to(npu_device)
    logits = _gate_logits(hidden, weight)
    reference_weights, reference_ids = kernels.moe_gate_routing(
        _logits_reference(hidden, weight),
        correction_bias,
        TOPK,
        TOPK_GROUP,
        NUM_EXPERT_GROUPS,
        True,
        ROUTED_SCALING,
    )
    weights, ids = kernels.moe_gate_routing(
        logits,
        correction_bias,
        TOPK,
        TOPK_GROUP,
        NUM_EXPERT_GROUPS,
        True,
        ROUTED_SCALING,
    )
    flips = int((ids != reference_ids).sum().item())
    weight_error = _maximum_relative_error(weights, reference_weights)
    logger.info("tokens=%d top-k id flips=%d, weight max relative error=%.3e", num_tokens, flips, weight_error)
    assert flips == 0
    assert weight_error < TOP_K_WEIGHT_MAX_RELATIVE_ERROR


class _TensorStateDict:
    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        self._tensors = tensors

    def has(self, name: str) -> bool:
        return name in self._tensors

    def get_tensor(self, name: str) -> torch.Tensor:
        return self._tensors[name]


def test_router_gate_loader_preserves_checkpoint_layout(npu_device: torch.device) -> None:
    from xllm.python.models.deepseek_v32 import RouterGate
    from xllm.python.models.weight_utils import WeightLoader

    hidden = _hidden(4, npu_device, seed=11)
    weight = _router_weight(torch.device("cpu"))
    model = torch.nn.Module()
    model.gate = RouterGate(HIDDEN_SIZE, NUM_EXPERTS, npu_device)
    original_storage = model.gate.weight.data_ptr()
    loader = WeightLoader(model, [_TensorStateDict({"gate.weight": weight})], tp_size=16, tp_rank=7)
    loader.copy_replicated("gate.weight")
    gate = model.gate
    assert gate.weight.dtype is torch.bfloat16
    assert gate.weight.shape == (NUM_EXPERTS, HIDDEN_SIZE)
    assert [name for name, _ in model.named_parameters()] == ["gate.weight"]
    assert list(model.state_dict()) == ["gate.weight"]
    assert list(model.named_buffers()) == []
    torch.testing.assert_close(gate.weight.cpu(), weight, rtol=0, atol=0)
    assert gate.weight.data_ptr() == original_storage
    assert not gate.weight.is_contiguous()
    assert gate.weight.t().is_contiguous()
    assert gate.weight.t().data_ptr() == gate.weight.data_ptr()
    logits = gate(hidden)
    assert logits.dtype is torch.float32
    relative_error = _maximum_relative_error(logits, _logits_reference(hidden, weight.to(npu_device)))
    logger.info("loaded RouterGate logits max relative error: %.3e", relative_error)
    assert relative_error < LOGITS_MAX_RELATIVE_ERROR


def test_router_addmm_beta_zero_ignores_poisoned_output(npu_device: torch.device) -> None:
    hidden = _hidden(4, npu_device, seed=19)
    weight = _router_weight(npu_device)
    output = torch.full((4, NUM_EXPERTS), float("nan"), dtype=torch.float32, device=npu_device)
    actual = torch.addmm(output, hidden, weight.t(), beta=0, alpha=1, out=output)
    assert actual.data_ptr() == output.data_ptr()
    assert torch.isfinite(actual).all()
    assert _maximum_relative_error(actual, _logits_reference(hidden, weight)) < LOGITS_MAX_RELATIVE_ERROR
