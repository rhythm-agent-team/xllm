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

"""Router gate A/B: the 16-in/32-out bf16 matmul against the replaced fp32 path.

Requires XLLM_TEST_NATIVE_LIBRARY (the built operator library) and
XLLM_TEST_NPU_DEVICE. Run tests/npu in its own pytest process so the CPU package
stubs from tests/python stay inactive:

    XLLM_TEST_NATIVE_LIBRARY=<built operator library> XLLM_TEST_NPU_DEVICE=0 \
        python3 -m pytest -q -s tests/npu/test_router_gate.py
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
# Both paths multiply the same bf16-exact products and accumulate in fp32, so
# they differ only by summation order (~1e-6 relative to the logits scale). The
# bf16-output control below sits ~1e-3, three orders of magnitude away.
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

    if not hasattr(runtime, "initialize_runtime"):
        pytest.fail("CPU package stubs are active; run tests/npu in a separate pytest process")
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
    """The replaced path: fp32 upcast of both bf16 operands, then an fp32 GEMM."""
    return F.linear(hidden.to(torch.float32), weight.to(torch.float32))


def _maximum_relative_error(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    return (candidate - reference).abs().max().item() / reference.abs().max().item()


@pytest.mark.parametrize("num_tokens", [1, 8, 512])
def test_logits_match_the_fp32_reference_path(npu_device: torch.device, num_tokens: int) -> None:
    from xllm.python import kernels

    hidden = _hidden(num_tokens, npu_device, seed=num_tokens)
    weight = _router_weight(npu_device)
    reference = _logits_reference(hidden, weight)
    logits = kernels.matmul_16in32out(hidden, weight.transpose(0, 1).contiguous())

    assert logits.dtype is torch.float32
    relative_error = _maximum_relative_error(logits, reference)
    logger.info("tokens=%d logits max relative error vs fp32 reference: %.3e", num_tokens, relative_error)
    assert relative_error < LOGITS_MAX_RELATIVE_ERROR

    # Control: a bf16-precision product (the 3-D aclnn form degrades to this)
    # stays far outside the same bound, so the assertion above can detect a
    # matmul that lost fp32 accumulation.
    bf16_error = _maximum_relative_error(torch.matmul(hidden, weight.transpose(0, 1).contiguous()), reference)
    logger.info("tokens=%d bf16-output control max relative error: %.3e", num_tokens, bf16_error)
    assert bf16_error > LOGITS_MAX_RELATIVE_ERROR


@pytest.mark.parametrize("num_tokens", [1, 8, 64, 512])
def test_expert_selection_matches_the_fp32_reference_path(npu_device: torch.device, num_tokens: int) -> None:
    from xllm.python import kernels

    hidden = _hidden(num_tokens, npu_device, seed=1000 + num_tokens)
    weight = _router_weight(npu_device)
    generator = torch.Generator().manual_seed(7)
    correction_bias = (torch.randn(NUM_EXPERTS, generator=generator) * 0.1).to(npu_device)
    logits = kernels.matmul_16in32out(hidden, weight.transpose(0, 1).contiguous())

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
    logger.info(
        "tokens=%d top-k ids: %d flips, top-k weight max relative error: %.3e",
        num_tokens,
        flips,
        weight_error,
    )
    assert flips == 0
    assert weight_error < TOP_K_WEIGHT_MAX_RELATIVE_ERROR


def test_router_gate_module_drives_the_op_with_bf16_operands(npu_device: torch.device) -> None:
    from xllm.python.models.deepseek_v32 import RouterGate

    hidden = _hidden(512, npu_device, seed=11)
    weight = _router_weight(npu_device)
    gate = RouterGate(HIDDEN_SIZE, NUM_EXPERTS, npu_device)
    gate.weight.data.copy_(weight)
    gate.process_weights_after_loading()

    # Checkpoint layout and dtype stay loader-addressable as ``gate.weight``.
    assert gate.weight.dtype is torch.bfloat16
    assert gate.weight.shape == (NUM_EXPERTS, HIDDEN_SIZE)
    assert [name for name, _ in gate.named_parameters()] == ["weight"]
    # The derived [H, E] operand is a plain attribute: absent from the state
    # dict, from named_buffers(), and therefore from the loader's snapshot.
    assert list(gate.named_buffers()) == []
    assert "weight_kn" not in gate.state_dict()
    assert gate.weight_kn is not None
    assert gate.weight_kn.dtype is torch.bfloat16
    assert gate.weight_kn.shape == (HIDDEN_SIZE, NUM_EXPERTS)
    assert gate.weight_kn.is_contiguous()
    assert torch.equal(gate.weight_kn, weight.transpose(0, 1))

    # A passing call also proves the operands were bf16: the op rejects fp32.
    logits = gate(hidden)
    assert logits.dtype is torch.float32
    relative_error = _maximum_relative_error(logits, _logits_reference(hidden, weight))
    logger.info("RouterGate logits max relative error vs fp32 reference: %.3e", relative_error)
    assert relative_error < LOGITS_MAX_RELATIVE_ERROR
