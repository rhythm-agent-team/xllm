# Copyright 2026 The xLLM Authors.
# SPDX-License-Identifier: Apache-2.0

"""GLM router: replicated [256,6144] weights, three full GEMM paths.

Run with the normal tests/python pytest entry point. XLLM_ROUTER_PROFILE_DIR
selects a fresh result directory. Numerical checks and warmup are outside
profiling; measured annotations contain enqueue and same-stream synchronization.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn.functional as F
import torch_npu

ROUTES = ("old", "addmm", "aclnn_matmul")
SHAPES = (1, 2, 4, 8, 512, 4096)
K, N = 6144, 256


@pytest.fixture(scope="module")
def router_weights() -> tuple[torch.Tensor, torch.Tensor, Path]:
    torch.npu.set_device(int(os.environ.get("XLLM_TEST_NPU_DEVICE", "0")))
    generator = torch.Generator().manual_seed(918)
    checkpoint_layout = (torch.randn(N, K, generator=generator) * 0.02).to(torch.bfloat16)
    # Match the old contiguous [E,H] FP32 weight and candidate contiguous [H,E].
    old_weight = checkpoint_layout.float().to("npu")
    new_weight = checkpoint_layout.t().contiguous().to("npu")
    result_root = Path(os.environ["XLLM_ROUTER_PROFILE_DIR"])
    result_root.mkdir(parents=True, exist_ok=False)
    return old_weight, new_weight, result_root


def _calls(
    hidden: torch.Tensor, old_weight: torch.Tensor, new_weight: torch.Tensor
) -> dict[str, Callable[[], torch.Tensor]]:
    def old() -> torch.Tensor:
        return F.linear(hidden.to(torch.float32), old_weight)

    def addmm() -> torch.Tensor:
        out = torch.empty((hidden.shape[0], N), dtype=torch.float32, device=hidden.device)
        return torch.addmm(out, hidden, new_weight, beta=0, alpha=1, out=out)

    def direct() -> torch.Tensor:
        return torch.ops.xllm_ops.matmul_16in32out(hidden, new_weight)

    return dict(zip(ROUTES, (old, addmm, direct), strict=True))


@pytest.mark.parametrize("m", SHAPES)
@pytest.mark.parametrize("repetition", (1, 2))
@torch.inference_mode()
def test_router_matmul(router_weights: tuple[torch.Tensor, torch.Tensor, Path], repetition: int, m: int) -> None:
    old_weight, new_weight, result_root = router_weights
    generator = torch.Generator().manual_seed(918 + m)
    hidden_cpu = (torch.randn(m, K, generator=generator) * 0.5).to(torch.bfloat16)
    hidden = hidden_cpu.to(old_weight.device)
    reference = F.linear(hidden_cpu.double(), old_weight.cpu().double())
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    profile_dir = result_root / f"eager-r{repetition}-m{m}"
    profile_dir.mkdir()
    numerical: dict[str, float] = {}
    samples: list[dict[str, Any]] = []
    with torch.npu.stream(stream):
        calls = _calls(hidden, old_weight, new_weight)
        for route in ROUTES:
            output = calls[route]()
            stream.synchronize()
            assert output.dtype == torch.float32 and output.shape == (m, N)
            cpu = output.cpu()
            assert torch.isfinite(cpu).all()
            error = float((cpu.double() - reference).abs().max() / reference.abs().max().clamp_min(1.0))
            assert error < 2e-5, f"{route}: FP64-reference relative error={error}"
            assert not torch.equal(cpu, cpu.to(torch.bfloat16).float()), "BF16-rounded values in FP32 output"
            numerical[route] = error
        for warmup in range(8):
            for route in ROUTES if warmup % 2 == 0 else tuple(reversed(ROUTES)):
                calls[route]()
                stream.synchronize()
        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
            record_shapes=True,
            experimental_config=torch_npu.profiler._ExperimentalConfig(
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                export_type=[torch_npu.profiler.ExportType.Text],
                data_simplification=False,
            ),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(profile_dir), async_mode=False),
        ):
            for round_index in range(32):
                rotation = (round_index + repetition - 1) % len(ROUTES)
                rotated = ROUTES[rotation:] + ROUTES[:rotation]
                for position, route in enumerate(rotated + tuple(reversed(rotated))):
                    index = len(samples)
                    label = f"router_probe/m={m}/route={route}/sample={index}"
                    with torch.profiler.record_function(label):
                        calls[route]()
                        stream.synchronize()
                    samples.append(
                        {"sample": index, "label": label, "route": route, "round": round_index, "position": position}
                    )
        stream.synchronize()
    manifest = {
        "source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "mode": "eager",
        "repetition": repetition,
        "m": m,
        "k": K,
        "n": N,
        "routes": ROUTES,
        "samples": samples,
        "numerical_relative_error": numerical,
        "warmup_per_route_outside_profile": 8,
        "measured_samples_per_route": 64,
        "weight_source": "seeded synthetic BF16; verified replicated online shape and matching layouts",
        "old_weight_stride": list(old_weight.stride()),
        "new_weight_stride": list(new_weight.stride()),
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "criterion": {"p90_p10_le": 1.5, "half_median_ratio_le": 1.2, "repeat_median_ratio_le": 1.2},
        "measurement": "complete device task sum; not host latency or model E2E",
    }
    (profile_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
