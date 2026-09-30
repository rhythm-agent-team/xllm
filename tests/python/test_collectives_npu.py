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

"""Direct current-stream NPU gather checks with two explicit test devices."""

from __future__ import annotations

import math
import os
import time
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist


def _gather_payload(shape: tuple[int, ...], rank: int, strided: bool, offset: int) -> torch.Tensor:
    backing = torch.arange(math.prod(shape) * 2, dtype=torch.float32).reshape(*shape[:-1], shape[-1] * 2)
    value = (backing + 100 * rank + offset).to(torch.bfloat16)
    return value[..., 1::2] if strided else value[..., 1::2].contiguous()


def _run_current_stream_gather(rank: int, devices: tuple[int, int], rendezvous_path: str) -> None:
    import torch_npu  # noqa: F401

    device = torch.device(f"npu:{devices[rank]}")
    torch.npu.set_device(device)
    from xllm import xllm_export  # noqa: F401
    from xllm.python import initialize_runtime

    initialize_runtime()
    from xllm.python.distributed import collectives

    dist.init_process_group(
        "hccl",
        init_method=f"file://{rendezvous_path}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    collectives._groups[("tp", str(device))] = dist.group.WORLD
    warmup = torch.tensor([rank + 1], dtype=torch.bfloat16, device=device)
    dist.all_reduce(warmup)
    torch.testing.assert_close(warmup.cpu(), torch.tensor([3.0], dtype=warmup.dtype), rtol=0, atol=0)
    cases = (
        ((1, 5), 1),
        ((1, 5), -1),
        ((2, 5), 1),
        ((4, 5), -1),
        ((2, 5), 0),
        ((1, 1, 5), -1),
        ((1, 2, 5), -1),
    )
    for shape, dim in cases:
        for strided in (False, True):
            first: torch.Tensor | None = None
            first_values: torch.Tensor | None = None
            for offset in (0, 20):
                local = _gather_payload(shape, rank, strided, offset).to(device)
                if strided:
                    backing = torch.empty((*shape[:-1], shape[-1] * 2), dtype=local.dtype, device=device)
                    backing[..., 1::2].copy_(local)
                    local = backing[..., 1::2]
                original = local.cpu()
                gathered = collectives.all_gather(local, dim, 2)
                # Consume immediately on the current stream, without a host wait.
                consumed = gathered + 1
                peers = [_gather_payload(shape, peer, strided, offset) for peer in range(2)]
                reference = torch.cat(peers, dim=dim)
                torch.testing.assert_close(gathered.cpu(), reference, rtol=0, atol=0)
                torch.testing.assert_close(consumed.cpu(), reference + 1, rtol=0, atol=0)
                torch.testing.assert_close(local.cpu(), original, rtol=0, atol=0)
                assert gathered.dtype == local.dtype and gathered.device == local.device
                assert gathered.is_contiguous() and gathered.data_ptr() != local.data_ptr()
                if first is None:
                    first, first_values = gathered, reference
                else:
                    assert first.data_ptr() != gathered.data_ptr()
                    torch.testing.assert_close(first.cpu(), first_values, rtol=0, atol=0)
    torch.npu.synchronize()
    collectives._groups.clear()
    dist.destroy_process_group()


def test_two_rank_current_stream_gather_layout(tmp_path: Path) -> None:
    device_list = os.environ.get("XLLM_TEST_HCCL_DEVICES")
    if device_list is None:
        pytest.skip("set XLLM_TEST_HCCL_DEVICES to two NPUs with sufficient free HBM")
    parts = device_list.split(",")
    assert len(parts) == 2 and all(part.strip().isdecimal() for part in parts), "expected two nonnegative NPU ids"
    devices = (int(parts[0]), int(parts[1]))
    assert devices[0] != devices[1], "HCCL ranks require distinct devices"
    assert os.environ.get("HCCL_HOST_SOCKET_PORT_RANGE"), "set a test-specific HCCL_HOST_SOCKET_PORT_RANGE"
    assert os.environ.get("HCCL_NPU_SOCKET_PORT_RANGE"), "set a test-specific HCCL_NPU_SOCKET_PORT_RANGE"
    process_context = torch.multiprocessing.start_processes(
        _run_current_stream_gather,
        args=(devices, str(tmp_path / "current-stream-gather")),
        nprocs=2,
        join=False,
        start_method="spawn",
    )
    deadline = time.monotonic() + 120.0
    try:
        while not process_context.join(timeout=max(0.0, deadline - time.monotonic()), grace_period=5.0):
            if time.monotonic() >= deadline:
                pytest.fail("current-stream NPU gather test exceeded 120s")
    finally:
        for process in process_context.processes:
            if process.is_alive():
                process.terminate()
        cleanup_deadline = time.monotonic() + 5.0
        for process in process_context.processes:
            process.join(timeout=max(0.0, cleanup_deadline - time.monotonic()))
        for process in process_context.processes:
            if process.is_alive():
                process.kill()
                process.join(timeout=5.0)
