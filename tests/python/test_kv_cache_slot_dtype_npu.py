# Copyright 2026 The xLLM Authors.
# SPDX-License-Identifier: Apache-2.0

"""Compare INT32 and INT64 indices in the installed fused PA cache writer."""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch_npu


@torch.inference_mode()
def test_kv_cache_slot_dtype() -> None:
    torch.npu.set_device(int(os.environ.get("XLLM_TEST_NPU_DEVICE", "0")))
    generator = torch.Generator().manual_seed(918)
    kv = torch.randn(4, 1, 1, 576, generator=generator).to(device="npu", dtype=torch.bfloat16)
    gamma = torch.ones(512, device="npu", dtype=torch.bfloat16)
    cos = torch.ones(4, 1, 1, 64, device="npu", dtype=torch.bfloat16)
    sin = torch.zeros_like(cos)
    slots = {
        "int64": torch.tensor([0, 3, 17, 19], device="npu", dtype=torch.int64),
        "int32": torch.tensor([0, 3, 17, 19], device="npu", dtype=torch.int32),
    }
    caches = {
        name: (
            torch.zeros(2, 128, 1, 64, device="npu", dtype=torch.bfloat16),
            torch.zeros(2, 128, 1, 512, device="npu", dtype=torch.bfloat16),
        )
        for name in slots
    }

    def call(name: str) -> None:
        k_cache, ckv_cache = caches[name]
        torch_npu.npu_kv_rmsnorm_rope_cache(
            kv,
            gamma,
            cos,
            sin,
            slots[name],
            k_cache,
            ckv_cache,
            epsilon=1e-5,
            cache_mode="PA",
            is_output_kv=False,
        )
        torch.npu.synchronize()

    for _ in range(4):
        call("int64")
    reference = tuple(cache.cpu() for cache in caches["int64"])
    assert all(torch.isfinite(cache).all() and torch.count_nonzero(cache) > 0 for cache in reference)
    print("INT64 control: successful nonempty finite cache writes", flush=True)
    result_root = Path(os.environ["XLLM_KV_SLOT_PROFILE_DIR"])
    result_root.mkdir(parents=True, exist_ok=False)
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        experimental_config=torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            export_type=[torch_npu.profiler.ExportType.Text],
            data_simplification=False,
        ),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(result_root), async_mode=False),
    ):
        for repetition in range(4):
            for name in slots:
                with torch.profiler.record_function(f"kv_slot_probe/{name}/{repetition}"):
                    call(name)
    for actual, expected in zip(caches["int32"], reference, strict=True):
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    print("INT32 matches INT64 cache writes exactly", flush=True)
