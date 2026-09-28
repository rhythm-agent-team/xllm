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

"""Numerical probe: bf16-input matmul with fp32 output on Ascend910_93 (A3).

Question under test: does an A3 device compute ``x(bf16) @ w(bf16)`` with fp32
accumulation and an fp32 result when the caller passes an fp32 ``out`` tensor?
Both operands are bf16-native in the router use case, so such a kernel would be
numerically equivalent to the current fp32 GEMM (lossless upcast + fp32
accumulate) while dropping the upcast copies.

The probe calls the aclnn two-phase API directly (no xLLM build required) and
compares every variant against an fp64 reference computed on the host:

* ``bmm bf16 x bf16 -> bf16 out``  (baseline: plain bf16 kernel)
* ``bmm bf16 x bf16 -> fp32 out``  (candidate: 16-in/32-out)
* ``mm  bf16 x bf16 -> fp32 out``  (2D MatMulV3: combo is not in the A3 op table)
* ``mm  fp32 x fp32 -> fp32 out``  (current router path in xLLM)

Build the wrapper inside the NPU container:

    g++ -std=c++17 -O2 -fPIC -shared tests/npu/aclnn_matmul_probe.cpp \
      -o /tmp/aclnn_matmul_probe.so \
      -I/usr/local/Ascend/cann-9.0.0/include \
      -L/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64 \
      -lopapi -lnnopbase -lascendcl

Run inside the container:

    XLLM_TEST_MATMUL_PROBE_LIB=/tmp/aclnn_matmul_probe.so \
      python3 tests/npu/test_matmul_16in32out_a3.py

Optional: ``XLLM_TEST_MATMUL_PROBE_TIME=1`` adds device timings. Only meaningful
on an idle device; the numbers are indicative when other workloads share it.
"""

from __future__ import annotations

import ctypes
import os
import sys
from functools import partial

import torch

# Gate shape of the GLM-5.3 router: K = hidden size, N = expert count.
HIDDEN_SIZE = 6144
NUM_EXPERTS = 256

# Distinct from the bf16 output rounding (~2^-9) by a wide margin.
FP32_ACCUMULATE_MAX_REL = 1e-4

DTYPE_FP32 = 0
DTYPE_BF16 = 2


def _load_probe(lib_path: str) -> ctypes.CDLL:
    lib = ctypes.CDLL(lib_path)
    f64 = ctypes.c_int64
    lib.probe_bmm.restype = ctypes.c_int32
    lib.probe_bmm.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        f64,
        f64,
        f64,
        f64,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_void_p,
    ]
    lib.probe_mm.restype = ctypes.c_int32
    lib.probe_mm.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        f64,
        f64,
        f64,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_void_p,
    ]
    lib.probe_last_error.restype = ctypes.c_char_p
    return lib


def _pick_device(requested: str | None) -> torch.device:
    if requested is not None:
        return torch.device(f"npu:{int(requested)}")
    best_device, best_free = 0, -1
    for index in range(torch.npu.device_count()):
        try:
            free, _total = torch.npu.mem_get_info(index)
        except Exception:  # mem_get_info is not available on every build
            free = 0
        if free > best_free:
            best_device, best_free = index, free
    return torch.device(f"npu:{best_device}")


def _make_inputs(m: int, device: torch.device, seed: int = 0):
    """bf16-native operands plus their lossless fp32 upcasts and fp64 reference."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x_bf = (torch.randn(m, HIDDEN_SIZE, generator=generator, dtype=torch.float32) * 0.5).to(torch.bfloat16)
    w_bf = (torch.randn(NUM_EXPERTS, HIDDEN_SIZE, generator=generator, dtype=torch.float32) * 0.02).to(torch.bfloat16)
    x32, w32 = x_bf.to(torch.float32), w_bf.to(torch.float32)
    reference = x32.double() @ w32.double().t()  # [m, N]
    inputs = {
        "x_bf16": x_bf.to(device),
        "x_fp32": x32.to(device),
        "w_bf16_kn": w_bf.t().contiguous().to(device),  # [K, N]
        "w_fp32_kn": w32.t().contiguous().to(device),
    }
    return inputs, reference


def _error_metrics(result: torch.Tensor, reference: torch.Tensor) -> dict:
    got = result.detach().to("cpu", torch.float32).double()
    diff = (got - reference).abs()
    scale = reference.abs().max().item()
    return {
        "max_abs": diff.max().item(),
        "max_rel": diff.max().item() / scale,
        "rms_rel": diff.pow(2).mean().sqrt().item() / scale,
    }


def _call_bmm(lib, inputs, out_dtype_code, m, device) -> tuple[int, torch.Tensor]:
    out = torch.empty(
        m, NUM_EXPERTS, dtype=(torch.float32 if out_dtype_code == DTYPE_FP32 else torch.bfloat16), device=device
    )
    stream = ctypes.c_void_p(torch.npu.current_stream().npu_stream)
    status = lib.probe_bmm(
        ctypes.c_void_p(inputs["x_bf16"].data_ptr()),
        ctypes.c_void_p(inputs["w_bf16_kn"].data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        1,
        m,
        HIDDEN_SIZE,
        NUM_EXPERTS,
        DTYPE_BF16,
        out_dtype_code,
        stream,
    )
    if status == 0:
        torch.npu.synchronize()
    return status, out


def _call_mm(lib, x, w, out, m, in_dtype_code, out_dtype_code) -> int:
    stream = ctypes.c_void_p(torch.npu.current_stream().npu_stream)
    status = lib.probe_mm(
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(w.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        m,
        HIDDEN_SIZE,
        NUM_EXPERTS,
        in_dtype_code,
        out_dtype_code,
        stream,
    )
    if status == 0:
        torch.npu.synchronize()
    return status


def _time_call(fn, iterations: int = 20, warmup: int = 3) -> float:
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    torch.npu.synchronize()
    return start.elapsed_time(end) / iterations


def run_probe(lib_path: str, requested_device: str | None = None) -> bool:
    lib = _load_probe(lib_path)
    device = _pick_device(requested_device)
    torch.npu.set_device(device)

    free_mb, total_mb = (value / (1024 * 1024) for value in torch.npu.mem_get_info(device))
    print(
        f"device={device} free={free_mb:.0f}MB total={total_mb:.0f}MB "
        f"cann={torch.version.__version__} torch_npu={torch.npu.__name__}"
    )
    print(f"shape: M in (512, 8, 1) K={HIDDEN_SIZE} N={NUM_EXPERTS}, operands bf16-native")

    ok = True

    for m in (512, 8, 1):
        inputs, reference = _make_inputs(m, device)
        asserted = m == 512
        print(f"\n--- M={m} ---")

        status, out_bf16 = _call_bmm(lib, inputs, DTYPE_BF16, m, device)
        if status != 0:
            print(f"  bmm bf16->bf16  STATUS={status} {lib.probe_last_error().decode()}")
            ok = ok and not asserted
        else:
            metrics = _error_metrics(out_bf16, reference)
            print(
                f"  bmm bf16->bf16  max_abs={metrics['max_abs']:.3e} "
                f"max_rel={metrics['max_rel']:.3e} rms_rel={metrics['rms_rel']:.3e}"
            )
            if asserted:
                # A bf16 result tensor must quantize the logits at ~2^-9.
                ok = ok and metrics["max_rel"] > FP32_ACCUMULATE_MAX_REL

        status, out_fp32 = _call_bmm(lib, inputs, DTYPE_FP32, m, device)
        if status != 0:
            print(f"  bmm bf16->fp32  STATUS={status} {lib.probe_last_error().decode()}")
            if asserted:
                ok = False
        else:
            metrics = _error_metrics(out_fp32, reference)
            print(
                f"  bmm bf16->fp32  max_abs={metrics['max_abs']:.3e} "
                f"max_rel={metrics['max_rel']:.3e} rms_rel={metrics['rms_rel']:.3e}"
            )
            print(f"    out dtype={out_fp32.dtype} (fp32 accumulate expected)")
            if asserted:
                ok = ok and metrics["max_rel"] < FP32_ACCUMULATE_MAX_REL

        mm_out_fp32 = torch.empty(m, NUM_EXPERTS, dtype=torch.float32, device=device)
        status = _call_mm(lib, inputs["x_bf16"], inputs["w_bf16_kn"], mm_out_fp32, m, DTYPE_BF16, DTYPE_FP32)
        if status != 0:
            print(f"  mm  bf16->fp32   STATUS={status} {lib.probe_last_error().decode()}")
        else:
            metrics = _error_metrics(mm_out_fp32, reference)
            print(
                f"  mm  bf16->fp32   max_abs={metrics['max_abs']:.3e} "
                f"max_rel={metrics['max_rel']:.3e} rms_rel={metrics['rms_rel']:.3e}"
            )

        mm_out_fp32_in = torch.empty(m, NUM_EXPERTS, dtype=torch.float32, device=device)
        status = _call_mm(lib, inputs["x_fp32"], inputs["w_fp32_kn"], mm_out_fp32_in, m, DTYPE_FP32, DTYPE_FP32)
        if status != 0:
            print(f"  mm  fp32->fp32   STATUS={status} {lib.probe_last_error().decode()}")
            if asserted:
                ok = False
        else:
            metrics = _error_metrics(mm_out_fp32_in, reference)
            print(
                f"  mm  fp32->fp32   max_abs={metrics['max_abs']:.3e} "
                f"max_rel={metrics['max_rel']:.3e} rms_rel={metrics['rms_rel']:.3e}"
                "   (current router path)"
            )

        if m == 512 and os.environ.get("XLLM_TEST_MATMUL_PROBE_TIME") == "1":
            bmm_ms = _time_call(partial(_call_bmm, lib, inputs, DTYPE_FP32, m, device))
            mm_bf16_ms = _time_call(
                partial(_call_mm, lib, inputs["x_bf16"], inputs["w_bf16_kn"], mm_out_fp32, m, DTYPE_BF16, DTYPE_FP32)
            )
            mm_fp32_ms = _time_call(
                partial(_call_mm, lib, inputs["x_fp32"], inputs["w_fp32_kn"], mm_out_fp32_in, m, DTYPE_FP32, DTYPE_FP32)
            )
            print(
                f"  [contended timing] bmm bf16->fp32 {bmm_ms:.3f}ms  "
                f"mm bf16->fp32 {mm_bf16_ms:.3f}ms  mm fp32->fp32 {mm_fp32_ms:.3f}ms"
            )

    print(f"\nresult: {'PASS' if ok else 'FAIL'}")
    return ok


def test_matmul_16in32out_a3() -> None:
    import pytest

    lib_path = os.environ.get("XLLM_TEST_MATMUL_PROBE_LIB")
    if not lib_path:
        pytest.skip("set XLLM_TEST_MATMUL_PROBE_LIB (build tests/npu/aclnn_matmul_probe.cpp)")
    if not torch.npu.is_available():
        pytest.skip("torch_npu device is unavailable")
    assert run_probe(lib_path, os.environ.get("XLLM_TEST_NPU_DEVICE"))


if __name__ == "__main__":
    library = os.environ.get("XLLM_TEST_MATMUL_PROBE_LIB", "/tmp/aclnn_matmul_probe.so")
    if not os.path.exists(library):
        sys.exit(f"probe library not found: {library}")
    sys.exit(0 if run_probe(library, os.environ.get("XLLM_TEST_NPU_DEVICE")) else 1)
