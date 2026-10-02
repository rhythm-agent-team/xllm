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

"""One rank of a direct Python TileLang SHMEM AllGather correctness test.

Launch one worker per PE with a fresh shared --artifact-dir and explicit NPU
--device. CPU FileStore coordination and HYBM/TCP bootstrap use no HCCL or MPI. The caller
must verify device availability and provide matching TileLang/SHMEM packages.
No xLLM native binary, AOT registration, package installation, or runtime shim
is used. A failure leaves the original traceback and saved phase; it does not
enter a cleanup rendezvous that could hide the failure behind a peer timeout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.logger import logger

_GUARD_BYTES = 128
_GUARD_VALUE = -123


def _save(result_path: Path, result: dict[str, Any], phase: str) -> None:
    result["phase"] = phase
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    logger.info("Rank %d: %s", result["rank"], phase)


def _rendezvous(store: dist.Store, rank: int, world_size: int, phase: str) -> None:
    store.set(f"{phase}/{rank}", b"ready")
    for peer in range(world_size):
        if store.get(f"{phase}/{peer}") != b"ready":
            raise RuntimeError(f"Invalid {phase} rendezvous for PE {peer}")


def _file_identity(path: str | Path) -> dict[str, str]:
    source = Path(path).resolve()
    return {"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}


def _payload(rank: int, count: int, iteration: int, dtype: torch.dtype) -> torch.Tensor:
    positions = torch.arange(count, dtype=torch.int64)
    rows = positions // 17
    columns = positions % 17
    # Distinct rank intervals and changing row/column markers stay exact in BF16.
    values = rank * 8 + (5 * rows + 3 * columns + 7 * iteration) % 8
    return values.to(dtype)


def _guarded(count: int, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, int]:
    guard_elements = _GUARD_BYTES // torch.empty((), dtype=dtype).element_size()
    padded_count = (count + guard_elements - 1) // guard_elements * guard_elements
    storage = torch.full((padded_count + 2 * guard_elements,), _GUARD_VALUE, dtype=dtype, device=device)
    view = storage[guard_elements : guard_elements + count]
    if storage.data_ptr() % _GUARD_BYTES or view.data_ptr() % _GUARD_BYTES:
        raise RuntimeError("Guarded allocation and payload must be 128-byte aligned")
    return storage, view, guard_elements


def _check_guards(storage: torch.Tensor, count: int, guard_elements: int) -> None:
    actual = storage.cpu()
    assert torch.all(actual[:guard_elements] == _GUARD_VALUE), "Leading guard changed"
    assert torch.all(actual[guard_elements + count :] == _GUARD_VALUE), "Tail padding or trailing guard changed"


def _check_state(
    controls: torch.Tensor, epochs: torch.Tensor, expected: torch.Tensor, lanes: int, world_size: int
) -> None:
    actual_epochs = epochs.cpu().view(lanes, 32)
    torch.testing.assert_close(actual_epochs[:, :8], expected[:, None].expand(-1, 8), rtol=0, atol=0)
    assert torch.count_nonzero(actual_epochs[:, 8:]) == 0, "Epoch slot padding changed"
    actual_controls = controls.cpu().view(2, world_size, lanes, 32)
    torch.testing.assert_close(
        actual_controls[:, :, :, :8],
        expected.view(1, 1, lanes, 1).expand(2, world_size, lanes, 8),
        rtol=0,
        atol=0,
    )
    assert torch.count_nonzero(actual_controls[:, :, :, 8:]) == 0, "Control slot padding changed"


def _compile_only(args: argparse.Namespace, result: dict[str, Any], result_path: Path) -> None:
    import tilelang

    from xllm.python.kernels_npu.tilelang import shmem_all_gather as kernel_module

    result["versions"] = {"tilelang": tilelang.__version__}
    result["kernel"] = _file_identity(kernel_module.__file__)
    result["tilelang_python"] = _file_identity(tilelang.__file__)
    result["tilelang_native"] = _file_identity(tilelang._LIB_PATH)
    for count in args.counts:
        _save(result_path, result, f"count-{count}/lower")
        function = kernel_module.build_shmem_all_gather_kernel(
            count, args.world_size, args.lanes, args.chunk_bytes, args.dtype
        )
        with tilelang.tvm.transform.PassContext(opt_level=3, config=kernel_module.SHMEM_PASS_CONFIGS):
            lowered = tilelang.lower(function, target="ascendc", platform="A3")
        source_path = args.artifact_dir / f"rank-{args.rank}-count-{count}.cpp"
        source_path.write_text(lowered.kernel_source, encoding="utf-8")
        result["cases"].append({"count": count, "generated_source": _file_identity(source_path)})
    result["status"] = "LOWERED_ONLY"
    _save(result_path, result, "lowered-only/no-device-execution")


def _run_case(
    args: argparse.Namespace,
    store: dist.Store,
    result: dict[str, Any],
    result_path: Path,
    count: int,
    receive: torch.Tensor,
    controls: torch.Tensor,
    epochs: torch.Tensor,
) -> None:
    import tilelang

    from xllm.python.kernels_npu.tilelang.shmem_all_gather import SHMEM_PASS_CONFIGS, build_shmem_all_gather_kernel

    dtype = getattr(torch, args.dtype)
    device = torch.device(f"npu:{args.device}")
    source_storage, source, source_guard = _guarded(count, dtype, device)
    output_storage, output, output_guard = _guarded(args.world_size * count, dtype, device)
    consumed = torch.empty_like(output)
    _save(result_path, result, f"count-{count}/compile")
    kernel = tilelang.compile(
        build_shmem_all_gather_kernel(count, args.world_size, args.lanes, args.chunk_bytes, args.dtype),
        out_idx=None,
        target="ascendc",
        platform="A3",
        pass_configs=SHMEM_PASS_CONFIGS,
    )
    generated = kernel.get_kernel_source()
    generated_path = args.artifact_dir / f"rank-{args.rank}-count-{count}.cpp"
    generated_path.write_text(generated, encoding="utf-8")
    case: dict[str, Any] = {
        "count": count,
        "generated_source": _file_identity(generated_path),
        "jit_library": _file_identity(kernel.adapter.libpath),
        "checked_iterations": [],
        "graph_replays": 0,
    }
    result["cases"].append(case)
    _save(result_path, result, f"count-{count}/compiled")
    _rendezvous(store, args.rank, args.world_size, f"count-{count}/compiled")
    rounds = (count + args.lanes * args.chunk_bytes // source.element_size() - 1) // (
        args.lanes * args.chunk_bytes // source.element_size()
    )

    def _submit() -> None:
        kernel(source, output.view(args.world_size, count), receive, controls, epochs, args.rank)
        # The consumer is on the same current stream immediately after gather.
        torch.add(output, 1, out=consumed)

    def _prepare(iteration: int) -> torch.Tensor:
        local = _payload(args.rank, count, iteration, dtype)
        source.copy_(local)
        output.fill_(float("nan"))
        consumed.fill_(float("nan"))
        return local

    def _check(iteration: int, local: torch.Tensor, previous_epochs: torch.Tensor) -> None:
        torch.npu.synchronize()
        expected = torch.cat([_payload(peer, count, iteration, dtype) for peer in range(args.world_size)])
        torch.testing.assert_close(source.cpu(), local, rtol=0, atol=0)
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(consumed.cpu(), expected + 1, rtol=0, atol=0)
        _check_guards(source_storage, count, source_guard)
        _check_guards(output_storage, args.world_size * count, output_guard)
        _check_state(controls, epochs, (previous_epochs + rounds) % 2, args.lanes, args.world_size)
        case["checked_iterations"].append(iteration)
        # Keep the next invocation from changing controls during another PE's
        # host-side state checks. This is once per collective, never per row.
        _rendezvous(store, args.rank, args.world_size, f"count-{count}/iteration-{iteration}/checked")

    _save(result_path, result, f"count-{count}/warmup")
    for iteration in (-2, -1):
        previous_epochs = epochs.cpu().view(args.lanes, 32)[:, 0].clone()
        local = _prepare(iteration)
        _submit()
        _check(iteration, local, previous_epochs)

    graph = None
    if args.mode == "graph":
        _prepare(0)
        torch.npu.synchronize()
        _rendezvous(store, args.rank, args.world_size, f"count-{count}/capture")
        _save(result_path, result, f"count-{count}/capture")
        graph = torch.npu.NPUGraph()
        capture_stream = torch.npu.Stream()
        with torch.npu.graph(graph, stream=capture_stream):
            _submit()
        torch.npu.synchronize()
        # Capture is not a numerical pass and its execution is not assumed.
        # Each replay is checked against the observed state immediately before it.
        case["epochs_after_capture"] = epochs.cpu().view(args.lanes, 32)[:, 0].tolist()
        _rendezvous(store, args.rank, args.world_size, f"count-{count}/captured")

    for iteration in range(args.repeats):
        _save(result_path, result, f"count-{count}/iteration-{iteration}")
        previous_epochs = epochs.cpu().view(args.lanes, 32)[:, 0].clone()
        assert torch.all((previous_epochs == 0) | (previous_epochs == 1)), "Invalid device epoch"
        local = _prepare(iteration)
        if graph is None:
            _submit()
        else:
            graph.replay()
            case["graph_replays"] += 1
        _check(iteration, local, previous_epochs)

    torch.npu.synchronize()
    if graph is not None:
        graph.reset()
        del graph
    _rendezvous(store, args.rank, args.world_size, f"count-{count}/checked")
    _save(result_path, result, f"count-{count}/checked")


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, choices=(2, 4, 8, 16), required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--mode", choices=("eager", "graph"), default="eager")
    parser.add_argument(
        "--compile-only", action="store_true", help="Lower DSL only; no SHMEM bootstrap or NPU execution"
    )
    parser.add_argument("--counts", type=int, nargs="+", default=[64, 65, 127, 128, 129])
    parser.add_argument("--lanes", type=int, default=2)
    parser.add_argument("--chunk-bytes", type=int, default=128)
    parser.add_argument("--heap-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--repeats", type=int, default=16)
    args = parser.parse_args()
    if not 0 <= args.rank < args.world_size or args.device < 0:
        parser.error("Require 0 <= rank < world-size and a nonnegative explicit device")
    if args.lanes <= 0 or args.lanes % 2 or args.repeats <= 0:
        parser.error("Require positive even lanes and positive repeats")
    if not 0 < args.chunk_bytes <= 64 * 1024 or args.chunk_bytes % 128:
        parser.error("Require chunk-bytes <= 64 KiB and a positive multiple of 128")
    if len(set(args.counts)) != len(args.counts) or any(
        not 0 < args.world_size * count <= (1 << 31) - 1 for count in args.counts
    ):
        parser.error("Require distinct positive counts with world-size * count <= INT32_MAX")
    required_bytes = args.world_size * args.lanes * args.chunk_bytes + (2 * args.world_size + 1) * args.lanes * 128
    if args.heap_bytes < required_bytes:
        parser.error(f"Symmetric buffers require at least {required_bytes} heap bytes")
    args.artifact_dir = args.artifact_dir.resolve()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.artifact_dir / f"rank-{args.rank}.json"
    result: dict[str, Any] = {
        "rank": args.rank,
        "world_size": args.world_size,
        "device": args.device,
        "mode": args.mode,
        "compile_only": args.compile_only,
        "dtype": args.dtype,
        "counts": args.counts,
        "lanes": args.lanes,
        "chunk_bytes": args.chunk_bytes,
        "heap_bytes": args.heap_bytes,
        "repeats": args.repeats,
        "worker": _file_identity(__file__),
        "python": sys.version,
        "executable": sys.executable,
        "cases": [],
        "environment": {
            name: os.environ.get(name)
            for name in ("ASCEND_RT_VISIBLE_DEVICES", "ASCEND_HOME_PATH", "TL_ROOT", "SHMEM_HOME_PATH")
        },
    }
    # Refuse an earlier attempt's rank result or rendezvous namespace.
    with result_path.open("x", encoding="utf-8") as result_file:
        result_file.write(json.dumps(result, indent=2) + "\n")
    _save(result_path, result, "imports")
    if args.compile_only:
        _compile_only(args, result, result_path)
        return
    import shmem
    import shmem._pyshmem as shmem_native
    import tilelang
    import torch_npu
    from shmem.construct_tensor import construct_tensor_from_ptr

    from xllm.python.kernels_npu.tilelang import shmem_all_gather as kernel_module

    torch.set_num_threads(1)
    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
    dtype = getattr(torch, args.dtype)
    result["versions"] = {
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "tilelang": tilelang.__version__,
    }
    result["kernel"] = _file_identity(kernel_module.__file__)
    result["shmem_python"] = _file_identity(shmem.__file__)
    result["shmem_native"] = _file_identity(shmem_native.__file__)
    result["tilelang_python"] = _file_identity(tilelang.__file__)
    result["tilelang_native"] = _file_identity(tilelang._LIB_PATH)
    _save(result_path, result, "bootstrap")
    store = dist.FileStore(str(args.artifact_dir / "bootstrap.store"), args.world_size)
    store.set_timeout(timedelta(seconds=120))
    if store.add(f"participant/{args.rank}", 1) != 1:
        raise RuntimeError("FileStore namespace already used by this rank")
    configuration_fields = (
        "world_size",
        "mode",
        "dtype",
        "counts",
        "lanes",
        "chunk_bytes",
        "heap_bytes",
        "repeats",
        "versions",
    )
    configuration_data = {name: result[name] for name in configuration_fields}
    configuration_data["source_sha256"] = {
        name: result[name]["sha256"]
        for name in ("worker", "kernel", "shmem_python", "shmem_native", "tilelang_python", "tilelang_native")
    }
    configuration = json.dumps(configuration_data, sort_keys=True).encode()
    store.set(f"config/{args.rank}", configuration)
    store.set(f"device/{args.rank}", str(args.device))
    devices = []
    for peer in range(args.world_size):
        if store.get(f"config/{peer}") != configuration:
            raise RuntimeError(f"AllGather configuration mismatch with PE {peer}")
        devices.append(int(store.get(f"device/{peer}")))
    if len(set(devices)) != args.world_size:
        raise RuntimeError(f"PEs require distinct devices, got {devices}")
    result["pe_devices"] = devices
    if shmem.aclshmemx_init_status() != shmem.InitStatus.NOT_INITIALIZED:
        raise RuntimeError("Test requires an uninitialized SHMEM process")
    # Match the existing TileLang JIT's HYBM device ABI and official TCP example.
    # This is selected before execution, not a retry or a backend fallback.
    listener = None
    if args.rank == 0:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        listener.listen(args.world_size)
        store.set("tcp_port", str(listener.getsockname()[1]))
    port = int(store.get("tcp_port"))
    attributes = shmem.InitAttr()
    attributes.my_rank = args.rank
    attributes.n_ranks = args.world_size
    attributes.local_mem_size = args.heap_bytes
    attributes.ip_port = f"tcp://127.0.0.1:{port}"
    attributes.option_attr.data_op_engine_type = shmem.OpEngineType.MTE
    if listener is not None:
        attributes.option_attr.sockFd = listener.fileno()
    ret = shmem.set_conf_store_tls(False, "")
    if ret != 0:
        raise RuntimeError(f"set_conf_store_tls failed: {ret}")
    ret = shmem.aclshmem_init(attributes)
    if ret != 0:
        raise RuntimeError(f"aclshmem_init failed: {ret}")
    result["bootstrap"] = "HYBM/TCP/MTE"
    if shmem.aclshmemx_init_status() != shmem.InitStatus.INITIALIZED:
        raise RuntimeError("SHMEM initialization returned without an initialized domain")
    if shmem.my_pe() != args.rank or shmem.pe_count() != args.world_size:
        raise RuntimeError("SHMEM PE mapping does not match the test")
    _save(result_path, result, "allocate")
    allocations = []
    views = []
    for elements, tensor_dtype in (
        (args.world_size * args.lanes * args.chunk_bytes // torch.empty((), dtype=dtype).element_size(), dtype),
        (2 * args.world_size * args.lanes * 32, torch.int32),
        (args.lanes * 32, torch.int32),
    ):
        nbytes = elements * torch.empty((), dtype=tensor_dtype).element_size()
        pointer = shmem.aclshmem_align(128, nbytes)
        if not pointer or pointer % 128:
            raise RuntimeError(f"Invalid symmetric allocation: bytes={nbytes}, pointer={pointer}")
        allocations.append(pointer)
        views.append(construct_tensor_from_ptr(pointer, [elements], tensor_dtype, device))
    receive, controls, epochs = views
    receive.fill_(float("nan"))
    controls.zero_()
    epochs.zero_()
    torch.npu.synchronize()
    _rendezvous(store, args.rank, args.world_size, "initialized")
    for count in args.counts:
        _run_case(args, store, result, result_path, count, receive, controls, epochs)
    _save(result_path, result, "close/drain")
    torch.npu.synchronize()
    _rendezvous(store, args.rank, args.world_size, "drained")
    del receive, controls, epochs
    views.clear()
    for pointer in reversed(allocations):
        shmem.aclshmem_free(pointer)
    ret = shmem.aclshmem_finialize()
    if ret != 0:
        raise RuntimeError(f"aclshmem_finialize failed: {ret}")
    if listener is not None:
        listener.detach()  # The official TCP store owns and closes the passed FD.
    _rendezvous(store, args.rank, args.world_size, "closed")
    result["status"] = "PASS"
    _save(result_path, result, "complete")


if __name__ == "__main__":
    try:
        _main()
    except Exception:
        logger.exception("SHMEM AllGather worker failed; inspect its saved phase and original traceback")
        raise
