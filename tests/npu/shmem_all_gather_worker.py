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

"""One rank of direct AllGather correctness and optional msprof capture.

TileLang and native Ascend C use CPU FileStore coordination and official
SHMEM TCP/MTE bootstrap, without an HCCL process group or MPI. Matched profiling
may explicitly select the existing current-stream native HCCL AllGather instead.
The caller must verify device availability, package/native identities and prior
required correctness evidence before requesting measurement.

A failure leaves the original traceback and saved phase; it does not enter a
cleanup rendezvous that could hide the failure behind a peer timeout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
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
_SKEW_PHASES = {"none": 0, "read": 1, "ack": 2}


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
    identity = rank + 16 * (iteration + 2)
    # Signed markers and their +1 consumer results are exact in BF16. For the
    # initial 16 repeats, position zero distinguishes all supported PE/call pairs.
    # The odd column stride and row marker expose both intra-row errors and
    # whole-row shifts; no scalar pattern claims global position uniqueness.
    rows = positions // 512
    columns = positions % 512
    values = (identity + (2 * iteration + 1) * columns + 31 * rows) % 512 - 256
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


def _prepare_scratch(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    band_elements = max(1, args.skew_iterations) * 32
    elements = 2 * band_elements
    storage, scratch, guard = _guarded(elements, torch.int32, device)
    initial = torch.full((elements,), -1, dtype=torch.int32)
    initial[:band_elements] = torch.arange(band_elements, dtype=torch.int32) + 1
    expected = initial.clone()
    if args.skew_phase != "none" and args.rank == 1:
        expected[band_elements:] = initial[:band_elements]
    scratch.copy_(initial)
    return {
        "storage": storage,
        "tensor": scratch,
        "guard": guard,
        "elements": elements,
        "initial": initial,
        "expected": expected,
    }


def _check_scratch(scratch: dict[str, Any]) -> None:
    torch.testing.assert_close(scratch["tensor"].cpu(), scratch["expected"], rtol=0, atol=0)
    _check_guards(scratch["storage"], scratch["elements"], scratch["guard"])


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
            count,
            args.world_size,
            args.lanes,
            args.chunk_bytes,
            args.dtype,
            skew_phase=args.skew_phase,
            skew_iterations=args.skew_iterations,
        )
        with tilelang.tvm.transform.PassContext(opt_level=3, config=kernel_module.SHMEM_PASS_CONFIGS):
            lowered = tilelang.lower(function, target="ascendc", platform="A3")
        source_path = args.artifact_dir / f"rank-{args.rank}-count-{count}.cpp"
        source_path.write_text(lowered.kernel_source, encoding="utf-8")
        result["cases"].append({"count": count, "generated_source": _file_identity(source_path)})
    result["status"] = "LOWERED_ONLY"
    _save(result_path, result, "lowered-only/no-device-execution")


def _compile_tilelang(args: argparse.Namespace, count: int) -> tuple[Any, dict[str, Any]]:
    import tilelang

    from xllm.python.kernels_npu.tilelang.shmem_all_gather import SHMEM_PASS_CONFIGS, build_shmem_all_gather_kernel

    kernel = tilelang.compile(
        build_shmem_all_gather_kernel(
            count,
            args.world_size,
            args.lanes,
            args.chunk_bytes,
            args.dtype,
            skew_phase=args.skew_phase,
            skew_iterations=args.skew_iterations,
        ),
        out_idx=None,
        target="ascendc",
        platform="A3",
        pass_configs=SHMEM_PASS_CONFIGS,
    )
    generated_path = args.artifact_dir / f"rank-{args.rank}-count-{count}.cpp"
    generated_path.write_text(kernel.get_kernel_source(), encoding="utf-8")
    return kernel, {
        "generated_source": _file_identity(generated_path),
        "jit_library": _file_identity(kernel.adapter.libpath),
    }


def _load_native(args: argparse.Namespace, result: dict[str, Any]) -> None:
    result["native_library"] = _file_identity(args.native_library)
    result["native_build_revision_evidence"] = "caller_assertion_not_verified_build_provenance"
    torch.ops.load_library(str(args.native_library))
    namespace = torch.ops.aclshmem_ops if args.backend == "ascendc" else torch.ops.xllm_ops
    name = "all_gather" if args.backend == "ascendc" else "npu_all_gather"
    operator = getattr(namespace, name, None)
    if operator is None or not callable(operator.default):
        raise RuntimeError(f"Native {args.backend} AllGather registration {name} is not callable")
    if result["native_library"] != _file_identity(args.native_library):
        raise RuntimeError(f"{args.backend} native library identity changed during loading")


def _gather_ascendc(
    args: argparse.Namespace,
    source: torch.Tensor,
    output: torch.Tensor,
    receive: torch.Tensor,
    controls: torch.Tensor,
    epochs: torch.Tensor,
    scratch: torch.Tensor,
) -> None:
    torch.ops.aclshmem_ops.all_gather(
        source,
        output,
        receive,
        controls,
        epochs,
        scratch,
        args.rank,
        args.world_size,
        args.lanes,
        args.chunk_bytes,
        _SKEW_PHASES[args.skew_phase],
        args.skew_iterations,
    )


def _run_case(
    args: argparse.Namespace,
    store: dist.Store,
    result: dict[str, Any],
    result_path: Path,
    count: int,
    receive: torch.Tensor | None,
    controls: torch.Tensor | None,
    epochs: torch.Tensor | None,
    comm: int | None = None,
) -> None:
    dtype = getattr(torch, args.dtype)
    device = torch.device(f"npu:{args.device}")
    source_storage, source, source_guard = _guarded(count, dtype, device)
    output_storage, output, output_guard = _guarded(args.world_size * count, dtype, device)
    consumer_storage, consumed, consumer_guard = _guarded(args.world_size * count, dtype, device)
    rank_major_output = output.view(args.world_size, count)
    case: dict[str, Any] = {"count": count, "checked_iterations": [], "graph_replays": 0}
    result["cases"].append(case)
    scratch = None
    kernel = None
    if args.backend in ("shmem", "ascendc"):
        if receive is None or controls is None or epochs is None:
            raise RuntimeError("SHMEM AllGather requires its initialized symmetric buffers")
        scratch = _prepare_scratch(args, device)
    if args.backend == "shmem":
        _save(result_path, result, f"count-{count}/compile")
        kernel, identity = _compile_tilelang(args, count)
        case.update(identity)
    elif args.backend == "hccl" and not comm:
        raise RuntimeError("HCCL AllGather requires a valid initialized communicator")
    ready_phase = f"count-{count}/{'compiled' if args.backend == 'shmem' else 'ready'}"
    _save(result_path, result, ready_phase)
    _rendezvous(store, args.rank, args.world_size, ready_phase)
    rounds = (count + args.lanes * args.chunk_bytes // source.element_size() - 1) // (
        args.lanes * args.chunk_bytes // source.element_size()
    )
    stream = torch.npu.Stream() if args.profile_samples else torch.npu.current_stream()

    def _gather(
        source_tensor: torch.Tensor = source,
        output_tensor: torch.Tensor = output,
        rank_major: torch.Tensor = rank_major_output,
    ) -> None:
        if args.backend == "shmem":
            kernel(source_tensor, rank_major, receive, controls, epochs, scratch["tensor"], args.rank)
        elif args.backend == "ascendc":
            _gather_ascendc(args, source_tensor, output_tensor, receive, controls, epochs, scratch["tensor"])
        else:
            torch.ops.xllm_ops.npu_all_gather(source_tensor, output_tensor, comm)

    def _submit() -> None:
        _gather()
        # The consumer is on the same current stream immediately after gather.
        torch.add(output, 1, out=consumed)

    def _prepare(iteration: int) -> torch.Tensor:
        local = _payload(args.rank, count, iteration, dtype)
        source.copy_(local)
        output.fill_(float("nan"))
        consumed.fill_(float("nan"))
        if scratch is not None:
            scratch["tensor"].copy_(scratch["initial"])
        return local

    def _previous_epochs() -> torch.Tensor | None:
        if epochs is None:
            return None
        previous = epochs.cpu().view(args.lanes, 32)[:, 0].clone()
        assert torch.all((previous == 0) | (previous == 1)), "Invalid device epoch"
        return previous

    def _check_guards_all() -> None:
        _check_guards(source_storage, count, source_guard)
        _check_guards(output_storage, args.world_size * count, output_guard)
        _check_guards(consumer_storage, args.world_size * count, consumer_guard)

    def _check_outputs(iteration: int, local: torch.Tensor, previous_epochs: torch.Tensor | None) -> None:
        expected = torch.cat([_payload(peer, count, iteration, dtype) for peer in range(args.world_size)])
        torch.testing.assert_close(source.cpu(), local, rtol=0, atol=0)
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(consumed.cpu(), expected + 1, rtol=0, atol=0)
        _check_guards_all()
        if scratch is not None:
            _check_scratch(scratch)
            _check_state(controls, epochs, (previous_epochs + rounds) % 2, args.lanes, args.world_size)

    def _check(iteration: int, local: torch.Tensor, previous_epochs: torch.Tensor | None) -> None:
        torch.npu.synchronize()
        _check_outputs(iteration, local, previous_epochs)
        case["checked_iterations"].append(iteration)
        # Keep the next invocation from changing controls during another PE's
        # host-side state checks. This is once per collective, never per row.
        _rendezvous(store, args.rank, args.world_size, f"count-{count}/iteration-{iteration}/checked")

    def _check_capture(local: torch.Tensor, previous_epochs: torch.Tensor | None) -> None:
        torch.testing.assert_close(source.cpu(), local, rtol=0, atol=0)
        _check_guards_all()
        if scratch is not None:
            band = scratch["elements"] // 2
            torch.testing.assert_close(scratch["tensor"][:band].cpu(), scratch["initial"][:band], rtol=0, atol=0)
            _check_guards(scratch["storage"], scratch["elements"], scratch["guard"])
            observed = _previous_epochs()
            assert torch.equal(observed, previous_epochs) or torch.equal(observed, (previous_epochs + rounds) % 2), (
                "Invalid capture epoch transition"
            )
            _check_state(controls, epochs, observed, args.lanes, args.world_size)

    _save(result_path, result, f"count-{count}/warmup")
    for iteration in (-2, -1):
        previous_epochs = _previous_epochs()
        local = _prepare(iteration)
        if args.profile_samples:
            torch.npu.synchronize()
            _rendezvous(store, args.rank, args.world_size, f"count-{count}/iteration-{iteration}/prepared")
        with torch.npu.stream(stream):
            _submit()
        _check(iteration, local, previous_epochs)

    graph = None
    capture_stream = stream
    if args.mode == "graph":
        local = _prepare(0)
        torch.npu.synchronize()
        previous_epochs = _previous_epochs()
        _rendezvous(store, args.rank, args.world_size, f"count-{count}/capture")
        _save(result_path, result, f"count-{count}/capture")
        graph = torch.npu.NPUGraph()
        capture_stream = stream if args.profile_samples else torch.npu.Stream()
        with torch.npu.graph(graph, stream=capture_stream):
            _submit()
        torch.npu.synchronize()
        _rendezvous(store, args.rank, args.world_size, f"count-{count}/captured")
        # Capture is not a numerical pass and its execution is not assumed.
        _check_capture(local, previous_epochs)
        if epochs is not None:
            case["epochs_after_capture"] = _previous_epochs().tolist()
        _rendezvous(store, args.rank, args.world_size, f"count-{count}/capture-observed")

    for iteration in range(args.repeats):
        _save(result_path, result, f"count-{count}/iteration-{iteration}")
        previous_epochs = _previous_epochs()
        local = _prepare(iteration)
        if args.profile_samples or graph is not None:
            torch.npu.synchronize()
            _rendezvous(store, args.rank, args.world_size, f"count-{count}/iteration-{iteration}/prepared")
        with torch.npu.stream(capture_stream if graph is not None else stream):
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
    if not args.profile_samples:
        return

    import torch_npu

    profile_dir = args.artifact_dir / f"profile-rank-{args.rank}-count-{count}"
    profile: dict[str, Any] = {
        "backend": args.backend,
        "mode": args.mode,
        "scope": "gather_only",
        "duration_source": "msprof_device_trace",
        "evidence_version": 1,
        "device_association": "UNVERIFIED",
        "comparison_id": args.comparison_id,
        "round_id": args.round_id,
        "stage_index": args.stage_index,
        "row_width": args.row_width,
        "rows": None if args.row_width is None else count // args.row_width,
        "warmup_samples": [],
        "samples": [],
        "warmup_excluded": True,
        "consumer_in_gather_range": False,
        "graph_reset": False if args.mode == "graph" else None,
        "stream_ptr": hex(stream.npu_stream),
        "stream_ptr_is_trace_stream_id": False,
        "directory": str(profile_dir),
    }
    case["profile"] = profile
    _save(result_path, result, f"count-{count}/profile/prepare")
    profile_graph = None
    if args.mode == "graph":
        local = _prepare(args.repeats)
        torch.npu.synchronize()
        previous_epochs = _previous_epochs()
        _rendezvous(store, args.rank, args.world_size, f"count-{count}/profile/capture-prepared")
        profile_graph = torch.npu.NPUGraph()
        with torch.npu.graph(profile_graph, stream=stream):
            _gather()
        torch.npu.synchronize()
        _rendezvous(store, args.rank, args.world_size, f"count-{count}/profile/captured")
        _check_capture(local, previous_epochs)
        assert torch.all(torch.isnan(consumed.cpu())), "Gather-only capture changed consumer poison"
        _rendezvous(store, args.rank, args.world_size, f"count-{count}/profile/capture-observed")

    profiler = torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(profile_dir)),
        experimental_config=torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            export_type=[torch_npu.profiler.ExportType.Db, torch_npu.profiler.ExportType.Text],
        ),
    )
    if args.mode == "eager":
        profile["submission_pattern"] = "retained_continuous"
        calls = []
        for ordinal in range(args.profile_warmup + args.profile_samples):
            sample_index = ordinal - args.profile_warmup
            iteration = args.repeats + ordinal + 1
            buffers = {
                "source": _guarded(count, dtype, device),
                "output": _guarded(args.world_size * count, dtype, device),
                "consumer": _guarded(args.world_size * count, dtype, device),
            }
            local = _payload(args.rank, count, iteration, dtype)
            expected = torch.cat([_payload(peer, count, iteration, dtype) for peer in range(args.world_size)])
            buffers["source"][1].copy_(local)
            buffers["output"][1].fill_(float("nan"))
            buffers["consumer"][1].fill_(float("nan"))
            calls.append(
                {
                    "buffers": buffers,
                    "rank_major": buffers["output"][1].view(args.world_size, count),
                    "local": local,
                    "expected": expected,
                    "record": {
                        "sample_index": sample_index,
                        "input_iteration": iteration,
                        "gather_range": f"all_gather/rank-{args.rank}/count-{count}/sample-{sample_index}",
                    },
                }
            )

        for name, group in (
            ("warmup_samples", calls[: args.profile_warmup]),
            ("samples", calls[args.profile_warmup :]),
        ):
            torch.npu.synchronize()
            starting_epochs = _previous_epochs()
            _save(result_path, result, f"count-{count}/profile/{name}/prepared")
            _rendezvous(store, args.rank, args.world_size, f"count-{count}/profile/{name}/prepared")
            if name == "samples":
                profiler.start()
                _rendezvous(store, args.rank, args.world_size, f"count-{count}/profile/started")
            with torch.npu.stream(stream):
                for call in group:
                    buffers = call["buffers"]
                    with torch.profiler.record_function(call["record"]["gather_range"]):
                        _gather(buffers["source"][1], buffers["output"][1], call["rank_major"])
                    torch.add(buffers["output"][1], 1, out=buffers["consumer"][1])
            torch.npu.synchronize()
            if name == "samples":
                profiler.stop()
            for call in group:
                for key, expected in (
                    ("source", call["local"]),
                    ("output", call["expected"]),
                    ("consumer", call["expected"] + 1),
                ):
                    storage, tensor, guard = call["buffers"][key]
                    torch.testing.assert_close(tensor.cpu(), expected, rtol=0, atol=0)
                    _check_guards(storage, tensor.numel(), guard)
            if scratch is not None:
                _check_scratch(scratch)
                _check_state(controls, epochs, (starting_epochs + len(group) * rounds) % 2, args.lanes, args.world_size)
            profile[name] = [call["record"] for call in group]
            _rendezvous(store, args.rank, args.world_size, f"count-{count}/profile/{name}/checked")
            _save(result_path, result, f"count-{count}/profile/{name}/checked")
    else:
        profile["submission_pattern"] = "checked_single_graph_replay"
        for ordinal in range(args.profile_warmup + args.profile_samples):
            sample_index = ordinal - args.profile_warmup
            iteration = args.repeats + ordinal + 1
            phase = f"count-{count}/profile/sample-{sample_index}"
            _save(result_path, result, phase)
            previous_epochs = _previous_epochs()
            local = _prepare(iteration)
            torch.npu.synchronize()
            _rendezvous(store, args.rank, args.world_size, f"{phase}/prepared")
            if sample_index == 0:
                profiler.start()
                _rendezvous(store, args.rank, args.world_size, f"count-{count}/profile/started")
            gather_range = f"all_gather/rank-{args.rank}/count-{count}/sample-{sample_index}"
            with torch.npu.stream(stream):
                with torch.profiler.record_function(gather_range):
                    profile_graph.replay()
                torch.add(output, 1, out=consumed)
            torch.npu.synchronize()
            _check_outputs(iteration, local, previous_epochs)
            _rendezvous(store, args.rank, args.world_size, f"{phase}/checked")
            records = profile["warmup_samples"] if sample_index < 0 else profile["samples"]
            records.append({"sample_index": sample_index, "input_iteration": iteration, "gather_range": gather_range})
            _save(result_path, result, f"{phase}/checked")
        torch.npu.synchronize()
        profiler.stop()
    _rendezvous(store, args.rank, args.world_size, f"count-{count}/profile/drained")
    traces = sorted(profile_dir.rglob("trace_view.json"))
    if not traces:
        raise RuntimeError(f"Profiler did not export a device trace: {profile_dir}")
    profile["traces"] = [_file_identity(path) for path in traces]
    databases = sorted(profile_dir.rglob(f"ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_{args.rank}.db"))
    if len(databases) != 1:
        raise RuntimeError(f"Expected one finalized rank-{args.rank} profiler database, found {databases}")
    completion = databases[0].parent / "analyse.done"
    if not completion.is_file():
        raise RuntimeError(f"Profiler database export did not complete: {completion}")
    profile["databases"] = [_file_identity(path) for path in databases]
    profile["export_complete"] = _file_identity(completion)
    if profile_graph is not None:
        graph_dump = args.artifact_dir / f"profile-graph-rank-{args.rank}-count-{count}.json"
        profile_graph.debug_dump(str(graph_dump))
        json.loads(graph_dump.read_text(encoding="utf-8"))
        profile["graph_dump"] = _file_identity(graph_dump)
        profile_graph.reset()
        profile["graph_reset"] = True
        del profile_graph
    _rendezvous(store, args.rank, args.world_size, f"count-{count}/profile/closed")
    _save(result_path, result, f"count-{count}/profile/complete")


def _run_batch(
    args: argparse.Namespace,
    store: dist.Store,
    result: dict[str, Any],
    result_path: Path,
    receive: torch.Tensor,
    controls: torch.Tensor,
    epochs: torch.Tensor,
) -> None:
    dtype = getattr(torch, args.dtype)
    device = torch.device(f"npu:{args.device}")
    kernels = {}
    if args.backend == "shmem":
        _save(result_path, result, "batch/compile")
    for count in args.counts:
        case = {"count": count}
        if args.backend == "shmem":
            kernel, identity = _compile_tilelang(args, count)
            case.update(identity)
            kernels[count] = kernel
        result["cases"].append(case)

    _save(result_path, result, "batch/prepare")
    calls = []
    round_elements = args.lanes * args.chunk_bytes // torch.empty((), dtype=dtype).element_size()
    batch_counts = args.counts * args.repeats
    retained_counts = batch_counts + ([] if args.alternate_count is None else [args.alternate_count])
    for call_id, count in enumerate(retained_counts):
        source_storage, source, source_guard = _guarded(count, dtype, device)
        output_storage, output, output_guard = _guarded(args.world_size * count, dtype, device)
        consumer_storage, consumer, consumer_guard = _guarded(args.world_size * count, dtype, device)
        scratch = _prepare_scratch(args, device)
        local = _payload(args.rank, count, call_id, dtype)
        expected = torch.cat([_payload(peer, count, call_id, dtype) for peer in range(args.world_size)])
        source.copy_(local)
        output.fill_(float("nan"))
        consumer.fill_(float("nan"))
        rounds = (count + round_elements - 1) // round_elements
        calls.append(
            {
                "call_id": call_id,
                "count": count,
                "rounds": rounds,
                "kernel": kernels[count] if args.backend == "shmem" else None,
                "source": source,
                "source_storage": source_storage,
                "source_guard": source_guard,
                "local": local,
                "output": output,
                "rank_major_output": output.view(args.world_size, count),
                "output_storage": output_storage,
                "output_guard": output_guard,
                "consumer": consumer,
                "consumer_storage": consumer_storage,
                "consumer_guard": consumer_guard,
                "scratch": scratch,
                "expected": expected,
            }
        )
    groups = [calls[: len(batch_counts)]]
    if args.alternate_count is not None:
        groups.append(calls[len(batch_counts) :])
    group_rounds = [sum(call["rounds"] for call in group) for group in groups]
    torch.npu.synchronize()
    starting_epochs = epochs.cpu().view(args.lanes, 32)[:, 0].clone()
    assert torch.all((starting_epochs == 0) | (starting_epochs == 1)), "Invalid device epoch"
    result["batch"] = {
        "counts": batch_counts,
        "rounds": [call["rounds"] for call in groups[0]],
        "starting_epochs": starting_epochs.tolist(),
        "total_rounds": group_rounds[0],
        "checked_calls": [],
        "graph_groups": [[call["count"] for call in group] for group in groups],
        "group_rounds": group_rounds,
        "graph_warmups": [],
        "graph_captures": [],
        "graph_replays": [],
        "graphs_reset": False,
    }
    _rendezvous(store, args.rank, args.world_size, "batch/prepared")

    def _submit(group: list[dict[str, Any]]) -> None:
        # No per-call host checks, resets, rendezvous, allocation or logging.
        for call in group:
            if args.backend == "shmem":
                call["kernel"](
                    call["source"],
                    call["rank_major_output"],
                    receive,
                    controls,
                    epochs,
                    call["scratch"]["tensor"],
                    args.rank,
                )
            else:
                _gather_ascendc(
                    args, call["source"], call["output"], receive, controls, epochs, call["scratch"]["tensor"]
                )
            torch.add(call["output"], 1, out=call["consumer"])

    def _check_outputs(group: list[dict[str, Any]], executed: bool = True) -> None:
        for call in group:
            torch.testing.assert_close(call["source"].cpu(), call["local"], rtol=0, atol=0)
            if executed:
                torch.testing.assert_close(call["output"].cpu(), call["expected"], rtol=0, atol=0)
                torch.testing.assert_close(call["consumer"].cpu(), call["expected"] + 1, rtol=0, atol=0)
                _check_scratch(call["scratch"])
            else:
                assert torch.all(torch.isnan(call["output"].cpu())), "Inactive graph output poison changed"
                assert torch.all(torch.isnan(call["consumer"].cpu())), "Inactive graph consumer poison changed"
                torch.testing.assert_close(call["scratch"]["tensor"].cpu(), call["scratch"]["initial"], rtol=0, atol=0)
            _check_guards(call["source_storage"], call["count"], call["source_guard"])
            _check_guards(call["output_storage"], args.world_size * call["count"], call["output_guard"])
            _check_guards(call["consumer_storage"], args.world_size * call["count"], call["consumer_guard"])
            _check_guards(call["scratch"]["storage"], call["scratch"]["elements"], call["scratch"]["guard"])

    def _prepare(group: list[dict[str, Any]], iteration: int) -> None:
        for call in group:
            identity = iteration + call["call_id"]
            call["local"] = _payload(args.rank, call["count"], identity, dtype)
            call["expected"] = torch.cat(
                [_payload(peer, call["count"], identity, dtype) for peer in range(args.world_size)]
            )
            call["source"].copy_(call["local"])
            call["output"].fill_(float("nan"))
            call["consumer"].fill_(float("nan"))
            call["scratch"]["tensor"].copy_(call["scratch"]["initial"])

    def _check_group(group_id: int, previous: torch.Tensor, phase: str) -> dict[str, Any]:
        torch.npu.synchronize()
        _rendezvous(store, args.rank, args.world_size, f"{phase}/drained")
        _check_outputs(groups[group_id])
        expected_epochs = (previous + group_rounds[group_id]) % 2
        _check_state(controls, epochs, expected_epochs, args.lanes, args.world_size)
        observed_epochs = epochs.cpu().view(args.lanes, 32)[:, 0].tolist()
        record = {
            "group": group_id,
            "starting_epochs": previous.tolist(),
            "ending_epochs": observed_epochs,
            "checked_calls": [call["call_id"] for call in groups[group_id]],
        }
        _rendezvous(store, args.rank, args.world_size, f"{phase}/checked")
        return record

    if args.mode == "eager":
        _save(result_path, result, "batch/submit-and-drain")
        _submit(groups[0])
        checked = _check_group(0, starting_epochs, "batch")
        result["batch"]["checked_calls"] = checked["checked_calls"]
        _save(result_path, result, "batch/checked")
        return

    for iteration in (-2, -1):
        for group_id, group in enumerate(groups):
            phase = f"batch/warmup-{iteration}/group-{group_id}"
            _save(result_path, result, phase)
            previous = epochs.cpu().view(args.lanes, 32)[:, 0].clone()
            _prepare(group, iteration)
            torch.npu.synchronize()
            _rendezvous(store, args.rank, args.world_size, f"{phase}/prepared")
            _submit(group)
            checked = _check_group(group_id, previous, phase)
            checked["iteration"] = iteration
            result["batch"]["graph_warmups"].append(checked)

    graphs = []
    capture_stream = torch.npu.Stream()
    for group_id, group in enumerate(groups):
        phase = f"batch/capture/group-{group_id}"
        _save(result_path, result, phase)
        _prepare(group, -3)
        torch.npu.synchronize()
        before_capture = epochs.cpu().view(args.lanes, 32)[:, 0].clone()
        _rendezvous(store, args.rank, args.world_size, f"{phase}/prepared")
        graph = torch.npu.NPUGraph()
        graphs.append(graph)
        with torch.npu.graph(graph, stream=capture_stream):
            _submit(group)
        torch.npu.synchronize()
        # Capture may or may not execute; only observed replay prestate is used.
        _rendezvous(store, args.rank, args.world_size, f"{phase}/captured")
        after_capture = epochs.cpu().view(args.lanes, 32)[:, 0].clone()
        assert torch.equal(after_capture, before_capture) or torch.equal(
            after_capture, (before_capture + group_rounds[group_id]) % 2
        ), "Invalid capture epoch transition"
        _check_state(controls, epochs, after_capture, args.lanes, args.world_size)
        for call in calls:
            torch.testing.assert_close(call["source"].cpu(), call["local"], rtol=0, atol=0)
            source_band = call["scratch"]["elements"] // 2
            torch.testing.assert_close(
                call["scratch"]["tensor"][:source_band].cpu(),
                call["scratch"]["initial"][:source_band],
                rtol=0,
                atol=0,
            )
            _check_guards(call["source_storage"], call["count"], call["source_guard"])
            _check_guards(call["output_storage"], args.world_size * call["count"], call["output_guard"])
            _check_guards(call["consumer_storage"], args.world_size * call["count"], call["consumer_guard"])
            _check_guards(call["scratch"]["storage"], call["scratch"]["elements"], call["scratch"]["guard"])
        result["batch"]["graph_captures"].append(
            {"group": group_id, "starting_epochs": before_capture.tolist(), "ending_epochs": after_capture.tolist()}
        )
        _rendezvous(store, args.rank, args.world_size, f"{phase}/observed")

    last_ending = after_capture.tolist()
    for group in groups:
        _prepare(group, -4)
    torch.npu.synchronize()
    _rendezvous(store, args.rank, args.world_size, "batch/poisoned-after-capture")
    for group in groups:
        _check_outputs(group, executed=False)
    retained_groups = set()
    for iteration in range(args.graph_replays):
        for group_id, group in enumerate(groups):
            phase = f"batch/replay-{iteration}/group-{group_id}"
            _save(result_path, result, phase)
            previous = epochs.cpu().view(args.lanes, 32)[:, 0].clone()
            assert previous.tolist() == last_ending, "Protocol state changed between checked operations"
            assert torch.all((previous == 0) | (previous == 1)), "Invalid replay prestate"
            _prepare(group, iteration)
            torch.npu.synchronize()
            _rendezvous(store, args.rank, args.world_size, f"{phase}/prepared")
            with torch.npu.stream(capture_stream):
                graphs[group_id].replay()
            checked = _check_group(group_id, previous, phase)
            inactive_groups = [other_id for other_id in range(len(groups)) if other_id != group_id]
            for inactive_id in inactive_groups:
                _check_outputs(groups[inactive_id], executed=inactive_id in retained_groups)
            checked["checked_inactive_groups"] = inactive_groups
            checked["iteration"] = iteration
            result["batch"]["graph_replays"].append(checked)
            last_ending = checked["ending_epochs"]
            retained_groups.add(group_id)
            _rendezvous(store, args.rank, args.world_size, f"{phase}/retained-checked")

    for group_id in range(len(groups)):
        starts = {
            tuple(record["starting_epochs"])
            for record in result["batch"]["graph_replays"]
            if record["group"] == group_id
        }
        assert starts == {(0,) * args.lanes, (1,) * args.lanes}, "Live graphs did not exercise both epoch parities"
    torch.npu.synchronize()
    _rendezvous(store, args.rank, args.world_size, "batch/graphs-drained")
    for graph in graphs:
        graph.reset()
    graphs.clear()
    _rendezvous(store, args.rank, args.world_size, "batch/graphs-reset")
    result["batch"]["graphs_reset"] = True
    result["batch"]["final_epochs"] = last_ending
    result["batch"]["checked_calls"] = list(range(len(batch_counts)))
    _save(result_path, result, "batch/checked")


def _bootstrap_store(args: argparse.Namespace, result: dict[str, Any], identities: tuple[str, ...]) -> dist.Store:
    store = dist.FileStore(str(args.artifact_dir / "bootstrap.store"), args.world_size)
    store.set_timeout(timedelta(seconds=120))
    if store.add(f"participant/{args.rank}", 1) != 1:
        raise RuntimeError("FileStore namespace already used by this rank")
    configuration_fields = (
        "backend",
        "profile_samples",
        "profile_warmup",
        "comparison_id",
        "round_id",
        "stage_index",
        "row_width",
        "native_build_revision",
        "native_build_revision_owner",
        "world_size",
        "mode",
        "stress_batch",
        "skew_phase",
        "skew_iterations",
        "graph_replays",
        "alternate_count",
        "dtype",
        "counts",
        "lanes",
        "chunk_bytes",
        "heap_bytes",
        "repeats",
        "versions",
        "environment",
    )
    configuration_data = {name: result[name] for name in configuration_fields}
    configuration_data["source_sha256"] = {name: result[name]["sha256"] for name in identities}
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
    return store


def _run_hccl(args: argparse.Namespace, result: dict[str, Any], result_path: Path) -> None:
    import torch_npu

    torch.set_num_threads(1)
    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
    result["versions"] = {"torch": torch.__version__, "torch_npu": torch_npu.__version__}
    _load_native(args, result)
    _save(result_path, result, "bootstrap")
    store = _bootstrap_store(args, result, ("worker", "native_library"))
    if dist.is_initialized():
        raise RuntimeError("Test requires an uninitialized HCCL process")
    dist.init_process_group(
        backend="hccl",
        init_method=(args.artifact_dir / "hccl.store").as_uri(),
        world_size=args.world_size,
        rank=args.rank,
        timeout=timedelta(seconds=120),
    )
    options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
    options.hccl_config = {"hccl_op_expansion_mode": 3}
    result["hccl_config"] = dict(options.hccl_config)
    _save(result_path, result, "bootstrap/hccl-config")
    group = dist.new_group(
        ranks=list(range(args.world_size)),
        backend="hccl",
        timeout=timedelta(seconds=120),
        pg_options=options,
    )
    warm = torch.ones(1, dtype=torch.float32, device=device)
    dist.all_reduce(warm, group=group)
    torch.npu.synchronize()
    comm = group._get_backend(device).get_hccl_comm(device.index)
    if not comm:
        raise RuntimeError(f"No HCCL communicator on {device}")
    result["bootstrap"] = "HCCL/ProcessGroup"
    result["hccl_comm"] = hex(comm)
    result["group_name"] = group.group_name
    result["pg_owner"] = "torch.distributed.new_group(ranks=all_PEs, backend='hccl')"
    _rendezvous(store, args.rank, args.world_size, "initialized")
    for count in args.counts:
        _run_case(args, store, result, result_path, count, None, None, None, comm)
    _save(result_path, result, "close/drain")
    torch.npu.synchronize()
    _rendezvous(store, args.rank, args.world_size, "drained")
    dist.destroy_process_group(group)
    dist.destroy_process_group()
    _rendezvous(store, args.rank, args.world_size, "closed")
    result["status"] = "PASS"
    _save(result_path, result, "complete")


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("shmem", "ascendc", "hccl"), default="shmem")
    parser.add_argument("--profile-samples", type=int, default=0)
    parser.add_argument("--profile-warmup", type=int, default=20)
    parser.add_argument("--comparison-id")
    parser.add_argument("--round-id", type=int)
    parser.add_argument("--stage-index", type=int)
    parser.add_argument("--row-width", type=int)
    parser.add_argument("--native-library", type=Path)
    parser.add_argument("--native-build-revision")
    parser.add_argument("--kernel-source", type=Path, help="Standalone Ascend C kernel source identity")
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, choices=(2, 4, 8, 16), required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--mode", choices=("eager", "graph"), default="eager")
    parser.add_argument("--skew-phase", choices=("none", "read", "ack"), default="none")
    parser.add_argument("--skew-iterations", type=int, default=0)
    parser.add_argument("--graph-replays", type=int, help="Checked retained graph replay rounds")
    parser.add_argument("--alternate-count", type=int, help="Second live graph contains one call of this count")
    parser.add_argument(
        "--stress-batch", action="store_true", help="Retain counts * repeats calls and check after each full drain"
    )
    parser.add_argument(
        "--compile-only", action="store_true", help="Lower DSL only; no SHMEM bootstrap or NPU execution"
    )
    parser.add_argument("--counts", type=int, nargs="+", default=[64, 65, 127, 128, 129])
    parser.add_argument("--lanes", type=int, default=2)
    parser.add_argument("--chunk-bytes", type=int, default=128)
    parser.add_argument("--heap-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--repeats", type=int, default=16)
    args = parser.parse_args()
    if args.profile_samples < 0:
        parser.error("profile-samples must be nonnegative")
    if args.profile_samples:
        if args.profile_samples < 2 or args.profile_warmup < 2:
            parser.error("Gather-only profiling requires at least two warmup and measured samples")
        if (
            args.compile_only
            or args.stress_batch
            or args.skew_phase != "none"
            or args.skew_iterations != 0
            or args.graph_replays is not None
            or args.alternate_count is not None
        ):
            parser.error("Profiling excludes compile-only, retained-batch and skew instrumentation")
        if args.comparison_id is None or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", args.comparison_id) is None:
            parser.error("comparison-id must be a nonempty 1..128 character alphanumeric/underscore/dot/hyphen ID")
        if args.round_id is None or args.round_id < 0 or args.stage_index is None or args.stage_index < 0:
            parser.error("Profiling requires nonnegative round-id and stage-index")
        if args.row_width is not None and (args.row_width <= 0 or any(count % args.row_width for count in args.counts)):
            parser.error("row-width must be positive and divide every count")
    elif args.profile_warmup != 20 or any(
        value is not None for value in (args.comparison_id, args.round_id, args.stage_index, args.row_width)
    ):
        parser.error("Profiling metadata requires positive profile-samples")
    if args.compile_only and args.backend != "shmem":
        parser.error("compile-only is available only for the TileLang SHMEM backend")
    if args.backend in ("hccl", "ascendc"):
        if args.native_library is None or not args.native_library.is_absolute() or not args.native_library.is_file():
            parser.error(f"{args.backend} requires native-library as an existing absolute file")
        args.native_library = args.native_library.resolve()
        if args.native_build_revision is None or re.fullmatch(r"[0-9a-f]{40}", args.native_build_revision) is None:
            parser.error(f"{args.backend} requires the full native-build-revision caller assertion")
    elif args.native_library is not None or args.native_build_revision is not None:
        parser.error("TileLang SHMEM must not load a native library")
    if args.backend == "ascendc":
        if args.kernel_source is None or not args.kernel_source.is_absolute() or not args.kernel_source.is_file():
            parser.error("Ascend C requires kernel-source as an existing absolute file")
        args.kernel_source = args.kernel_source.resolve()
    elif args.kernel_source is not None:
        parser.error("kernel-source is only valid for the standalone Ascend C backend")
    if args.backend == "hccl":
        if not args.profile_samples:
            parser.error("The HCCL backend is only available for matched gather-only measurement")
        if os.environ.get("HCCL_OP_EXPANSION_MODE") != "AIV" or any(
            not os.environ.get(name) for name in ("HCCL_HOST_SOCKET_PORT_RANGE", "HCCL_NPU_SOCKET_PORT_RANGE")
        ):
            parser.error("HCCL requires explicit AIV expansion and host/NPU socket port ranges")
    if not 0 <= args.rank < args.world_size or args.device < 0:
        parser.error("Require 0 <= rank < world-size and a nonnegative explicit device")
    if args.stress_batch and args.compile_only:
        parser.error("Retained batch validation requires device execution")
    retained_graph = args.stress_batch and args.mode == "graph"
    if not retained_graph and (args.graph_replays is not None or args.alternate_count is not None):
        parser.error("Graph replay options require --stress-batch --mode graph")
    if retained_graph:
        if args.graph_replays is None:
            args.graph_replays = 16
        if args.graph_replays < 2:
            parser.error("Retained graphs require at least two replay rounds")
        if args.alternate_count is not None and args.alternate_count not in args.counts:
            parser.error("Alternate count must belong to --counts")
    if (args.skew_phase == "none" and args.skew_iterations != 0) or (
        args.skew_phase != "none" and not 0 < args.skew_iterations <= 32768
    ):
        parser.error("Require skew-iterations=0 for none, or 1..32768 for read/ack instrumentation")
    if args.repeats <= 0 or args.lanes <= 0:
        parser.error("Require positive repeats and positive lanes")
    if args.backend == "ascendc" and args.lanes > 48:
        parser.error("Ascend C supports lanes in 1..48; native launch also checks hardware capacity")
    if args.backend != "ascendc" and args.lanes % 2:
        parser.error("TileLang MIX and matched HCCL configuration require even lanes")
    if not 0 < args.chunk_bytes <= 64 * 1024 or args.chunk_bytes % 128:
        parser.error("Require chunk-bytes <= 64 KiB and a positive multiple of 128")
    if len(set(args.counts)) != len(args.counts) or any(
        not 0 < args.world_size * count <= (1 << 31) - 1 for count in args.counts
    ):
        parser.error("Require distinct positive counts with world-size * count <= INT32_MAX")
    if retained_graph:
        round_elements = args.lanes * args.chunk_bytes // (4 if args.dtype == "float32" else 2)
        total_rounds = sum((count + round_elements - 1) // round_elements for count in args.counts) * args.repeats
        if args.alternate_count is not None:
            total_rounds += (args.alternate_count + round_elements - 1) // round_elements
        if total_rounds % 2 != 1:
            parser.error("Retained graph sequence must have odd total rounds to exercise both epoch parities")
    required_bytes = args.world_size * args.lanes * args.chunk_bytes + (2 * args.world_size + 1) * args.lanes * 128
    if args.heap_bytes < required_bytes:
        parser.error(f"Symmetric buffers require at least {required_bytes} heap bytes")
    args.artifact_dir = args.artifact_dir.resolve()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.artifact_dir / f"rank-{args.rank}.json"
    result: dict[str, Any] = {
        "backend": args.backend,
        "profile_samples": args.profile_samples,
        "profile_warmup": args.profile_warmup,
        "comparison_id": args.comparison_id,
        "round_id": args.round_id,
        "stage_index": args.stage_index,
        "row_width": args.row_width,
        "native_build_revision": args.native_build_revision,
        "native_build_revision_owner": {"ascendc": "workspace_source", "hccl": "xllm_source"}.get(args.backend),
        "rank": args.rank,
        "world_size": args.world_size,
        "device": args.device,
        "mode": args.mode,
        "stress_batch": args.stress_batch,
        "skew_phase": args.skew_phase,
        "skew_iterations": args.skew_iterations,
        "graph_replays": args.graph_replays,
        "alternate_count": args.alternate_count,
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
            for name in (
                "ASCEND_RT_VISIBLE_DEVICES",
                "ASCEND_HOME_PATH",
                "TL_ROOT",
                "SHMEM_HOME_PATH",
                "HCCL_OP_EXPANSION_MODE",
                "HCCL_HOST_SOCKET_PORT_RANGE",
                "HCCL_NPU_SOCKET_PORT_RANGE",
                "HCCL_IF_IP",
                "HCCL_ALGO",
                "TORCH_HCCL_ZERO_COPY",
            )
        },
    }
    # Refuse an earlier attempt's rank result or rendezvous namespace.
    with result_path.open("x", encoding="utf-8") as result_file:
        result_file.write(json.dumps(result, indent=2) + "\n")
    _save(result_path, result, "imports")
    if args.compile_only:
        _compile_only(args, result, result_path)
        return
    if args.backend == "hccl":
        _run_hccl(args, result, result_path)
        return
    # Native initialization resets its log level; the environment overrides it.
    # File logging preserves diagnostics if native failure also crashes at exit.
    os.environ["SHMEM_LOG_LEVEL"] = "DEBUG"
    os.environ["SHMEM_LOG_TO_STDOUT"] = "0"
    os.environ["SHMEM_LOG_PATH"] = str(args.artifact_dir)
    result["environment"].update(
        {name: os.environ[name] for name in ("SHMEM_LOG_LEVEL", "SHMEM_LOG_TO_STDOUT", "SHMEM_LOG_PATH")}
    )
    import shmem
    import torch_npu
    from shmem import InitAttr
    from shmem.construct_tensor import construct_tensor_from_ptr

    torch.set_num_threads(1)
    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
    dtype = getattr(torch, args.dtype)
    result["versions"] = {"torch": torch.__version__, "torch_npu": torch_npu.__version__}
    result["shmem_python"] = _file_identity(shmem.__file__)
    result["shmem_native"] = _file_identity(sys.modules["shmem._pyshmem"].__file__)
    identities = ("worker", "kernel", "shmem_python", "shmem_native")
    if args.backend == "ascendc":
        result["kernel"] = _file_identity(args.kernel_source)
        _load_native(args, result)
        identities += ("native_library",)
    else:
        import tilelang

        from xllm.python.kernels_npu.tilelang import shmem_all_gather as kernel_module

        result["versions"]["tilelang"] = tilelang.__version__
        result["kernel"] = _file_identity(kernel_module.__file__)
        result["tilelang_python"] = _file_identity(tilelang.__file__)
        result["tilelang_native"] = _file_identity(tilelang._LIB_PATH)
        identities += ("tilelang_python", "tilelang_native")
    _save(result_path, result, "bootstrap")
    store = _bootstrap_store(args, result, identities)
    if shmem.aclshmemx_init_status() != shmem.InitStatus.NOT_INITIALIZED:
        raise RuntimeError("Test requires an uninitialized SHMEM process")
    # Use the selected SHMEM version's official config-store TCP/MTE bootstrap.
    # This is selected before execution, not a retry or a backend fallback.
    listener = None
    if args.rank == 0:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        listener.listen(args.world_size)
        store.set("tcp_port", str(listener.getsockname()[1]))
    port = int(store.get("tcp_port"))
    attributes = InitAttr()
    attributes.my_rank = args.rank
    attributes.n_ranks = args.world_size
    attributes.local_mem_size = args.heap_bytes
    attributes.ip_port = f"tcp://127.0.0.1:{port}"
    attributes.option_attr.data_op_engine_type = shmem.OpEngineType.MTE
    if listener is not None:
        # Transfer ownership before native init; its error cleanup may close FD.
        attributes.option_attr.sockFd = listener.detach()
    ret = shmem.set_conf_store_tls(False, "")
    if ret != 0:
        raise RuntimeError(f"set_conf_store_tls failed: {ret}")
    _save(result_path, result, "bootstrap/native-init")
    ret = shmem.aclshmem_init(attributes)
    result["shmem_init_return"] = ret
    if ret != 0:
        result["status"] = "FAIL"
        _save(result_path, result, "bootstrap/native-init-failed")
        raise RuntimeError(f"aclshmem_init failed: {ret}")
    result["bootstrap"] = "SHMEM/TCP/MTE"
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
    if args.stress_batch:
        _run_batch(args, store, result, result_path, receive, controls, epochs)
    else:
        for count in args.counts:
            _run_case(args, store, result, result_path, count, receive, controls, epochs)
    _save(result_path, result, "close/drain")
    torch.npu.synchronize()
    _rendezvous(store, args.rank, args.world_size, "drained")
    del receive, controls, epochs
    views.clear()
    for pointer in reversed(allocations):
        shmem.aclshmem_free(pointer)
    ret = shmem.aclshmem_finalize()
    if ret != 0:
        raise RuntimeError(f"aclshmem_finalize failed: {ret}")
    _rendezvous(store, args.rank, args.world_size, "closed")
    result["status"] = "PASS"
    _save(result_path, result, "complete")


if __name__ == "__main__":
    try:
        _main()
    except Exception:
        logger.exception("AllGather worker failed; inspect its saved phase and original traceback")
        raise
