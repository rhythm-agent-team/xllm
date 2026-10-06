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

TileLang and native SHMEM Ascend C use CPU FileStore coordination and official
SHMEM TCP/MTE bootstrap, without an HCCL process group or MPI. The hccl_aiv
backend uses a ProcessGroup only to prepare HCCL/MC2 resources; its AllGather
payload executes in the custom AIV kernel. Its test entry permits eager numerical
checks, basic PE2/8/16 normal graph entries and a bounded two-live-graph PE2 entry.
Larger-rank entry support is not runtime qualification; capture alone is not a
numerical pass. Normal MC2 aligned profiling reuses the AlltoAll-to-50 graph with
20 full-graph warmups; entry support is not device-duration or performance proof.
MC2 uses communicator-owned lane state initialized once outside capture. Read/ack
skew is available for correctness only; its bounded per-lane PIPE_ALL iterations
are not time units or SHMEM scratch-copy steps. Eager stress remains unsupported.
The retired whole-kernel-barrier diagnostic entries are not part of this protocol.
Matched profiling may select the existing current-stream native HCCL AllGather
instead. The caller must verify device availability, package/native identities
and prior required correctness evidence before requesting measurement.

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
import threading
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.logger import logger

_GUARD_BYTES = 128
_GUARD_VALUE = -123
_SKEW_PHASES = {"none": 0, "read": 1, "ack": 2, "ready": 3}
_PREFIX_ELEMENTS = 32
_MC2_PROTOCOL = "mc2_lane_fanout_v2"
# Retain failed MC2 state until the process boundary reports the original error.
_active_prepared: Any | None = None
_active_graph: dict[str, Any] | None = None


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


def _prefix_payload(sender: int, destination: int, iteration: int, world_size: int) -> torch.Tensor:
    marker = ((iteration % 128) * world_size + sender) * world_size + destination
    return torch.arange(_PREFIX_ELEMENTS, dtype=torch.float32) + marker * _PREFIX_ELEMENTS


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
    if args.backend == "hccl_aiv":
        if not callable(torch.classes.hccl_aiv_ops.PreparedAllGather):
            raise RuntimeError("Native hccl_aiv PreparedAllGather class is not callable")
    else:
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
    prefix_group: dist.ProcessGroup | None = None,
    hccl_comm_name: str | None = None,
) -> None:
    global _active_prepared, _active_graph

    dtype = getattr(torch, args.dtype)
    device = torch.device(f"npu:{args.device}")
    source_storage, source, source_guard = _guarded(count, dtype, device)
    output_storage, output, output_guard = _guarded(args.world_size * count, dtype, device)
    consumer_storage, consumed, consumer_guard = _guarded(args.world_size * count, dtype, device)
    rank_major_output = output.view(args.world_size, count)
    case: dict[str, Any] = {"count": count, "checked_iterations": [], "graph_replays": 0}
    if args.mc2_submission_identity:
        case["submission_identities"] = []
    result["cases"].append(case)
    scratch = None
    kernel = None
    prepared = None
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
    elif args.backend == "hccl_aiv":
        if not hccl_comm_name:
            raise RuntimeError("MC2 AIV AllGather requires the initialized HCCL communicator name")
        if _active_prepared is not None:
            raise RuntimeError("A previous MC2 prepared object was not successfully closed")
        if (args.mode == "graph" or args.profile_samples) and _active_graph is not None:
            raise RuntimeError("A previous MC2 graph was not successfully reset")
        _save(result_path, result, f"count-{count}/prepare")
        prepared = torch.classes.hccl_aiv_ops.PreparedAllGather(
            source,
            output,
            hccl_comm_name,
            args.world_size,
            args.chunk_bytes,
            args.lanes,
            _SKEW_PHASES[args.skew_phase],
            args.skew_iterations,
            args.row_width or 0,
        )
        _active_prepared = prepared
        case["lane_schedule"] = dict(prepared.schedule())
    ready_phase = f"count-{count}/{'compiled' if args.backend == 'shmem' else 'ready'}"
    _save(result_path, result, ready_phase)
    _rendezvous(store, args.rank, args.world_size, ready_phase)
    if scratch is not None:
        round_elements = args.lanes * args.chunk_bytes // source.element_size()
        rounds = (count + round_elements - 1) // round_elements
    stream = torch.npu.Stream() if args.profile_samples and args.backend != "hccl_aiv" else torch.npu.current_stream()

    def _gather(
        source_tensor: torch.Tensor = source,
        output_tensor: torch.Tensor = output,
        rank_major: torch.Tensor = rank_major_output,
    ) -> None:
        if args.backend == "shmem":
            kernel(source_tensor, rank_major, receive, controls, epochs, scratch["tensor"], args.rank)
        elif args.backend == "ascendc":
            _gather_ascendc(args, source_tensor, output_tensor, receive, controls, epochs, scratch["tensor"])
        elif args.backend == "hccl_aiv":
            if args.mc2_submission_identity:
                phase = result["phase"]
                observation = {
                    "submission_index": len(case["submission_identities"]),
                    "phase": phase,
                    "host_native_thread_id": threading.get_native_id(),
                    "run_returned": False,
                }
                case["submission_identities"].append(observation)
                observation["before"] = dict(prepared.eager_submission_identity())
                _save(result_path, result, phase)
                try:
                    prepared.run()
                    observation["run_returned"] = True
                    observation["after"] = dict(prepared.eager_submission_identity())
                finally:
                    # Preserve partial observations and propagate launch/query errors.
                    # Queries and file writes precede the consumer/drain and perturb
                    # submission timing; these records do not qualify performance.
                    _save(result_path, result, phase)
            else:
                prepared.run()
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
        # Keep the next invocation from changing state during another PE's host
        # checks. hccl_aiv passes here do not qualify continuous-call safety.
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
        if args.backend == "hccl_aiv":
            _active_graph = {"graph": graph, "stream": stream, "consumer_storage": consumer_storage}
        capture_stream = stream if args.profile_samples or args.backend == "hccl_aiv" else torch.npu.Stream()
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

    if args.backend == "hccl_aiv" and args.mode == "eager":
        iteration = args.repeats
        local = _prepare(iteration)
        torch.npu.synchronize()
        phase = f"count-{count}/continuous-pair"
        case["continuous_pair"] = {"calls": 2, "checked_final_output": False}
        _save(result_path, result, phase)
        _rendezvous(store, args.rank, args.world_size, phase)
        with torch.npu.stream(stream):
            _gather()
            _gather()
            torch.add(output, 1, out=consumed)
        torch.npu.synchronize()
        _check_outputs(iteration, local, None)
        case["continuous_pair"]["checked_final_output"] = True
    else:
        torch.npu.synchronize()
    if graph is not None:
        graph.reset()
        if args.backend == "hccl_aiv":
            _active_graph = None
        del graph
    if prepared is not None:
        _save(result_path, result, f"count-{count}/close")
        prepared.close()
        _active_prepared = None
        prepared = None
        case["prepared_closed"] = True
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
        "profiler": f"CPU+NPU_{args.profile_level}",
        "profile_level": args.profile_level,
        "evidence_version": 1,
        "device_association": "UNVERIFIED",
        "comparison_id": args.comparison_id,
        "round_id": args.round_id,
        "stage_index": args.stage_index,
        "profile_rank0_delay_ms": None,
        "profile_submission": args.profile_submission,
        "profile_graph_warmup": args.profile_graph_warmup,
        "row_width": args.row_width,
        "rows": None if args.row_width is None else count // args.row_width,
        "warmup_samples": [],
        "samples": [],
        "warmup_excluded": True,
        "consumer_in_gather_range": False,
        "graph_reset": False,
        "stream_ptr": hex(stream.npu_stream),
        "stream_ptr_is_trace_stream_id": False,
        "directory": str(profile_dir),
    }
    if args.backend == "hccl_aiv":
        profile.update(
            {
                "mc2_profile_level": args.profile_level,
                "mc2_capture_identity": args.mc2_capture_identity,
                "protocol": result["backend_identity"]["protocol"],
                "lane_schedule": case["lane_schedule"],
            }
        )
        if args.mc2_capture_identity:
            profile.update(
                {
                    "purpose": "performance_measurement",
                    "performance_verdict": "UNASSESSED",
                    "capture_identity_records": [],
                    "replay_thread_identity": {},
                }
            )
    case["profile"] = profile
    _save(result_path, result, f"count-{count}/profile/prepare")
    profiler = torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(profile_dir)),
        experimental_config=torch_npu.profiler._ExperimentalConfig(
            profiler_level=(
                torch_npu.profiler.ProfilerLevel.Level2
                if args.profile_level == "Level2"
                else torch_npu.profiler.ProfilerLevel.Level1
            ),
            export_type=[torch_npu.profiler.ExportType.Db, torch_npu.profiler.ExportType.Text],
        ),
    )
    profile["submission_pattern"] = "alltoall_aligned_graph"
    calls = []
    prefix_buffers = {}
    if args.backend == "hccl_aiv":
        _active_graph = {
            "stream": stream,
            "calls": calls,
            "prefix_group": prefix_group,
            "prefix_buffers": prefix_buffers,
            "source_storage": source_storage,
            "output_storage": output_storage,
            "consumer_storage": consumer_storage,
            "consumed": consumed,
        }
        profile["prepared_closed"] = False
    for sample_index in range(args.profile_samples):
        iteration = args.repeats + sample_index + 1
        buffers = {
            "source": _guarded(count, dtype, device),
            "output": _guarded(args.world_size * count, dtype, device),
        }
        local = _payload(args.rank, count, iteration, dtype)
        expected = torch.cat([_payload(peer, count, iteration, dtype) for peer in range(args.world_size)])
        buffers["source"][1].copy_(local)
        buffers["output"][1].fill_(float("nan"))
        calls.append(
            {
                "buffers": buffers,
                "rank_major": buffers["output"][1].view(args.world_size, count),
                "local": local,
                "expected": expected,
                "record": {"sample_index": sample_index, "input_iteration": iteration},
            }
        )
        if args.backend == "hccl_aiv":
            calls[-1]["prepared"] = torch.classes.hccl_aiv_ops.PreparedAllGather(
                buffers["source"][1],
                buffers["output"][1],
                hccl_comm_name,
                args.world_size,
                args.chunk_bytes,
                args.lanes,
                _SKEW_PHASES[args.skew_phase],
                args.skew_iterations,
                args.row_width or 0,
            )
            if dict(calls[-1]["prepared"].schedule()) != case["lane_schedule"]:
                raise RuntimeError("MC2 profiling owners disagree on the static lane schedule")
            if args.mc2_capture_identity:
                calls[-1]["prepared"].enable_capture_identity()

    def _check_profile_group(executed: bool | None) -> str:
        observed = set()
        for call in calls:
            for key, expected in (("source", call["local"]), ("output", call["expected"])):
                storage, tensor, guard = call["buffers"][key]
                actual = tensor.cpu()
                if key == "output" and torch.all(torch.isnan(actual)):
                    observed.add("POISONED")
                else:
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    if key == "output":
                        observed.add("CORRECT")
                _check_guards(storage, tensor.numel(), guard)
        assert len(observed) == 1, "Retained group has partially executed outputs"
        output_state = observed.pop()
        if executed is not None:
            assert output_state == ("CORRECT" if executed else "POISONED"), "Unexpected retained output state"
        return output_state

    if prefix_group is None:
        raise RuntimeError("AlltoAll-aligned graph requires its initialized HCCL prefix group")
    prefix_elements = _PREFIX_ELEMENTS
    prefix_buffers["source"] = _guarded(args.world_size * prefix_elements, torch.float32, device)
    prefix_buffers["output"] = _guarded(args.world_size * prefix_elements, torch.float32, device)
    prefix_local = torch.empty(0)
    prefix_expected = torch.empty(0)

    def _prepare_prefix(iteration: int) -> None:
        nonlocal prefix_local, prefix_expected
        prefix_local = torch.cat(
            [_prefix_payload(args.rank, peer, iteration, args.world_size) for peer in range(args.world_size)]
        )
        prefix_expected = torch.cat(
            [_prefix_payload(peer, args.rank, iteration, args.world_size) for peer in range(args.world_size)]
        )
        prefix_buffers["source"][1].copy_(prefix_local)
        prefix_buffers["output"][1].fill_(float("nan"))

    def _submit_prefix() -> None:
        work = dist.all_to_all_single(
            prefix_buffers["output"][1], prefix_buffers["source"][1], group=prefix_group, async_op=True
        )
        # The public wait orders this current stream after the HCCL stream.
        work.wait()

    def _check_prefix(executed: bool | None) -> str:
        torch.testing.assert_close(prefix_buffers["source"][1].cpu(), prefix_local, rtol=0, atol=0)
        actual = prefix_buffers["output"][1].cpu()
        state = "POISONED" if torch.all(torch.isnan(actual)) else "CORRECT"
        if state == "CORRECT":
            torch.testing.assert_close(actual, prefix_expected, rtol=0, atol=0)
        if executed is not None:
            assert state == ("CORRECT" if executed else "POISONED"), "Unexpected AlltoAll prefix output"
        for storage, tensor, guard in prefix_buffers.values():
            _check_guards(storage, tensor.numel(), guard)
        return state

    def _prepare_aligned(iteration_offset: int, prefix_iteration: int) -> list[int]:
        iterations = []
        for call in calls:
            iteration = call["record"]["input_iteration"] + iteration_offset
            local = _payload(args.rank, count, iteration, dtype)
            assert not torch.equal(local, call["local"]), "Graph input did not change between phases"
            call["local"] = local
            call["expected"] = torch.cat([_payload(peer, count, iteration, dtype) for peer in range(args.world_size)])
            call["buffers"]["source"][1].copy_(local)
            call["buffers"]["output"][1].fill_(float("nan"))
            iterations.append(iteration)
        _prepare_prefix(prefix_iteration)
        torch.npu.synchronize()
        _check_profile_group(executed=False)
        _check_prefix(executed=False)
        return iterations

    def _check_aligned(previous: torch.Tensor | None, executed: bool | None) -> dict[str, Any]:
        output_state = _check_profile_group(executed)
        prefix_state = _check_prefix(executed)
        assert output_state != "CORRECT" or prefix_state == "CORRECT", "AllGather executed without its prefix"
        if scratch is not None:
            _check_scratch(scratch)
            advanced = len(calls) * rounds if output_state == "CORRECT" else 0
            _check_state(controls, epochs, (previous + advanced) % 2, args.lanes, args.world_size)
        checked: dict[str, Any] = {"outputs": output_state, "prefix_outputs": prefix_state}
        if args.backend != "hccl_aiv":
            ending = _previous_epochs()
            checked.update(
                {
                    "starting_epochs": None if previous is None else previous.tolist(),
                    "ending_epochs": None if ending is None else ending.tolist(),
                }
            )
        return checked

    _prepare_prefix(-args.profile_graph_warmup - 2)
    torch.npu.synchronize()
    phase = f"count-{count}/profile/prefix-bootstrap"
    _rendezvous(store, args.rank, args.world_size, f"{phase}/prepared")
    _save(result_path, result, phase)
    with torch.npu.stream(stream):
        _submit_prefix()
    torch.npu.synchronize()
    _check_prefix(executed=True)
    _rendezvous(store, args.rank, args.world_size, f"{phase}/checked")
    aligned: dict[str, Any] = {
        "prefix": {
            "api": "torch.distributed.all_to_all_single",
            "collective": "all_to_all",
            "dtype": "float32",
            "elements_per_peer": prefix_elements,
            "hccl_op_expansion_mode": 0,
            "async_op": True,
            "wait": "current_stream_dependency",
            "bootstrap_checked": True,
        },
        "capture": {},
        "warmups": [],
        "warmup_graph_replays": 0,
        "warmup_allgather_calls": 0,
        "measured_graph_replays": 0,
        "measured_prefix_input_iteration": 0,
        "measured_prefix_checked": False,
        "checked_samples": [],
        "replay_range": f"all_gather_aligned/rank-{args.rank}/count-{count}/replay-0",
        "consumer_in_graph": False,
    }
    if args.backend != "hccl_aiv":
        aligned.update({"starting_epochs": None, "ending_epochs": None})
    profile["aligned_graph"] = aligned
    # Disjoint negative phases advance by 65 so even count=1 changes;
    # the last warmup offset is -1 mod 32 before the measured inputs.
    last_offset = -(32 * (calls[-1]["record"]["input_iteration"] // 32 + 2) + 1)
    capture_offset = last_offset - 65 * args.profile_graph_warmup
    if capture_offset % 32 == 0:
        capture_offset -= 1
    capture_iterations = _prepare_aligned(capture_offset, -args.profile_graph_warmup - 1)
    previous_epochs = _previous_epochs()
    phase = f"count-{count}/profile/aligned-capture"
    _save(result_path, result, f"{phase}/prepared")
    _rendezvous(store, args.rank, args.world_size, f"{phase}/prepared")
    profile_graph = torch.npu.NPUGraph()
    if args.backend == "hccl_aiv":
        _active_graph["graph"] = profile_graph
    with torch.npu.graph(profile_graph, stream=stream):
        _submit_prefix()
        for call in calls:
            if args.backend == "hccl_aiv":
                if args.mc2_capture_identity:
                    call["capture_host_thread_id"] = threading.get_native_id()
                call["prepared"].run()
            else:
                _gather(call["buffers"]["source"][1], call["buffers"]["output"][1], call["rank_major"])
    torch.npu.synchronize()
    _rendezvous(store, args.rank, args.world_size, f"{phase}/drained")
    if args.mc2_capture_identity:
        identity_dump = profile_dir / "capture-instance-graph.json"
        if identity_dump.exists() or identity_dump.is_symlink():
            raise RuntimeError(f"MC2 capture identity dump path is not fresh: {identity_dump}")
        profile_dir.mkdir(parents=True, exist_ok=True)
        calls[0]["prepared"].dump_capture_identity_graph(str(identity_dump))
        profile["capture_identity_graph_json"] = json.loads(identity_dump.read_text(encoding="utf-8"))
        profile["capture_identity_graph"] = _file_identity(identity_dump)
        for call, iteration in zip(calls, capture_iterations, strict=True):
            profile["capture_identity_records"].append(
                {
                    "sample_index": call["record"]["sample_index"],
                    "input_iteration": iteration,
                    "host_thread_id": call["capture_host_thread_id"],
                    **dict(call["prepared"].take_capture_identity_record()),
                }
            )
        profile["capture_identity_dump_instance_handle"] = profile["capture_identity_records"][0][
            "before_capture_instance_handle"
        ]
        _save(result_path, result, f"{phase}/identity-observed")
    aligned["capture"] = {
        "input_iterations": capture_iterations,
        "prefix_input_iteration": -args.profile_graph_warmup - 1,
        **_check_aligned(previous_epochs, executed=None),
    }
    _save(result_path, result, f"{phase}/observed")
    _rendezvous(store, args.rank, args.world_size, f"{phase}/observed")
    for replay_index in range(args.profile_graph_warmup + 1):
        measured = replay_index == args.profile_graph_warmup
        prefix_iteration = replay_index - args.profile_graph_warmup
        offset = 0 if measured else last_offset - 65 * (args.profile_graph_warmup - 1 - replay_index)
        input_iterations = _prepare_aligned(offset, prefix_iteration)
        previous_epochs = _previous_epochs()
        phase = f"count-{count}/profile/{'measured' if measured else f'graph-warmup-{replay_index}'}"
        _save(result_path, result, f"{phase}/prepared")
        _rendezvous(store, args.rank, args.world_size, f"{phase}/prepared")
        if measured:
            profiler.start()
            _rendezvous(store, args.rank, args.world_size, f"count-{count}/profile/started")
        with torch.npu.stream(stream):
            if measured:
                with torch.profiler.record_function(aligned["replay_range"]):
                    if args.mc2_capture_identity:
                        profile["replay_thread_identity"]["before"] = {
                            "host_thread_id": threading.get_native_id(),
                            **dict(calls[0]["prepared"].thread_identity()),
                        }
                    profile_graph.replay()
                    if args.mc2_capture_identity:
                        profile["replay_thread_identity"]["after"] = {
                            **dict(calls[0]["prepared"].thread_identity()),
                            "host_thread_id": threading.get_native_id(),
                        }
            else:
                profile_graph.replay()
        torch.npu.synchronize()
        if measured:
            profiler.stop()
        _rendezvous(store, args.rank, args.world_size, f"{phase}/drained")
        checked = _check_aligned(previous_epochs, executed=True)
        record = {
            "replay_index": replay_index,
            "input_iterations": input_iterations,
            "prefix_input_iteration": prefix_iteration,
            "prefix_checked": True,
            "checked_samples": list(range(len(calls))),
        }
        if args.backend != "hccl_aiv":
            record.update({"starting_epochs": checked["starting_epochs"], "ending_epochs": checked["ending_epochs"]})
        if measured:
            aligned.update(
                {
                    "measured_graph_replays": 1,
                    "measured_prefix_checked": True,
                    "checked_samples": record["checked_samples"],
                }
            )
            if args.backend != "hccl_aiv":
                aligned.update({"starting_epochs": record["starting_epochs"], "ending_epochs": record["ending_epochs"]})
            profile["samples"] = [call["record"] for call in calls]
        else:
            aligned["warmups"].append(record)
            aligned["warmup_graph_replays"] += 1
            aligned["warmup_allgather_calls"] += len(calls)
        _save(result_path, result, f"{phase}/checked")
        _rendezvous(store, args.rank, args.world_size, f"{phase}/checked")
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
    if args.backend == "hccl_aiv":
        stream.synchronize()
    graph_dump = profile_dir / "graph.json"
    profile_graph.debug_dump(str(graph_dump))
    json.loads(graph_dump.read_text(encoding="utf-8"))
    profile["graph_dump"] = _file_identity(graph_dump)
    profile_graph.reset()
    profile["graph_reset"] = True
    if args.backend == "hccl_aiv":
        del _active_graph["graph"]
    del profile_graph
    if args.backend == "hccl_aiv":
        for call in calls:
            call["prepared"].close()
        profile["prepared_closed"] = True
        _active_graph = None
    _rendezvous(store, args.rank, args.world_size, f"count-{count}/profile/closed")
    _save(result_path, result, f"count-{count}/profile/complete")


def _run_batch(
    args: argparse.Namespace,
    store: dist.Store,
    result: dict[str, Any],
    result_path: Path,
    receive: torch.Tensor | None,
    controls: torch.Tensor | None,
    epochs: torch.Tensor | None,
    hccl_comm_name: str | None = None,
) -> None:
    global _active_graph

    dtype = getattr(torch, args.dtype)
    device = torch.device(f"npu:{args.device}")
    uses_shmem = args.backend in ("shmem", "ascendc")
    if uses_shmem and (receive is None or controls is None or epochs is None):
        raise RuntimeError("SHMEM retained AllGather requires its initialized symmetric buffers")
    if args.backend == "hccl_aiv":
        if not hccl_comm_name:
            raise RuntimeError("MC2 retained AllGather requires the initialized HCCL communicator name")
        if _active_prepared is not None or _active_graph is not None:
            raise RuntimeError("Previous MC2 prepared/graph resources were not successfully released")
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
    graphs = []
    stream = torch.npu.current_stream()
    if args.backend == "hccl_aiv":
        _active_graph = {"stream": stream, "graphs": graphs, "calls": calls}
    if uses_shmem:
        round_elements = args.lanes * args.chunk_bytes // torch.empty((), dtype=dtype).element_size()
    batch_counts = args.counts * args.repeats
    retained_counts = batch_counts + ([] if args.alternate_count is None else [args.alternate_count])
    for call_id, count in enumerate(retained_counts):
        call_dtype = getattr(torch, args.alternate_dtype or args.dtype) if call_id >= len(batch_counts) else dtype
        source_storage, source, source_guard = _guarded(count, call_dtype, device)
        output_storage, output, output_guard = _guarded(args.world_size * count, call_dtype, device)
        consumer_storage, consumer, consumer_guard = _guarded(args.world_size * count, call_dtype, device)
        scratch = _prepare_scratch(args, device) if uses_shmem else None
        local = _payload(args.rank, count, call_id, call_dtype)
        expected = torch.cat([_payload(peer, count, call_id, call_dtype) for peer in range(args.world_size)])
        source.copy_(local)
        output.fill_(float("nan"))
        consumer.fill_(float("nan"))
        calls.append(
            {
                "call_id": call_id,
                "count": count,
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
        if args.backend == "hccl_aiv":
            calls[-1]["prepared"] = torch.classes.hccl_aiv_ops.PreparedAllGather(
                source,
                output,
                hccl_comm_name,
                args.world_size,
                args.chunk_bytes,
                args.lanes,
                _SKEW_PHASES[args.skew_phase],
                args.skew_iterations,
                args.row_width or 0,
            )
            calls[-1]["lane_schedule"] = dict(calls[-1]["prepared"].schedule())
        else:
            calls[-1]["rounds"] = (count + round_elements - 1) // round_elements
    groups = [calls[: len(batch_counts)]]
    if args.alternate_count is not None:
        groups.append(calls[len(batch_counts) :])
    torch.npu.synchronize()
    starting_epochs = None
    if uses_shmem:
        group_rounds = [sum(call["rounds"] for call in group) for group in groups]
        starting_epochs = epochs.cpu().view(args.lanes, 32)[:, 0].clone()
        assert torch.all((starting_epochs == 0) | (starting_epochs == 1)), "Invalid device epoch"
    result["batch"] = {
        "counts": batch_counts,
        "checked_calls": [],
        "graph_groups": [[call["count"] for call in group] for group in groups],
        "graph_warmups": [],
        "graph_captures": [],
        "graph_replays": [],
        "graphs_reset": False,
    }
    if uses_shmem:
        result["batch"].update(
            {
                "rounds": [call["rounds"] for call in groups[0]],
                "total_rounds": group_rounds[0],
                "group_rounds": group_rounds,
                "starting_epochs": starting_epochs.tolist(),
            }
        )
    else:
        result["batch"].update(
            {
                "lane_schedules": [call["lane_schedule"] for call in calls],
                "call_dtypes": [str(call["source"].dtype).removeprefix("torch.") for call in calls],
                "prepared_closed": False,
            }
        )
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
            elif args.backend == "hccl_aiv":
                call["prepared"].run()
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
                if uses_shmem:
                    _check_scratch(call["scratch"])
            else:
                assert torch.all(torch.isnan(call["output"].cpu())), "Inactive graph output poison changed"
                assert torch.all(torch.isnan(call["consumer"].cpu())), "Inactive graph consumer poison changed"
                if uses_shmem:
                    torch.testing.assert_close(
                        call["scratch"]["tensor"].cpu(), call["scratch"]["initial"], rtol=0, atol=0
                    )
            _check_guards(call["source_storage"], call["count"], call["source_guard"])
            _check_guards(call["output_storage"], args.world_size * call["count"], call["output_guard"])
            _check_guards(call["consumer_storage"], args.world_size * call["count"], call["consumer_guard"])
            if uses_shmem:
                _check_guards(call["scratch"]["storage"], call["scratch"]["elements"], call["scratch"]["guard"])

    def _prepare(group: list[dict[str, Any]], iteration: int) -> None:
        for call in group:
            identity = iteration + call["call_id"]
            call_dtype = call["source"].dtype
            call["local"] = _payload(args.rank, call["count"], identity, call_dtype)
            call["expected"] = torch.cat(
                [_payload(peer, call["count"], identity, call_dtype) for peer in range(args.world_size)]
            )
            call["source"].copy_(call["local"])
            call["output"].fill_(float("nan"))
            call["consumer"].fill_(float("nan"))
            if uses_shmem:
                call["scratch"]["tensor"].copy_(call["scratch"]["initial"])

    def _check_group(group_id: int, previous: torch.Tensor | None, phase: str) -> dict[str, Any]:
        torch.npu.synchronize()
        _rendezvous(store, args.rank, args.world_size, f"{phase}/drained")
        _check_outputs(groups[group_id])
        record = {
            "group": group_id,
            "checked_calls": [call["call_id"] for call in groups[group_id]],
        }
        if uses_shmem:
            expected_epochs = (previous + group_rounds[group_id]) % 2
            _check_state(controls, epochs, expected_epochs, args.lanes, args.world_size)
            record["starting_epochs"] = previous.tolist()
            record["ending_epochs"] = epochs.cpu().view(args.lanes, 32)[:, 0].tolist()
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
            previous = epochs.cpu().view(args.lanes, 32)[:, 0].clone() if uses_shmem else None
            _prepare(group, iteration)
            torch.npu.synchronize()
            _rendezvous(store, args.rank, args.world_size, f"{phase}/prepared")
            _submit(group)
            checked = _check_group(group_id, previous, phase)
            checked["iteration"] = iteration
            result["batch"]["graph_warmups"].append(checked)

    capture_stream = stream if args.backend == "hccl_aiv" else torch.npu.Stream()
    for group_id, group in enumerate(groups):
        phase = f"batch/capture/group-{group_id}"
        _save(result_path, result, phase)
        _prepare(group, -3)
        torch.npu.synchronize()
        before_capture = epochs.cpu().view(args.lanes, 32)[:, 0].clone() if uses_shmem else None
        _rendezvous(store, args.rank, args.world_size, f"{phase}/prepared")
        graph = torch.npu.NPUGraph()
        graphs.append(graph)
        with torch.npu.graph(graph, stream=capture_stream):
            _submit(group)
        torch.npu.synchronize()
        # Capture may or may not execute; only observed replay prestate is used.
        _rendezvous(store, args.rank, args.world_size, f"{phase}/captured")
        capture = {"group": group_id}
        if uses_shmem:
            after_capture = epochs.cpu().view(args.lanes, 32)[:, 0].clone()
            assert torch.equal(after_capture, before_capture) or torch.equal(
                after_capture, (before_capture + group_rounds[group_id]) % 2
            ), "Invalid capture epoch transition"
            _check_state(controls, epochs, after_capture, args.lanes, args.world_size)
            capture.update({"starting_epochs": before_capture.tolist(), "ending_epochs": after_capture.tolist()})
        for call in calls:
            torch.testing.assert_close(call["source"].cpu(), call["local"], rtol=0, atol=0)
            if uses_shmem:
                source_band = call["scratch"]["elements"] // 2
                torch.testing.assert_close(
                    call["scratch"]["tensor"][:source_band].cpu(),
                    call["scratch"]["initial"][:source_band],
                    rtol=0,
                    atol=0,
                )
                _check_guards(call["scratch"]["storage"], call["scratch"]["elements"], call["scratch"]["guard"])
            _check_guards(call["source_storage"], call["count"], call["source_guard"])
            _check_guards(call["output_storage"], args.world_size * call["count"], call["output_guard"])
            _check_guards(call["consumer_storage"], args.world_size * call["count"], call["consumer_guard"])
        result["batch"]["graph_captures"].append(capture)
        _rendezvous(store, args.rank, args.world_size, f"{phase}/observed")

    last_ending = after_capture.tolist() if uses_shmem else None
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
            previous = None
            if uses_shmem:
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
            if uses_shmem:
                last_ending = checked["ending_epochs"]
            retained_groups.add(group_id)
            _rendezvous(store, args.rank, args.world_size, f"{phase}/retained-checked")

    if uses_shmem:
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
    if uses_shmem:
        result["batch"]["final_epochs"] = last_ending
    else:
        _save(result_path, result, "batch/close")
        for call in calls:
            call["prepared"].close()
        result["batch"]["prepared_closed"] = True
        _active_graph = None
    result["batch"]["checked_calls"] = list(range(len(batch_counts)))
    _save(result_path, result, "batch/checked")


def _bootstrap_store(args: argparse.Namespace, result: dict[str, Any], identities: tuple[str, ...]) -> dist.Store:
    store = dist.FileStore(str(args.artifact_dir / "bootstrap.store"), args.world_size)
    store.set_timeout(timedelta(seconds=120))
    if store.add(f"participant/{args.rank}", 1) != 1:
        raise RuntimeError("FileStore namespace already used by this rank")
    configuration_fields = (
        "backend",
        "mc2_probe_stage",
        "mc2_delay_iterations",
        "profile_level",
        "mc2_profile_level",
        "mc2_capture_identity",
        "mc2_submission_identity",
        "profile_samples",
        "profile_warmup",
        "profile_rank0_delay_ms",
        "profile_submission",
        "profile_graph_warmup",
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
        "alternate_dtype",
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
    if args.mc2_capture_identity:
        configuration_data["purpose"] = result["purpose"]
    if args.backend in ("hccl", "hccl_aiv"):
        configuration_data["hccl_config"] = result["hccl_config"]
    if args.profile_submission == "alltoall_graph":
        configuration_data["prefix_hccl_config"] = result["prefix_hccl_config"]
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


def _init_hccl_process_group(args: argparse.Namespace, result: dict[str, Any]) -> dist.ProcessGroup | None:
    import torch_npu

    if dist.is_initialized():
        raise RuntimeError("Test requires an uninitialized HCCL process")
    options = {}
    if args.profile_submission == "alltoall_graph":
        prefix_options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
        prefix_options.hccl_config = result["prefix_hccl_config"]
        options["pg_options"] = prefix_options
    dist.init_process_group(
        backend="hccl",
        init_method=(args.artifact_dir / "hccl.store").as_uri(),
        world_size=args.world_size,
        rank=args.rank,
        timeout=timedelta(seconds=120),
        **options,
    )
    if args.profile_submission != "alltoall_graph":
        return None
    group = dist.group.WORLD
    if group is None:
        raise RuntimeError("HCCL prefix process group was not initialized")
    result["prefix_group"] = {
        "hccl_config": dict(result["prefix_hccl_config"]),
        "group_name": group.group_name,
        "owner": "torch.distributed.group.WORLD",
        "pe_devices": result["pe_devices"],
    }
    return group


def _materialize_mc2_prefix(
    args: argparse.Namespace,
    store: dist.Store,
    result: dict[str, Any],
    result_path: Path,
    prefix_group: dist.ProcessGroup,
    device: torch.device,
) -> str:
    global _active_graph

    if _active_graph is not None:
        raise RuntimeError("Previous MC2 graph resources were not successfully released")
    buffers = {
        "source": _guarded(args.world_size * _PREFIX_ELEMENTS, torch.float32, device),
        "output": _guarded(args.world_size * _PREFIX_ELEMENTS, torch.float32, device),
    }
    _active_graph = {"prefix_group": prefix_group, "buffers": buffers, "stream": torch.npu.current_stream()}
    local = torch.cat([_prefix_payload(args.rank, peer, -1, args.world_size) for peer in range(args.world_size)])
    expected = torch.cat([_prefix_payload(peer, args.rank, -1, args.world_size) for peer in range(args.world_size)])
    buffers["source"][1].copy_(local)
    buffers["output"][1].fill_(float("nan"))
    torch.npu.synchronize()
    phase = "bootstrap/prefix-materialization"
    _save(result_path, result, phase)
    _rendezvous(store, args.rank, args.world_size, f"{phase}/prepared")
    work = dist.all_to_all_single(buffers["output"][1], buffers["source"][1], group=prefix_group, async_op=True)
    _active_graph["work"] = work
    work.wait()
    torch.npu.synchronize()
    torch.testing.assert_close(buffers["source"][1].cpu(), local, rtol=0, atol=0)
    torch.testing.assert_close(buffers["output"][1].cpu(), expected, rtol=0, atol=0)
    for storage, tensor, guard in buffers.values():
        _check_guards(storage, tensor.numel(), guard)
    backend = prefix_group._get_backend(device)
    if not callable(getattr(backend, "get_hccl_comm_name", None)):
        raise RuntimeError("Prefix ProcessGroupHCCL does not expose get_hccl_comm_name")
    name = backend.get_hccl_comm_name(dist.get_rank(), init_comm=False)
    if not isinstance(name, str) or not 0 < len(name) < 128:
        raise RuntimeError(f"Invalid materialized prefix HCCL communicator name: {name!r}")
    result["prefix_group"].update(hccl_comm_name=name, materialization_checked=True)
    _rendezvous(store, args.rank, args.world_size, f"{phase}/checked")
    _active_graph = None
    return name


def _run_hccl(args: argparse.Namespace, result: dict[str, Any], result_path: Path) -> None:
    global _active_prepared

    import torch_npu

    torch.set_num_threads(1)
    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
    result["versions"] = {"torch": torch.__version__, "torch_npu": torch_npu.__version__}
    _load_native(args, result)
    identities = ("worker", "native_library")
    if args.backend == "hccl_aiv":
        result["kernel"] = _file_identity(args.kernel_source)
        identities += ("kernel",)
    options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
    options.hccl_config = {"hccl_op_expansion_mode": args.hccl_op_expansion_mode} if args.backend == "hccl" else {}
    result["hccl_config"] = dict(options.hccl_config)
    _save(result_path, result, "bootstrap")
    store = _bootstrap_store(args, result, identities)
    prefix_group = _init_hccl_process_group(args, result)
    group = dist.new_group(
        ranks=list(range(args.world_size)),
        backend="hccl",
        timeout=timedelta(seconds=120),
        pg_options=options,
    )
    _save(result_path, result, "bootstrap/communicator-allreduce")
    warm = torch.ones(1, dtype=torch.float32, device=device)
    # This initializes communicator resources, never the test's AllGather payload.
    dist.all_reduce(warm, group=group)
    torch.npu.synchronize()
    backend = group._get_backend(device)
    comm = None
    hccl_comm_name = None
    if args.backend == "hccl_aiv":
        if not callable(getattr(backend, "get_hccl_comm_name", None)):
            raise RuntimeError("ProcessGroupHCCL does not expose get_hccl_comm_name")
        hccl_comm_name = backend.get_hccl_comm_name(dist.get_rank(), init_comm=False)
        if not isinstance(hccl_comm_name, str) or not 0 < len(hccl_comm_name) < 128:
            raise RuntimeError(f"Invalid initialized HCCL communicator name: {hccl_comm_name!r}")
        result["hccl_comm_name"] = hccl_comm_name
        result["bootstrap_allreduce_role"] = "communicator_resource_initialization_only"
    else:
        comm = backend.get_hccl_comm(device.index)
        if not comm:
            raise RuntimeError(f"No HCCL communicator on {device}")
        result["hccl_comm"] = hex(comm)
    result["bootstrap"] = "HCCL/ProcessGroup"
    result["group_name"] = group.group_name
    result["pg_owner"] = "torch.distributed.new_group(ranks=all_PEs, backend='hccl')"
    _rendezvous(store, args.rank, args.world_size, "initialized")
    run_stream = (
        torch.npu.Stream()
        if args.backend == "hccl_aiv" and (args.mode == "graph" or args.profile_samples)
        else torch.npu.current_stream()
    )
    with torch.npu.stream(run_stream):
        if args.backend == "hccl_aiv":
            prefix_name = None
            if args.profile_samples:
                if prefix_group is None:
                    raise RuntimeError("MC2 profiling requires its prefix process group before initialization")
                prefix_name = _materialize_mc2_prefix(args, store, result, result_path, prefix_group, device)
            if _active_prepared is not None:
                raise RuntimeError("Previous MC2 prepared resources were not successfully released")
            _save(result_path, result, "bootstrap/mc2-prepare-state")
            dtype = getattr(torch, args.dtype)
            setup_source_storage, setup_source, setup_source_guard = _guarded(1, dtype, device)
            setup_output_storage, setup_output, setup_output_guard = _guarded(args.world_size, dtype, device)
            setup_source.fill_(1)
            setup_output.fill_(float("nan"))
            setup = torch.classes.hccl_aiv_ops.PreparedAllGather(
                setup_source, setup_output, hccl_comm_name, args.world_size, args.chunk_bytes, args.lanes, 0, 0, 0
            )
            _active_prepared = setup
            protocol = setup.protocol()
            result["backend_identity"] = {"protocol": protocol}
            _save(result_path, result, "bootstrap/mc2-protocol")
            if protocol != _MC2_PROTOCOL:
                raise RuntimeError(f"Expected {_MC2_PROTOCOL}, loaded native protocol {protocol!r}")
            if prefix_name is not None:
                setup.verify_disjoint_buffers(prefix_name)
                result["prefix_group"]["target_buffers_disjoint"] = True
            result["mc2_state_initialization"] = {
                "calls": 1,
                "outside_capture": True,
                "drained": False,
                "prepared_closed": False,
            }
            torch.npu.synchronize()
            _rendezvous(store, args.rank, args.world_size, "bootstrap/mc2-state-ready")
            _save(result_path, result, "bootstrap/mc2-initialize-state")
            setup.initialize_state()
            torch.npu.synchronize()
            result["mc2_state_initialization"]["drained"] = True
            torch.testing.assert_close(setup_source.cpu(), torch.ones(1, dtype=dtype), rtol=0, atol=0)
            assert torch.all(torch.isnan(setup_output.cpu())), "MC2 state initialization produced AllGather output"
            _check_guards(setup_source_storage, 1, setup_source_guard)
            _check_guards(setup_output_storage, args.world_size, setup_output_guard)
            _rendezvous(store, args.rank, args.world_size, "bootstrap/mc2-state-initialized")
            _save(result_path, result, "bootstrap/mc2-close-setup")
            setup.close()
            _active_prepared = None
            result["mc2_state_initialization"]["prepared_closed"] = True
            _rendezvous(store, args.rank, args.world_size, "bootstrap/mc2-setup-closed")
        if args.stress_batch:
            _run_batch(args, store, result, result_path, None, None, None, hccl_comm_name)
        else:
            for count in args.counts:
                _run_case(args, store, result, result_path, count, None, None, None, comm, prefix_group, hccl_comm_name)
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
    parser.add_argument("--backend", choices=("shmem", "ascendc", "hccl", "hccl_aiv"), default="shmem")
    parser.add_argument(
        "--mc2-probe-stage",
        choices=("resource", "before-barrier", "after-barrier", "final-zero"),
        help="Retired whole-kernel-barrier diagnostic; rejected before execution",
    )
    parser.add_argument(
        "--mc2-delay-iterations",
        type=int,
        default=None,
        help="Retired final-zero diagnostic option; rejected before execution",
    )
    parser.add_argument("--hccl-op-expansion-mode", type=int, choices=(4,), help="Matched HCCL AIV Only (4)")
    parser.add_argument("--profile-samples", type=int, default=0)
    parser.add_argument("--profile-warmup", type=int, help="Logical warmups; aligned profiling requires 0")
    parser.add_argument(
        "--profile-submission", choices=("alltoall_graph",), help="Capture one AlltoAll followed by 50 AllGather calls"
    )
    parser.add_argument("--profile-graph-warmup", type=int, help="Full aligned-graph warmup replays (default 20)")
    parser.add_argument(
        "--profile-level",
        choices=("Level1", "Level2"),
        default=None,
        help="Aligned profiling coverage (default Level1)",
    )
    parser.add_argument(
        "--mc2-capture-identity",
        action="store_true",
        help="Collect unqualified Level2 identities for the PE2 control or TP16 GLM5.2 embedding/logits shapes at reserved L8/L48/chunk65536",
    )
    parser.add_argument(
        "--mc2-submission-identity",
        action="store_true",
        help="Observe public eager host submission identities; not execution/completion or performance proof",
    )
    parser.add_argument("--comparison-id")
    parser.add_argument("--round-id", type=int)
    parser.add_argument("--stage-index", type=int)
    parser.add_argument("--row-width", type=int, help="Per-rank MC2 scheduling row width, or profiling shape metadata")
    parser.add_argument("--native-library", type=Path)
    parser.add_argument("--native-build-revision")
    parser.add_argument("--kernel-source", type=Path, help="Standalone Ascend C kernel source identity")
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, choices=(2, 4, 8, 16), required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--mode", choices=("eager", "graph"), default="eager")
    parser.add_argument("--skew-phase", choices=("none", "read", "ack", "ready"), default="none")
    parser.add_argument(
        "--skew-iterations", type=int, default=0, help="Bounded correctness delay steps; not cycles or time units"
    )
    parser.add_argument("--graph-replays", type=int, help="Checked retained graph replay rounds")
    parser.add_argument("--alternate-count", type=int, help="Second live graph contains one call of this count")
    parser.add_argument(
        "--alternate-dtype",
        choices=("float32", "float16", "bfloat16"),
        help="Second retained MC2 graph dtype, sharing the initialized communicator",
    )
    parser.add_argument(
        "--stress-batch", action="store_true", help="Retain counts * repeats calls and check after each full drain"
    )
    parser.add_argument(
        "--compile-only", action="store_true", help="Lower DSL only; no SHMEM bootstrap or NPU execution"
    )
    parser.add_argument("--counts", type=int, nargs="+", default=[64, 65, 127, 128, 129])
    parser.add_argument("--lanes", type=int, help="MC2 reserved lanes 1..48 (default 48); other backends default to 2")
    parser.add_argument("--chunk-bytes", type=int, default=128)
    parser.add_argument("--heap-bytes", type=int, help="SHMEM heap bytes (default 64 MiB); not used by hccl_aiv")
    parser.add_argument("--repeats", type=int, default=16)
    args = parser.parse_args()
    if args.mc2_probe_stage is not None or args.mc2_delay_iterations is not None:
        parser.error("The whole-kernel-barrier MC2 diagnostics are retired; they cannot run the lane protocol")
    if args.alternate_dtype is not None and (
        args.backend != "hccl_aiv"
        or not args.stress_batch
        or args.mode != "graph"
        or args.alternate_count is None
        or args.alternate_dtype == args.dtype
        or args.profile_samples
        or args.profile_submission is not None
    ):
        parser.error("alternate-dtype requires two retained MC2 graphs of different dtypes without profiling")
    if args.backend == "hccl_aiv":
        if args.mode == "graph" and (args.world_size not in (2, 8, 16) or args.mc2_probe_stage is not None):
            parser.error("hccl_aiv basic graph entry requires PE2/8/16 normal AllGather without diagnostics")
        if args.stress_batch and (
            args.mode != "graph"
            or args.world_size != 2
            or args.mc2_probe_stage is not None
            or len(args.counts) != 1
            or args.repeats != 1
            or args.alternate_count is None
            or (args.alternate_count == args.counts[0] and args.alternate_dtype is None)
            or not 0 < args.world_size * args.alternate_count <= (1 << 31) - 1
        ):
            parser.error(
                "hccl_aiv retained graphs require normal PE2 graph, one primary count, "
                "repeats=1 and a distinct alternate count or dtype"
            )
        if args.world_size not in (2, 8, 16):
            parser.error("hccl_aiv requires 2/8/16 ranks")
        if args.heap_bytes is not None:
            parser.error("hccl_aiv uses MC2 windows, not a SHMEM heap")
    elif args.heap_bytes is None:
        args.heap_bytes = 64 * 1024 * 1024
    if args.lanes is None:
        args.lanes = 48 if args.backend == "hccl_aiv" else 2
    if args.profile_warmup is None:
        args.profile_warmup = 0 if args.profile_submission == "alltoall_graph" else 20
    if args.profile_samples < 0:
        parser.error("profile-samples must be nonnegative")
    if args.profile_samples or args.profile_submission is not None:
        if (
            args.profile_submission != "alltoall_graph"
            or args.profile_samples != 50
            or args.profile_warmup != 0
            or args.backend not in ("hccl", "ascendc", "hccl_aiv")
            or args.mode != "eager"
        ):
            parser.error("Profiling requires native eager mode, explicit alltoall_graph, 50 samples and warmup=0")
        if args.profile_graph_warmup is None:
            args.profile_graph_warmup = 20
        if args.profile_graph_warmup < 1:
            parser.error("profile-graph-warmup must be positive")
        if args.backend == "hccl_aiv" and (args.mc2_probe_stage is not None or args.profile_graph_warmup != 20):
            parser.error(
                "hccl_aiv aligned profiling requires normal uninstrumented AllGather and 20 full-graph warmups"
            )
        for name in ("TORCH_HCCL_BLOCKING_WAIT", "HCCL_BLOCKING_WAIT"):
            if os.environ.get(name) not in (None, "0"):
                parser.error(f"alltoall_graph requires {name} unset or 0 for device-only stream ordering")
    elif args.profile_graph_warmup is not None:
        parser.error("profile-graph-warmup requires alltoall_graph submission")
    if args.profile_samples:
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
    elif (
        args.profile_warmup != 20
        or any(value is not None for value in (args.comparison_id, args.round_id, args.stage_index))
        or (args.row_width is not None and args.backend != "hccl_aiv")
    ):
        parser.error("Profiling metadata requires positive profile-samples")
    if args.row_width is not None and (
        args.row_width <= 0
        or any(count % args.row_width for count in args.counts)
        or (args.alternate_count is not None and args.alternate_count % args.row_width != 0)
    ):
        parser.error("row-width must be positive and divide every primary and alternate count")
    if args.compile_only and args.backend != "shmem":
        parser.error("compile-only is available only for the TileLang SHMEM backend")
    if args.backend in ("hccl", "ascendc", "hccl_aiv"):
        if args.native_library is None or not args.native_library.is_absolute() or not args.native_library.is_file():
            parser.error(f"{args.backend} requires native-library as an existing absolute file")
        args.native_library = args.native_library.resolve()
        if args.native_build_revision is None or re.fullmatch(r"[0-9a-f]{40}", args.native_build_revision) is None:
            parser.error(f"{args.backend} requires the full native-build-revision caller assertion")
    elif args.native_library is not None or args.native_build_revision is not None:
        parser.error("TileLang SHMEM must not load a native library")
    if args.backend in ("ascendc", "hccl_aiv"):
        if args.kernel_source is None or not args.kernel_source.is_absolute() or not args.kernel_source.is_file():
            parser.error(f"{args.backend} requires kernel-source as an existing absolute file")
        args.kernel_source = args.kernel_source.resolve()
    elif args.kernel_source is not None:
        parser.error("kernel-source is only valid for the standalone ascendc/hccl_aiv backends")
    if args.backend != "hccl" and args.hccl_op_expansion_mode is not None:
        parser.error("hccl-op-expansion-mode requires the HCCL backend")
    if args.backend == "hccl":
        if not args.profile_samples:
            parser.error("The HCCL backend is only available for matched gather-only measurement")
        if args.hccl_op_expansion_mode != 4 or args.dtype not in ("float16", "bfloat16"):
            parser.error("Matched alltoall_graph requires FP16/BF16 HCCL with explicit expansion mode 4")
        if args.dtype == "float16" and args.profile_graph_warmup != 20:
            parser.error("FP16 HCCL raw baseline requires exactly 20 full-graph warmups")
    if args.backend == "hccl" or args.profile_submission == "alltoall_graph":
        if os.environ.get("HCCL_OP_EXPANSION_MODE") != "AIV" or any(
            not os.environ.get(name) for name in ("HCCL_HOST_SOCKET_PORT_RANGE", "HCCL_NPU_SOCKET_PORT_RANGE")
        ):
            parser.error("HCCL requires explicit AIV expansion and host/NPU socket port ranges")
    if args.backend == "hccl_aiv" and any(
        not os.environ.get(name) for name in ("HCCL_HOST_SOCKET_PORT_RANGE", "HCCL_NPU_SOCKET_PORT_RANGE")
    ):
        parser.error("hccl_aiv resource bootstrap requires host/NPU socket port ranges")
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
        if args.backend != "hccl_aiv" and args.alternate_count is not None and args.alternate_count not in args.counts:
            parser.error("Alternate count must belong to --counts")
    if args.skew_phase == "ready" and args.backend != "hccl_aiv":
        parser.error("READY publication skew requires the hccl_aiv backend")
    if (args.skew_phase == "none" and args.skew_iterations != 0) or (
        args.skew_phase != "none" and not 0 < args.skew_iterations <= 32768
    ):
        parser.error("Require skew-iterations=0 for none, or 1..32768 for ready/read/ack instrumentation")
    if args.repeats <= 0 or args.lanes <= 0:
        parser.error("Require positive repeats and positive lanes")
    if args.backend == "hccl_aiv" and args.lanes > 48:
        parser.error("MC2 supports reserved lanes in 1..48; native launch also checks hardware capacity")
    if args.backend == "ascendc" and args.lanes > 48:
        parser.error("Ascend C supports lanes in 1..48; native launch also checks hardware capacity")
    if args.backend in ("shmem", "hccl") and args.lanes % 2:
        parser.error("TileLang MIX and matched HCCL configuration require even lanes")
    chunk_limit = 64 * 1024
    if not 0 < args.chunk_bytes <= chunk_limit or args.chunk_bytes % 128:
        parser.error(f"Require chunk-bytes <= {chunk_limit} and a positive multiple of 128")
    if len(set(args.counts)) != len(args.counts) or any(
        not 0 < args.world_size * count <= (1 << 31) - 1 for count in args.counts
    ):
        parser.error("Require distinct positive counts with world-size * count <= INT32_MAX")
    if retained_graph and args.backend in ("shmem", "ascendc"):
        round_elements = args.lanes * args.chunk_bytes // (4 if args.dtype == "float32" else 2)
        total_rounds = sum((count + round_elements - 1) // round_elements for count in args.counts) * args.repeats
        if args.alternate_count is not None:
            total_rounds += (args.alternate_count + round_elements - 1) // round_elements
        if total_rounds % 2 != 1:
            parser.error("Retained graph sequence must have odd total rounds to exercise both epoch parities")
    if args.backend != "hccl_aiv":
        required_bytes = args.world_size * args.lanes * args.chunk_bytes + (2 * args.world_size + 1) * args.lanes * 128
        if args.heap_bytes < required_bytes:
            parser.error(f"Symmetric buffers require at least {required_bytes} heap bytes")
    if args.profile_level is not None and not args.profile_samples:
        parser.error("profile-level requires alltoall_graph profiling")
    if args.profile_samples and args.profile_level is None:
        args.profile_level = "Level1"
    capture_identity_preset = (
        args.world_size == 2
        and args.dtype == "float16"
        and args.counts == [64]
        and args.lanes == 2
        and args.chunk_bytes == 128
    ) or (
        args.world_size == 16
        and args.chunk_bytes == 65536
        and (
            (args.dtype == "bfloat16" and args.counts == [309760] and args.lanes in (4, 8, 16))
            or (
                args.dtype in ("float16", "bfloat16")
                and args.lanes in (8, 48)
                and args.row_width in (384, 9680)
                and len(args.counts) == 1
                and args.counts[0]
                in [args.row_width * tokens for tokens in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)]
            )
        )
    )
    if args.mc2_capture_identity and (
        args.backend != "hccl_aiv"
        or args.profile_submission != "alltoall_graph"
        or args.profile_samples != 50
        or args.profile_warmup != 0
        or args.profile_graph_warmup != 20
        or args.profile_level != "Level2"
        or args.mc2_probe_stage is not None
        or args.mode != "eager"
        or not capture_identity_preset
        or args.repeats != 2
    ):
        parser.error(
            "mc2-capture-identity requires repeats2 Level2 aligned profiling and the PE2 control, "
            "TP16 BF16/count309760 at L4/L8/L16, or one GLM5.2 TP16 embedding/logits shape at "
            "reserved L8/L48/chunk65536; completion metadata remains unqualified"
        )
    if args.mc2_submission_identity and (
        args.backend != "hccl_aiv"
        or args.mode != "eager"
        or args.stress_batch
        or args.profile_samples
        or args.profile_submission is not None
        or args.mc2_capture_identity
        or args.compile_only
    ):
        parser.error(
            "mc2-submission-identity requires ordinary hccl_aiv eager numerical calls without profiling/capture"
        )
    args.artifact_dir = args.artifact_dir.resolve()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.artifact_dir / f"rank-{args.rank}.json"
    result: dict[str, Any] = {
        "backend": args.backend,
        "mc2_probe_stage": args.mc2_probe_stage,
        "mc2_delay_iterations": args.mc2_delay_iterations,
        "profile_level": args.profile_level,
        "mc2_profile_level": args.profile_level if args.backend == "hccl_aiv" else None,
        "mc2_capture_identity": args.mc2_capture_identity,
        "mc2_submission_identity": args.mc2_submission_identity,
        "profile_samples": args.profile_samples,
        "profile_warmup": args.profile_warmup,
        "comparison_id": args.comparison_id,
        "round_id": args.round_id,
        "stage_index": args.stage_index,
        "profile_rank0_delay_ms": None,
        "profile_submission": args.profile_submission,
        "profile_graph_warmup": args.profile_graph_warmup,
        "row_width": args.row_width,
        "native_build_revision": args.native_build_revision,
        "native_build_revision_owner": {
            "ascendc": "workspace_source",
            "hccl_aiv": "workspace_source",
            "hccl": "xllm_source",
        }.get(args.backend),
        "rank": args.rank,
        "world_size": args.world_size,
        "device": args.device,
        "mode": args.mode,
        "stress_batch": args.stress_batch,
        "skew_phase": args.skew_phase,
        "skew_iterations": args.skew_iterations,
        "graph_replays": args.graph_replays,
        "alternate_count": args.alternate_count,
        "alternate_dtype": args.alternate_dtype,
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
                "ASCEND_CUSTOM_OPP_PATH",
                "LD_LIBRARY_PATH",
                "TL_ROOT",
                "SHMEM_HOME_PATH",
                "HCCL_OP_EXPANSION_MODE",
                "HCCL_HOST_SOCKET_PORT_RANGE",
                "HCCL_NPU_SOCKET_PORT_RANGE",
                "HCCL_IF_IP",
                "HCCL_ALGO",
                "TORCH_HCCL_ZERO_COPY",
                "TORCH_HCCL_BLOCKING_WAIT",
                "HCCL_BLOCKING_WAIT",
            )
        },
    }
    if args.backend == "hccl_aiv":
        result["skew_mechanism"] = "per_lane_PIPE_ALL_iterations"
    if args.mc2_capture_identity:
        result["purpose"] = "performance_measurement"
        result["performance_verdict"] = "UNASSESSED"
    if args.profile_submission == "alltoall_graph":
        result["prefix_hccl_config"] = {"hccl_op_expansion_mode": 0}
    # Refuse an earlier attempt's rank result or rendezvous namespace.
    with result_path.open("x", encoding="utf-8") as result_file:
        result_file.write(json.dumps(result, indent=2) + "\n")
    _save(result_path, result, "imports")
    if args.compile_only:
        _compile_only(args, result, result_path)
        return
    if args.backend in ("hccl", "hccl_aiv"):
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
    prefix_group = _init_hccl_process_group(args, result) if args.profile_submission == "alltoall_graph" else None
    if args.stress_batch:
        _run_batch(args, store, result, result_path, receive, controls, epochs)
    else:
        for count in args.counts:
            _run_case(args, store, result, result_path, count, receive, controls, epochs, prefix_group=prefix_group)
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
    if prefix_group is not None:
        dist.destroy_process_group()
    _rendezvous(store, args.rank, args.world_size, "closed")
    result["status"] = "PASS"
    _save(result_path, result, "complete")


if __name__ == "__main__":
    try:
        _main()
    except Exception:
        logger.exception("AllGather worker failed; inspect its saved phase and original traceback")
        if _active_prepared is not None or _active_graph is not None:
            # No device drain or destructor may replace this failure with a hang
            # or abort. The launcher still owns failure exit and peer cleanup.
            logger.error("MC2 prepared/graph state retained; exiting without device cleanup after failure")
            for handler in logger.handlers:
                handler.flush()
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(1)
        raise
