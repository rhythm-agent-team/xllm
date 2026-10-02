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

"""Standard AllGather using the existing TileLang SHMEM dispatch primitives.

Each lane puts its input shard into every PE's symmetric receive window, then
publishes ready. A receiver copies that shard into ordinary rank-major output
before acknowledging it. All acknowledgements, including the final chunk's,
are required before the sender can reuse the window. Slots have one writer and
128-byte spacing. The alternating sequence advances on device, not at capture.

Like the existing dispatch/combine example, this uses the existing MIX launch
and its Vector scope. It does not claim to be a pure AIV kernel. No compiler
extension, extern kernel, AOT registry, or xLLM native wrapper is required.
"""

import tilelang.language as T
from tilelang import tvm

SIGNAL_STRIDE = 128 // 4
SIGNAL_ELEMENTS = 32 // 4
SUPPORTED_DTYPES = {"float32": 4, "float16": 2, "bfloat16": 2}
SHMEM_PASS_CONFIGS = {
    "tl.ascend_auto_sync": False,
    "tl.ascend_memory_planning": True,
    "tl.ascend_auto_cross_core_sync": False,
    "tl.ascend_auto_cv_combine": False,
}


def build_shmem_all_gather_kernel(
    count: int,
    world_size: int,
    lanes: int,
    chunk_bytes: int,
    dtype: str,
) -> tvm.tir.PrimFunc:
    """Return one kernel; every PE must use identical count and specialization."""
    if dtype not in SUPPORTED_DTYPES:
        raise ValueError(f"Unsupported SHMEM AllGather dtype: {dtype}")
    if world_size not in (2, 4, 8, 16):
        raise ValueError(f"Unsupported SHMEM PE count: {world_size}")
    if count <= 0 or world_size * count > (1 << 31) - 1:
        raise ValueError(f"Existing TileLang copies require 0 < world_size * count <= INT32_MAX, got {count}")
    if lanes <= 0 or lanes % 2:
        raise ValueError(f"Existing MIX launch requires a positive even lane count, got {lanes}")
    if chunk_bytes <= 0 or chunk_bytes % 128:
        raise ValueError(f"SHMEM chunk_bytes must be a positive multiple of 128, got {chunk_bytes}")
    if chunk_bytes > 64 * 1024:
        raise ValueError("SHMEM payload exceeds the initial 64 KiB per-lane UB budget")
    chunk_elements = chunk_bytes // SUPPORTED_DTYPES[dtype]
    rounds = (count + lanes * chunk_elements - 1) // (lanes * chunk_elements)
    window_elements = world_size * lanes * chunk_elements
    control_elements = 2 * world_size * lanes * SIGNAL_STRIDE

    @T.prim_func
    def shmem_all_gather(
        source: T.Tensor((count,), dtype),
        output: T.Tensor((world_size, count), dtype),
        receive: T.Tensor((window_elements,), dtype),
        controls: T.Tensor((control_elements,), "int32"),
        epochs: T.Tensor((lanes * SIGNAL_STRIDE,), "int32"),
        rank: T.int32,
    ):
        with T.Kernel(lanes // 2, is_npu=True) as (core, subcore):
            payload = T.alloc_ub((chunk_elements,), dtype)
            publication = T.alloc_ub((SIGNAL_ELEMENTS,), "int32")
            probe = T.alloc_ub((SIGNAL_ELEMENTS,), "int32")
            epoch = T.alloc_ub((SIGNAL_ELEMENTS,), "int32")
            generation = T.alloc_ub((SIGNAL_ELEMENTS,), "int32")
            observed = T.alloc_ub((SIGNAL_ELEMENTS,), "int32")
            with T.Scope("V"):
                lane = T.Cast("int32", core) * 2 + T.Cast("int32", subcore)
                T.copy(epochs[lane * SIGNAL_STRIDE], epoch)
                T.barrier_all()
                generation[0] = epoch[0]
                for chunk in T.serial(rounds):
                    offset = chunk * lanes * chunk_elements + lane * chunk_elements
                    remaining = count - offset
                    length = T.if_then_else(
                        remaining <= 0,
                        0,
                        T.if_then_else(remaining < chunk_elements, remaining, chunk_elements),
                    )
                    generation[0] = 1 - generation[0]
                    # Scalar writes into UB must precede the MTE3 publication.
                    for element in T.serial(SIGNAL_ELEMENTS):
                        publication[element] = generation[0]
                    T.barrier_all()
                    if length > 0:
                        T.copy(source[offset], payload)
                        T.barrier_all()
                        for peer in T.serial(world_size):
                            T.shmem_ub_put_nbi(
                                payload,
                                receive,
                                length,
                                peer,
                                (rank * lanes + lane) * chunk_elements,
                            )
                            T.barrier_all()
                    # Every lane participates even when its final shard is empty.
                    for peer in T.serial(world_size):
                        T.shmem_ub_put_nbi(
                            publication,
                            controls,
                            SIGNAL_ELEMENTS,
                            peer,
                            (rank * lanes + lane) * SIGNAL_STRIDE,
                        )
                        T.barrier_all()
                    for peer_offset in T.serial(world_size):
                        peer = (rank + peer_offset) % world_size
                        observed[0] = -1
                        while observed[0] != generation[0]:
                            T.copy(controls[(peer * lanes + lane) * SIGNAL_STRIDE], probe)
                            T.barrier_all()
                            observed[0] = probe[0]
                        if length > 0:
                            start = (peer * lanes + lane) * chunk_elements
                            T.copy(receive[start], payload)
                            T.barrier_all()
                            T.copy(payload, output[peer, offset])
                            T.barrier_all()
                        T.shmem_ub_put_nbi(
                            publication,
                            controls,
                            SIGNAL_ELEMENTS,
                            peer,
                            (world_size * lanes + rank * lanes + lane) * SIGNAL_STRIDE,
                        )
                        T.barrier_all()
                    for peer in T.serial(world_size):
                        observed[0] = -1
                        while observed[0] != generation[0]:
                            T.copy(
                                controls[(world_size * lanes + peer * lanes + lane) * SIGNAL_STRIDE],
                                probe,
                            )
                            T.barrier_all()
                            observed[0] = probe[0]
                for element in T.serial(SIGNAL_ELEMENTS):
                    epoch[element] = generation[0]
                T.barrier_all()
                T.copy(epoch, epochs[lane * SIGNAL_STRIDE])
                T.barrier_all()

    return shmem_all_gather
