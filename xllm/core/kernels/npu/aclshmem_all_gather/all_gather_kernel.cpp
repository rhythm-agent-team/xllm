/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "device/gm2gm/shmem_device_mo.h"
#include "device/ub2gm/engine/shmem_device_mte.h"
#include "kernel_operator.h"

namespace {

constexpr int32_t kSignalStride = 128 / sizeof(int32_t);
constexpr int32_t kSignalElements = 32 / sizeof(int32_t);

__aicore__ inline void wait_generation(AscendC::GlobalTensor<int32_t> slot,
                                       AscendC::LocalTensor<int32_t> probe,
                                       int32_t generation) {
  const AscendC::DataCopyExtParams params{1, 32, 0, 0, 0};
  const AscendC::DataCopyPadExtParams<int32_t> padding{};
  do {
    // MTE polling reloads local GM rather than retaining a scalar cache line.
    AscendC::DataCopyPad(probe, slot, params, padding);
    AscendC::PipeBarrier<PIPE_ALL>();
  } while (probe.GetValue(0) != generation);
}

__aicore__ inline void copy_skew_tiles(AscendC::GlobalTensor<int32_t> scratch,
                                       AscendC::LocalTensor<int32_t> tile,
                                       int32_t iterations) {
  const AscendC::DataCopyExtParams params{1, 128, 0, 0, 0};
  const AscendC::DataCopyPadExtParams<int32_t> padding{};
  const int64_t band_elements =
      static_cast<int64_t>(iterations) * kSignalStride;
  for (int32_t step = 0; step < iterations; ++step) {
    const int64_t begin = static_cast<int64_t>(step) * kSignalStride;
    AscendC::DataCopyPad(tile, scratch[begin], params, padding);
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataCopyPad(scratch[band_elements + begin], tile, params);
    AscendC::PipeBarrier<PIPE_ALL>();
  }
}

template <typename T>
__aicore__ inline void all_gather(GM_ADDR source_ptr,
                                  GM_ADDR output_ptr,
                                  GM_ADDR receive_ptr,
                                  GM_ADDR controls_ptr,
                                  GM_ADDR epochs_ptr,
                                  GM_ADDR scratch_ptr,
                                  int64_t count,
                                  int32_t rank,
                                  int32_t world_size,
                                  int32_t lanes,
                                  uint32_t chunk_bytes,
                                  int32_t skew_phase,
                                  int32_t skew_iterations) {
  AscendC::GlobalTensor<T> source;
  source.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(source_ptr));
  AscendC::GlobalTensor<T> output;
  output.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(output_ptr));
  AscendC::GlobalTensor<T> receive;
  receive.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(receive_ptr));
  AscendC::GlobalTensor<int32_t> controls;
  controls.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(controls_ptr));
  AscendC::GlobalTensor<int32_t> epochs;
  epochs.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(epochs_ptr));
  AscendC::GlobalTensor<int32_t> scratch;
  scratch.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(scratch_ptr));

  AscendC::TPipe pipe;
  AscendC::TBuf<AscendC::TPosition::VECCALC> payload_buffer;
  AscendC::TBuf<AscendC::TPosition::VECCALC> signal_buffer;
  AscendC::TBuf<AscendC::TPosition::VECCALC> skew_buffer;
  pipe.InitBuffer(payload_buffer, chunk_bytes);
  pipe.InitBuffer(signal_buffer, 3 * 32);
  pipe.InitBuffer(skew_buffer, 128);
  const auto payload = payload_buffer.Get<T>();
  const auto epoch = signal_buffer.GetWithOffset<int32_t>(kSignalElements, 0);
  const auto publication =
      signal_buffer.GetWithOffset<int32_t>(kSignalElements, 32);
  const auto probe = signal_buffer.GetWithOffset<int32_t>(kSignalElements, 64);
  const auto skew_tile = skew_buffer.Get<int32_t>();

  const int32_t lane = static_cast<int32_t>(AscendC::GetBlockIdx());
  const int64_t chunk_elements = chunk_bytes / sizeof(T);
  const int64_t round_elements = lanes * chunk_elements;
  const int64_t rounds = (count - 1) / round_elements + 1;
  const AscendC::DataCopyExtParams signal_params{1, 32, 0, 0, 0};
  const AscendC::DataCopyPadExtParams<int32_t> signal_padding{};
  const AscendC::DataCopyPadExtParams<T> payload_padding{};
  AscendC::DataCopyPad(
      epoch, epochs[lane * kSignalStride], signal_params, signal_padding);
  AscendC::PipeBarrier<PIPE_ALL>();
  int32_t generation = epoch.GetValue(0);

  for (int64_t round = 0; round < rounds; ++round) {
    const int64_t offset = round * round_elements + lane * chunk_elements;
    const int64_t remaining = count - offset;
    const uint32_t length = static_cast<uint32_t>(
        remaining <= 0
            ? 0
            : (remaining < chunk_elements ? remaining : chunk_elements));
    const AscendC::DataCopyExtParams payload_params{
        1, static_cast<uint32_t>(length * sizeof(T)), 0, 0, 0};
    generation = 1 - generation;
    for (int32_t element = 0; element < kSignalElements; ++element) {
      publication.SetValue(element, generation);
    }
    AscendC::PipeBarrier<PIPE_ALL>();

    if (length > 0) {
      AscendC::DataCopyPad(
          payload, source[offset], payload_params, payload_padding);
      AscendC::PipeBarrier<PIPE_ALL>();
      for (int32_t peer = 0; peer < world_size; ++peer) {
        aclshmemx_mte_put_nbi(receive[(rank * lanes + lane) * chunk_elements],
                              payload,
                              length,
                              peer,
                              EVENT_ID0);
        AscendC::PipeBarrier<PIPE_ALL>();
      }
      // Public symmetric-data completion precedes every ready publication.
      aclshmem_quiet();
    }
    // Empty tail lanes still publish ready and acknowledge every producer.
    for (int32_t peer = 0; peer < world_size; ++peer) {
      aclshmemx_mte_put_nbi(controls[(rank * lanes + lane) * kSignalStride],
                            publication,
                            kSignalElements,
                            peer,
                            EVENT_ID0);
      AscendC::PipeBarrier<PIPE_ALL>();
    }
    for (int32_t peer_offset = 0; peer_offset < world_size; ++peer_offset) {
      const int32_t peer = (rank + peer_offset) % world_size;
      wait_generation(
          controls[(peer * lanes + lane) * kSignalStride], probe, generation);
      if (skew_phase == 1 && rank == 1 && lane == 0 && peer == 0 &&
          round == 0) {
        copy_skew_tiles(scratch, skew_tile, skew_iterations);
      }
      if (length > 0) {
        AscendC::DataCopyPad(payload,
                             receive[(peer * lanes + lane) * chunk_elements],
                             payload_params,
                             payload_padding);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopyPad(
            output[peer * count + offset], payload, payload_params);
        // Ordinary output completion is required before granting window reuse.
        AscendC::PipeBarrier<PIPE_ALL>();
      }
      if (skew_phase == 2 && rank == 1 && lane == 0 && peer == 0 &&
          round == rounds - 1) {
        copy_skew_tiles(scratch, skew_tile, skew_iterations);
      }
      aclshmemx_mte_put_nbi(
          controls[(world_size * lanes + rank * lanes + lane) * kSignalStride],
          publication,
          kSignalElements,
          peer,
          EVENT_ID0);
      AscendC::PipeBarrier<PIPE_ALL>();
    }
    // This includes the final round: the next call may immediately reuse GM.
    for (int32_t peer = 0; peer < world_size; ++peer) {
      wait_generation(
          controls[(world_size * lanes + peer * lanes + lane) * kSignalStride],
          probe,
          generation);
    }
  }
  for (int32_t element = 0; element < kSignalElements; ++element) {
    epoch.SetValue(element, generation);
  }
  AscendC::PipeBarrier<PIPE_ALL>();
  AscendC::DataCopyPad(epochs[lane * kSignalStride], epoch, signal_params);
  AscendC::PipeBarrier<PIPE_ALL>();
}

}  // namespace

#define XLLM_ACLSHMEM_ALL_GATHER_KERNEL(name, type)                         \
  extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void name( \
      GM_ADDR source,                                                       \
      GM_ADDR output,                                                       \
      GM_ADDR receive,                                                      \
      GM_ADDR controls,                                                     \
      GM_ADDR epochs,                                                       \
      GM_ADDR scratch,                                                      \
      int64_t count,                                                        \
      int32_t rank,                                                         \
      int32_t world_size,                                                   \
      int32_t lanes,                                                        \
      uint32_t chunk_bytes,                                                 \
      int32_t skew_phase,                                                   \
      int32_t skew_iterations) {                                            \
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);                         \
    all_gather<type>(source,                                                \
                     output,                                                \
                     receive,                                               \
                     controls,                                              \
                     epochs,                                                \
                     scratch,                                               \
                     count,                                                 \
                     rank,                                                  \
                     world_size,                                            \
                     lanes,                                                 \
                     chunk_bytes,                                           \
                     skew_phase,                                            \
                     skew_iterations);                                      \
  }

XLLM_ACLSHMEM_ALL_GATHER_KERNEL(aclshmem_all_gather_fp32, float)
XLLM_ACLSHMEM_ALL_GATHER_KERNEL(aclshmem_all_gather_fp16, half)
XLLM_ACLSHMEM_ALL_GATHER_KERNEL(aclshmem_all_gather_bf16, bfloat16_t)

#undef XLLM_ACLSHMEM_ALL_GATHER_KERNEL
