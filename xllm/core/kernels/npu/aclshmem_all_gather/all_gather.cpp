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

#include "core/kernels/npu/aclshmem_all_gather/all_gather.h"

#include <ATen/MemoryOverlap.h>
#include <acl/acl.h>
#include <glog/logging.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <array>
#include <limits>

#ifdef TORCH_HIGHER_THAN_PTA6
#include <torch_npu/csrc/aten/CustomFunctions.h>
#else
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#endif

#include "aclrtlaunch_aclshmem_all_gather_bf16.h"
#include "aclrtlaunch_aclshmem_all_gather_fp16.h"
#include "aclrtlaunch_aclshmem_all_gather_fp32.h"

namespace xllm::kernel::npu {
namespace {

void check_buffer(const torch::Tensor& tensor,
                  const torch::Tensor& input,
                  torch::ScalarType dtype,
                  const char* name) {
  CHECK(tensor.defined()) << name << " must be defined";
  CHECK(tensor.device().is_privateuseone()) << name << " must be on NPU";
  CHECK_EQ(tensor.device(), input.device()) << name << " device mismatch";
  CHECK(tensor.layout() == torch::kStrided && tensor.is_contiguous())
      << name << " must be dense and contiguous";
  CHECK_GT(tensor.numel(), 0) << name << " must be nonempty";
  CHECK_EQ(tensor.scalar_type(), dtype) << name << " dtype mismatch";
#ifdef TORCH_HIGHER_THAN_PTA6
  const int64_t format = at_npu::native::get_npu_format(tensor);
#else
  const int64_t format =
      at_npu::native::NPUNativeFunctions::get_npu_format(tensor);
#endif
  CHECK_EQ(format, ACL_FORMAT_ND) << name << " requires ND storage";
}

void check_symmetric_alignment(const torch::Tensor& tensor, const char* name) {
  CHECK_EQ(reinterpret_cast<uintptr_t>(tensor.data_ptr()) % 128, 0)
      << name << " must have a 128-byte-aligned symmetric address";
}

}  // namespace

void aclshmem_all_gather_on_current_stream(const torch::Tensor& input,
                                           torch::Tensor& output,
                                           torch::Tensor& receive,
                                           torch::Tensor& controls,
                                           torch::Tensor& epochs,
                                           torch::Tensor& scratch,
                                           int64_t rank,
                                           int64_t world_size,
                                           int64_t lanes,
                                           int64_t chunk_bytes,
                                           int64_t skew_phase,
                                           int64_t skew_iterations) {
  CHECK(input.defined()) << "ACLSHMEM AllGather input must be defined";
  const torch::ScalarType dtype = input.scalar_type();
  CHECK(dtype == torch::kFloat || dtype == torch::kHalf ||
        dtype == torch::kBFloat16)
      << "ACLSHMEM AllGather supports FP32, FP16 and BF16 only";
  check_buffer(input, input, dtype, "input");
  check_buffer(output, input, dtype, "output");
  check_buffer(receive, input, dtype, "receive");
  check_buffer(controls, input, torch::kInt, "controls");
  check_buffer(epochs, input, torch::kInt, "epochs");
  check_buffer(scratch, input, torch::kInt, "scratch");
  const std::array<const torch::Tensor*, 6> buffers{
      &input, &output, &receive, &controls, &epochs, &scratch};
  for (size_t first = 0; first < buffers.size(); ++first) {
    for (size_t second = first + 1; second < buffers.size(); ++second) {
      CHECK(torch::get_overlap_status(*buffers[first], *buffers[second]) ==
            torch::MemOverlapStatus::No)
          << "ACLSHMEM AllGather buffers must not overlap: " << first << ", "
          << second;
    }
  }
  CHECK(world_size == 2 || world_size == 4 || world_size == 8 ||
        world_size == 16)
      << "ACLSHMEM AllGather requires 2, 4, 8 or 16 PEs";
  CHECK_GE(rank, 0);
  CHECK_LT(rank, world_size);
  CHECK_GT(lanes, 0);
  CHECK_GT(chunk_bytes, 0);
  CHECK_LE(chunk_bytes, 65536);
  CHECK_EQ(chunk_bytes % 128, 0);
  CHECK_GE(skew_phase, 0);
  CHECK_LE(skew_phase, 2);
  CHECK((skew_phase == 0 && skew_iterations == 0) ||
        (skew_phase != 0 && skew_iterations > 0 && skew_iterations <= 32768))
      << "Invalid controlled-skew configuration";
  CHECK_LE(input.numel(), std::numeric_limits<int32_t>::max());
  CHECK_EQ(output.numel(), world_size * input.numel())
      << "Standard AllGather requires one contiguous input-sized block per PE";

  const auto stream = c10_npu::getCurrentNPUStream();
  CHECK_EQ(stream.device_index(), input.device().index())
      << "ACLSHMEM current stream and input device mismatch";
  int32_t device_id = -1;
  CHECK_EQ(aclrtGetDevice(&device_id), ACL_SUCCESS);
  CHECK_EQ(device_id, input.device().index());
  int64_t vector_cores = 0;
  CHECK_EQ(aclrtGetDeviceInfo(static_cast<uint32_t>(device_id),
                              ACL_DEV_ATTR_VECTOR_CORE_NUM,
                              &vector_cores),
           ACL_SUCCESS)
      << "Cannot query ACLSHMEM AllGather vector core capacity";
  CHECK_LE(lanes, vector_cores)
      << "Each lane requires one resident Vector core";
  CHECK_LE(lanes, std::numeric_limits<int32_t>::max());
  const int64_t chunk_elements = chunk_bytes / input.element_size();
  CHECK_EQ(receive.numel(), world_size * lanes * chunk_elements);
  CHECK_EQ(controls.numel(), 2 * world_size * lanes * 32);
  CHECK_EQ(epochs.numel(), lanes * 32);
  CHECK_GE(scratch.numel(),
           2 * (skew_iterations == 0 ? 1 : skew_iterations) * 32);
  check_symmetric_alignment(receive, "receive");
  check_symmetric_alignment(controls, "controls");
  check_symmetric_alignment(epochs, "epochs");

  const uint32_t block_dim = static_cast<uint32_t>(lanes);
  const int32_t rank_arg = static_cast<int32_t>(rank);
  const int32_t world_arg = static_cast<int32_t>(world_size);
  const int32_t lanes_arg = static_cast<int32_t>(lanes);
  const uint32_t chunk_arg = static_cast<uint32_t>(chunk_bytes);
  const int32_t phase_arg = static_cast<int32_t>(skew_phase);
  const int32_t iterations_arg = static_cast<int32_t>(skew_iterations);
  aclError result = ACL_SUCCESS;
#define XLLM_LAUNCH_ACLSHMEM_ALL_GATHER(kernel)    \
  ACLRT_LAUNCH_KERNEL(kernel)(block_dim,           \
                              stream.stream(),     \
                              input.data_ptr(),    \
                              output.data_ptr(),   \
                              receive.data_ptr(),  \
                              controls.data_ptr(), \
                              epochs.data_ptr(),   \
                              scratch.data_ptr(),  \
                              input.numel(),       \
                              rank_arg,            \
                              world_arg,           \
                              lanes_arg,           \
                              chunk_arg,           \
                              phase_arg,           \
                              iterations_arg)
  switch (dtype) {
    case torch::kFloat:
      result = XLLM_LAUNCH_ACLSHMEM_ALL_GATHER(aclshmem_all_gather_fp32);
      break;
    case torch::kHalf:
      result = XLLM_LAUNCH_ACLSHMEM_ALL_GATHER(aclshmem_all_gather_fp16);
      break;
    case torch::kBFloat16:
      result = XLLM_LAUNCH_ACLSHMEM_ALL_GATHER(aclshmem_all_gather_bf16);
      break;
    default:
      LOG(FATAL) << "Unsupported ACLSHMEM AllGather dtype " << dtype;
  }
#undef XLLM_LAUNCH_ACLSHMEM_ALL_GATHER
  CHECK_EQ(result, ACL_SUCCESS) << "ACLSHMEM AllGather kernel launch failed";
}

}  // namespace xllm::kernel::npu
