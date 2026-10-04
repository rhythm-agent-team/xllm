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

#pragma once

#include <torch/types.h>

namespace xllm::kernel::npu {

// Borrowed buffers from an initialized public ACLSHMEM MTE context. All PEs
// must agree on rank mapping, count and configuration. Receive/controls/epochs
// are persistent symmetric allocations initialized before capture. Their owner
// serializes submissions on one stream (including retained graph replay) and
// keeps buffers and the SHMEM context alive until all submitted work is
// drained. This raw submission neither allocates nor synchronizes nor changes
// host epochs.
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
                                           int64_t skew_iterations);

}  // namespace xllm::kernel::npu
