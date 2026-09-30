/* Copyright 2025-2026 The xLLM Authors.

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

#include <torch/nn/functional/linear.h>

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "core/kernels/npu/npu_ops_api.h"
#include "ops_npu/npu_ops.h"

namespace xllm::kernel::npu {
namespace {

constexpr int8_t kCubeMathTypeKeepDtype = 0;

void check_matmul_operand(const torch::Tensor& tensor, const char* name) {
  CHECK(tensor.defined()) << name << " must be defined";
  CHECK(tensor.device().is_privateuseone()) << name << " must be on NPU";
  CHECK_EQ(tensor.dim(), 2) << name << " must be a 2-D tensor";
  CHECK_EQ(tensor.scalar_type(), torch::kBFloat16) << name << " must be BF16";
  CHECK(tensor.layout() == torch::kStrided) << name << " must be dense";
  CHECK(tensor.is_contiguous()) << name << " must be contiguous";
  CHECK_GT(tensor.numel(), 0) << name << " must be nonempty";
  CHECK_EQ(at_npu::native::custom_ops::get_npu_format(tensor), ACL_FORMAT_ND)
      << name << " must have ND storage";
}

}  // namespace

torch::Tensor matmul(const torch::Tensor& a,
                     const torch::Tensor& b,
                     const std::optional<torch::Tensor>& bias) {
  if (!bias.has_value()) {
    return torch::nn::functional::linear(a, b);
  } else {
    return torch::nn::functional::linear(a, b, bias.value());
  }
}

torch::Tensor matmul_16in32out(const torch::Tensor& x1,
                               const torch::Tensor& x2) {
  check_matmul_operand(x1, "matmul_16in32out x1 [M,K]");
  check_matmul_operand(x2, "matmul_16in32out x2 [K,N]");
  CHECK_EQ(x1.device(), x2.device())
      << "matmul_16in32out operands must use the same device";
  CHECK_EQ(x1.size(1), x2.size(0))
      << "matmul_16in32out reduction dimension mismatch: x1=" << x1.sizes()
      << ", x2=" << x2.sizes();
  CHECK_EQ(c10_npu::getCurrentNPUStream().device_index(), x1.device().index())
      << "matmul_16in32out current stream does not match " << x1.device();
  torch::Tensor output =
      torch::empty({x1.size(0), x2.size(1)}, x1.options().dtype(torch::kFloat));
  CHECK_EQ(at_npu::native::custom_ops::get_npu_format(output), ACL_FORMAT_ND)
      << "matmul_16in32out output must have ND storage";
  // The FP32 output selects the 2-D BF16-input/FP32-output ACLNN route.
  EXEC_NPU_CMD(aclnnMatmul, x1, x2, output, kCubeMathTypeKeepDtype);
  return output;
}

}  // namespace xllm::kernel::npu
