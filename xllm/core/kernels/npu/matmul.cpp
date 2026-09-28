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

// KEEP_DTYPE: keep the input dtype as the compute dtype. The caller-supplied
// FP32 output is what selects the 16-bit-in / 32-bit-out kernel on A3, which
// accumulates in fp32 (measured max relative error ~2e-07 against fp64).
constexpr int8_t kCubeMathTypeKeepDtype = 0;

int64_t get_tensor_npu_format(const torch::Tensor& tensor) {
#ifdef TORCH_HIGHER_THAN_PTA6
  return at_npu::native::get_npu_format(tensor);
#else
  return at_npu::native::NPUNativeFunctions::get_npu_format(tensor);
#endif
}

void check_operands(const torch::Tensor& x1, const torch::Tensor& x2) {
  // Reject 3-D on purpose: those aclnn matmul forms accept a bf16 x bf16 ->
  // fp32 request and return bf16-precision values inside the fp32 tensor
  // without reporting an error, which is exactly what this operator avoids.
  CHECK_EQ(x1.dim(), 2) << "matmul_16in32out x1 must be [M,K]";
  CHECK_EQ(x2.dim(), 2) << "matmul_16in32out x2 must be [K,N]";
  CHECK_EQ(x1.scalar_type(), torch::kBFloat16)
      << "matmul_16in32out x1 must be bf16";
  CHECK_EQ(x2.scalar_type(), x1.scalar_type())
      << "matmul_16in32out operand dtypes must match";
  CHECK_EQ(x1.device(), x2.device())
      << "matmul_16in32out operands must use the same device";
  CHECK_EQ(x1.size(1), x2.size(0))
      << "matmul_16in32out reduction dimension mismatch";
  CHECK(x2.is_contiguous()) << "matmul_16in32out x2 must be contiguous";
  // The aclnn descriptor keeps a fractal-NZ tensor as NZ and only re-checks its
  // tile alignment, so an NZ operand would be reinterpreted tile by tile
  // instead of failing.
  CHECK_EQ(get_tensor_npu_format(x1), ACL_FORMAT_ND)
      << "matmul_16in32out x1 must be ND";
  CHECK_EQ(get_tensor_npu_format(x2), ACL_FORMAT_ND)
      << "matmul_16in32out x2 must be ND";
}

}  // namespace

// Generic dtype-following matmul used by xllm::kernel::matmul (the DiT model
// path); the router gate uses matmul_16in32out instead.
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
  check_operands(x1, x2);
  torch::Tensor contiguous_x1 = x1.contiguous();
  torch::Tensor out =
      torch::empty({x1.size(0), x2.size(1)}, x1.options().dtype(torch::kFloat));
  EXEC_NPU_CMD(aclnnMatmul, contiguous_x1, x2, out, kCubeMathTypeKeepDtype);
  return out;
}

}  // namespace xllm::kernel::npu
