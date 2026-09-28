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

#include <gtest/gtest.h>
#include <torch/torch.h>
#include <torch_npu/csrc/libs/init_npu.h>
#include <torch_npu/torch_npu.h>

#include <string>
#include <vector>

#include "core/kernels/npu/npu_ops_api.h"

namespace xllm::kernel::npu {
namespace {

// The bf16 products are exact in fp32, so the fp64 reference isolates the
// accumulator precision. A bf16 accumulator would land around 2e-03 instead.
constexpr double kMaxRelativeError = 1e-4;
// MoE router gate shape: hidden_size x num_experts.
constexpr int64_t kHiddenSize = 6144;
constexpr int64_t kNumExperts = 256;

class Matmul16In32OutTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { torch_npu::init_npu("npu:0"); }

  static void TearDownTestSuite() { torch_npu::finalize_npu(); }
};

torch::Tensor operand(torch::IntArrayRef sizes) {
  return torch::randn(sizes, torch::kFloat).to(torch::kBFloat16);
}

double max_relative_error(const torch::Tensor& out,
                          const torch::Tensor& reference) {
  return (out.double() - reference).abs().max().item<double>() /
         reference.abs().max().item<double>();
}

void expect_fp32_accumulation(int64_t num_tokens) {
  const torch::Device device("npu:0");
  const torch::Tensor x_cpu = operand({num_tokens, kHiddenSize});
  const torch::Tensor weight_cpu = operand({kHiddenSize, kNumExperts});
  const torch::Tensor reference = x_cpu.double().matmul(weight_cpu.double());

  const torch::Tensor out =
      matmul_16in32out(x_cpu.to(device), weight_cpu.to(device)).cpu();
  ASSERT_EQ(out.sizes(), torch::IntArrayRef({num_tokens, kNumExperts}));
  ASSERT_EQ(out.scalar_type(), torch::kFloat);

  const double relative_error = max_relative_error(out, reference);
  EXPECT_LT(relative_error, kMaxRelativeError)
      << "max relative error against the fp64 reference: " << relative_error;
  // An fp32 tensor holding a bf16-precision result is bit-identical to its own
  // bf16 round trip; a real fp32 accumulator never is.
  EXPECT_FALSE(torch::equal(out, out.to(torch::kBFloat16).to(torch::kFloat)))
      << "output lies on the bf16 grid: the matmul degraded to bf16";
}

TEST_F(Matmul16In32OutTest, AccumulatesInFp32) {
  const std::vector<int64_t> num_tokens = {1, 8, 512};
  for (int64_t m : num_tokens) {
    SCOPED_TRACE("num_tokens=" + std::to_string(m));
    expect_fp32_accumulation(m);
  }
}

// Control for the test above: the bf16-out route stays far outside the
// tolerance, so the accuracy assertion can actually detect a degraded matmul.
TEST_F(Matmul16In32OutTest, Bf16OutputRouteIsOutsideTheTolerance) {
  const torch::Device device("npu:0");
  const torch::Tensor x_cpu = operand({512, kHiddenSize});
  const torch::Tensor weight_cpu = operand({kHiddenSize, kNumExperts});
  const torch::Tensor reference = x_cpu.double().matmul(weight_cpu.double());

  const torch::Tensor bf16_out =
      torch::matmul(x_cpu.to(device), weight_cpu.to(device)).cpu();
  EXPECT_GT(max_relative_error(bf16_out, reference), kMaxRelativeError);
}

TEST_F(Matmul16In32OutTest, RejectsUnsupportedOperands) {
  const torch::Device device("npu:0");
  const torch::Tensor x = operand({8, kHiddenSize}).to(device);
  const torch::Tensor weight = operand({kHiddenSize, kNumExperts}).to(device);

  ASSERT_DEATH(
      { (void)matmul_16in32out(x.unsqueeze(0), weight); },
      "matmul_16in32out x1 must be");
  ASSERT_DEATH(
      { (void)matmul_16in32out(x.to(torch::kFloat), weight); },
      "matmul_16in32out x1 must be bf16");
  ASSERT_DEATH(
      { (void)matmul_16in32out(x, weight.narrow(0, 0, kHiddenSize / 2)); },
      "matmul_16in32out reduction dimension mismatch");
  ASSERT_DEATH(
      { (void)matmul_16in32out(x, weight.transpose(0, 1)); },
      "matmul_16in32out x2 must be contiguous");
}

}  // namespace
}  // namespace xllm::kernel::npu
