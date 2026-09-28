/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Minimal aclnn entry points used by the matmul dtype probe
// (tests/npu/test_matmul_16in32out_a3.py). The probe checks whether an A3
// (Ascend910_93) device computes a bf16 x bf16 matmul with fp32 accumulation
// and an fp32 result when the caller supplies an fp32 `out` tensor.
//
// The wrapper only builds aclTensor descriptors over caller-owned device
// buffers and issues the two-phase aclnn call; memory management and
// synchronization stay on the Python side (torch_npu).
//
// Build inside the NPU container:
//   g++ -std=c++17 -O2 -fPIC -shared tests/npu/aclnn_matmul_probe.cpp \
//     -o /tmp/aclnn_matmul_probe.so \
//     -I/usr/local/Ascend/cann-9.0.0/include \
//     -L/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64 \
//     -lopapi -lnnopbase -lascendcl

#include <cstdint>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_batch_matmul.h"
#include "aclnnop/aclnn_matmul.h"
#include "aclnnop/aclnn_mm.h"

namespace {

// Probe dtype code: 0 = float32, 1 = float16, 2 = bfloat16.
constexpr int32_t kDtypeFp32 = 0;
constexpr int32_t kDtypeFp16 = 1;
constexpr int32_t kDtypeBf16 = 2;

// KEEP_DTYPE: inputs keep their own dtype; an fp32 `out` is kept as fp32.
constexpr int8_t kCubeMathTypeKeepDtype = 0;

thread_local std::string g_last_error;

aclDataType to_acl_dtype(int32_t code) {
  switch (code) {
    case kDtypeFp32:
      return ACL_FLOAT;
    case kDtypeFp16:
      return ACL_FLOAT16;
    case kDtypeBf16:
      return ACL_BF16;
    default:
      return ACL_DT_UNDEFINED;
  }
}

void record_error(const char* stage, int32_t status) {
  const char* msg = aclGetRecentErrMsg();
  g_last_error = std::string(stage) +
                 " failed, aclnnStatus=" + std::to_string(status) +
                 ", msg=" + (msg == nullptr ? "" : msg);
}

int32_t execute(
    uint64_t workspace_size,
    aclOpExecutor* executor,
    void* stream,
    const char* stage,
    int32_t (*launch)(void*, uint64_t, aclOpExecutor*, aclrtStream)) {
  void* workspace = nullptr;
  if (workspace_size > 0) {
    if (aclrtMalloc(&workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST) !=
        ACL_SUCCESS) {
      g_last_error = std::string(stage) + ": aclrtMalloc(workspace) failed";
      return -1;
    }
  }
  int32_t status = launch(
      workspace, workspace_size, executor, static_cast<aclrtStream>(stream));
  if (status != 0) {
    record_error(stage, status);
  }
  if (workspace != nullptr) {
    aclrtFree(workspace);
  }
  return status;
}

}  // namespace

extern "C" {

// 3D batched matmul: x [batch, m, k] @ w [batch, k, n] -> y [batch, m, n].
// All three tensors use ND format; dtype of x/w follows in_dtype, y follows
// out_dtype.
int32_t probe_bmm(const void* x,
                  const void* w,
                  void* y,
                  int64_t batch,
                  int64_t m,
                  int64_t k,
                  int64_t n,
                  int32_t in_dtype,
                  int32_t out_dtype,
                  void* stream) {
  const aclDataType acl_in = to_acl_dtype(in_dtype);
  const aclDataType acl_out = to_acl_dtype(out_dtype);
  if (acl_in == ACL_DT_UNDEFINED || acl_out == ACL_DT_UNDEFINED) {
    g_last_error = "probe_bmm: unsupported dtype code";
    return -1;
  }
  int64_t x_shape[3] = {batch, m, k};
  int64_t w_shape[3] = {batch, k, n};
  int64_t y_shape[3] = {batch, m, n};
  aclTensor* x_t = aclCreateTensor(x_shape,
                                   3,
                                   acl_in,
                                   nullptr,
                                   0,
                                   ACL_FORMAT_ND,
                                   x_shape,
                                   3,
                                   const_cast<void*>(x));
  aclTensor* w_t = aclCreateTensor(w_shape,
                                   3,
                                   acl_in,
                                   nullptr,
                                   0,
                                   ACL_FORMAT_ND,
                                   w_shape,
                                   3,
                                   const_cast<void*>(w));
  aclTensor* y_t = aclCreateTensor(
      y_shape, 3, acl_out, nullptr, 0, ACL_FORMAT_ND, y_shape, 3, y);
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  int32_t status = aclnnBatchMatMulGetWorkspaceSize(
      x_t, w_t, y_t, kCubeMathTypeKeepDtype, &workspace_size, &executor);
  if (status != 0) {
    record_error("aclnnBatchMatMulGetWorkspaceSize", status);
  } else {
    status = execute(workspace_size,
                     executor,
                     stream,
                     "aclnnBatchMatMul",
                     &aclnnBatchMatMul);
    if (status == 0) {
      g_last_error.clear();
    }
  }
  aclDestroyTensor(x_t);
  aclDestroyTensor(w_t);
  aclDestroyTensor(y_t);
  return status;
}

// 2D matmul: x [m, k] @ w [k, n] -> y [m, n].
int32_t probe_mm(const void* x,
                 const void* w,
                 void* y,
                 int64_t m,
                 int64_t k,
                 int64_t n,
                 int32_t in_dtype,
                 int32_t out_dtype,
                 void* stream) {
  const aclDataType acl_in = to_acl_dtype(in_dtype);
  const aclDataType acl_out = to_acl_dtype(out_dtype);
  if (acl_in == ACL_DT_UNDEFINED || acl_out == ACL_DT_UNDEFINED) {
    g_last_error = "probe_mm: unsupported dtype code";
    return -1;
  }
  int64_t x_shape[2] = {m, k};
  int64_t w_shape[2] = {k, n};
  int64_t y_shape[2] = {m, n};
  aclTensor* x_t = aclCreateTensor(x_shape,
                                   2,
                                   acl_in,
                                   nullptr,
                                   0,
                                   ACL_FORMAT_ND,
                                   x_shape,
                                   2,
                                   const_cast<void*>(x));
  aclTensor* w_t = aclCreateTensor(w_shape,
                                   2,
                                   acl_in,
                                   nullptr,
                                   0,
                                   ACL_FORMAT_ND,
                                   w_shape,
                                   2,
                                   const_cast<void*>(w));
  aclTensor* y_t = aclCreateTensor(
      y_shape, 2, acl_out, nullptr, 0, ACL_FORMAT_ND, y_shape, 2, y);
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  int32_t status = aclnnMmGetWorkspaceSize(
      x_t, w_t, y_t, kCubeMathTypeKeepDtype, &workspace_size, &executor);
  if (status != 0) {
    record_error("aclnnMmGetWorkspaceSize", status);
  } else {
    status = execute(workspace_size, executor, stream, "aclnnMm", &aclnnMm);
    if (status == 0) {
      g_last_error.clear();
    }
  }
  aclDestroyTensor(x_t);
  aclDestroyTensor(w_t);
  aclDestroyTensor(y_t);
  return status;
}

// Multidimensional matmul: batch <= 0 selects x [m, k] @ w [k, n] -> y [m, n];
// a positive batch selects x [batch, m, k] @ w [batch, k, n] -> y
// [batch, m, n]. Same 3D shapes as probe_bmm, but through aclnnMatmul, which
// mirrors the 2D aclnnMm path.
int32_t probe_matmul(const void* x,
                     const void* w,
                     void* y,
                     int64_t batch,
                     int64_t m,
                     int64_t k,
                     int64_t n,
                     int32_t in_dtype,
                     int32_t out_dtype,
                     void* stream) {
  const aclDataType acl_in = to_acl_dtype(in_dtype);
  const aclDataType acl_out = to_acl_dtype(out_dtype);
  if (acl_in == ACL_DT_UNDEFINED || acl_out == ACL_DT_UNDEFINED) {
    g_last_error = "probe_matmul: unsupported dtype code";
    return -1;
  }
  int64_t x_shape[3] = {m, k, 0};
  int64_t w_shape[3] = {k, n, 0};
  int64_t y_shape[3] = {m, n, 0};
  int64_t dim_num = 2;
  if (batch > 0) {
    x_shape[0] = batch;
    x_shape[1] = m;
    x_shape[2] = k;
    w_shape[0] = batch;
    w_shape[1] = k;
    w_shape[2] = n;
    y_shape[0] = batch;
    y_shape[1] = m;
    y_shape[2] = n;
    dim_num = 3;
  }
  aclTensor* x_t = aclCreateTensor(x_shape,
                                   dim_num,
                                   acl_in,
                                   nullptr,
                                   0,
                                   ACL_FORMAT_ND,
                                   x_shape,
                                   dim_num,
                                   const_cast<void*>(x));
  aclTensor* w_t = aclCreateTensor(w_shape,
                                   dim_num,
                                   acl_in,
                                   nullptr,
                                   0,
                                   ACL_FORMAT_ND,
                                   w_shape,
                                   dim_num,
                                   const_cast<void*>(w));
  aclTensor* y_t = aclCreateTensor(y_shape,
                                   dim_num,
                                   acl_out,
                                   nullptr,
                                   0,
                                   ACL_FORMAT_ND,
                                   y_shape,
                                   dim_num,
                                   y);
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  int32_t status = aclnnMatmulGetWorkspaceSize(
      x_t, w_t, y_t, kCubeMathTypeKeepDtype, &workspace_size, &executor);
  if (status != 0) {
    record_error("aclnnMatmulGetWorkspaceSize", status);
  } else {
    status =
        execute(workspace_size, executor, stream, "aclnnMatmul", &aclnnMatmul);
    if (status == 0) {
      g_last_error.clear();
    }
  }
  aclDestroyTensor(x_t);
  aclDestroyTensor(w_t);
  aclDestroyTensor(y_t);
  return status;
}

const char* probe_last_error(void) { return g_last_error.c_str(); }

}  // extern "C"
