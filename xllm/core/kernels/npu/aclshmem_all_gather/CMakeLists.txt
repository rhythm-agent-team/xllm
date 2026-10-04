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

set(ASCENDC_CMAKE_PATH "" CACHE FILEPATH "Official CANN ascendc.cmake module")
set(ACLSHMEM_ROOT "" CACHE PATH "Complete public ACLSHMEM SDK root")
foreach(required_path
    "${ASCENDC_CMAKE_PATH}"
    "${ACLSHMEM_ROOT}/include/device/ub2gm/engine/shmem_device_mte.h"
    "${ACLSHMEM_ROOT}/include/device/gm2gm/shmem_device_mo.h"
    "${ACLSHMEM_ROOT}/src/device/shmemi_device_common.hpp")
  if(NOT EXISTS "${required_path}")
    message(FATAL_ERROR "ACLSHMEM AllGather prerequisite missing: ${required_path}")
  endif()
endforeach()
if(NOT DEFINED SOC_VERSION OR NOT DEFINED ASCEND_CANN_PACKAGE_PATH)
  message(FATAL_ERROR "ACLSHMEM AllGather requires SOC_VERSION and ASCEND_CANN_PACKAGE_PATH")
endif()

# The SDK owns device compilation and generated ACLRT launch stubs. Torch
# adapters remain on xLLM's ordinary host compiler and ABI.
set(ASCEND_KERNEL_LAUNCH_ONLY ON)
include("${ASCENDC_CMAKE_PATH}")
ascendc_library(aclshmem_all_gather_device STATIC all_gather_kernel.cpp)
ascendc_include_directories(aclshmem_all_gather_device PRIVATE
  "${ACLSHMEM_ROOT}/include"
  "${ACLSHMEM_ROOT}/src"
  "${ACLSHMEM_ROOT}/src/device"
  "${ACLSHMEM_ROOT}/src/device_simt"
)
