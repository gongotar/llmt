# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Masoud Jami
cmake_minimum_required(VERSION 3.24)
project(llmt VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Warnings everywhere (host C++ directly; CUDA host-side code via -Xcompiler).
add_compile_options(
    "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra>"
    "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>"
)

# CUDA is optional at configure time so the repo also configures on the
# (CUDA-less) macOS dev host; all GPU targets are gated on LLMT_HAS_CUDA.
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    # Must be set BEFORE enable_language(CUDA): enablement initializes
    # CMAKE_CUDA_ARCHITECTURES to the compiler default (52) if unset, which
    # would make this guard dead code. Pods are disposable and the GPU model
    # varies per session: build for whatever is present. Override with
    # -DCMAKE_CUDA_ARCHITECTURES.
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
    enable_language(CUDA)
    set(LLMT_HAS_CUDA ON)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    option(LLMT_LINEINFO "Add -lineinfo to CUDA builds (Nsight source correlation)" OFF)
    message(STATUS "llmt: CUDA enabled (arch ${CMAKE_CUDA_ARCHITECTURES})")
else()
    set(LLMT_HAS_CUDA OFF)
    message(STATUS "llmt: CUDA compiler not found — building host-only targets")
endif()

add_library(llmt STATIC
    src/core/version.cpp
)
target_include_directories(llmt PUBLIC include)

if(LLMT_HAS_CUDA)
    target_sources(llmt PRIVATE
        src/core/arena.cu
        src/core/device.cu
        src/core/rng.cu
        src/kernels/attention.cpp
        src/kernels/cross_entropy.cu
        src/kernels/embedding.cu
        src/kernels/fill.cu
        src/kernels/gelu.cu
        src/kernels/matmul.cpp
        src/kernels/permute.cu
        src/kernels/residual.cu
        src/kernels/rmsnorm.cu
        src/kernels/rope.cu
        src/kernels/softmax.cu
        src/layers/linear.cpp
        src/layers/mlp.cpp
        src/model/init.cpp
        src/model/param_store.cpp
    )
    target_compile_definitions(llmt PUBLIC LLMT_HAS_CUDA=1)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(llmt PUBLIC CUDA::cudart CUDA::cublasLt)
    if(LLMT_LINEINFO)
        target_compile_options(llmt PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
    endif()
endif()

enable_testing()
add_subdirectory(tests)
add_subdirectory(bench)
