// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Masoud Jami
#pragma once

#ifdef LLMT_HAS_CUDA

#include <cublasLt.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "llmt/core/precision.h"

namespace llmt::kernels {
struct AlgoCache;
}

namespace llmt {

class Arena;

/**
 * Which kernel implementations to dispatch (DESIGN §7.1 RunConfig).
 * Machinery only — must never affect convergence.
 */
enum class KernelBackend : uint8_t { Naive, Fused };

/**
 * Everything a launch needs, threaded explicitly through every call
 * (DESIGN §7.2). A value type: cheap to copy, no ownership.
 */
struct RunCtx {
    cudaStream_t stream = nullptr;
    cublasLtHandle_t blas = nullptr;
    void* blas_workspace = nullptr;  // Device-owned scratch for cuBLASLt
    size_t blas_workspace_bytes = 0;
    kernels::AlgoCache* algo_cache = nullptr;  // Device-owned (invariant 5)
    Arena* activations = nullptr;  // owned by the model; set after planning
    PrecisionPolicy precision;
    KernelBackend backend = KernelBackend::Naive;
    uint64_t seed = 0;  // counter-based RNG root
    int64_t step = 0;   // RNG offset component (dropout replay, data order)
};

}  // namespace llmt

#endif  // LLMT_HAS_CUDA
