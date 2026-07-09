// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Masoud Jami
#pragma once

#ifdef LLMT_HAS_CUDA

#include <cuda_runtime.h>

#include <cstdint>

namespace llmt::kernels {

/** Fill dst[0..n) with value on stream s (non-zero floats aren't memset-able). */
void fill_value(cudaStream_t s, float* dst, int64_t n, float value) noexcept;

}  // namespace llmt::kernels

#endif  // LLMT_HAS_CUDA
