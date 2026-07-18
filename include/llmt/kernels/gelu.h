// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Masoud Jami
#pragma once

#ifdef LLMT_HAS_CUDA

#include <cuda_runtime.h>

#include "llmt/core/tensor.h"

namespace llmt::kernels {

/**
 * Elementwise GELU, tanh approximation (matches torch gelu(approximate=
 * 'tanh') — recorded micro-decision, DESIGN invariant 10). Shapes of x and
 * y must match; y may alias x. Arithmetic runs in kernel_compute_t (the
 * non-GEMM fp32 rule); no policy axis applies, so no ctx is taken.
 */
void gelu_fwd(cudaStream_t s, Tensor& y, const Tensor& x) noexcept;

}  // namespace llmt::kernels

#endif  // LLMT_HAS_CUDA
