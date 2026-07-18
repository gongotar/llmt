// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Masoud Jami
#pragma once

#ifdef LLMT_HAS_CUDA

#include <cuda_runtime.h>

#include "llmt/core/tensor.h"

namespace llmt::kernels {

/**
 * Elementwise add: out = a + b (the residual-stream write). Shapes must
 * match; out may alias a or b. Arithmetic runs in kernel_compute_t (the
 * non-GEMM fp32 rule); no policy axis applies, so no ctx is taken.
 */
void residual_fwd(cudaStream_t s, Tensor& out, const Tensor& a, const Tensor& b) noexcept;

}  // namespace llmt::kernels

#endif  // LLMT_HAS_CUDA
