// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Masoud Jami
#pragma once

#ifdef LLMT_HAS_CUDA

#include <cuda_runtime.h>

#include "llmt/core/tensor.h"

namespace llmt::kernels {

/**
 * Rotary position embedding applied IN PLACE to the q and k head views
 * ([B, H, T, hd] each, hd even): position t rotates the pair
 * (d, d + hd/2) by angle t · theta^(-2d/hd) for d < hd/2.
 * The pairing is the rotate-half convention (recorded micro-decision):
 * mathematically interchangeable with the interleaved (2d, 2d+1)
 * convention via a weight permutation, but checkpoints are welded to
 * their convention — this one matches the
 * HF-transformers ecosystem and gives the kernel contiguous instead of
 * strided pair access. The golden oracle implements the same formula.
 * Elementwise arithmetic runs in kernel_compute_t (the non-GEMM fp32
 * rule); no policy axis applies, so no ctx is taken.
 */
void rope_fwd(cudaStream_t s, Tensor& q, Tensor& k, float theta) noexcept;

}  // namespace llmt::kernels

#endif  // LLMT_HAS_CUDA
