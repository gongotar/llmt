// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Masoud Jami
#pragma once

#ifdef LLMT_HAS_CUDA

#include "llmt/core/run_ctx.h"
#include "llmt/core/tensor.h"

namespace llmt::kernels {

/**
 * C[M,N] = alpha · op(A) · op(B) + beta · C, for row-major tensors.
 * op(X) is X or Xᵀ per the trans flag: op(A) is [M,K], op(B) is [K,N].
 * All three tensors must share one rank: rank 2 is a single GEMM; rank 3
 * ([batch, rows, cols], contiguous) runs one independent GEMM per batch
 * index with the same op/alpha/beta semantics. Shapes are validated against
 * the flags. alpha and beta are arbitrary scales; the two cases training
 * uses are beta = 0 (overwrite C) and beta = 1 (accumulate into C).
 * Deterministic per run: the algorithm for each problem is resolved once
 * and cached.
 */
void matmul(const RunCtx& ctx, Tensor& c, const Tensor& a, const Tensor& b, bool trans_a,
            bool trans_b, float alpha, float beta) noexcept;

}  // namespace llmt::kernels

#endif  // LLMT_HAS_CUDA
