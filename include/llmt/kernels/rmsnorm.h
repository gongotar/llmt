// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Masoud Jami
#pragma once

#ifdef LLMT_HAS_CUDA

#include "llmt/core/run_ctx.h"
#include "llmt/core/tensor.h"

namespace llmt::kernels {

/**
 * Row-wise RMS normalization — each row r of the [N, C] input is scaled by
 * its own scalar statistic: y[r][c] = x[r][c] · rstd[r] · g[c], where
 * rstd[r] = 1/sqrt(mean over c of x[r][c]² + eps). y may alias x; g is the
 * learned per-channel gain [C], shared by all rows; rstd [N] holds one
 * statistic per row, saved for the backward pass — its dtype must equal
 * ctx.precision.reduce, the dtype the reduction accumulates in.
 * Elementwise arithmetic is kernel_compute_t (the non-GEMM fp32 rule).
 */
void rmsnorm_fwd(const RunCtx& ctx, Tensor& y, Tensor& rstd, const Tensor& x, const Tensor& g,
                 float eps) noexcept;

}  // namespace llmt::kernels

#endif  // LLMT_HAS_CUDA
