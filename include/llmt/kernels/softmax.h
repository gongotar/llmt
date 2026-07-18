// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Masoud Jami
#pragma once

#ifdef LLMT_HAS_CUDA

#include "llmt/core/run_ctx.h"
#include "llmt/core/tensor.h"
#include "llmt/kernels/mask.h"

namespace llmt::kernels {

/**
 * Row-wise softmax over the last dimension of [batch, Tq, Tk] tensors
 * (attention scores): each row of raw scores becomes a probability
 * distribution, y[i] = exp(x[i]) / Σ_j exp(x[j]) — non-negative, summing
 * to 1, larger scores → larger weights. Numerically stabilized by max
 * subtraction.
 *
 * T is the sequence length in tokens; row q = "token q asking" (query),
 * column k = "token k being looked at" (key). MaskKind::Causal keeps
 * yourself-and-earlier (k <= q) and gives the future probability exactly 0
 * (masked = ---):
 *
 *              k0   k1   k2   k3
 *     q0  [   s00  ---  ---  ---  ]
 *     q1  [   s10  s11  ---  ---  ]
 *     q2  [   s20  s21  s22  ---  ]
 *     q3  [   s30  s31  s32  s33  ]
 *
 * The cut at "column q = my own position" only lines up when rows and
 * columns index the same sequence (self-attention, square scores) — hence
 * Causal requires Tq == Tk.
 *
 * y may alias x (in place). The max/sum reductions accumulate in
 * ctx.precision.reduce; elementwise arithmetic is kernel_compute_t (the
 * non-GEMM fp32 rule). Callers pass scores already scaled by 1/sqrt(hd) —
 * apply it via the preceding matmul's alpha, not here.
 */
void softmax_fwd(const RunCtx& ctx, Tensor& y, const Tensor& x, MaskSpec mask) noexcept;

}  // namespace llmt::kernels

#endif  // LLMT_HAS_CUDA
