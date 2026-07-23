// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Masoud Jami
#pragma once

#ifdef LLMT_HAS_CUDA

#include "llmt/core/run_ctx.h"
#include "llmt/core/tensor.h"
#include "llmt/kernels/mask.h"

namespace llmt::kernels {

/**
 * Naive attention forward over per-head views (DESIGN §5: the mask is a
 * parameter, never baked in):
 *   scores  = q · kᵀ / sqrt(hd)      (batched-strided GEMM, scale via alpha)
 *   probs   = softmax(scores, mask)  — probs OVERWRITE scores in place
 *   y_heads = probs · v              (batched-strided GEMM)
 *   y       = merge(y_heads)         — heads back into token rows
 * q, k, v and y_heads are [B, H, T, hd]; scores is [B, H, T, T]; y is
 * [B, T, H·hd]. scores and y_heads are caller-provided scratch (planner-
 * owned): after the call scores holds the probabilities (the naive
 * backward reads them) and y_heads the pre-merge per-head output. RoPE,
 * if any, is applied to q/k before this call.
 *
 * Backend contract (recorded micro-decision): this signature is shared
 * with the future fused variant (KernelBackend seam) — a fused
 * kernel writes token-row output directly and materializes no
 * probabilities, so the scratch tensors are naive-only: they must be valid
 * under Naive and are expected INVALID (unallocated) under Fused; the
 * planner allocates them per backend.
 */
void attention_fwd(const RunCtx& ctx, Tensor& y, Tensor& scores, Tensor& y_heads,
                   const Tensor& q, const Tensor& k, const Tensor& v, MaskSpec mask) noexcept;

}  // namespace llmt::kernels

#endif  // LLMT_HAS_CUDA
