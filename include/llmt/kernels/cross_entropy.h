// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Masoud Jami
#pragma once

#ifdef LLMT_HAS_CUDA

#include "llmt/core/run_ctx.h"
#include "llmt/core/tensor.h"

namespace llmt::kernels {

/**
 * Cross-entropy loss, computed directly from the raw logits tensor
 * [B, T, V] (V vocabulary scores per token position). "Fused" = this
 * kernel never builds the [B, T, V] softmax-probability tensor the
 * textbook recipe would allocate; the identity
 * −log softmax(x)[tgt] = lse(x) − x[tgt] lets it get each position's loss
 * from one row reduction instead.
 *
 * What it computes, per row logits[b][t] (V scores), all in one pass:
 *   1. lse = log Σᵥ exp(logits[b][t][v]), via row max subtraction so the
 *      exps cannot overflow;
 *   2. nll[b][t] = lse − logits[b][t][targets[b][t]]   (that token's loss);
 *   3. loss = Σ nll·mask / Σ mask — one scalar, the mask-weighted mean.
 *
 * targets is [B, T] (I32); mask is [B, T] in logits' dtype (0 excludes a
 * token, weights in between are honored); mask must not be all zero.
 * row_nll is caller-provided [B, T] scratch that afterwards holds the
 * masked per-token NLL (diagnostics; the backward pass recomputes what it
 * needs). loss [1] and row_nll are published to the host: their dtype is
 * the fixed host_result_t, NOT the policy's reduce dtype (precision.h —
 * reduce governs how the sums are accumulated, which it does here, but
 * must not narrow what the host reads). The final mean is reduced
 * deterministically (fixed order — invariant 5, no atomics).
 */
void cross_entropy_fwd(const RunCtx& ctx, Tensor& loss, Tensor& row_nll, const Tensor& logits,
                       const Tensor& targets, const Tensor& mask) noexcept;

}  // namespace llmt::kernels

#endif  // LLMT_HAS_CUDA
