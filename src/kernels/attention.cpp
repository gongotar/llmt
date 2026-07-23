// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Masoud Jami
//
// Naive attention forward (task 6.5): a host-side composition of the
// batched-strided matmul, the masked softmax and the head merge — no
// device code of its own, which is exactly what KernelBackend::Naive
// means. The fused/flash variant (M3) replaces this composition behind
// the same signature.
#ifdef LLMT_HAS_CUDA

#include <cmath>

#include <initializer_list>

#include "llmt/core/error.h"
#include "llmt/kernels/attention.h"
#include "llmt/kernels/matmul.h"
#include "llmt/kernels/permute.h"
#include "llmt/kernels/softmax.h"

namespace llmt::kernels {

namespace {

// Rank-3 view of a [B, H, T, x] head tensor as [B·H, T, x] — same bytes;
// B and H are adjacent and contiguous, so the flatten is free. This is the
// shape the batched matmul and the softmax operate on.
Tensor fold_heads(const Tensor& t) noexcept {
    return Tensor{t.data, t.dtype, {t.shape[0] * t.shape[1], t.shape[2], t.shape[3]}};
}

}  // namespace

void attention_fwd(const RunCtx& ctx, Tensor& y, Tensor& scores, Tensor& y_heads,
                   const Tensor& q, const Tensor& k, const Tensor& v, MaskSpec mask) noexcept {
    // Capability guard on the backend axis: this file is the Naive
    // composition; under Fused the scratch contract below doesn't even
    // apply, so refuse rather than silently run the naive path.
    if (ctx.backend != KernelBackend::Naive)
        detail::fatal("attention_fwd", "no fused variant implemented yet");
    for (const Tensor* t :
         std::initializer_list<const Tensor*>{&y, &scores, &y_heads, &q, &k, &v})
        if (!t->valid()) detail::fatal("attention_fwd", "invalid tensor (null data)");
    for (const Tensor* t : std::initializer_list<const Tensor*>{&scores, &y_heads, &q, &k, &v})
        if (t->shape.rank != 4) detail::fatal("attention_fwd", "expected rank-4 head tensors");
    if (k.shape != q.shape || v.shape != q.shape || y_heads.shape != q.shape)
        detail::fatal("attention_fwd", "q/k/v/y_heads must share one [B, H, T, hd] shape");
    const int64_t b = q.shape[0], h = q.shape[1], seq = q.shape[2], hd = q.shape[3];
    if (scores.shape != Shape{b, h, seq, seq})
        detail::fatal("attention_fwd", "scores must be [B, H, T, T]");
    if (y.shape != Shape{b, seq, h * hd})
        detail::fatal("attention_fwd", "y must be [B, T, H·hd]");

    Tensor scores3 = fold_heads(scores), y_heads3 = fold_heads(y_heads);
    const Tensor q3 = fold_heads(q), k3 = fold_heads(k), v3 = fold_heads(v);

    // scores = q · kᵀ / sqrt(hd) — the softmax contract expects the scale
    // to ride this GEMM's alpha.
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    matmul(ctx, scores3, q3, k3, /*trans_a=*/false, /*trans_b=*/true, scale, /*beta=*/0.0f);
    softmax_fwd(ctx, scores3, scores3, mask);  // probs overwrite scores in place
    matmul(ctx, y_heads3, scores3, v3, /*trans_a=*/false, /*trans_b=*/false, 1.0f, /*beta=*/0.0f);
    permute_merge(ctx.stream, y, y_heads);
}

}  // namespace llmt::kernels

#endif  // LLMT_HAS_CUDA
