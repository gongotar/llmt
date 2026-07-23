// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Masoud Jami
// The Phase 6 attention path vs the PyTorch oracle, per stage so a failure
// localizes itself: head split, RoPE, scaled scores, causal probs, PV
// output — plus the composed attention_fwd op incl. its head merge (out
// cross-checked against F.scaled_dot_product_attention at generation).
#include <cmath>

#include "check.h"
#include "doctest.h"
#include "llmt/core/arena.h"
#include "llmt/core/device.h"
#include "llmt/core/error.h"
#include "llmt/kernels/attention.h"
#include "llmt/kernels/matmul.h"
#include "llmt/kernels/permute.h"
#include "llmt/kernels/rope.h"
#include "llmt/kernels/softmax.h"

using namespace llmt;
using namespace llmt::testing;

TEST_CASE("permute: head split and merge match oracle") {
    Device dev(0);
    Arena arena(1 << 20, "perm-test");
    const GoldenCase g("permute");

    const Tensor qkv = upload(arena, g.tensor("qkv"), dev.stream());
    Tensor q = arena.alloc({2, 2, 3, 4}, DType::FP32);
    Tensor k = arena.alloc({2, 2, 3, 4}, DType::FP32);
    Tensor v = arena.alloc({2, 2, 3, 4}, DType::FP32);
    kernels::permute_split(dev.stream(), q, k, v, qkv, /*n_head=*/2);
    dev.synchronize();
    check_close(q, g.tensor("q"), 1e-6, 1e-7);
    check_close(k, g.tensor("k"), 1e-6, 1e-7);
    check_close(v, g.tensor("v"), 1e-6, 1e-7);

    const Tensor x = upload(arena, g.tensor("x"), dev.stream());
    Tensor y = arena.alloc({2, 3, 8}, DType::FP32);
    kernels::permute_merge(dev.stream(), y, x);
    dev.synchronize();
    check_close(y, g.tensor("y"), 1e-6, 1e-7);
}

TEST_CASE("rope_fwd: rotate-half matches oracle (in place, q and k)") {
    Device dev(0);
    Arena arena(1 << 20, "rope-test");
    const GoldenCase g("rope");

    Tensor q = upload(arena, g.tensor("q"), dev.stream());
    Tensor k = upload(arena, g.tensor("k"), dev.stream());
    kernels::rope_fwd(dev.stream(), q, k, /*theta=*/10000.0f);
    dev.synchronize();
    check_close(q, g.tensor("q_rot"), 1e-5, 1e-6);
    check_close(k, g.tensor("k_rot"), 1e-5, 1e-6);
}

TEST_CASE("attention: every stage matches oracle; composed op agrees") {
    Device dev(0);
    const RunCtx ctx = dev.make_ctx(1, KernelBackend::Naive, PrecisionPolicy::fp32());
    Arena arena(1 << 20, "attn-test");
    const GoldenCase g("attention");
    constexpr int64_t kB = 2, kH = 2, kT = 4, kHd = 6;

    const Tensor q = upload(arena, g.tensor("q"), ctx.stream);
    const Tensor k = upload(arena, g.tensor("k"), ctx.stream);
    const Tensor v = upload(arena, g.tensor("v"), ctx.stream);

    // Stage by stage, on rank-3 head-folded views (as attention_fwd does).
    const Tensor q3{q.data, q.dtype, {kB * kH, kT, kHd}};
    const Tensor k3{k.data, k.dtype, {kB * kH, kT, kHd}};
    const Tensor v3{v.data, v.dtype, {kB * kH, kT, kHd}};
    Tensor scores = arena.alloc({kB * kH, kT, kT}, DType::FP32);
    const float scale = 1.0f / std::sqrt(static_cast<float>(kHd));
    kernels::matmul(ctx, scores, q3, k3, false, true, scale, 0.0f);
    dev.synchronize();
    check_close(scores, g.tensor("scores"), 1e-5, 1e-6);

    kernels::softmax_fwd(ctx, scores, scores, kernels::MaskSpec{kernels::MaskKind::Causal});
    dev.synchronize();
    check_close(scores, g.tensor("probs"), 1e-5, 1e-6);

    Tensor out3 = arena.alloc({kB * kH, kT, kHd}, DType::FP32);
    kernels::matmul(ctx, out3, scores, v3, false, false, 1.0f, 0.0f);
    dev.synchronize();
    check_close(out3, g.tensor("out"), 1e-5, 1e-6);

    // The composed op: merged token-row output, plus the documented scratch
    // contents — probs left in scores, pre-merge per-head output in y_heads.
    Tensor y = arena.alloc({kB, kT, kH * kHd}, DType::FP32);
    Tensor scores4 = arena.alloc({kB, kH, kT, kT}, DType::FP32);
    Tensor y_heads = arena.alloc({kB, kH, kT, kHd}, DType::FP32);
    kernels::attention_fwd(ctx, y, scores4, y_heads, q, k, v,
                           kernels::MaskSpec{kernels::MaskKind::Causal});
    dev.synchronize();
    check_close(y, g.tensor("merged"), 1e-5, 1e-6);
    check_close(scores4, g.tensor("probs"), 1e-5, 1e-6);
    check_close(y_heads, g.tensor("out"), 1e-5, 1e-6);
}
