// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Masoud Jami
// Fused cross-entropy forward vs the PyTorch oracle: masked per-token NLL
// (mask contains zeros) and the weighted-mean scalar; bitwise-deterministic
// reduction (invariant 5).
#include <cstring>
#include <vector>

#include "check.h"
#include "doctest.h"
#include "llmt/core/arena.h"
#include "llmt/core/device.h"
#include "llmt/core/error.h"
#include "llmt/kernels/cross_entropy.h"

using namespace llmt;
using namespace llmt::testing;

TEST_CASE("cross_entropy_fwd: masked NLL and weighted mean match oracle") {
    Device dev(0);
    const RunCtx ctx = dev.make_ctx(1, KernelBackend::Naive, PrecisionPolicy::fp32());
    Arena arena(1 << 20, "ce-test");
    const GoldenCase g("cross_entropy");

    const Tensor logits = upload(arena, g.tensor("logits"), ctx.stream);
    const Tensor targets = upload(arena, g.tensor("targets"), ctx.stream);
    const Tensor mask = upload(arena, g.tensor("mask"), ctx.stream);
    Tensor row_nll = arena.alloc({2, 4}, DType::FP32);
    Tensor loss = arena.alloc({1}, DType::FP32);

    kernels::cross_entropy_fwd(ctx, loss, row_nll, logits, targets, mask);
    dev.synchronize();
    check_close(row_nll, g.tensor("wnll"), 1e-5, 1e-6);
    check_close(loss, g.tensor("loss"), 1e-5, 1e-6);
}

TEST_CASE("cross_entropy_fwd: bitwise deterministic (fixed reduction order)") {
    Device dev(0);
    const RunCtx ctx = dev.make_ctx(1, KernelBackend::Naive, PrecisionPolicy::fp32());
    Arena arena(1 << 20, "ce-test");
    const GoldenCase g("cross_entropy");

    const Tensor logits = upload(arena, g.tensor("logits"), ctx.stream);
    const Tensor targets = upload(arena, g.tensor("targets"), ctx.stream);
    const Tensor mask = upload(arena, g.tensor("mask"), ctx.stream);
    Tensor row_nll = arena.alloc({2, 4}, DType::FP32);
    Tensor l1 = arena.alloc({1}, DType::FP32), l2 = arena.alloc({1}, DType::FP32);

    kernels::cross_entropy_fwd(ctx, l1, row_nll, logits, targets, mask);
    kernels::cross_entropy_fwd(ctx, l2, row_nll, logits, targets, mask);
    dev.synchronize();
    const std::vector<float> h1 = download(l1), h2 = download(l2);
    CHECK(std::memcmp(h1.data(), h2.data(), sizeof(float)) == 0);
}
