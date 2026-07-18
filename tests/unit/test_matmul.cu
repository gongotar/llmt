// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Masoud Jami
// GEMM wrapper (Phase 4): all transpose combos, beta accumulation, rank-3
// batching, and per-run determinism — all vs the PyTorch oracle.
#include <cstring>
#include <vector>

#include "check.h"
#include "doctest.h"
#include "golden_io.h"
#include "llmt/core/arena.h"
#include "llmt/core/device.h"
#include "llmt/core/error.h"
#include "llmt/kernels/matmul.h"
#include "transfer.h"

using namespace llmt;
using namespace llmt::testing;

TEST_CASE("matmul: four transpose combos vs oracle") {
    Device dev(0);
    const RunCtx ctx = dev.make_ctx(1, KernelBackend::Naive, PrecisionPolicy::fp32());
    Arena arena(1 << 20, "matmul-test");
    const GoldenCase g("matmul");

    const Tensor a = upload(arena, g.tensor("a"), ctx.stream);
    const Tensor at = upload(arena, g.tensor("at"), ctx.stream);
    const Tensor b = upload(arena, g.tensor("b"), ctx.stream);
    const Tensor bt = upload(arena, g.tensor("bt"), ctx.stream);
    Tensor c = arena.alloc({7, 9}, DType::FP32);

    kernels::matmul(ctx, c, a, b, false, false, 1.0f, 0.0f);
    dev.synchronize();
    check_close(c, g.tensor("c_nn"), 1e-5, 1e-6);

    kernels::matmul(ctx, c, a, bt, false, true, 1.0f, 0.0f);
    dev.synchronize();
    check_close(c, g.tensor("c_nt"), 1e-5, 1e-6);

    kernels::matmul(ctx, c, at, b, true, false, 1.0f, 0.0f);
    dev.synchronize();
    check_close(c, g.tensor("c_tn"), 1e-5, 1e-6);

    kernels::matmul(ctx, c, at, bt, true, true, 1.0f, 0.0f);
    dev.synchronize();
    check_close(c, g.tensor("c_tt"), 1e-5, 1e-6);
}

TEST_CASE("matmul: beta=1 accumulates into C") {
    Device dev(0);
    const RunCtx ctx = dev.make_ctx(1, KernelBackend::Naive, PrecisionPolicy::fp32());
    Arena arena(1 << 20, "matmul-test");
    const GoldenCase g("matmul");

    const Tensor a = upload(arena, g.tensor("a"), ctx.stream);
    const Tensor b = upload(arena, g.tensor("b"), ctx.stream);
    Tensor c = upload(arena, g.tensor("c0"), ctx.stream);  // pre-existing contents

    kernels::matmul(ctx, c, a, b, false, false, 1.0f, /*beta=*/1.0f);
    dev.synchronize();
    check_close(c, g.tensor("c_beta"), 1e-5, 1e-6);
}

TEST_CASE("matmul rank-3: plain batch and the QK^T pattern") {
    Device dev(0);
    const RunCtx ctx = dev.make_ctx(1, KernelBackend::Naive, PrecisionPolicy::fp32());
    Arena arena(1 << 20, "matmul-test");
    const GoldenCase g("matmul_rank3");

    const Tensor a = upload(arena, g.tensor("a"), ctx.stream);
    const Tensor b = upload(arena, g.tensor("b"), ctx.stream);
    Tensor c = arena.alloc({3, 4, 5}, DType::FP32);
    kernels::matmul(ctx, c, a, b, false, false, 1.0f, 0.0f);
    dev.synchronize();
    check_close(c, g.tensor("c"), 1e-5, 1e-6);

    // scores = q @ k^T per batch — attention's Phase 6 shape.
    const Tensor q = upload(arena, g.tensor("q"), ctx.stream);
    const Tensor k = upload(arena, g.tensor("k"), ctx.stream);
    Tensor scores = arena.alloc({3, 4, 4}, DType::FP32);
    kernels::matmul(ctx, scores, q, k, false, true, 1.0f, 0.0f);
    dev.synchronize();
    check_close(scores, g.tensor("scores"), 1e-5, 1e-6);
}

TEST_CASE("matmul: bitwise deterministic across calls (cached algo)") {
    Device dev(0);
    const RunCtx ctx = dev.make_ctx(1, KernelBackend::Naive, PrecisionPolicy::fp32());
    Arena arena(1 << 20, "matmul-test");
    const GoldenCase g("matmul");

    const Tensor a = upload(arena, g.tensor("a"), ctx.stream);
    const Tensor b = upload(arena, g.tensor("b"), ctx.stream);
    Tensor c1 = arena.alloc({7, 9}, DType::FP32);
    Tensor c2 = arena.alloc({7, 9}, DType::FP32);

    kernels::matmul(ctx, c1, a, b, false, false, 1.0f, 0.0f);
    kernels::matmul(ctx, c2, a, b, false, false, 1.0f, 0.0f);
    dev.synchronize();

    const std::vector<float> h1 = download(c1), h2 = download(c2);
    CHECK(std::memcmp(h1.data(), h2.data(), h1.size() * sizeof(float)) == 0);
}
