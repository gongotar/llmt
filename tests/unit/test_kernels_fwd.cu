// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Masoud Jami
// Phase 5 forward kernels vs the PyTorch oracle: rmsnorm (+ saved rstd),
// tanh-GELU, residual add, masked softmax — including each kernel's
// documented in-place (y aliases x) contract, and reduction determinism.
#include <cstring>
#include <vector>

#include "check.h"
#include "doctest.h"
#include "llmt/core/arena.h"
#include "llmt/core/device.h"
#include "llmt/core/error.h"
#include "llmt/kernels/gelu.h"
#include "llmt/kernels/residual.h"
#include "llmt/kernels/rmsnorm.h"
#include "llmt/kernels/softmax.h"

using namespace llmt;
using namespace llmt::testing;

namespace {

// Device-to-device copy into a fresh arena tensor (for in-place variants).
Tensor clone(Arena& arena, const Tensor& t, cudaStream_t s) {
    Tensor c = arena.alloc(t.shape, t.dtype);
    CUDA_CHECK(cudaMemcpyAsync(c.data, t.data, t.bytes(), cudaMemcpyDeviceToDevice, s));
    return c;
}

}  // namespace

TEST_CASE("rmsnorm_fwd: y and rstd match oracle; in-place works") {
    Device dev(0);
    const RunCtx ctx = dev.make_ctx(1, KernelBackend::Naive, PrecisionPolicy::fp32());
    Arena arena(1 << 20, "kern-test");
    const GoldenCase g("rmsnorm");
    constexpr float kEps = 1e-5f;  // matches the golden case

    const Tensor x = upload(arena, g.tensor("x"), dev.stream());
    const Tensor gw = upload(arena, g.tensor("g"), dev.stream());
    Tensor y = arena.alloc(x.shape, DType::FP32);
    Tensor rstd = arena.alloc({x.shape[0]}, DType::FP32);

    kernels::rmsnorm_fwd(ctx, y, rstd, x, gw, kEps);
    dev.synchronize();
    check_close(y, g.tensor("y"), 1e-5, 1e-6);
    check_close(rstd, g.tensor("rstd"), 1e-5, 1e-6);

    // In-place: x is both input and output.
    Tensor xc = clone(arena, x, dev.stream());
    kernels::rmsnorm_fwd(ctx, xc, rstd, xc, gw, kEps);
    dev.synchronize();
    check_close(xc, g.tensor("y"), 1e-5, 1e-6);
}

TEST_CASE("rmsnorm_fwd: bitwise deterministic (fixed reduction order)") {
    Device dev(0);
    const RunCtx ctx = dev.make_ctx(1, KernelBackend::Naive, PrecisionPolicy::fp32());
    Arena arena(1 << 20, "kern-test");
    const GoldenCase g("rmsnorm");
    const Tensor x = upload(arena, g.tensor("x"), dev.stream());
    const Tensor gw = upload(arena, g.tensor("g"), dev.stream());
    Tensor y1 = arena.alloc(x.shape, DType::FP32), y2 = arena.alloc(x.shape, DType::FP32);
    Tensor r1 = arena.alloc({x.shape[0]}, DType::FP32), r2 = arena.alloc({x.shape[0]}, DType::FP32);

    kernels::rmsnorm_fwd(ctx, y1, r1, x, gw, 1e-5f);
    kernels::rmsnorm_fwd(ctx, y2, r2, x, gw, 1e-5f);
    dev.synchronize();
    const std::vector<float> h1 = download(y1), h2 = download(y2);
    CHECK(std::memcmp(h1.data(), h2.data(), h1.size() * sizeof(float)) == 0);
}

TEST_CASE("gelu_fwd: matches oracle (tanh approximation); in-place works") {
    Device dev(0);
    Arena arena(1 << 20, "kern-test");
    const GoldenCase g("gelu");

    const Tensor x = upload(arena, g.tensor("x"), dev.stream());
    Tensor y = arena.alloc(x.shape, DType::FP32);
    kernels::gelu_fwd(dev.stream(), y, x);
    dev.synchronize();
    check_close(y, g.tensor("y"), 1e-5, 1e-6);

    Tensor xc = clone(arena, x, dev.stream());
    kernels::gelu_fwd(dev.stream(), xc, xc);
    dev.synchronize();
    check_close(xc, g.tensor("y"), 1e-5, 1e-6);
}

TEST_CASE("residual_fwd: out = a + b; in-place accumulation works") {
    Device dev(0);
    Arena arena(1 << 20, "kern-test");
    const GoldenCase g("residual");

    const Tensor a = upload(arena, g.tensor("a"), dev.stream());
    const Tensor b = upload(arena, g.tensor("b"), dev.stream());
    Tensor out = arena.alloc(a.shape, DType::FP32);
    kernels::residual_fwd(dev.stream(), out, a, b);
    dev.synchronize();
    check_close(out, g.tensor("out"), 1e-6, 1e-7);

    // In-place: the residual-stream pattern, a += b.
    Tensor ac = clone(arena, a, dev.stream());
    kernels::residual_fwd(dev.stream(), ac, ac, b);
    dev.synchronize();
    check_close(ac, g.tensor("out"), 1e-6, 1e-7);
}

TEST_CASE("softmax_fwd: unmasked and causal match oracle; in-place works") {
    Device dev(0);
    const RunCtx ctx = dev.make_ctx(1, KernelBackend::Naive, PrecisionPolicy::fp32());
    Arena arena(1 << 20, "kern-test");
    const GoldenCase g("softmax");

    const Tensor x = upload(arena, g.tensor("x"), dev.stream());
    Tensor y = arena.alloc(x.shape, DType::FP32);

    kernels::softmax_fwd(ctx, y, x, kernels::MaskSpec{kernels::MaskKind::None});
    dev.synchronize();
    check_close(y, g.tensor("y_none"), 1e-5, 1e-6);

    kernels::softmax_fwd(ctx, y, x, kernels::MaskSpec{kernels::MaskKind::Causal});
    dev.synchronize();
    check_close(y, g.tensor("y_causal"), 1e-5, 1e-6);

    // Masked entries are EXACTLY zero, not merely tiny.
    const std::vector<float> h = download(y);
    const int64_t t = x.shape[1];
    for (int64_t bq = 0; bq < x.shape[0] * t; ++bq)
        for (int64_t k = (bq % t) + 1; k < t; ++k) REQUIRE(h[bq * t + k] == 0.0f);

    // In-place: scores become probabilities in their own buffer.
    Tensor xc = clone(arena, x, dev.stream());
    kernels::softmax_fwd(ctx, xc, xc, kernels::MaskSpec{kernels::MaskKind::Causal});
    dev.synchronize();
    check_close(xc, g.tensor("y_causal"), 1e-5, 1e-6);
}
