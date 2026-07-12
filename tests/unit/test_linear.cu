// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Masoud Jami
// Linear layer (Phase 4): forward and the first hand-derived backward pass
// vs PyTorch autograd; grad ACCUMULATION proven by running backward twice.
#include <vector>

#include "doctest.h"
#include "golden_io.h"
#include "llmt/core/arena.h"
#include "llmt/core/device.h"
#include "llmt/core/error.h"
#include "llmt/layers/linear.h"
#include "transfer.h"

using namespace llmt;
using namespace llmt::testing;

namespace {

void check_close(const Tensor& actual, const HostTensor& expected, double scale = 1.0) {
    const std::vector<float> h = download(actual);
    std::vector<float> e(expected.ptr<float>(), expected.ptr<float>() + expected.numel());
    for (float& v : e) v *= static_cast<float>(scale);
    const CloseReport r = allclose(h.data(), e.data(), expected.numel(), 1e-5, 1e-6);
    CHECK_MESSAGE(r.ok, to_string(r));
}

}  // namespace

TEST_CASE("Linear: forward and backward match the oracle; dW accumulates") {
    Device dev(0);
    const RunCtx ctx = dev.make_ctx(1, KernelBackend::Naive, PrecisionPolicy::fp32());
    Arena arena(1 << 20, "linear-test");
    const GoldenCase g("linear");

    ParamStore ps(StateDtypes::uniform(DType::FP32));
    const Linear lin(ps, "w", /*in_features=*/12, /*out_features=*/10, Role::Matrix);
    ps.finalize();

    // Load the oracle's weight into the registered parameter.
    const HostTensor w = g.tensor("weight");
    REQUIRE(lin.param().weight.shape == w.shape);
    CUDA_CHECK(cudaMemcpyAsync(lin.param().weight.data, w.bytes.data(), w.bytes.size(),
                               cudaMemcpyHostToDevice, ctx.stream));

    const Tensor x = upload(arena, g.tensor("x"), ctx.stream);
    const Tensor dy = upload(arena, g.tensor("dy"), ctx.stream);
    Tensor y = arena.alloc({8, 10}, DType::FP32);
    Tensor dx = arena.alloc({8, 12}, DType::FP32);

    lin.forward(ctx, y, x);
    dev.synchronize();
    check_close(y, g.tensor("y"));

    lin.backward(ctx, dx, dy, x);
    dev.synchronize();
    check_close(dx, g.tensor("dx"));
    check_close(lin.param().grad, g.tensor("dw"));

    // The accumulation contract (invariant 9): a second backward must ADD —
    // the grad is now exactly twice the oracle's.
    lin.backward(ctx, dx, dy, x);
    dev.synchronize();
    check_close(lin.param().grad, g.tensor("dw"), /*scale=*/2.0);
}
