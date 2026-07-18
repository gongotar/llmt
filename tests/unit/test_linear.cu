// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Masoud Jami
// Linear layer (Phase 4): forward and the first hand-derived backward pass
// vs PyTorch autograd; grad ACCUMULATION proven by running backward twice.
#include <vector>

#include "check.h"
#include "doctest.h"
#include "golden_io.h"
#include "llmt/core/arena.h"
#include "llmt/core/device.h"
#include "llmt/core/error.h"
#include "llmt/layers/linear.h"
#include "transfer.h"

using namespace llmt;
using namespace llmt::testing;

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
    check_close(y, g.tensor("y"), 1e-5, 1e-6);

    lin.backward(ctx, dx, dy, x);
    dev.synchronize();
    check_close(dx, g.tensor("dx"), 1e-5, 1e-6);
    check_close(lin.param().grad, g.tensor("dw"), 1e-5, 1e-6);

    // The accumulation contract (invariant 9): a second backward must ADD —
    // the grad is now exactly twice the oracle's.
    lin.backward(ctx, dx, dy, x);
    dev.synchronize();
    HostTensor dw2 = g.tensor("dw");
    float* v = reinterpret_cast<float*>(dw2.bytes.data());
    for (int64_t i = 0; i < dw2.numel(); ++i) v[i] *= 2.0f;
    check_close(lin.param().grad, dw2, 1e-5, 1e-6);
}
