// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Masoud Jami
// MLP layer forward (Linear → GELU → Linear) vs the PyTorch oracle.
#include <utility>

#include "check.h"
#include "doctest.h"
#include "llmt/core/arena.h"
#include "llmt/core/device.h"
#include "llmt/core/error.h"
#include "llmt/layers/mlp.h"

using namespace llmt;
using namespace llmt::testing;

TEST_CASE("MLP: forward matches the oracle") {
    Device dev(0);
    const RunCtx ctx = dev.make_ctx(1, KernelBackend::Naive, PrecisionPolicy::fp32());
    Arena arena(1 << 20, "mlp-test");
    const GoldenCase g("mlp");

    ParamStore ps(StateDtypes::uniform(DType::FP32));
    const MLP mlp(ps, "mlp", /*d_model=*/8, /*d_ff=*/16);
    ps.finalize();

    // Load the oracle's weights into the registered parameters.
    const std::pair<const Param*, const char*> weights[] = {{&mlp.up(), "wu"},
                                                            {&mlp.down(), "wd"}};
    for (const auto& [param, name] : weights) {
        const HostTensor w = g.tensor(name);
        REQUIRE(param->weight.shape == w.shape);
        CUDA_CHECK(cudaMemcpyAsync(param->weight.data, w.bytes.data(), w.bytes.size(),
                                   cudaMemcpyHostToDevice, ctx.stream));
    }

    const Tensor x = upload(arena, g.tensor("x"), ctx.stream);
    Tensor h = arena.alloc({6, 16}, DType::FP32);
    Tensor y = arena.alloc({6, 8}, DType::FP32);
    mlp.forward(ctx, y, h, x);
    dev.synchronize();
    check_close(y, g.tensor("y"), 1e-5, 1e-6);
}
