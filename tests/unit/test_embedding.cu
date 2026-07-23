// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Masoud Jami
// embedding_fwd vs the PyTorch oracle: gather incl. id 0, id V-1 and
// repeated tokens (repeated rows must be bitwise identical).
#include <cstring>
#include <vector>

#include "check.h"
#include "doctest.h"
#include "llmt/core/arena.h"
#include "llmt/core/device.h"
#include "llmt/core/error.h"
#include "llmt/kernels/embedding.h"

using namespace llmt;
using namespace llmt::testing;

TEST_CASE("embedding_fwd: gather matches oracle; repeated tokens identical") {
    Device dev(0);
    Arena arena(1 << 20, "embed-test");
    const GoldenCase g("embedding");

    const Tensor wte = upload(arena, g.tensor("wte"), dev.stream());
    const Tensor tokens = upload(arena, g.tensor("tokens"), dev.stream());
    Tensor y = arena.alloc({2, 5, 8}, DType::FP32);

    kernels::embedding_fwd(dev.stream(), y, tokens, wte);
    dev.synchronize();
    check_close(y, g.tensor("y"), 1e-6, 1e-7);

    // tokens[0][1] and tokens[0][3] are both id 5: a gather has no
    // arithmetic, so the two rows must match bit for bit.
    const std::vector<float> h = download(y);
    CHECK(std::memcmp(&h[1 * 8], &h[3 * 8], 8 * sizeof(float)) == 0);
}
