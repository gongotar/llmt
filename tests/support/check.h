// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Masoud Jami
// Shared "device tensor vs golden" assertion for GPU tests.
#pragma once

#ifdef LLMT_HAS_CUDA

#include <vector>

#include "doctest.h"
#include "golden_io.h"
#include "transfer.h"

namespace llmt::testing {

// Downloads `actual` and asserts numpy-style closeness against the golden
// tensor; failures report the worst element (index, values, abs/rel diff).
inline void check_close(const Tensor& actual, const HostTensor& expected, double rtol,
                        double atol) {
    REQUIRE(actual.numel() == expected.numel());
    const std::vector<float> h = download(actual);
    const CloseReport r = allclose(h.data(), expected.ptr<float>(), expected.numel(), rtol, atol);
    CHECK_MESSAGE(r.ok, to_string(r));
}

}  // namespace llmt::testing

#endif  // LLMT_HAS_CUDA
