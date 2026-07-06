// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Masoud Jami
#pragma once

#include "llmt/core/dtype.h"

namespace llmt {

/**
 * Cross-cutting precision policy (DESIGN invariant 3). Layers query this —
 * no dtype is hard-coded in a signature. M1: fp32 everywhere; M2 adds bf16
 * compute with fp32 master weights plus per-layer/per-role overrides.
 */
struct PrecisionPolicy {
    DType master = DType::FP32;   // parameter storage ("master weights")
    DType compute = DType::FP32;  // activations / GEMM inputs
    DType reduce = DType::FP32;   // accumulations (norms, softmax sums, loss)

    static PrecisionPolicy fp32() noexcept { return {}; }
};

}  // namespace llmt
