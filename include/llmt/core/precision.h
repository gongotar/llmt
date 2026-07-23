// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Masoud Jami
#pragma once

#include "llmt/core/dtype.h"

namespace llmt {

/**
 * Cross-cutting precision policy (DESIGN invariant 3). Layers query this —
 * no dtype is hard-coded in a signature. The policy carries the axes
 * hardware can actually vary per run; the fixed side of the precision story
 * is kernel_compute_t below. M1: fp32 everywhere; M2 adds bf16 gemm_compute
 * with fp32 master weights plus per-layer/per-role overrides.
 */
struct PrecisionPolicy {
    DType master = DType::FP32;        // parameter storage ("master weights")
    DType gemm_compute = DType::FP32;  // tensor-core multiply mode for GEMMs
    DType reduce = DType::FP32;  // accumulations (GEMM accumulator, norms, softmax sums, loss)

    static PrecisionPolicy fp32() noexcept { return {}; }
};

/**
 * The FIXED precision choice, deliberately not a PrecisionPolicy field:
 * non-GEMM kernel arithmetic (elementwise math in gelu/residual/rmsnorm/
 * softmax bodies) runs in this type. Those kernels are memory-bound, so
 * narrower arithmetic saves nothing and only adds rounding, and fp32 keeps
 * transcendentals (tanhf/expf/rsqrtf) exact to hardware. Changing it is a
 * compile-time design decision made here, once — kernels use the alias,
 * never bare float.
 */
using kernel_compute_t = float;

/**
 * Storage type of results whose designed consumer is HOST code, not another
 * kernel (the loss scalar, per-token NLL diagnostics) — also fixed,
 * deliberately not policy-driven. Tensors consumed on-device (rstd,
 * activations) may follow the policy: producer and consumer change
 * together. These don't: the host reader sits outside the policy, and the
 * tensors are tiny, so narrowing buys no bandwidth and only quantizes the
 * numbers training is steered by (logging, LR decisions, golden compares).
 */
using host_result_t = float;

}  // namespace llmt
