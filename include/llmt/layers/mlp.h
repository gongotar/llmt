// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Masoud Jami
#pragma once

#ifdef LLMT_HAS_CUDA

#include <cstdint>
#include <string>

#include "llmt/core/run_ctx.h"
#include "llmt/core/tensor.h"
#include "llmt/layers/linear.h"
#include "llmt/model/param_store.h"

namespace llmt {

/**
 * Bias-free GELU MLP: y = gelu_tanh(x · Wupᵀ) · Wdownᵀ, applied per token
 * row (x is [M, d_model], M = the flattened token count, e.g. B·T).
 * Registers "<name>.up" [d_ff, d_model] (Matrix) and "<name>.down"
 * [d_model, d_ff] (ResidualProj: it writes into the residual stream).
 * h is caller-provided [M, d_ff] scratch; after the call it holds the
 * post-GELU hidden activation. y may alias x (x is consumed before y is
 * written); h may alias neither.
 */
class MLP {
   public:
    MLP(ParamStore& ps, const std::string& name, int64_t d_model, int64_t d_ff) noexcept;

    void forward(const RunCtx& ctx, Tensor& y, Tensor& h, const Tensor& x) const noexcept;

    const Param& up() const noexcept { return m_up.param(); }
    const Param& down() const noexcept { return m_down.param(); }

   private:
    Linear m_up, m_down;
};

}  // namespace llmt

#endif  // LLMT_HAS_CUDA
