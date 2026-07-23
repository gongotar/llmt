// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Masoud Jami
#ifdef LLMT_HAS_CUDA

#include "llmt/kernels/gelu.h"
#include "llmt/layers/mlp.h"

namespace llmt {

MLP::MLP(ParamStore& ps, const std::string& name, int64_t d_model, int64_t d_ff) noexcept
    : m_up(ps, name + ".up", d_model, d_ff, Role::Matrix),
      m_down(ps, name + ".down", d_ff, d_model, Role::ResidualProj) {}

void MLP::forward(const RunCtx& ctx, Tensor& y, Tensor& h, const Tensor& x) const noexcept {
    m_up.forward(ctx, h, x);              // h = x · Wupᵀ   [M, d_ff]
    kernels::gelu_fwd(ctx.stream, h, h);  // in place
    m_down.forward(ctx, y, h);            // y = h · Wdownᵀ [M, d_model]
}

}  // namespace llmt

#endif  // LLMT_HAS_CUDA
