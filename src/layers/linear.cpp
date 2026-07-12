// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Masoud Jami
#ifdef LLMT_HAS_CUDA

#include <utility>

#include "llmt/kernels/matmul.h"
#include "llmt/layers/linear.h"

namespace llmt {

Linear::Linear(ParamStore& ps, std::string name, int64_t in_features, int64_t out_features,
               Role role) noexcept
    : m_param(ps.add(std::move(name), {out_features, in_features}, role)) {}

void Linear::forward(const RunCtx& ctx, Tensor& y, const Tensor& x) const noexcept {
    // y = x · Wᵀ : [M, in] · [out, in]ᵀ → [M, out]
    kernels::matmul(ctx, y, x, m_param.weight, /*trans_a=*/false, /*trans_b=*/true,
                    /*alpha=*/1.0f, /*beta=*/0.0f);
}

void Linear::backward(const RunCtx& ctx, Tensor& dx, const Tensor& dy, const Tensor& x)
    const noexcept {
    // dx = dy · W : [M, out] · [out, in] → [M, in]
    kernels::matmul(ctx, dx, dy, m_param.weight, /*trans_a=*/false, /*trans_b=*/false,
                    /*alpha=*/1.0f, /*beta=*/0.0f);
    // dW += dyᵀ · x : [out, M] · [M, in] → [out, in], accumulated (beta = 1).
    kernels::matmul(ctx, m_param.grad, dy, x, /*trans_a=*/true, /*trans_b=*/false,
                    /*alpha=*/1.0f, /*beta=*/1.0f);
}

}  // namespace llmt

#endif  // LLMT_HAS_CUDA
