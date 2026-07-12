// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Masoud Jami
#pragma once

#ifdef LLMT_HAS_CUDA

#include <cstdint>
#include <string>

#include "llmt/core/run_ctx.h"
#include "llmt/core/tensor.h"
#include "llmt/model/param_store.h"

namespace llmt {

/**
 * Bias-free linear projection with weight W[out, in] (PyTorch convention):
 *   forward:  y[M, out] = x[M, in] · Wᵀ
 *   backward: dx[M, in] = dy[M, out] · W ;  dW += dyᵀ · x   (accumulates —
 *             the grad view may be shared, invariant 9)
 * Registers its weight in the given ParamStore at construction; the caller
 * supplies x to backward (activation ownership stays with the caller).
 */
class Linear {
   public:
    Linear(ParamStore& ps, std::string name, int64_t in_features, int64_t out_features,
           Role role) noexcept;

    void forward(const RunCtx& ctx, Tensor& y, const Tensor& x) const noexcept;
    void backward(const RunCtx& ctx, Tensor& dx, const Tensor& dy, const Tensor& x) const noexcept;

    const Param& param() const noexcept { return m_param; }

   private:
    Param& m_param;  // stable reference into the store
};

}  // namespace llmt

#endif  // LLMT_HAS_CUDA
