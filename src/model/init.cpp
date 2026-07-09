// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Masoud Jami
#ifdef LLMT_HAS_CUDA

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "llmt/core/rng.h"
#include "llmt/kernels/fill.h"
#include "llmt/model/init.h"

namespace llmt {

void init_params(const ParamStore& store, cudaStream_t s, uint64_t seed, int n_layer) noexcept {
    if (!store.finalized() || n_layer <= 0) {
        std::fprintf(stderr, "[llmt] init_params: %s\n",
                     n_layer <= 0 ? "n_layer must be positive" : "store is not finalized");
        std::abort();
    }
    const float resid_std = 0.02f / std::sqrt(2.0f * static_cast<float>(n_layer));

    for (const Param& p : store.params()) {
        float* const w = p.weight.ptr<float>();
        const int64_t n = p.weight.numel();
        switch (p.role) {
            case Role::Embedding:
            case Role::Matrix:
                rng::fill_normal(s, w, n, seed, p.stream_id(), 0.0f, 0.02f);
                break;
            case Role::ResidualProj:
                rng::fill_normal(s, w, n, seed, p.stream_id(), 0.0f, resid_std);
                break;
            case Role::Norm:
                kernels::fill_value(s, w, n, 1.0f);
                break;
        }
    }
}

}  // namespace llmt

#endif  // LLMT_HAS_CUDA
