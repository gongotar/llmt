// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Masoud Jami
#include <cmath>

#include <initializer_list>

#include "llmt/core/error.h"
#include "llmt/core/precision.h"
#include "llmt/kernels/softmax.h"
#include "warp.cuh"

namespace llmt::kernels {

namespace {

constexpr int kWarpsPerBlock = 8;

// One warp per score row (one (batch, q) pair). Entries [0, limit) are
// valid; the rest get probability 0. Three passes over the row (max, sum,
// write) — the write pass only reads its own element after both reductions,
// which is what makes y == x (in-place) safe. T is the storage type; TR the
// reduce dtype (max/sum accumulators, from the policy); elementwise
// arithmetic is kernel_compute_t.
template <typename T, typename TR>
__global__ void softmax_fwd_kernel(T* y, const T* x, int64_t n_rows, int64_t tq, int64_t tk,
                                   bool causal) {
    using TC = kernel_compute_t;
    const int64_t row =
        static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + threadIdx.x / kWarpSize;
    if (row >= n_rows) return;
    const int lane = threadIdx.x % kWarpSize;
    const int64_t q = row % tq;
    const int64_t limit = causal ? q + 1 : tk;
    const T* xr = x + row * tk;
    T* yr = y + row * tk;

    TR m = -INFINITY;
    for (int64_t i = lane; i < limit; i += kWarpSize)
        m = fmaxf(m, static_cast<TR>(xr[i]));
    m = warp_max(m);

    TR sum = TR(0);
    for (int64_t i = lane; i < limit; i += kWarpSize)
        sum += expf(static_cast<TC>(xr[i]) - static_cast<TC>(m));
    sum = warp_sum(sum);
    const TC inv = TC(1) / static_cast<TC>(sum);

    for (int64_t i = lane; i < tk; i += kWarpSize)
        yr[i] = static_cast<T>(
            i < limit ? expf(static_cast<TC>(xr[i]) - static_cast<TC>(m)) * inv : TC(0));
}

}  // namespace

void softmax_fwd(const RunCtx& ctx, Tensor& y, const Tensor& x, MaskSpec mask) noexcept {
    if (x.shape.rank != 3) detail::fatal("softmax_fwd", "x must be [batch, Tq, Tk]");
    if (y.shape != x.shape) detail::fatal("softmax_fwd", "y/x shape mismatch");
    if (y.dtype != x.dtype) detail::fatal("softmax_fwd", "y/x dtype mismatch");
    const int64_t tq = x.shape[1], tk = x.shape[2];
    if (mask.kind == MaskKind::Causal && tq != tk)
        detail::fatal("softmax_fwd", "causal mask requires square scores (Tq=%lld Tk=%lld)",
                      static_cast<long long>(tq), static_cast<long long>(tk));
    for (const Tensor* t : std::initializer_list<const Tensor*>{&y, &x})
        if (!t->valid()) detail::fatal("softmax_fwd", "invalid tensor (null data)");
    const int64_t n_rows = x.shape[0] * tq;
    const auto grid = static_cast<unsigned>((n_rows + kWarpsPerBlock - 1) / kWarpsPerBlock);
    // Two-axis dispatch (invariant 3): outer switch = reduce precision,
    // inner = storage dtype; capability = the arms that launch. TR is named
    // inside the arm that proved it, so check and instantiation cannot
    // drift apart.
    switch (ctx.precision.reduce) {
        case DType::FP32:
            switch (x.dtype) {
                case DType::FP32:
                    softmax_fwd_kernel<float, float>
                        <<<grid, kWarpsPerBlock * kWarpSize, 0, ctx.stream>>>(
                            y.ptr<float>(), x.ptr<float>(), n_rows, tq, tk,
                            mask.kind == MaskKind::Causal);
                    break;
                case DType::BF16:
                case DType::I32:
                    detail::fatal("softmax_fwd", "no kernel instantiated for dtype %s",
                                  dtype_name(x.dtype));
            }
            break;
        case DType::BF16:
        case DType::I32:
            detail::fatal("softmax_fwd", "no kernel instantiated for reduce precision %s",
                          dtype_name(ctx.precision.reduce));
    }
    CUDA_CHECK_LAST();
}

}  // namespace llmt::kernels
