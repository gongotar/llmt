// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Masoud Jami
#include <initializer_list>

#include "llmt/core/error.h"
#include "llmt/core/precision.h"
#include "llmt/kernels/rmsnorm.h"
#include "warp.cuh"

namespace llmt::kernels {

namespace {

constexpr int kWarpsPerBlock = 8;  // 256 threads; one warp owns one row

// One warp per row: lanes stride the C channels, sum of squares reduced via
// xor-shuffle (result lands in every lane — no broadcast needed).
// T is the storage type; TR the reduce dtype (accumulator and rstd storage,
// from the policy); elementwise arithmetic is kernel_compute_t.
template <typename T, typename TR>
__global__ void rmsnorm_fwd_kernel(T* y, TR* rstd, const T* x, const T* g, int64_t n_rows,
                                   int64_t n_cols, float eps) {
    using TC = kernel_compute_t;
    const int64_t row =
        static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + threadIdx.x / kWarpSize;
    if (row >= n_rows) return;
    const int lane = threadIdx.x % kWarpSize;
    const T* xr = x + row * n_cols;

    TR sumsq = TR(0);
    for (int64_t i = lane; i < n_cols; i += kWarpSize) {
        const TR v = static_cast<TR>(xr[i]);
        sumsq += v * v;
    }
    sumsq = warp_sum(sumsq);
    const TR r = rsqrtf(sumsq / static_cast<TR>(n_cols) + static_cast<TR>(eps));
    if (lane == 0) rstd[row] = r;

    T* yr = y + row * n_cols;
    for (int64_t i = lane; i < n_cols; i += kWarpSize)
        yr[i] = static_cast<T>(static_cast<TC>(xr[i]) * static_cast<TC>(r) *
                               static_cast<TC>(g[i]));
}

}  // namespace

void rmsnorm_fwd(const RunCtx& ctx, Tensor& y, Tensor& rstd, const Tensor& x, const Tensor& g,
                 float eps) noexcept {
    if (x.shape.rank != 2) detail::fatal("rmsnorm_fwd", "x must be [N, C]");
    const int64_t n_rows = x.shape[0], n_cols = x.shape[1];
    if (y.shape != x.shape) detail::fatal("rmsnorm_fwd", "y/x shape mismatch");
    if (g.shape != Shape{n_cols}) detail::fatal("rmsnorm_fwd", "g must be [C]");
    if (rstd.shape != Shape{n_rows}) detail::fatal("rmsnorm_fwd", "rstd must be [N]");
    if (y.dtype != x.dtype || g.dtype != x.dtype)
        detail::fatal("rmsnorm_fwd", "y/g/x dtype mismatch");
    // rstd is a reduction product: its dtype is the policy's reduce dtype.
    if (rstd.dtype != ctx.precision.reduce)
        detail::fatal("rmsnorm_fwd", "rstd dtype %s does not match reduce precision %s",
                      dtype_name(rstd.dtype), dtype_name(ctx.precision.reduce));
    for (const Tensor* t : std::initializer_list<const Tensor*>{&y, &rstd, &x, &g})
        if (!t->valid()) detail::fatal("rmsnorm_fwd", "invalid tensor (null data)");
    // Capability guard on the reduce axis (invariant 3): only the FP32
    // instantiation exists so far.
    if (ctx.precision.reduce != DType::FP32)
        detail::fatal("rmsnorm_fwd", "no kernel instantiated for reduce precision %s",
                      dtype_name(ctx.precision.reduce));

    const auto grid = static_cast<unsigned>((n_rows + kWarpsPerBlock - 1) / kWarpsPerBlock);
    switch (x.dtype) {
        case DType::FP32:
            rmsnorm_fwd_kernel<float, float><<<grid, kWarpsPerBlock * kWarpSize, 0, ctx.stream>>>(
                y.ptr<float>(), rstd.ptr<float>(), x.ptr<float>(), g.ptr<float>(), n_rows, n_cols,
                eps);
            break;
        case DType::BF16:
        case DType::I32:
            detail::fatal("rmsnorm_fwd", "no kernel instantiated for dtype %s",
                          dtype_name(x.dtype));
    }
    CUDA_CHECK_LAST();
}

}  // namespace llmt::kernels
