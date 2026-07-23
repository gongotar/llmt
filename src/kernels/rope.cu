// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Masoud Jami
#include <initializer_list>

#include "llmt/core/error.h"
#include "llmt/core/precision.h"
#include "llmt/kernels/rope.h"

namespace llmt::kernels {

namespace {

constexpr int kBlock = 256;

// One thread per rotation pair (d, d + hd/2): each thread owns both
// elements of its pair exclusively, which is what makes the in-place
// update race-free. T is the storage type; arithmetic is kernel_compute_t.
template <typename T>
__global__ void rope_fwd_kernel(T* x, int64_t n, int64_t seq, int64_t hd, float theta) {
    using TC = kernel_compute_t;
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;  // n = B·H·T·(hd/2)
    const int64_t half = hd / 2;
    const int64_t d = i % half;
    const int64_t row = i / half;          // (b, h, t) flattened
    const int64_t t = row % seq;           // position index
    const TC ang = static_cast<TC>(t) *
                   powf(static_cast<TC>(theta),
                        TC(-2) * static_cast<TC>(d) / static_cast<TC>(hd));
    const TC c = cosf(ang), s = sinf(ang);
    T* xr = x + row * hd;
    const TC x1 = static_cast<TC>(xr[d]);
    const TC x2 = static_cast<TC>(xr[d + half]);
    xr[d] = static_cast<T>(x1 * c - x2 * s);
    xr[d + half] = static_cast<T>(x2 * c + x1 * s);
}

void launch(cudaStream_t s, Tensor& x, float theta, const char* who) noexcept {
    const int64_t seq = x.shape[2], hd = x.shape[3];
    const int64_t n = x.numel() / 2;  // one thread per pair
    const auto grid = static_cast<unsigned>((n + kBlock - 1) / kBlock);
    switch (x.dtype) {
        case DType::FP32:
            rope_fwd_kernel<float><<<grid, kBlock, 0, s>>>(x.ptr<float>(), n, seq, hd, theta);
            break;
        case DType::BF16:
        case DType::I32:
            detail::fatal(who, "no kernel instantiated for dtype %s", dtype_name(x.dtype));
    }
    CUDA_CHECK_LAST();
}

}  // namespace

void rope_fwd(cudaStream_t s, Tensor& q, Tensor& k, float theta) noexcept {
    for (const Tensor* t : std::initializer_list<const Tensor*>{&q, &k}) {
        if (t->shape.rank != 4) detail::fatal("rope_fwd", "q/k must be [B, H, T, hd]");
        if (t->shape[3] % 2 != 0) detail::fatal("rope_fwd", "hd must be even");
        if (!t->valid()) detail::fatal("rope_fwd", "invalid tensor (null data)");
    }
    if (q.dtype != k.dtype) detail::fatal("rope_fwd", "q/k dtype mismatch");
    if (theta <= 0.0f) detail::fatal("rope_fwd", "theta must be positive");

    launch(s, q, theta, "rope_fwd");
    launch(s, k, theta, "rope_fwd");
}

}  // namespace llmt::kernels
