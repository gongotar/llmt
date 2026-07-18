// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Masoud Jami
#include <initializer_list>

#include "llmt/core/error.h"
#include "llmt/core/precision.h"
#include "llmt/kernels/gelu.h"

namespace llmt::kernels {

namespace {

constexpr int kBlock = 256;

// tanh-approximation GELU: 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³))).
// T is the storage type; arithmetic is kernel_compute_t, converting at the
// load/store edges.
template <typename T>
__global__ void gelu_fwd_kernel(T* y, const T* x, int64_t n) {
    using TC = kernel_compute_t;
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    constexpr TC kSqrt2OverPi = TC(0.7978845608028654);
    constexpr TC kCoeff = TC(0.044715);
    const TC v = static_cast<TC>(x[i]);
    y[i] = static_cast<T>(TC(0.5) * v * (TC(1) + tanhf(kSqrt2OverPi * (v + kCoeff * v * v * v))));
}

}  // namespace

void gelu_fwd(cudaStream_t s, Tensor& y, const Tensor& x) noexcept {
    if (y.shape != x.shape) detail::fatal("gelu_fwd", "y/x shape mismatch");
    if (y.dtype != x.dtype) detail::fatal("gelu_fwd", "y/x dtype mismatch");
    for (const Tensor* t : std::initializer_list<const Tensor*>{&y, &x})
        if (!t->valid()) detail::fatal("gelu_fwd", "invalid tensor (null data)");

    const int64_t n = x.numel();
    const auto grid = static_cast<unsigned>((n + kBlock - 1) / kBlock);
    switch (x.dtype) {
        case DType::FP32:
            gelu_fwd_kernel<float><<<grid, kBlock, 0, s>>>(y.ptr<float>(), x.ptr<float>(), n);
            break;
        case DType::BF16:
        case DType::I32:
            detail::fatal("gelu_fwd", "no kernel instantiated for dtype %s", dtype_name(x.dtype));
    }
    CUDA_CHECK_LAST();
}

}  // namespace llmt::kernels
