// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Masoud Jami
#include <initializer_list>

#include "llmt/core/error.h"
#include "llmt/core/precision.h"
#include "llmt/kernels/residual.h"

namespace llmt::kernels {

namespace {

constexpr int kBlock = 256;

// T is the storage type; arithmetic is kernel_compute_t.
template <typename T>
__global__ void residual_fwd_kernel(T* out, const T* a, const T* b, int64_t n) {
    using TC = kernel_compute_t;
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = static_cast<T>(static_cast<TC>(a[i]) + static_cast<TC>(b[i]));
}

}  // namespace

void residual_fwd(cudaStream_t s, Tensor& out, const Tensor& a, const Tensor& b) noexcept {
    if (out.shape != a.shape || a.shape != b.shape)
        detail::fatal("residual_fwd", "shape mismatch");
    if (out.dtype != a.dtype || a.dtype != b.dtype)
        detail::fatal("residual_fwd", "dtype mismatch");
    for (const Tensor* t : std::initializer_list<const Tensor*>{&out, &a, &b})
        if (!t->valid()) detail::fatal("residual_fwd", "invalid tensor (null data)");

    const int64_t n = a.numel();
    const auto grid = static_cast<unsigned>((n + kBlock - 1) / kBlock);
    switch (a.dtype) {
        case DType::FP32:
            residual_fwd_kernel<float><<<grid, kBlock, 0, s>>>(out.ptr<float>(), a.ptr<float>(),
                                                               b.ptr<float>(), n);
            break;
        case DType::BF16:
        case DType::I32:
            detail::fatal("residual_fwd", "no kernel instantiated for dtype %s",
                          dtype_name(a.dtype));
    }
    CUDA_CHECK_LAST();
}

}  // namespace llmt::kernels
