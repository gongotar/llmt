// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Masoud Jami
#include "llmt/core/error.h"
#include "llmt/kernels/fill.h"

namespace llmt::kernels {

namespace {

constexpr int kBlock = 256;

template <typename T>
__global__ void fill_value_kernel(T* dst, int64_t n, T value) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = value;
}

}  // namespace

template <TensorElement T>
void fill_value(cudaStream_t s, Tensor& dst, T value) noexcept {
    if (!dst.valid()) detail::fatal("fill_value", "invalid tensor (null data)");
    if (dst.dtype != dtype_of<T>::value)
        detail::fatal("fill_value", "value type does not match tensor dtype %s",
                      dtype_name(dst.dtype));

    const int64_t n = dst.numel();
    const auto grid = static_cast<unsigned>((n + kBlock - 1) / kBlock);
    fill_value_kernel<T><<<grid, kBlock, 0, s>>>(dst.ptr<T>(), n, value);
    CUDA_CHECK_LAST();
}

template void fill_value<float>(cudaStream_t, Tensor&, float) noexcept;
template void fill_value<int32_t>(cudaStream_t, Tensor&, int32_t) noexcept;

}  // namespace llmt::kernels
