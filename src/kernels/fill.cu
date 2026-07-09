// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Masoud Jami
#include "llmt/core/error.h"
#include "llmt/kernels/fill.h"

namespace llmt::kernels {

namespace {

__global__ void fill_value_kernel(float* dst, int64_t n, float value) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = value;
}

constexpr int kBlock = 256;

}  // namespace

void fill_value(cudaStream_t s, float* dst, int64_t n, float value) noexcept {
    if (n <= 0) return;
    const auto grid = static_cast<unsigned>((n + kBlock - 1) / kBlock);
    fill_value_kernel<<<grid, kBlock, 0, s>>>(dst, n, value);
    CUDA_CHECK_LAST();
}

}  // namespace llmt::kernels
