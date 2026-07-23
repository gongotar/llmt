// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Masoud Jami
#include <initializer_list>

#include "llmt/core/error.h"
#include "llmt/kernels/embedding.h"

namespace llmt::kernels {

namespace {

constexpr int kBlock = 256;

// One thread per output element: adjacent lanes read/write adjacent
// channels of one row — coalesced on both sides of the gather.
template <typename T>
__global__ void embedding_fwd_kernel(T* y, const int32_t* tokens, const T* wte, int64_t n,
                                     int64_t c) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t row = i / c, col = i % c;
    y[i] = wte[static_cast<int64_t>(tokens[row]) * c + col];
}

}  // namespace

void embedding_fwd(cudaStream_t s, Tensor& y, const Tensor& tokens, const Tensor& wte) noexcept {
    if (tokens.shape.rank != 2) detail::fatal("embedding_fwd", "tokens must be [B, T]");
    if (wte.shape.rank != 2) detail::fatal("embedding_fwd", "wte must be [V, C]");
    if (y.shape.rank != 3 || y.shape[0] != tokens.shape[0] || y.shape[1] != tokens.shape[1] ||
        y.shape[2] != wte.shape[1])
        detail::fatal("embedding_fwd", "y must be [B, T, C] matching tokens and wte");
    if (tokens.dtype != DType::I32) detail::fatal("embedding_fwd", "tokens must be I32");
    if (y.dtype != wte.dtype) detail::fatal("embedding_fwd", "y/wte dtype mismatch");
    for (const Tensor* t : std::initializer_list<const Tensor*>{&y, &tokens, &wte})
        if (!t->valid()) detail::fatal("embedding_fwd", "invalid tensor (null data)");

    const int64_t n = y.numel(), c = wte.shape[1];
    const auto grid = static_cast<unsigned>((n + kBlock - 1) / kBlock);
    switch (wte.dtype) {
        case DType::FP32:
            embedding_fwd_kernel<float><<<grid, kBlock, 0, s>>>(
                y.ptr<float>(), tokens.ptr<int32_t>(), wte.ptr<float>(), n, c);
            break;
        case DType::BF16:
        case DType::I32:
            detail::fatal("embedding_fwd", "no kernel instantiated for dtype %s",
                          dtype_name(wte.dtype));
    }
    CUDA_CHECK_LAST();
}

}  // namespace llmt::kernels
