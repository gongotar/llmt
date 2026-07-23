// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Masoud Jami
#include <initializer_list>

#include "llmt/core/error.h"
#include "llmt/kernels/permute.h"

namespace llmt::kernels {

namespace {

constexpr int kBlock = 256;

// One thread per (b, h, t, d) coordinate, writing that element of q, k and
// v: adjacent threads differ in d, so all three writes and all three reads
// are coalesced (source elements sit h·hd+d apart within one token row).
template <typename T>
__global__ void permute_split_kernel(T* q, T* k, T* v, const T* qkv, int64_t n, int64_t n_head,
                                     int64_t seq, int64_t hd) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t d = i % hd;
    const int64_t t = (i / hd) % seq;
    const int64_t h = (i / (hd * seq)) % n_head;
    const int64_t b = i / (hd * seq * n_head);
    const int64_t a = n_head * hd;  // attention inner width
    const int64_t src = (b * seq + t) * 3 * a + h * hd + d;
    q[i] = qkv[src];
    k[i] = qkv[src + a];
    v[i] = qkv[src + 2 * a];
}

// One thread per element of y [B, T, H·hd], decoded as (b, t, h, d) —
// h and d split y's last axis (column h·hd + d = head h's channel d).
template <typename T>
__global__ void permute_merge_kernel(T* y, const T* x, int64_t n, int64_t n_head, int64_t seq,
                                     int64_t hd) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t a = n_head * hd;  // attention inner width
    const int64_t d = i % hd;
    const int64_t h = (i / hd) % n_head;
    const int64_t t = (i / a) % seq;
    const int64_t b = i / (a * seq);
    y[i] = x[((b * n_head + h) * seq + t) * hd + d];
}

}  // namespace

void permute_split(cudaStream_t s, Tensor& q, Tensor& k, Tensor& v, const Tensor& qkv,
                   int64_t n_head) noexcept {
    if (qkv.shape.rank != 3) detail::fatal("permute_split", "qkv must be [B, T, 3·H·hd]");
    const int64_t b = qkv.shape[0], seq = qkv.shape[1], a3 = qkv.shape[2];
    if (n_head <= 0 || a3 % (3 * n_head) != 0)
        detail::fatal("permute_split", "3·H·hd = %lld not divisible by 3·n_head = %lld",
                      static_cast<long long>(a3), static_cast<long long>(3 * n_head));
    const int64_t hd = a3 / (3 * n_head);
    const Shape head_shape{b, n_head, seq, hd};
    for (const Tensor* t : std::initializer_list<const Tensor*>{&q, &k, &v}) {
        if (t->shape != head_shape) detail::fatal("permute_split", "q/k/v must be [B, H, T, hd]");
        if (t->dtype != qkv.dtype) detail::fatal("permute_split", "q/k/v/qkv dtype mismatch");
    }
    for (const Tensor* t : std::initializer_list<const Tensor*>{&q, &k, &v, &qkv})
        if (!t->valid()) detail::fatal("permute_split", "invalid tensor (null data)");

    const int64_t n = q.numel();
    const auto grid = static_cast<unsigned>((n + kBlock - 1) / kBlock);
    switch (qkv.dtype) {
        case DType::FP32:
            permute_split_kernel<float><<<grid, kBlock, 0, s>>>(
                q.ptr<float>(), k.ptr<float>(), v.ptr<float>(), qkv.ptr<float>(), n, n_head, seq,
                hd);
            break;
        case DType::BF16:
        case DType::I32:
            detail::fatal("permute_split", "no kernel instantiated for dtype %s",
                          dtype_name(qkv.dtype));
    }
    CUDA_CHECK_LAST();
}

void permute_merge(cudaStream_t s, Tensor& y, const Tensor& x) noexcept {
    if (x.shape.rank != 4) detail::fatal("permute_merge", "x must be [B, H, T, hd]");
    const int64_t b = x.shape[0], n_head = x.shape[1], seq = x.shape[2], hd = x.shape[3];
    if (y.shape != Shape{b, seq, n_head * hd})
        detail::fatal("permute_merge", "y must be [B, T, H·hd]");
    if (y.dtype != x.dtype) detail::fatal("permute_merge", "y/x dtype mismatch");
    for (const Tensor* t : std::initializer_list<const Tensor*>{&y, &x})
        if (!t->valid()) detail::fatal("permute_merge", "invalid tensor (null data)");

    const int64_t n = y.numel();
    const auto grid = static_cast<unsigned>((n + kBlock - 1) / kBlock);
    switch (x.dtype) {
        case DType::FP32:
            permute_merge_kernel<float>
                <<<grid, kBlock, 0, s>>>(y.ptr<float>(), x.ptr<float>(), n, n_head, seq, hd);
            break;
        case DType::BF16:
        case DType::I32:
            detail::fatal("permute_merge", "no kernel instantiated for dtype %s",
                          dtype_name(x.dtype));
    }
    CUDA_CHECK_LAST();
}

}  // namespace llmt::kernels
