// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Masoud Jami
#include <cmath>

#include <initializer_list>

#include "llmt/core/error.h"
#include "llmt/core/precision.h"
#include "llmt/kernels/cross_entropy.h"
#include "warp.cuh"

namespace llmt::kernels {

namespace {

constexpr int kWarpsPerBlock = 8;
constexpr int kReduceBlock = 256;

// One warp per token row (V ≈ 50k channels, lanes stride by 32). Two
// reductions: m = row max, then sum = Σ exp(x−m); the header's
// lse = log Σ exp(x) is recovered overflow-safely as m + log(sum).
// T is the storage type; TR the reduce dtype (accumulators, from the
// policy); row_nll is host-published, so it stores as host_result_t;
// elementwise arithmetic is kernel_compute_t.
template <typename T, typename TR>
__global__ void ce_row_kernel(host_result_t* row_nll, const T* logits, const int32_t* targets,
                              const T* mask, int64_t n_rows, int64_t v) {
    using TC = kernel_compute_t;
    const int64_t row =
        static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + threadIdx.x / kWarpSize;
    if (row >= n_rows) return;
    const int lane = threadIdx.x % kWarpSize;
    const T* xr = logits + row * v;

    TR m = -INFINITY;
    for (int64_t i = lane; i < v; i += kWarpSize) m = fmaxf(m, static_cast<TR>(xr[i]));
    m = warp_max(m);

    TR sum = TR(0);
    for (int64_t i = lane; i < v; i += kWarpSize)
        sum += expf(static_cast<TC>(xr[i]) - static_cast<TC>(m));
    sum = warp_sum(sum);

    if (lane == 0) {
        const TC lse = static_cast<TC>(m) + logf(static_cast<TC>(sum));
        const TC nll = lse - static_cast<TC>(xr[targets[row]]);
        row_nll[row] = static_cast<host_result_t>(static_cast<TC>(mask[row]) * nll);
    }
}

// Single-block deterministic mean: fixed grid-stride order per thread,
// fixed shared-memory tree — same bits every run (invariant 5, no atomics).
// Accumulates in TR (the reduce dtype); the finished scalar stores as
// host_result_t. An all-zero mask yields 0/0 = NaN, loud by design (header
// contract).
template <typename T, typename TR>
__global__ void ce_reduce_kernel(host_result_t* loss, const host_result_t* row_nll,
                                 const T* mask, int64_t n) {
    __shared__ TR s_nll[kReduceBlock];
    __shared__ TR s_mask[kReduceBlock];
    TR nll = TR(0), msum = TR(0);
    for (int64_t i = threadIdx.x; i < n; i += kReduceBlock) {
        nll += static_cast<TR>(row_nll[i]);
        msum += static_cast<TR>(mask[i]);
    }
    s_nll[threadIdx.x] = nll;
    s_mask[threadIdx.x] = msum;
    __syncthreads();
    for (int off = kReduceBlock / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) {
            s_nll[threadIdx.x] += s_nll[threadIdx.x + off];
            s_mask[threadIdx.x] += s_mask[threadIdx.x + off];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) loss[0] = static_cast<host_result_t>(s_nll[0] / s_mask[0]);
}

}  // namespace

void cross_entropy_fwd(const RunCtx& ctx, Tensor& loss, Tensor& row_nll, const Tensor& logits,
                       const Tensor& targets, const Tensor& mask) noexcept {
    if (logits.shape.rank != 3) detail::fatal("cross_entropy_fwd", "logits must be [B, T, V]");
    const int64_t b = logits.shape[0], seq = logits.shape[1], v = logits.shape[2];
    const Shape row_shape{b, seq};
    if (targets.shape != row_shape) detail::fatal("cross_entropy_fwd", "targets must be [B, T]");
    if (mask.shape != row_shape) detail::fatal("cross_entropy_fwd", "mask must be [B, T]");
    if (row_nll.shape != row_shape) detail::fatal("cross_entropy_fwd", "row_nll must be [B, T]");
    if (loss.numel() != 1) detail::fatal("cross_entropy_fwd", "loss must be a [1] scalar");
    if (targets.dtype != DType::I32) detail::fatal("cross_entropy_fwd", "targets must be I32");
    if (mask.dtype != logits.dtype)
        detail::fatal("cross_entropy_fwd", "mask/logits dtype mismatch");
    // loss and row_nll are host-published: fixed host_result_t, never the
    // policy's reduce dtype (precision.h).
    if (loss.dtype != dtype_of<host_result_t>::value ||
        row_nll.dtype != dtype_of<host_result_t>::value)
        detail::fatal("cross_entropy_fwd", "loss/row_nll must be %s (host_result_t)",
                      dtype_name(dtype_of<host_result_t>::value));
    for (const Tensor* t :
         std::initializer_list<const Tensor*>{&loss, &row_nll, &logits, &targets, &mask})
        if (!t->valid()) detail::fatal("cross_entropy_fwd", "invalid tensor (null data)");
    const int64_t n_rows = b * seq;
    const auto grid = static_cast<unsigned>((n_rows + kWarpsPerBlock - 1) / kWarpsPerBlock);
    // Two-axis dispatch (invariant 3): outer switch = reduce precision,
    // inner = storage dtype; capability = the arms that launch. TR is named
    // inside the arm that proved it, so check and instantiation cannot
    // drift apart.
    switch (ctx.precision.reduce) {
        case DType::FP32:
            switch (logits.dtype) {
                case DType::FP32:
                    ce_row_kernel<float, float>
                        <<<grid, kWarpsPerBlock * kWarpSize, 0, ctx.stream>>>(
                            row_nll.ptr<host_result_t>(), logits.ptr<float>(),
                            targets.ptr<int32_t>(), mask.ptr<float>(), n_rows, v);
                    ce_reduce_kernel<float, float><<<1, kReduceBlock, 0, ctx.stream>>>(
                        loss.ptr<host_result_t>(), row_nll.ptr<host_result_t>(),
                        mask.ptr<float>(), n_rows);
                    break;
                case DType::BF16:
                case DType::I32:
                    detail::fatal("cross_entropy_fwd", "no kernel instantiated for dtype %s",
                                  dtype_name(logits.dtype));
            }
            break;
        case DType::BF16:
        case DType::I32:
            detail::fatal("cross_entropy_fwd", "no kernel instantiated for reduce precision %s",
                          dtype_name(ctx.precision.reduce));
    }
    CUDA_CHECK_LAST();
}

}  // namespace llmt::kernels
