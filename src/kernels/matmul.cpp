// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Masoud Jami
//
// The one file that knows cuBLASLt (DESIGN §8.2). Everything cuBLASLt —
// layouts, transposes, heuristics, workspace — is quarantined here.
//
// ── Row-major ↔ column-major mapping (THE convention, single source of truth)
//
// All llmt tensors are row-major; cuBLASLt computes column-major. A row-major
// [r, c] buffer read column-major is its transpose [c, r] — same bytes. So a
// row-major request
//     C[M,N] = op(A)·op(B)
// is executed column-major as its transpose
//     Cᵀ[N,M] = op(B)ᵀ · op(A)ᵀ
// which in cuBLASLt terms means, mechanically:
//   · operands SWAP slots: B goes into the A-slot, A into the B-slot;
//   · each operand KEEPS its own trans flag (the cm view supplies one
//     transpose, the algebra above the other — they compose);
//   · result dims swap: the cm problem is (m,n,k) = (N, M, K);
//   · every leading dimension is the row-major COLUMN count of its buffer.
// C's buffer needs no post-processing: Cᵀ column-major IS C row-major.
//
// ── Determinism (invariant 5)
//
// The algorithm per problem key is resolved via the heuristic ONCE and cached
// for the process lifetime; repeated calls replay the identical algorithm.
// The cache is not thread-safe: all launches happen on one host thread.
//
// ── Precision
//
// Storage dtypes flow from the tensors; compute/scale types are queried from
// the ctx's PrecisionPolicy (invariant 3). Each has a single capability guard
// for the paths implemented so far. Notably TF32 is not a silent default:
// its 10-bit mantissa would break the 1e-5-rel golden tolerances — enabling
// it is a policy decision.

#ifdef LLMT_HAS_CUDA

#include <cublasLt.h>

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <unordered_map>

#include "llmt/core/error.h"
#include "llmt/kernels/algo_cache.h"
#include "llmt/kernels/matmul.h"

namespace llmt::kernels {

namespace {

[[noreturn]] void fatal(const char* fmt, ...) noexcept {
    va_list args;
    va_start(args, fmt);
    std::fprintf(stderr, "[llmt] matmul: ");
    std::vfprintf(stderr, fmt, args);
    std::fprintf(stderr, "\n");
    va_end(args);
    std::abort();
}

// llmt dtype → cuBLASLt data type (kept exhaustive by -Wall).
cudaDataType_t lt_dtype(DType d) noexcept {
    switch (d) {
        case DType::FP32: return CUDA_R_32F;
        case DType::BF16: return CUDA_R_16BF;
        case DType::I32: return CUDA_R_32I;
    }
    return CUDA_R_32F;  // unreachable
}

// PrecisionPolicy → the cuBLASLt compute type (arithmetic precision of the
// multiply-accumulate) and the scale type (alpha/beta dtype) Lt mandates for
// it — the pairing is Lt's rule, so they are decided together, here only.
// Capability guard (single point): only strict-FP32 compute is implemented.
struct LtComputeSpec {
    cublasComputeType_t compute;
    cudaDataType_t scale;
};
LtComputeSpec lt_compute_spec(const PrecisionPolicy& p) noexcept {
    if (p.compute != DType::FP32 || p.reduce != DType::FP32)
        fatal("compute/reduce precision %s/%s not yet supported", dtype_name(p.compute),
              dtype_name(p.reduce));
    return {CUBLAS_COMPUTE_32F, CUDA_R_32F};
}

// Problem dimensions of one GEMM, as row-major semantics.
struct Problem {
    int64_t m, n, k, batch;
    int64_t stride_a, stride_b, stride_c;  // elements between batch items (0 = unbatched)
    DType dtype = DType::FP32;             // uniform across A, B, C (per call)
    bool trans_a, trans_b;
};

// RAII for the per-call cuBLASLt descriptors (host-side, cheap to recreate).
struct LtDescriptors {
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
    ~LtDescriptors() {
        if (a) cublasLtMatrixLayoutDestroy(a);
        if (b) cublasLtMatrixLayoutDestroy(b);
        if (c) cublasLtMatrixLayoutDestroy(c);
        if (op) cublasLtMatmulDescDestroy(op);
    }
};

// Column-major layout of a row-major [rows, cols] buffer (= its cm transpose),
// with batching attributes when batch > 1.
cublasLtMatrixLayout_t make_layout(DType dtype, int64_t rows, int64_t cols, int64_t batch,
                                   int64_t stride) noexcept {
    cublasLtMatrixLayout_t l = nullptr;
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&l, lt_dtype(dtype), static_cast<uint64_t>(cols),
                                            static_cast<uint64_t>(rows), cols));
    if (batch > 1) {
        const int32_t bc = static_cast<int32_t>(batch);
        CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(l, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &bc,
                                                      sizeof bc));
        CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
            l, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride, sizeof stride));
    }
    return l;
}

void run(const RunCtx& ctx, const Problem& p, float* c_data, const float* a_data,
         const float* b_data, float alpha, float beta) noexcept {
    // Row-major buffer dims (before the user's op is applied).
    const int64_t a_rows = p.trans_a ? p.k : p.m, a_cols = p.trans_a ? p.m : p.k;
    const int64_t b_rows = p.trans_b ? p.n : p.k, b_cols = p.trans_b ? p.k : p.n;

    LtDescriptors d;
    const LtComputeSpec spec = lt_compute_spec(ctx.precision);
    CUBLAS_CHECK(cublasLtMatmulDescCreate(&d.op, spec.compute, spec.scale));
    // Operand swap (see file comment): B takes the A-slot, A the B-slot,
    // each keeping its own trans flag.
    const cublasOperation_t op_slot_a = p.trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
    const cublasOperation_t op_slot_b = p.trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    CUBLAS_CHECK(
        cublasLtMatmulDescSetAttribute(d.op, CUBLASLT_MATMUL_DESC_TRANSA, &op_slot_a,
                                       sizeof op_slot_a));
    CUBLAS_CHECK(
        cublasLtMatmulDescSetAttribute(d.op, CUBLASLT_MATMUL_DESC_TRANSB, &op_slot_b,
                                       sizeof op_slot_b));

    d.a = make_layout(p.dtype, b_rows, b_cols, p.batch, p.stride_b);  // B in the A-slot
    d.b = make_layout(p.dtype, a_rows, a_cols, p.batch, p.stride_a);  // A in the B-slot
    d.c = make_layout(p.dtype, p.m, p.n, p.batch, p.stride_c);

    if (ctx.algo_cache == nullptr)
        fatal("RunCtx has no algo cache (build the ctx via Device::make_ctx)");
    const AlgoKey key{p.m, p.n, p.k, p.batch,
                      (p.trans_a ? 1LL : 0LL) | (p.trans_b ? 2LL : 0LL),
                      static_cast<int64_t>(p.dtype), static_cast<int64_t>(spec.compute),
                      static_cast<int64_t>(spec.scale)};
    auto& cache = ctx.algo_cache->map;
    auto it = cache.find(key);
    if (it == cache.end()) {
        cublasLtMatmulPreference_t pref = nullptr;
        CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
        const uint64_t ws = ctx.blas_workspace_bytes;
        CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
            pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof ws));

        cublasLtMatmulHeuristicResult_t result{};
        int n_results = 0;
        CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(ctx.blas, d.op, d.a, d.b, d.c, d.c, pref, 1,
                                                    &result, &n_results));
        cublasLtMatmulPreferenceDestroy(pref);
        if (n_results == 0)
            fatal("no algorithm for m=%lld n=%lld k=%lld batch=%lld",
                  static_cast<long long>(p.m), static_cast<long long>(p.n),
                  static_cast<long long>(p.k), static_cast<long long>(p.batch));
        it = cache.emplace_hint(it, key, result.algo);
    }

    CUBLAS_CHECK(cublasLtMatmul(ctx.blas, d.op, &alpha, b_data, d.a, a_data, d.b, &beta, c_data,
                                d.c, c_data, d.c, &it->second, ctx.blas_workspace,
                                ctx.blas_workspace_bytes, ctx.stream));
}

// Validates one operand set and produces the Problem. The rank is structure,
// not intent: 2 = one GEMM, 3 = strided-batched (derived from the tensors,
// which must agree).
Problem check(const Tensor& c, const Tensor& a, const Tensor& b, bool trans_a,
              bool trans_b) noexcept {
    const int rank = c.shape.rank;
    if (rank != 2 && rank != 3) fatal("expected rank-2 or rank-3 tensors (got %d)", rank);
    for (const Tensor* t : {&c, &a, &b}) {
        if (t->shape.rank != rank) fatal("mixed ranks in one matmul");
        if (t->dtype != a.dtype) fatal("mixed dtypes in one matmul");
        if (!t->valid()) fatal("invalid tensor (null data)");
    }
    // Capability guard (single point): the FP32 compute path is the only one
    // implemented; layouts/keys/plumbing are dtype-generic above and below.
    if (a.dtype != DType::FP32) fatal("dtype %s not yet supported", dtype_name(a.dtype));
    const int r = rank - 2;  // row axis (0 for 2-D, 1 for 3-D)
    Problem p{};
    p.dtype = a.dtype;
    p.trans_a = trans_a;
    p.trans_b = trans_b;
    p.m = trans_a ? a.shape[r + 1] : a.shape[r];
    p.k = trans_a ? a.shape[r] : a.shape[r + 1];
    const int64_t kb = trans_b ? b.shape[r + 1] : b.shape[r];
    p.n = trans_b ? b.shape[r] : b.shape[r + 1];
    if (p.k != kb) fatal("inner dims disagree: %lld vs %lld", static_cast<long long>(p.k),
                         static_cast<long long>(kb));
    if (c.shape[r] != p.m || c.shape[r + 1] != p.n) fatal("C shape mismatch");

    if (rank == 3) {
        p.batch = a.shape[0];
        if (b.shape[0] != p.batch || c.shape[0] != p.batch) fatal("batch sizes disagree");
        p.stride_a = a.shape[1] * a.shape[2];
        p.stride_b = b.shape[1] * b.shape[2];
        p.stride_c = c.shape[1] * c.shape[2];
    } else {
        p.batch = 1;
    }
    return p;
}

}  // namespace

void matmul(const RunCtx& ctx, Tensor& c, const Tensor& a, const Tensor& b, bool trans_a,
            bool trans_b, float alpha, float beta) noexcept {
    const Problem p = check(c, a, b, trans_a, trans_b);
    run(ctx, p, c.ptr<float>(), a.ptr<float>(), b.ptr<float>(), alpha, beta);
}

}  // namespace llmt::kernels

#endif  // LLMT_HAS_CUDA
