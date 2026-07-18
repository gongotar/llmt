// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Masoud Jami
// GEMM micro-benchmark (task 4.5): achieved TFLOPs vs device peak for the
// model-relevant shapes — records the "cuBLAS-bound" ceiling MFU is judged
// against. Strict FP32 (no TF32), matching the training configuration.
//
// Usage: llmt_bench_matmul   (run on the GPU box; prints a table)
#include <cstdio>
#include <vector>

#include "llmt/core/arena.h"
#include "llmt/core/device.h"
#include "llmt/core/error.h"
#include "llmt/core/rng.h"
#include "llmt/kernels/matmul.h"
#include "timing.h"

using namespace llmt;

namespace {

struct Case {
    const char* label;
    int64_t batch, m, n, k;  // batch = 1 → plain GEMM
};

}  // namespace

int main() {
    Device dev(0);
    const RunCtx ctx = dev.make_ctx(1, KernelBackend::Naive, PrecisionPolicy::fp32());
    std::printf("device: %s — peak fp32 %.1f TFLOPs (strict FP32, no TF32)\n",
                dev.props().name.c_str(), dev.props().peak_fp32_tflops);

    // Model-relevant shapes for the M1 config (B=32, T=256 → M = B·T = 8192,
    // C=384, d_ff=1536, V=50304, heads: B·H=192 batches of [T,hd]·[hd,T]).
    const std::vector<Case> cases = {
        // Empirical architecture ceiling: huge square GEMM — maximal
        // arithmetic intensity, no tile/wave quantization. Judge every other
        // shape against min(this, its memory roofline), not nominal peak.
        {"CEILING ref   [4096,4096]·[4096,4096]ᵀ", 1, 4096, 4096, 4096},
        {"proj  x·Wqᵀ   [8192,384]·[384,384]ᵀ", 1, 8192, 384, 384},
        {"mlp_up x·Wᵀ   [8192,384]·[1536,384]ᵀ", 1, 8192, 1536, 384},
        {"mlp_dn h·Wᵀ   [8192,1536]·[384,1536]ᵀ", 1, 8192, 384, 1536},
        {"lm_head x·Wᵀ  [8192,384]·[50304,384]ᵀ", 1, 8192, 50304, 384},
        {"attn QKᵀ      192×[256,64]·[256,64]ᵀ", 192, 256, 256, 64},
    };

    Arena arena(2ull << 30, "bench");
    std::printf("%-42s %10s %10s %8s\n", "case", "ms/iter", "TFLOPs", "% peak");
    for (const Case& cs : cases) {
        arena.reset();
        const Shape sa = cs.batch > 1 ? Shape{cs.batch, cs.m, cs.k} : Shape{cs.m, cs.k};
        const Shape sb = cs.batch > 1 ? Shape{cs.batch, cs.n, cs.k} : Shape{cs.n, cs.k};
        const Shape sc = cs.batch > 1 ? Shape{cs.batch, cs.m, cs.n} : Shape{cs.m, cs.n};
        Tensor a = arena.alloc(sa, DType::FP32);
        Tensor b = arena.alloc(sb, DType::FP32);
        Tensor c = arena.alloc(sc, DType::FP32);
        rng::fill_normal(ctx.stream, a.ptr<float>(), a.numel(), 1, 1, 0.0f, 1.0f);
        rng::fill_normal(ctx.stream, b.ptr<float>(), b.numel(), 1, 2, 0.0f, 1.0f);

        const double ms = bench::time_ms(dev, ctx.stream, /*warmup=*/3, /*iters=*/20, [&] {
            kernels::matmul(ctx, c, a, b, false, true, 1.0f, 0.0f);
        });
        const double tflops = 2.0 * cs.batch * cs.m * cs.n * cs.k / (ms * 1e-3) / 1e12;
        std::printf("%-42s %10.3f %10.2f %7.1f%%\n", cs.label, ms, tflops,
                    100.0 * tflops / dev.props().peak_fp32_tflops);
    }
    return 0;
}
