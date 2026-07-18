// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Masoud Jami
// Bandwidth benchmark for the Phase 5 kernels (task 5.6): these are
// memory-bound, so the metric is GB/s vs the device's peak bandwidth —
// bytes are counted explicitly per case (reads + writes, one pass each).
//
// Usage: llmt_bench_bandwidth [results.json]
#include <cstdio>

#include "llmt/core/arena.h"
#include "llmt/core/device.h"
#include "llmt/core/rng.h"
#include "llmt/kernels/gelu.h"
#include "llmt/kernels/residual.h"
#include "llmt/kernels/rmsnorm.h"
#include "llmt/kernels/softmax.h"
#include "timing.h"

using namespace llmt;

namespace {

struct Result {
    const char* name;
    double ms, gb, gbs, pct_peak;
};

}  // namespace

int main(int argc, char** argv) {
    Device dev(0);
    const RunCtx ctx = dev.make_ctx(1, KernelBackend::Naive, PrecisionPolicy::fp32());
    const cudaStream_t s = ctx.stream;
    std::printf("device: %s — peak bandwidth %.0f GB/s, L2 %zu MiB\n",
                dev.props().name.c_str(), dev.props().peak_bw_gbs,
                dev.props().l2_bytes >> 20);

    // Model shapes scaled up 16x in the batch axes (N = B·T rows, B·H score
    // batches) so every tensor exceeds any current L2 (Ada ships up to
    // 72 MiB; a working set that fits L2 measures L2 bandwidth, not DRAM —
    // observed 327-570% "of peak" on RTX 4000 Ada with the unscaled shapes).
    // Per-element arithmetic is unchanged, so %-of-peak is still the
    // kernel's DRAM story. C = 384; d_ff = 1536; T = 256. All buffers fp32.
    Arena arena(4ull << 30, "bench");
    constexpr int64_t kN = 131072, kC = 384, kFf = 1536, kBh = 3072, kT = 256;

    Tensor x = arena.alloc({kN, kC}, DType::FP32);
    Tensor y = arena.alloc({kN, kC}, DType::FP32);
    Tensor g = arena.alloc({kC}, DType::FP32);
    Tensor rstd = arena.alloc({kN}, DType::FP32);
    Tensor h = arena.alloc({kN, kFf}, DType::FP32);
    Tensor h2 = arena.alloc({kN, kFf}, DType::FP32);
    Tensor scores = arena.alloc({kBh, kT, kT}, DType::FP32);
    Tensor probs = arena.alloc({kBh, kT, kT}, DType::FP32);
    for (Tensor* t : {&x, &g, &h, &scores})
        rng::fill_normal(s, t->ptr<float>(), t->numel(), 1, 7, 0.0f, 1.0f);

    constexpr int kWarmup = 3, kIters = 50;
    constexpr size_t kCases = 4;
    const double bw_peak = dev.props().peak_bw_gbs;
    Result results[kCases];
    char names[kCases][48];
    std::snprintf(names[0], sizeof names[0], "rmsnorm  [%lld,%lld]", static_cast<long long>(kN),
                  static_cast<long long>(kC));
    std::snprintf(names[1], sizeof names[1], "gelu     [%lld,%lld]", static_cast<long long>(kN),
                  static_cast<long long>(kFf));
    std::snprintf(names[2], sizeof names[2], "residual [%lld,%lld]", static_cast<long long>(kN),
                  static_cast<long long>(kC));
    std::snprintf(names[3], sizeof names[3], "softmax  [%lld,%lld,%lld] causal",
                  static_cast<long long>(kBh), static_cast<long long>(kT),
                  static_cast<long long>(kT));

    {  // rmsnorm: read x + g, write y + rstd
        const double ms = bench::time_ms(dev, s, kWarmup, kIters, [&] {
            kernels::rmsnorm_fwd(ctx, y, rstd, x, g, 1e-5f);
        });
        const double gb = 4.0 * (x.numel() + g.numel() + y.numel() + rstd.numel()) / 1e9;
        results[0] = {names[0], ms, gb, gb / (ms * 1e-3), 0};
    }
    {  // gelu on the MLP activation: read h, write h2
        const double ms =
            bench::time_ms(dev, s, kWarmup, kIters, [&] { kernels::gelu_fwd(s, h2, h); });
        const double gb = 4.0 * (h.numel() + h2.numel()) / 1e9;
        results[1] = {names[1], ms, gb, gb / (ms * 1e-3), 0};
    }
    {  // residual: read x + y, write y (in-place accumulate pattern)
        const double ms =
            bench::time_ms(dev, s, kWarmup, kIters, [&] { kernels::residual_fwd(s, y, y, x); });
        const double gb = 4.0 * (3 * x.numel()) / 1e9;
        results[2] = {names[2], ms, gb, gb / (ms * 1e-3), 0};
    }
    {  // causal softmax: reads only the valid triangle (~half), writes all
        const double ms = bench::time_ms(dev, s, kWarmup, kIters, [&] {
            kernels::softmax_fwd(ctx, probs, scores,
                                 kernels::MaskSpec{kernels::MaskKind::Causal});
        });
        const double gb = 4.0 * (scores.numel() / 2 + probs.numel()) / 1e9;
        results[3] = {names[3], ms, gb, gb / (ms * 1e-3), 0};
    }

    std::printf("%-32s %10s %10s %10s %8s\n", "case", "ms/iter", "GB moved", "GB/s", "% peak");
    for (Result& r : results) {
        r.pct_peak = 100.0 * r.gbs / bw_peak;
        std::printf("%-32s %10.3f %10.3f %10.1f %7.1f%%\n", r.name, r.ms, r.gb, r.gbs, r.pct_peak);
    }

    if (argc > 1) {
        FILE* f = std::fopen(argv[1], "w");
        if (f == nullptr) {
            std::fprintf(stderr, "cannot open %s for writing\n", argv[1]);
            return 1;
        }
        std::fprintf(f, "{\n  \"device\": \"%s\",\n  \"peak_bw_gbs\": %.1f,\n  \"cases\": [\n",
                     dev.props().name.c_str(), bw_peak);
        for (size_t i = 0; i < kCases; ++i)
            std::fprintf(f,
                         "    {\"name\": \"%s\", \"ms\": %.4f, \"gbs\": %.1f, "
                         "\"pct_peak\": %.1f}%s\n",
                         results[i].name, results[i].ms, results[i].gbs, results[i].pct_peak,
                         i + 1 < kCases ? "," : "");
        std::fprintf(f, "  ]\n}\n");
        std::fclose(f);
        std::printf("wrote %s\n", argv[1]);
    }
    return 0;
}
