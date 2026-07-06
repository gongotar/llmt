// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Masoud Jami
// GPU-side core tests: Arena, Device, Rng device fills (Phase 1 exit).
#include <cstring>
#include <vector>

#include "doctest.h"
#include "llmt/core/arena.h"
#include "llmt/core/device.h"
#include "llmt/core/error.h"
#include "llmt/core/rng.h"

using namespace llmt;

TEST_CASE("Arena: alignment, offsets, reset, high water") {
    Arena a(1 << 20, "test");
    CHECK(a.capacity() == 1 << 20);
    CHECK(a.used() == 0);

    void* const p1 = a.alloc_bytes(100);
    CHECK(reinterpret_cast<uintptr_t>(p1) % Arena::kAlign == 0);
    CHECK(a.used() == 100);

    void* const p2 = a.alloc_bytes(50);  // starts at 256 (100 rounded up)
    CHECK(reinterpret_cast<uintptr_t>(p2) % Arena::kAlign == 0);
    CHECK(static_cast<char*>(p2) - static_cast<char*>(p1) == 256);
    CHECK(a.used() == 256 + 50);

    const Tensor t = a.alloc({8, 16}, DType::FP32);
    CHECK(t.bytes() == 8 * 16 * 4);
    CHECK(reinterpret_cast<uintptr_t>(t.data) % Arena::kAlign == 0);

    const size_t hw = a.high_water();
    CHECK(hw == a.used());
    a.reset();
    CHECK(a.used() == 0);
    CHECK(a.high_water() == hw);  // survives reset

    // memory is reusable and identical after reset
    void* const q1 = a.alloc_bytes(100);
    CHECK(q1 == p1);
}

TEST_CASE("Device: sane properties, stream, blas handle, ctx wiring") {
    Device dev(0);
    const DeviceProps& p = dev.props();
    MESSAGE("device: ", p.name, " sm_", p.sm_major, p.sm_minor, ", ", p.sm_count, " SMs, ",
            p.peak_fp32_tflops, " TFLOPs fp32, ", p.peak_bw_gbs, " GB/s");

    CHECK(!p.name.empty());
    CHECK(p.sm_major >= 7);          // anything older can't run us well anyway
    CHECK(p.sm_count > 0);
    CHECK(p.vram_bytes > (4ull << 30));
    CHECK(p.peak_fp32_tflops > 1.0);  // any real GPU
    CHECK(p.peak_fp32_tflops < 1000.0);
    CHECK(p.peak_bw_gbs > 100.0);
    CHECK(p.peak_bw_gbs < 10000.0);

    CHECK(dev.stream() != nullptr);
    CHECK(dev.blas() != nullptr);

    const RunCtx ctx =
        dev.make_ctx(/*seed=*/1337, KernelBackend::Naive, PrecisionPolicy::fp32());
    CHECK(ctx.stream == dev.stream());
    CHECK(ctx.blas == dev.blas());
    CHECK(ctx.seed == 1337);
    CHECK(ctx.activations == nullptr);  // model attaches it after planning
    CHECK(ctx.backend == KernelBackend::Naive);
}

TEST_CASE("Rng: device fill_uniform matches host bitwise; runs are identical") {
    Device dev(0);
    constexpr int64_t n = 4096;
    constexpr uint64_t seed = 99, sid = 3;

    Arena a(1 << 20, "rng-test");
    const Tensor d1 = a.alloc({n}, DType::FP32), d2 = a.alloc({n}, DType::FP32);

    rng::fill_uniform(dev.stream(), d1.ptr<float>(), n, seed, sid);
    rng::fill_uniform(dev.stream(), d2.ptr<float>(), n, seed, sid);
    dev.synchronize();

    std::vector<float> h1(n), h2(n);
    CUDA_CHECK(cudaMemcpy(h1.data(), d1.data, d1.bytes(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h2.data(), d2.data, d2.bytes(), cudaMemcpyDeviceToHost));

    // bitwise: device run == device run == host reference (pure integer path)
    CHECK(std::memcmp(h1.data(), h2.data(), n * sizeof(float)) == 0);
    for (int64_t i = 0; i < n; ++i) {
        const float expect = rng::uniform(seed, sid, i);
        REQUIRE(h1[i] == expect);
    }
}

TEST_CASE("Rng: device fill_normal statistics and determinism") {
    Device dev(0);
    constexpr int64_t n = 100000;

    Arena a(1 << 20, "rng-test");
    const Tensor d1 = a.alloc({n}, DType::FP32), d2 = a.alloc({n}, DType::FP32);
    rng::fill_normal(dev.stream(), d1.ptr<float>(), n, 7, 1, 0.0f, 0.02f);
    rng::fill_normal(dev.stream(), d2.ptr<float>(), n, 7, 1, 0.0f, 0.02f);
    dev.synchronize();

    std::vector<float> h1(n), h2(n);
    CUDA_CHECK(cudaMemcpy(h1.data(), d1.data, d1.bytes(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h2.data(), d2.data, d2.bytes(), cudaMemcpyDeviceToHost));
    CHECK(std::memcmp(h1.data(), h2.data(), n * sizeof(float)) == 0);  // bitwise repeatable

    double sum = 0, sq = 0;
    for (const float x : h1) {
        sum += x;
        sq += double(x) * x;
    }
    const double mean = sum / n;
    const double var = sq / n - mean * mean;
    CHECK(std::abs(mean) < 0.001);
    CHECK(var == doctest::Approx(0.02 * 0.02).epsilon(0.03));
}
