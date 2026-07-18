// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Masoud Jami
// CUDA-event timing shared by the benchmarks.
#pragma once

#include <concepts>

#include "llmt/core/device.h"
#include "llmt/core/error.h"

namespace llmt::bench {

// Mean wall time (ms) per call over `iters` timed iterations, after `warmup`
// untimed ones (first calls pay one-time costs — algo caches, module loads,
// clock ramp — that steady-state training never sees).
template <std::invocable F>
double time_ms(const Device& dev, cudaStream_t s, int warmup, int iters, F&& call) {
    for (int i = 0; i < warmup; ++i) call();
    dev.synchronize();

    cudaEvent_t t0 = nullptr, t1 = nullptr;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0, s));
    for (int i = 0; i < iters; ++i) call();
    CUDA_CHECK(cudaEventRecord(t1, s));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    return static_cast<double>(ms) / iters;
}

}  // namespace llmt::bench
