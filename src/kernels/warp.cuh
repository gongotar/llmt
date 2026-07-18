// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Masoud Jami
// Warp-level reduction toolbox — internal to kernels/ (device code only).
// The xor-shuffle butterfly leaves the result in EVERY lane: no shared
// memory, no __syncthreads, no broadcast step.
#pragma once

namespace llmt::kernels {

// Lanes per warp — hardware constant on every NVIDIA architecture to date,
// shared here as fact. Block sizes / warps-per-block are per-kernel tuning
// and stay file-local (knowledge.md "Blocks, SMs, and choosing a block size").
constexpr int kWarpSize = 32;

__device__ inline float warp_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_xor_sync(0xFFFFFFFFu, v, offset);
    return v;
}

__device__ inline float warp_max(float v) {
    for (int offset = 16; offset > 0; offset >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, offset));
    return v;
}

}  // namespace llmt::kernels
