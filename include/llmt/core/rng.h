// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Masoud Jami
// Counter-based RNG (DESIGN invariants 4 & 5): Philox4x32-10.
//
// Stateless: value = f(seed, stream_id, index). Same inputs → same bits,
// on host and device alike (pure integer math), which is what makes init
// reproducible, dropout replayable (M2), and runs bitwise-repeatable.
//
// stream_id decouples consumers (e.g. each parameter tensor gets its own
// stream_id at init, so values don't shift when layouts change).
#pragma once

#include <cmath>
#include <cstdint>

#include "llmt/core/defs.h"

namespace llmt::rng {

/** One 128-bit Philox output block: four independent uniform u32s. */
struct U32x4 {
    uint32_t x, y, z, w;
};

namespace detail {

constexpr uint32_t kPhiloxM0 = 0xD2511F53u;
constexpr uint32_t kPhiloxM1 = 0xCD9E8D57u;
constexpr uint32_t kPhiloxW0 = 0x9E3779B9u;  // golden ratio
constexpr uint32_t kPhiloxW1 = 0xBB67AE85u;  // sqrt(3) - 1

struct U32HiLo {
    uint32_t lo, hi;
};

/** Full 64-bit product of two u32s, split into halves (Philox uses both). */
LLMT_HD constexpr U32HiLo mulhilo(uint32_t a, uint32_t b) noexcept {
    const uint64_t p = static_cast<uint64_t>(a) * b;
    return {static_cast<uint32_t>(p), static_cast<uint32_t>(p >> 32)};
}

/** One Philox round: multiply-based scramble + key XOR + lane swap. */
LLMT_HD constexpr U32x4 philox_round(U32x4 c, uint32_t k0, uint32_t k1) noexcept {
    const auto [lo0, hi0] = mulhilo(kPhiloxM0, c.x);
    const auto [lo1, hi1] = mulhilo(kPhiloxM1, c.z);
    return {hi1 ^ c.y ^ k0, lo1, hi0 ^ c.w ^ k1, lo0};
}

}  // namespace detail

/**
 * Philox4x32-10: encrypts a 128-bit counter under a 64-bit key (10 rounds).
 * @param seed       the key — which universe of randomness (one per run)
 * @param stream_id  counter high half — which consumer (a parameter tensor,
 *                   a dropout site); decouples consumers from memory layout
 * @param index      counter low half — which element within that consumer
 * @return four independent uniform u32s
 */
LLMT_HD constexpr U32x4 philox(uint64_t seed, uint64_t stream_id, uint64_t index) noexcept {
    U32x4 c{static_cast<uint32_t>(index), static_cast<uint32_t>(index >> 32),
            static_cast<uint32_t>(stream_id), static_cast<uint32_t>(stream_id >> 32)};
    uint32_t k0 = static_cast<uint32_t>(seed);
    uint32_t k1 = static_cast<uint32_t>(seed >> 32);
#ifdef __CUDA_ARCH__  // device pass only: nvcc's host pass (g++) doesn't know it
#pragma unroll
#endif
    for (int r = 0; r < 10; ++r) {
        c = detail::philox_round(c, k0, k1);
        k0 += detail::kPhiloxW0;
        k1 += detail::kPhiloxW1;
    }
    return c;
}

/** u32 → float in [0, 1): top 24 bits (float's mantissa width) scaled down. */
LLMT_HD constexpr float to_uniform(uint32_t u) noexcept {
    return static_cast<float>(u >> 8) * (1.0f / 16777216.0f);  // 2^-24
}

/** One uniform float in [0, 1) per (seed, stream_id, index). */
LLMT_HD constexpr float uniform(uint64_t seed, uint64_t stream_id, uint64_t index) noexcept {
    return to_uniform(philox(seed, stream_id, index).x);
}

/**
 * One standard-normal float per (seed, stream_id, index), via Box–Muller.
 * Uses logf/cosf/sqrtf, whose device implementations may differ from host
 * libm by ULPs — bitwise host↔device identity is guaranteed only for
 * philox()/uniform(); normal() is bitwise-reproducible per side.
 */
LLMT_HD inline float normal(uint64_t seed, uint64_t stream_id, uint64_t index) noexcept {
    const U32x4 r = philox(seed, stream_id, index);
    // NOTE: shifts BEFORE the +1 so the result is in (0,1] with no possible
    // u32 wraparound (to_uniform(r.x + 256) would wrap for r.x >= 2^32-256).
    const float u1 = static_cast<float>((r.x >> 8) + 1) * (1.0f / 16777216.0f);
    const float u2 = to_uniform(r.y);
    constexpr float k2Pi = 6.28318530717958648f;
    return sqrtf(-2.0f * logf(u1)) * cosf(k2Pi * u2);
}

}  // namespace llmt::rng

#ifdef LLMT_HAS_CUDA
#include <cuda_runtime.h>

namespace llmt::rng {

/**
 * Fill dst[0..n) on stream s with one RNG value per index. Future consumers:
 * weight init (Phase 3), dropout (M2).
 */
void fill_uniform(cudaStream_t s, float* dst, int64_t n, uint64_t seed,
                  uint64_t stream_id) noexcept;
void fill_normal(cudaStream_t s, float* dst, int64_t n, uint64_t seed, uint64_t stream_id,
                 float mean, float stddev) noexcept;

}  // namespace llmt::rng
#endif
