// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Masoud Jami
#pragma once

#ifdef LLMT_HAS_CUDA

#include <cublasLt.h>

#include <cstddef>
#include <cstdint>
#include <unordered_map>

namespace llmt::kernels {

/** Identity of one GEMM problem — everything an algorithm choice depends on
 *  (dims, transposes, storage dtype, compute mode, scale type). All-int64,
 *  no padding — bytewise hashable. */
struct AlgoKey {
    int64_t m, n, k, batch, flags, dtype, compute, scale;
    bool operator==(const AlgoKey&) const = default;
};

struct AlgoKeyHash {
    size_t operator()(const AlgoKey& key) const noexcept {
        const auto* p = reinterpret_cast<const unsigned char*>(&key);
        uint64_t h = 0xCBF29CE484222325ull;  // FNV-1a
        for (size_t i = 0; i < sizeof key; ++i) {
            h ^= p[i];
            h *= 0x100000001B3ull;
        }
        return static_cast<size_t>(h);
    }
};

/**
 * Per-device cache of resolved GEMM algorithms (invariant 5: the heuristic
 * is consulted once per problem, then replayed). A plain value type — owned
 * by Device as a member, observed by RunCtx, filled by kernels::matmul.
 */
struct AlgoCache {
    std::unordered_map<AlgoKey, cublasLtMatmulAlgo_t, AlgoKeyHash> map;
};

}  // namespace llmt::kernels

#endif  // LLMT_HAS_CUDA
