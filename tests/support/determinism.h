// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Masoud Jami
// Bitwise-determinism harness (task 2.4): run the same GPU work twice into
// fresh buffers and locate the first diverging byte, if any.
#pragma once

#ifdef LLMT_HAS_CUDA

#include <concepts>
#include <cstring>
#include <vector>

#include "llmt/core/arena.h"
#include "llmt/core/device.h"
#include "llmt/core/error.h"

namespace llmt::testing {

// `fill(stream, dst)` must enqueue identical work on the given stream for
// both calls (it receives a different destination each time). Returns the
// byte offset of the first divergence, or -1 if bitwise identical.
template <std::invocable<cudaStream_t, void*> F>
int64_t first_divergence(const Device& dev, Arena& arena, size_t bytes, F&& fill) {
    void* const b1 = arena.alloc_bytes(bytes);
    void* const b2 = arena.alloc_bytes(bytes);
    fill(dev.stream(), b1);
    fill(dev.stream(), b2);

    // Copies are queued on the same stream as the fills (ordered after them);
    // one sync point makes everything host-visible.
    std::vector<char> h1(bytes), h2(bytes);
    CUDA_CHECK(cudaMemcpyAsync(h1.data(), b1, bytes, cudaMemcpyDeviceToHost, dev.stream()));
    CUDA_CHECK(cudaMemcpyAsync(h2.data(), b2, bytes, cudaMemcpyDeviceToHost, dev.stream()));
    dev.synchronize();

    if (std::memcmp(h1.data(), h2.data(), bytes) == 0) return -1;
    for (size_t i = 0; i < bytes; ++i)
        if (h1[i] != h2[i]) return static_cast<int64_t>(i);
    return -1;  // unreachable
}

}  // namespace llmt::testing

#endif  // LLMT_HAS_CUDA
