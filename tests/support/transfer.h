// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Masoud Jami
// Host↔device tensor transfer helpers shared by GPU tests.
#pragma once

#ifdef LLMT_HAS_CUDA

#include <cuda_runtime.h>

#include <vector>

#include "golden_io.h"
#include "llmt/core/arena.h"
#include "llmt/core/error.h"

namespace llmt::testing {

// Allocates a same-shaped device tensor from the arena and copies the golden
// bytes into it (stream-ordered).
inline Tensor upload(Arena& arena, const HostTensor& h, cudaStream_t s) {
    Tensor t = arena.alloc(h.shape, h.dtype);
    CUDA_CHECK(cudaMemcpyAsync(t.data, h.bytes.data(), h.bytes.size(), cudaMemcpyHostToDevice, s));
    return t;
}

// Copies a device tensor's floats to the host (synchronous).
inline std::vector<float> download(const Tensor& t) {
    std::vector<float> h(t.numel());
    CUDA_CHECK(cudaMemcpy(h.data(), t.data, t.bytes(), cudaMemcpyDeviceToHost));
    return h;
}

}  // namespace llmt::testing

#endif  // LLMT_HAS_CUDA
