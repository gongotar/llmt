// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Masoud Jami
// Phase 0 GPU smoke test: device visible, kernel launches, D2H roundtrip works.
#include "doctest.h"
#include "llmt/core/error.h"

#include <vector>

namespace {
__global__ void scale_kernel(float* x, float a, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= a;
}
}  // namespace

TEST_CASE("GPU smoke: device present, kernel launch, memcpy roundtrip") {
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    REQUIRE(count > 0);

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    MESSAGE("device: ", prop.name, " sm_", prop.major, prop.minor);

    constexpr int n = 1024;
    std::vector<float> h(n, 3.0f);
    float* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    scale_kernel<<<(n + 255) / 256, 256>>>(d, 2.0f, n);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemcpy(h.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d));

    for (int i = 0; i < n; ++i) REQUIRE(h[i] == 6.0f);
}
