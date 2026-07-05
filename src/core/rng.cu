#include "llmt/core/error.h"
#include "llmt/core/rng.h"

namespace llmt::rng {

namespace {

__global__ void fill_uniform_kernel(float* dst, int64_t n, uint64_t seed, uint64_t stream_id) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = uniform(seed, stream_id, static_cast<uint64_t>(i));
}

__global__ void fill_normal_kernel(float* dst, int64_t n, uint64_t seed, uint64_t stream_id,
                                   float mean, float stddev) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = mean + stddev * normal(seed, stream_id, static_cast<uint64_t>(i));
}

constexpr int kBlock = 256;
constexpr unsigned grid_for(int64_t n) noexcept {
    return static_cast<unsigned>((n + kBlock - 1) / kBlock);
}

}  // namespace

void fill_uniform(cudaStream_t s, float* dst, int64_t n, uint64_t seed,
                  uint64_t stream_id) noexcept {
    if (n <= 0) return;
    fill_uniform_kernel<<<grid_for(n), kBlock, 0, s>>>(dst, n, seed, stream_id);
    CUDA_CHECK_LAST();
}

void fill_normal(cudaStream_t s, float* dst, int64_t n, uint64_t seed, uint64_t stream_id,
                 float mean, float stddev) noexcept {
    if (n <= 0) return;
    fill_normal_kernel<<<grid_for(n), kBlock, 0, s>>>(dst, n, seed, stream_id, mean, stddev);
    CUDA_CHECK_LAST();
}

}  // namespace llmt::rng
