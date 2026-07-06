// End-to-end proof of the golden pipeline on the GPU (task 2.6):
// PyTorch wrote x and y=2x → our kernel computes 2x on device → allclose.
// Also exercises the determinism harness against the Rng fills.
#include <vector>

#include "doctest.h"
#include "determinism.h"
#include "golden_io.h"
#include "llmt/core/arena.h"
#include "llmt/core/device.h"
#include "llmt/core/error.h"
#include "llmt/core/rng.h"

using namespace llmt;
using namespace llmt::testing;

namespace {
__global__ void scale2_kernel(float* y, const float* x, int64_t n) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) y[i] = 2.0f * x[i];
}
}  // namespace

TEST_CASE("golden e2e: device kernel vs PyTorch oracle (scale2x)") {
    const GoldenCase g("scale2x");
    const HostTensor x = g.tensor("x");
    const HostTensor y = g.tensor("y");
    const int64_t n = x.numel();

    Device dev(0);
    Arena arena(1 << 20, "golden-test");
    const Tensor dx = arena.alloc(x.shape, x.dtype);
    const Tensor dy = arena.alloc(y.shape, y.dtype);

    CUDA_CHECK(cudaMemcpyAsync(dx.data, x.bytes.data(), x.bytes.size(), cudaMemcpyHostToDevice,
                               dev.stream()));
    scale2_kernel<<<(n + 255) / 256, 256, 0, dev.stream()>>>(dy.ptr<float>(), dx.ptr<float>(), n);
    CUDA_CHECK_LAST();
    dev.synchronize();

    std::vector<float> result(n);
    CUDA_CHECK(cudaMemcpy(result.data(), dy.data, dy.bytes(), cudaMemcpyDeviceToHost));

    const CloseReport r = allclose(result.data(), y.ptr<float>(), n, 0.0, 0.0);  // ×2 is exact
    CHECK_MESSAGE(r.ok, to_string(r));
}

TEST_CASE("determinism harness: rng fills are bitwise repeatable") {
    Device dev(0);
    Arena arena(1 << 20, "det-test");
    constexpr int64_t n = 10000;

    const int64_t div_u =
        first_divergence(dev, arena, n * sizeof(float), [](cudaStream_t s, void* dst) {
            rng::fill_uniform(s, static_cast<float*>(dst), n, 42, 9);
        });
    CHECK(div_u == -1);

    const int64_t div_n =
        first_divergence(dev, arena, n * sizeof(float), [](cudaStream_t s, void* dst) {
            rng::fill_normal(s, static_cast<float*>(dst), n, 42, 9, 0.0f, 1.0f);
        });
    CHECK(div_n == -1);
}
