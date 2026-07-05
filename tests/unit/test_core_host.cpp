// Host-side core tests: run on the Mac too (no CUDA needed).
#include <cmath>

#include "doctest.h"
#include "llmt/core/rng.h"
#include "llmt/core/shape.h"
#include "llmt/core/tensor.h"

using namespace llmt;

TEST_CASE("Shape: rank, numel, equality") {
    const Shape s0;
    CHECK(s0.rank == 0);
    CHECK(s0.numel() == 0);  // uninitialized shape is loud, not silently scalar

    const Shape s{32, 256, 384};
    CHECK(s.rank == 3);
    CHECK(s.numel() == 32 * 256 * 384);
    CHECK(s[2] == 384);

    CHECK(s == Shape{32, 256, 384});
    CHECK(s != Shape{32, 256});
    CHECK(s != Shape{32, 256, 385});

    const Shape r4{32, 6, 256, 64};
    CHECK(r4.rank == 4);
    CHECK(r4.numel() == 32ll * 6 * 256 * 64);
}

// dtype_of traits are compile-time facts.
static_assert(dtype_of<float>::value == DType::FP32);
static_assert(dtype_of<int32_t>::value == DType::I32);
static_assert(dtype_size(DType::I32) == 4);

TEST_CASE("Tensor: bytes, dtype, ptr") {
    Tensor t{nullptr, DType::FP32, {8, 4}};
    CHECK(t.numel() == 32);
    CHECK(t.bytes() == 32 * 4);
    CHECK(!t.valid());

    float dummy;
    t.data = &dummy;
    CHECK(t.valid());
    CHECK(t.ptr<float>() == &dummy);

    const Tensor b{&dummy, DType::BF16, {8, 4}};
    CHECK(b.bytes() == 32 * 2);
}

// The integer Philox chain is constexpr: these run at COMPILE time —
// the binary doesn't build unless the RNG is deterministic and in-range.
static_assert(rng::philox(1337, 0, 0).x == rng::philox(1337, 0, 0).x);
static_assert(rng::philox(1, 0, 0).x != rng::philox(2, 0, 0).x);
static_assert(rng::uniform(42, 7, 0) >= 0.0f && rng::uniform(42, 7, 0) < 1.0f);

TEST_CASE("Rng: philox is deterministic and input-sensitive") {
    const auto a = rng::philox(1337, 0, 0);
    const auto b = rng::philox(1337, 0, 0);
    CHECK(a.x == b.x);
    CHECK(a.y == b.y);
    CHECK(a.z == b.z);
    CHECK(a.w == b.w);

    // any input change → different output
    CHECK(rng::philox(1338, 0, 0).x != a.x);
    CHECK(rng::philox(1337, 1, 0).x != a.x);
    CHECK(rng::philox(1337, 0, 1).x != a.x);
}

TEST_CASE("Rng: uniform statistics") {
    constexpr int n = 100000;
    double sum = 0, sq = 0;
    for (int i = 0; i < n; ++i) {
        const float u = rng::uniform(42, 7, i);
        CHECK(u >= 0.0f);
        CHECK(u < 1.0f);
        sum += u;
        sq += double(u) * u;
    }
    const double mean = sum / n;
    const double var = sq / n - mean * mean;
    CHECK(mean == doctest::Approx(0.5).epsilon(0.01));       // expect 1/2
    CHECK(var == doctest::Approx(1.0 / 12).epsilon(0.02));   // expect 1/12
}

TEST_CASE("Rng: normal statistics") {
    constexpr int n = 100000;
    double sum = 0, sq = 0;
    for (int i = 0; i < n; ++i) {
        const float x = rng::normal(42, 7, i);
        sum += x;
        sq += double(x) * x;
    }
    const double mean = sum / n;
    const double var = sq / n - mean * mean;
    CHECK(std::abs(mean) < 0.02);
    CHECK(var == doctest::Approx(1.0).epsilon(0.02));
}
