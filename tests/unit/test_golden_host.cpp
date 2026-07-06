// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Masoud Jami
// Host-side proof of the golden pipeline: loader + allclose + finite-diff
// checker, no GPU needed (runs on the Mac).
#include <cmath>
#include <vector>

#include "doctest.h"
#include "golden_io.h"
#include "finite_diff.h"

using namespace llmt;
using namespace llmt::testing;

TEST_CASE("golden: scale2x loads with correct header facts") {
    const GoldenCase g("scale2x");
    const HostTensor x = g.tensor("x");
    const HostTensor y = g.tensor("y");

    CHECK(x.dtype == DType::FP32);
    CHECK(x.shape == Shape{8, 16});
    CHECK(y.shape == x.shape);
    CHECK(x.bytes.size() == 8 * 16 * 4);

    // y = 2x holds exactly: ×2 only increments the exponent in binary fp.
    const float* xf = x.ptr<float>();
    const float* yf = y.ptr<float>();
    for (int64_t i = 0; i < x.numel(); ++i) REQUIRE(yf[i] == 2.0f * xf[i]);

    // and the data is genuinely random-ish, not zeros (catches empty payloads)
    double sq = 0;
    for (int64_t i = 0; i < x.numel(); ++i) sq += double(xf[i]) * xf[i];
    CHECK(sq / x.numel() == doctest::Approx(1.0).epsilon(0.35));  // randn variance, n=128
}

TEST_CASE("golden: allclose reports failures precisely") {
    const GoldenCase g("scale2x");
    const HostTensor y = g.tensor("y");
    const float* yf = y.ptr<float>();
    const int64_t n = y.numel();

    CHECK(allclose(yf, yf, n, 0.0, 0.0).ok);  // identity, zero tolerance

    std::vector<float> tweaked(yf, yf + n);
    tweaked[37] += 0.5f;
    const CloseReport r = allclose(tweaked.data(), yf, n, 1e-5, 1e-8);
    CHECK(!r.ok);
    CHECK(r.n_bad == 1);
    CHECK(r.worst == 37);
    CHECK(r.max_abs == doctest::Approx(0.5).epsilon(1e-4));
    MESSAGE(to_string(r));  // exercise the reporter formatting

    std::vector<float> poisoned(yf, yf + n);
    poisoned[3] = NAN;
    CHECK(!allclose(poisoned.data(), yf, n, 1.0, 1e9).ok);  // NaN fails any tolerance
}

TEST_CASE("finite-diff: checker validates a known gradient") {
    // f(x) = Σ x², ∇f = 2x — exact analytic gradient must pass…
    std::vector<float> x = {0.5f, -1.25f, 2.0f, 0.1f, -0.7f};
    const auto f = [n = x.size()](const float* v) {
        double s = 0;
        for (size_t i = 0; i < n; ++i) s += double(v[i]) * v[i];
        return s;
    };
    std::vector<float> grad(x.size());
    for (size_t i = 0; i < x.size(); ++i) grad[i] = 2.0f * x[i];
    CHECK(max_grad_rel_error(f, x, grad.data()) < 1e-4);

    // …and a wrong gradient must fail loudly (checker self-test).
    grad[2] *= 1.05f;
    CHECK(max_grad_rel_error(f, x, grad.data()) > 1e-2);
}
