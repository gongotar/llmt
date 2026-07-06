// Central-difference gradient checker (task 2.5). Host-side; used from
// Phase 4 onward to sanity-check analytic backward passes on micro shapes
// (catches oracle-script bugs that golden tests alone would miss).
#pragma once

#include <cmath>
#include <concepts>
#include <cstdint>
#include <vector>

namespace llmt::testing {

// A scalar field: reads n floats, yields one scalar "loss" value.
template <typename F>
concept ScalarField = requires(F f, const float* x) {
    { f(x) } -> std::convertible_to<double>;
};

// Compares the analytic gradient against (f(x+h·e_i) − f(x−h·e_i)) / 2h
// per coordinate; returns the maximum relative error across coordinates.
// fp32 inputs + double loss ⇒ h around 1e-3 is the sweet spot (truncation
// vs rounding); expect agreement to ~1e-3..1e-4 rel on smooth functions.
template <ScalarField F>
double max_grad_rel_error(F&& f, std::vector<float> x, const float* analytic_grad,
                          double h = 1e-3) {
    const int64_t n = static_cast<int64_t>(x.size());
    double worst = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const float xi = x[i];
        x[i] = static_cast<float>(xi + h);
        const double fp = f(x.data());
        x[i] = static_cast<float>(xi - h);
        const double fm = f(x.data());
        x[i] = xi;

        const double numeric = (fp - fm) / (2.0 * h);
        const double analytic = analytic_grad[i];
        const double scale = std::max({std::abs(numeric), std::abs(analytic), 1e-8});
        worst = std::max(worst, std::abs(numeric - analytic) / scale);
    }
    return worst;
}

}  // namespace llmt::testing
