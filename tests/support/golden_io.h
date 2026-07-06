// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Masoud Jami
// Golden-file loading + numeric comparison (tasks 2.3). Test infrastructure —
// lives under tests/, never ships. Format: docs/golden.md.
#pragma once

#include <cassert>
#include <cstdint>
#include <string>
#include <vector>

#include "llmt/core/dtype.h"
#include "llmt/core/shape.h"

namespace llmt::testing {

// Host-side tensor owned by a byte buffer (unlike llmt::Tensor, which is a
// non-owning device view).
struct HostTensor {
    DType dtype = DType::FP32;
    Shape shape;
    std::vector<char> bytes;

    int64_t numel() const noexcept { return shape.numel(); }

    template <typename T>
    const T* ptr() const noexcept {
        assert(dtype == dtype_of<T>::value);
        return reinterpret_cast<const T*>(bytes.data());
    }
};

// Aborts with a clear message on missing/malformed files: this is test
// infrastructure — failing loudly beats failing subtly.
HostTensor load_tensor(const std::string& path);

// One golden case directory: resolves <LLMT_GOLDEN_DIR>/<case>/<tensor>.bin.
class GoldenCase {
   public:
    explicit GoldenCase(std::string name) noexcept;
    HostTensor tensor(const std::string& name) const;
    const std::string& dir() const noexcept { return m_dir; }

   private:
    std::string m_dir;
};

// numpy.allclose semantics: element is bad when |a - e| > atol + rtol·|e|,
// and any NaN on either side is bad (no NaN equality).
struct CloseReport {
    bool ok = true;
    int64_t n = 0;
    int64_t n_bad = 0;
    double max_abs = 0.0;   // worst absolute difference
    double max_rel = 0.0;   // relative difference at the worst element
    int64_t worst = -1;     // index of the worst element
    float actual_at_worst = 0.0f;
    float expected_at_worst = 0.0f;
};

CloseReport allclose(const float* actual, const float* expected, int64_t n, double rtol,
                     double atol) noexcept;

// One-line human summary for CHECK_MESSAGE.
std::string to_string(const CloseReport& r);

}  // namespace llmt::testing
