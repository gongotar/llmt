#include "golden_io.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>

#ifndef LLMT_GOLDEN_DIR
#error "LLMT_GOLDEN_DIR must be defined by the build (tests/CMakeLists.txt)"
#endif

namespace llmt::testing {

namespace {

constexpr uint32_t kMagic = 0x544D4C4C;  // "LLMT"
constexpr uint32_t kVersion = 1;

[[noreturn]] void die(const std::string& path, const char* why) noexcept {
    std::fprintf(stderr, "[llmt-test] golden file '%s': %s\n", path.c_str(), why);
    std::abort();
}

}  // namespace

HostTensor load_tensor(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) die(path, "cannot open (regenerate with tools/gen_golden.py?)");

    uint32_t head[4] = {};
    f.read(reinterpret_cast<char*>(head), sizeof(head));
    if (!f) die(path, "truncated header");
    if (head[0] != kMagic) die(path, "bad magic (not a golden tensor file)");
    if (head[1] != kVersion) die(path, "unsupported format version");
    // Validity is derived from dtype_size() (0 for unknown codes) — its switch
    // is kept exhaustive by -Wall, so this needs no update when DType grows.
    // (The range check guards the u32→u8 enum cast from truncating, e.g. 256→0.)
    if (head[2] > 0xFF || dtype_size(static_cast<DType>(head[2])) == 0)
        die(path, "unknown dtype code");
    if (head[3] > static_cast<uint32_t>(Shape::kMaxRank)) die(path, "rank > 4");

    HostTensor t;
    t.dtype = static_cast<DType>(head[2]);
    t.shape.rank = static_cast<int>(head[3]);
    f.read(reinterpret_cast<char*>(t.shape.d),
           t.shape.rank * static_cast<std::streamsize>(sizeof(int64_t)));
    if (!f) die(path, "truncated dims");

    const size_t payload = static_cast<size_t>(t.numel()) * dtype_size(t.dtype);
    t.bytes.resize(payload);
    f.read(t.bytes.data(), static_cast<std::streamsize>(payload));
    if (!f) die(path, "truncated payload");
    return t;
}

GoldenCase::GoldenCase(std::string name) noexcept
    : m_dir(std::string(LLMT_GOLDEN_DIR) + "/" + std::move(name)) {}

HostTensor GoldenCase::tensor(const std::string& name) const {
    return load_tensor(m_dir + "/" + name + ".bin");
}

CloseReport allclose(const float* actual, const float* expected, int64_t n, double rtol,
                     double atol) noexcept {
    CloseReport r;
    r.n = n;
    for (int64_t i = 0; i < n; ++i) {
        const double a = actual[i];
        const double e = expected[i];
        const bool nan_involved = std::isnan(a) || std::isnan(e);
        const double abs_diff = nan_involved ? INFINITY : std::abs(a - e);
        if (abs_diff > atol + rtol * std::abs(e)) {
            ++r.n_bad;
            if (abs_diff > r.max_abs || r.worst < 0) {
                r.max_abs = abs_diff;
                r.max_rel = abs_diff / std::max(std::abs(e), 1e-12);
                r.worst = i;
                r.actual_at_worst = actual[i];
                r.expected_at_worst = expected[i];
            }
        }
    }
    r.ok = (r.n_bad == 0);
    return r;
}

std::string to_string(const CloseReport& r) {
    if (r.ok) return "allclose: ok (" + std::to_string(r.n) + " elements)";
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "allclose: %lld/%lld bad; worst at [%lld]: actual=%g expected=%g "
                  "(abs=%g, rel=%g)",
                  static_cast<long long>(r.n_bad), static_cast<long long>(r.n),
                  static_cast<long long>(r.worst), static_cast<double>(r.actual_at_worst),
                  static_cast<double>(r.expected_at_worst), r.max_abs, r.max_rel);
    return buf;
}

}  // namespace llmt::testing
