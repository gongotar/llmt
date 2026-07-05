#pragma once

#include <cassert>
#include <cstdint>
#include <initializer_list>

#include "llmt/core/defs.h"

namespace llmt {

// Fixed-capacity shape: rank <= 4 covers every tensor a decoder-only
// transformer needs (see docs/knowledge.md). A stack value type — copyable
// straight into kernel arguments, no heap.
struct Shape {
    static constexpr int kMaxRank = 4;

    int64_t d[kMaxRank] = {0, 0, 0, 0};
    int rank = 0;

    Shape() = default;
    Shape(std::initializer_list<int64_t> dims) noexcept {
        assert(dims.size() <= kMaxRank);
        for (const int64_t v : dims) d[rank++] = v;
    }

    LLMT_HD int64_t operator[](int i) const noexcept {
        assert(i >= 0 && i < rank);
        return d[i];
    }

    // Number of elements: product of dims. Rank 0 (default-constructed,
    // "no shape yet") deliberately yields 0 so uninitialized use is loud.
    LLMT_HD int64_t numel() const noexcept {
        if (rank == 0) return 0;
        int64_t n = 1;
        for (int i = 0; i < rank; ++i) n *= d[i];
        return n;
    }

    LLMT_HD bool operator==(const Shape& o) const noexcept {
        if (rank != o.rank) return false;
        for (int i = 0; i < rank; ++i)
            if (d[i] != o.d[i]) return false;
        return true;
    }
    LLMT_HD bool operator!=(const Shape& o) const noexcept { return !(*this == o); }
};

}  // namespace llmt
