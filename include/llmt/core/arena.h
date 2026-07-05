#pragma once

#ifdef LLMT_HAS_CUDA

#include <cstddef>

#include "llmt/core/tensor.h"

namespace llmt {

// Bump allocator over one big cudaMalloc (DESIGN invariant 2): all
// steady-state memory comes from two Arena instances — persistent
// (params/optimizer state; never reset) and activations (re-planned/reset
// wholesale when the layout changes, e.g. train vs eval plans). One arena
// per lifetime class. No cudaMalloc in the training loop, ever. Allocation
// is O(1) pointer arithmetic; individual frees don't exist on purpose.
class Arena {
   public:
    static constexpr size_t kAlign = 256;  // safe for all CUDA access patterns

    Arena(size_t capacity, const char* name = "arena") noexcept;
    ~Arena();
    Arena(const Arena&) = delete;
    Arena& operator=(const Arena&) = delete;

    // Aborts with a clear message on exhaustion — capacity is planned
    // up front (ActivationPlanner); running out is a plan bug, not a
    // recoverable condition.
    void* alloc_bytes(size_t bytes) noexcept;
    // dtype is deliberately explicit (no default): in M2, buffer dtypes come
    // from PrecisionPolicy — a silent FP32 default would hide forgotten queries.
    Tensor alloc(const Shape& shape, DType dtype) noexcept;

    void reset() noexcept { m_used = 0; }

    size_t used() const noexcept { return m_used; }
    size_t capacity() const noexcept { return m_capacity; }
    size_t high_water() const noexcept { return m_high_water; }
    const char* name() const noexcept { return m_name; }

   private:
    char* m_base = nullptr;
    size_t m_capacity = 0;
    size_t m_used = 0;
    size_t m_high_water = 0;
    const char* m_name;
};

}  // namespace llmt

#endif  // LLMT_HAS_CUDA
