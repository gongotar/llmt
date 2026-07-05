#pragma once

#include <cassert>

#include "llmt/core/dtype.h"
#include "llmt/core/shape.h"

namespace llmt {

// Non-owning view of contiguous, row-major device memory (DESIGN §7.2).
// No strides, no autograd, no ownership — memory belongs to an Arena or
// the ParamStore. Deliberately a dumb value type.
struct Tensor {
    void* data = nullptr;
    DType dtype = DType::FP32;
    Shape shape;

    int64_t numel() const noexcept { return shape.numel(); }
    size_t bytes() const noexcept { return static_cast<size_t>(numel()) * dtype_size(dtype); }
    bool valid() const noexcept { return data != nullptr && shape.rank > 0; }

    // Typed accessor for kernel launches: t.ptr<float>().
    // Debug builds abort on a T/dtype mismatch; unmapped T is a compile error.
    template <typename T>
    T* ptr() const noexcept {
        assert(dtype == dtype_of<T>::value);
        return static_cast<T*>(data);
    }
};

}  // namespace llmt
