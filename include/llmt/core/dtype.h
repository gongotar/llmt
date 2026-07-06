// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Masoud Jami
#pragma once

#include <cstddef>
#include <cstdint>

namespace llmt {

/**
 * Element type of tensor data. FP32 is all M1 computes in; I32 carries token
 * ids (Phases 6/10); BF16 is declared for the M2 mixed-precision seam but
 * accepted by no kernel yet.
 */
enum class DType : uint8_t { FP32, BF16, I32 };

/** Bytes per element; 0 for out-of-range codes (usable as a validity probe). */
constexpr size_t dtype_size(DType d) noexcept {
    switch (d) {
        case DType::FP32: return 4;
        case DType::BF16: return 2;
        case DType::I32: return 4;
    }
    return 0;  // unreachable
}

/** Short lowercase name ("fp32") for logs, reports and file formats. */
constexpr const char* dtype_name(DType d) noexcept {
    switch (d) {
        case DType::FP32: return "fp32";
        case DType::BF16: return "bf16";
        case DType::I32: return "i32";
    }
    return "?";
}

/**
 * Compile-time C++ type ↔ DType pairing. Deliberately has no primary
 * definition: ptr<T>() with an unmapped T is a compile error.
 */
template <typename T>
struct dtype_of;
template <>
struct dtype_of<float> {
    static constexpr DType value = DType::FP32;
};
template <>
struct dtype_of<int32_t> {
    static constexpr DType value = DType::I32;
};
// bf16's C++ type (__nv_bfloat16) gets its specialization in M2.

}  // namespace llmt
