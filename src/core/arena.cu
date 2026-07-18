// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Masoud Jami
#include "llmt/core/arena.h"
#include "llmt/core/error.h"

namespace llmt {

Arena::Arena(size_t capacity, const char* name) noexcept : m_capacity(capacity), m_name(name) {
    CUDA_CHECK(cudaMalloc(&m_base, m_capacity));
}

Arena::~Arena() {
    if (m_base) cudaFree(m_base);  // no CUDA_CHECK: destructors run during teardown
}

void* Arena::alloc_bytes(size_t bytes) noexcept {
    const size_t aligned = (m_used + kAlign - 1) / kAlign * kAlign;
    if (aligned + bytes > m_capacity)
        detail::fatal("arena", "'%s' exhausted: request %zu B at offset %zu, capacity %zu B",
                      m_name, bytes, aligned, m_capacity);
    m_used = aligned + bytes;
    if (m_used > m_high_water) m_high_water = m_used;
    return m_base + aligned;
}

Tensor Arena::alloc(const Shape& shape, DType dtype) noexcept {
    Tensor t{nullptr, dtype, shape};
    t.data = alloc_bytes(t.bytes());
    return t;
}

}  // namespace llmt
