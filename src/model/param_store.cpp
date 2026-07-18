// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Masoud Jami
#ifdef LLMT_HAS_CUDA

#include <utility>
#include <vector>

#include "llmt/core/error.h"
#include "llmt/model/param_store.h"

namespace llmt {

namespace {

constexpr size_t align_up(size_t n) noexcept {
    return (n + Arena::kAlign - 1) / Arena::kAlign * Arena::kAlign;
}

}  // namespace

ParamStore::ParamStore(StateDtypes dtypes) noexcept : m_dtypes(dtypes) {
    for (int k = 0; k < kNumStateKinds; ++k)
        if (dtypes[StateKind(k)] != DType::FP32)
            detail::fatal("ParamStore", "storage dtype %s for state kind %d is not supported",
                          dtype_name(dtypes[StateKind(k)]), k);
}

Param& ParamStore::add(std::string name, const Shape& shape, Role role) noexcept {
    if (finalized()) detail::fatal("ParamStore", "add('%s') after finalize()", name.c_str());
    if (name.empty()) detail::fatal("ParamStore", "empty parameter name");
    if (shape.numel() <= 0) detail::fatal("ParamStore", "add('%s'): empty shape", name.c_str());
    if (m_index.count(name))
        detail::fatal("ParamStore", "duplicate parameter name '%s'", name.c_str());

    Param p;
    p.name = std::move(name);
    p.role = role;
    for (int k = 0; k < kNumStateKinds; ++k)
        p.state(StateKind(k)) = Tensor{nullptr, m_dtypes[StateKind(k)], shape};
    m_params.push_back(std::move(p));
    m_index.emplace(m_params.back().name, m_params.size() - 1);
    m_param_numel += shape.numel();
    return m_params.back();
}

void ParamStore::alias(const std::string& alias_name, const std::string& target) noexcept {
    const auto it = m_index.find(target);
    if (it == m_index.end())
        detail::fatal("ParamStore", "alias target '%s' is not registered", target.c_str());
    if (!m_index.try_emplace(alias_name, it->second).second)
        detail::fatal("ParamStore", "alias name '%s' already exists", alias_name.c_str());
}

void ParamStore::finalize() noexcept {
    if (finalized()) detail::fatal("ParamStore", "finalize() called twice");
    if (m_params.empty()) detail::fatal("ParamStore", "finalize() with no registered parameters");

    // Per-kind packing: each parameter starts 256B-aligned within its span;
    // sizes come from each tensor's own dtype, so kinds may pack differently.
    std::vector<size_t> offsets(kNumStateKinds * m_params.size());
    size_t total = 0;
    for (int k = 0; k < kNumStateKinds; ++k) {
        size_t off = 0;
        for (size_t i = 0; i < m_params.size(); ++i) {
            offsets[k * m_params.size() + i] = off;
            off = align_up(off + m_params[i].state(StateKind(k)).bytes());
        }
        m_span_bytes[k] = off;
        total += off;
    }

    // One device allocation carved into the four spans (see header diagram).
    m_mem.emplace(total, "param-store");
    for (int k = 0; k < kNumStateKinds; ++k) {
        m_base[k] = static_cast<char*>(m_mem->alloc_bytes(m_span_bytes[k]));
        // Zeroing contract (see class comment): spans start zero, and the
        // alignment padding inside them stays zero for the store's lifetime.
        CUDA_CHECK(cudaMemset(m_base[k], 0, m_span_bytes[k]));
    }

    for (int k = 0; k < kNumStateKinds; ++k)
        for (size_t i = 0; i < m_params.size(); ++i)
            m_params[i].state(StateKind(k)).data = m_base[k] + offsets[k * m_params.size() + i];
}

const Param& ParamStore::at(const std::string& name) const noexcept {
    const auto it = m_index.find(name);
    if (it == m_index.end()) detail::fatal("ParamStore", "unknown parameter '%s'", name.c_str());
    return m_params[it->second];
}

Param& ParamStore::at(const std::string& name) noexcept {
    return const_cast<Param&>(std::as_const(*this).at(name));
}

Tensor ParamStore::flat(StateKind k) const noexcept {
    if (!finalized())
        detail::fatal("ParamStore", "flat(%d) requested before finalize()", static_cast<int>(k));
    const int ki = static_cast<int>(k);
    const DType dt = m_dtypes[k];
    return Tensor{m_base[ki], dt, {static_cast<int64_t>(m_span_bytes[ki] / dtype_size(dt))}};
}

}  // namespace llmt

#endif  // LLMT_HAS_CUDA
