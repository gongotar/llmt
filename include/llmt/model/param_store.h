// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Masoud Jami
#pragma once

#ifdef LLMT_HAS_CUDA

#include <cstdint>
#include <deque>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

#include "llmt/core/arena.h"
#include "llmt/core/tensor.h"

namespace llmt {

/**
 * Parameter role. Drives the init scheme (init_params) and the optimizer's
 * param groups (weight decay applies to Matrix/ResidualProj only).
 * ResidualProj marks projections writing into the residual stream (attention
 * output, MLP down-projection).
 */
enum class Role : uint8_t { Embedding, Matrix, ResidualProj, Norm };

namespace detail {

/**
 * FNV-1a 64-bit name hash, used as a parameter's RNG stream_id: init values
 * depend on the parameter's identity, not on registration order or memory
 * layout, so adding/reordering parameters never shifts anyone's randomness.
 */
constexpr uint64_t fnv1a(std::string_view s) noexcept {
    uint64_t h = 0xCBF29CE484222325ull;
    for (const char c : s) {
        h ^= static_cast<unsigned char>(c);
        h *= 0x100000001B3ull;
    }
    return h;
}

}  // namespace detail

/**
 * The four twin spans every parameter has a slice in. Fixed by the training
 * algorithm itself (weights, their gradients, Adam's two moments) — a closed
 * set, hence an enum.
 */
enum class StateKind : uint8_t { Weight, Grad, AdamM, AdamV };
constexpr int kNumStateKinds = 4;

/**
 * Storage dtype per state kind — the store's precision policy, decided by
 * the caller and injected whole.
 */
struct StateDtypes {
    DType of[kNumStateKinds] = {};

    static constexpr StateDtypes uniform(DType d) noexcept {
        StateDtypes s;
        for (DType& x : s.of) x = d;
        return s;
    }
    constexpr DType operator[](StateKind k) const noexcept {
        return of[static_cast<int>(k)];
    }
};

/**
 * One registered parameter: one self-describing Tensor per state kind, all
 * filled by ParamStore::finalize() in a single loop (shape/dtype set at
 * ParamStore::add(); data pointers null until finalize). Each kind carries
 * its own shape+dtype deliberately: their current equality is a coincidence,
 * not an invariant — kinds may diverge in dtype (mixed precision) or shape
 * (optimizers with factored/absent moments).
 */
struct Param {
    std::string name;
    Role role = Role::Matrix;
    Tensor weight;  ///< slice of the Weight span
    Tensor grad;    ///< slice of the Grad span; backward ACCUMULATES (+=)
    Tensor adam_m;  ///< slice of the AdamM span (optimizer's 1st moment)
    Tensor adam_v;  ///< slice of the AdamV span (optimizer's 2nd moment)

    /** Kind-generic access for loops (tests, checkpoint-resume); the named
     *  fields are the same objects. */
    const Tensor& state(StateKind k) const noexcept {
        switch (k) {
            case StateKind::Weight: return weight;
            case StateKind::Grad: return grad;
            case StateKind::AdamM: return adam_m;
            case StateKind::AdamV: return adam_v;
        }
        return weight;  // unreachable
    }
    Tensor& state(StateKind k) noexcept {
        return const_cast<Tensor&>(std::as_const(*this).state(k));
    }

    /** RNG stream identity for init — derived from the name on demand, so it
     *  can never fall out of sync with it. */
    uint64_t stream_id() const noexcept { return detail::fnv1a(name); }
};

/**
 * Flat parameter storage (DESIGN invariant 1).
 *
 * Memory layout — ONE device allocation, carved into four spans, each
 * packing every parameter in registration order. Spans whose kinds share a
 * dtype are LAYOUT TWINS — a parameter's slice sits at the same byte offset
 * in both (with one dtype for all kinds, as now, all four are twins).
 * Example with two parameters, wte (512 B) and g (256 B), base 0x10000:
 *
 *   0x10000 ┌──────────────────────┐ ◄─ Weight span   flat(Weight)
 *           │ wte weights  (512 B) │    ← wte.weight.data
 *   0x10200 │ g weights    (256 B) │    ← g.weight.data
 *   0x10300 ├──────────────────────┤ ◄─ Grad span     flat(Grad)
 *           │ wte grads    (512 B) │
 *   0x10500 │ g grads      (256 B) │
 *   0x10600 ├──────────────────────┤ ◄─ AdamM span    flat(AdamM)
 *           │ wte m │ g m          │
 *   0x10900 ├──────────────────────┤ ◄─ AdamV span    flat(AdamV)
 *           │ wte v │ g v          │
 *   0x10C00 └──────────────────────┘
 *
 * g's slice is 512 B past the start of EVERY span — one offset, four homes
 * (Param carries the four resulting Tensors; a byte offset, where needed,
 * is pointer minus span base).
 * Layers use per-parameter views (layout-neutral); the layout exists for the
 * whole-model per-step operations, which see each kind as ONE contiguous
 * vector: fused optimizer step, global grad-norm, zero_grad, checkpoint
 * writes, future NCCL — one kernel/reduction/memset/transfer each instead
 * of one per tensor.
 *
 * Lifecycle: add()/alias() during model construction → finalize() once →
 * view data pointers valid, registration closed.
 *
 * Zeroing contract: finalize() zero-fills all four spans. Alignment padding
 * inside the spans must STAY zero — flat-span reductions read it, and zeros
 * are inert there (invariant 5). Gradients: backward accumulates (+=) into
 * grad views; zeroing between steps is the optimizer's job (zero_grad).
 *
 * Weight tying (invariant 9): alias() maps a second name onto an existing
 * parameter — one span, two consumers. The shared grad view then receives
 * one (+=) contribution from EACH consumer's backward per step; correctness
 * requires accumulation, never overwrite.
 */
class ParamStore {
   public:
    /** dtypes: storage dtype of each state kind (policy-driven, invariant 3).
     *  Aborts if a requested dtype is not supported by the kernels. */
    explicit ParamStore(StateDtypes dtypes) noexcept;

    /**
     * Registers a parameter; aborts on duplicate names or after finalize().
     * The returned reference is stable for the store's lifetime; its views
     * are filled in by finalize().
     */
    Param& add(std::string name, const Shape& shape, Role role) noexcept;

    /** Maps alias_name onto an existing parameter (weight tying — see class
     *  comment). No storage is added; both names resolve to the same Param. */
    void alias(const std::string& alias_name, const std::string& target) noexcept;

    /** Computes offsets, makes the single device allocation, zero-fills all
     *  spans and fills every Param's views. Registration is closed afterwards. */
    void finalize() noexcept;

    bool finalized() const noexcept { return m_mem.has_value(); }

    /** Lookup by name — real or alias; aborts on unknown names. */
    Param& at(const std::string& name) noexcept;
    const Param& at(const std::string& name) const noexcept;

    /** Unique parameters in registration order (aliases add no entries). */
    const std::deque<Param>& params() const noexcept { return m_params; }

    /** Sum of real elements across parameters (excludes alignment padding). */
    int64_t param_numel() const noexcept { return m_param_numel; }

    /** Whole-model view of one state kind (valid after finalize). numel()
     *  includes alignment padding — zero and inert by the zeroing contract. */
    Tensor flat(StateKind k) const noexcept;

   private:
    StateDtypes m_dtypes;
    std::deque<Param> m_params;                       // stable references
    std::unordered_map<std::string, size_t> m_index;  // real + alias names
    int64_t m_param_numel = 0;
    size_t m_span_bytes[kNumStateKinds] = {};         // per-kind span size incl. padding
    char* m_base[kNumStateKinds] = {};                // span bases — indexed by StateKind
    std::optional<Arena> m_mem;                       // engaged by finalize()
};

}  // namespace llmt

#endif  // LLMT_HAS_CUDA
