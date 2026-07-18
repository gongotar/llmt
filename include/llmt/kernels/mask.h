// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Masoud Jami
#pragma once

#ifdef LLMT_HAS_CUDA

#include <cstdint>

namespace llmt::kernels {

/**
 * Which attention-score entries are valid. A parameter of the score path
 * (softmax, and later the attention op) — never baked into a kernel, so
 * mask variants stay swappable (DESIGN §5 seams).
 */
enum class MaskKind : uint8_t {
    None,   ///< every entry valid
    Causal  ///< query q may see keys k <= q (requires square score matrices)
};

struct MaskSpec {
    MaskKind kind = MaskKind::None;
};

}  // namespace llmt::kernels

#endif  // LLMT_HAS_CUDA
