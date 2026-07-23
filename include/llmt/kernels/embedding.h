// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Masoud Jami
#pragma once

#ifdef LLMT_HAS_CUDA

#include <cuda_runtime.h>

#include "llmt/core/tensor.h"

namespace llmt::kernels {

/**
 * Token-embedding lookup: tokens[b][t] holds a token id, and the output
 * row y[b][t] (C channels) becomes a copy of that token's embedding row,
 * wte[tokens[b][t]]. tokens is [B, T] (I32, values in [0, V)); wte is
 * [V, C]; y is [B, T, C] in wte's dtype. A pure row gather — no arithmetic,
 * no policy axis, so no ctx is taken. Token ids are trusted (out-of-range
 * reads are the data pipeline's bug to prevent).
 */
void embedding_fwd(cudaStream_t s, Tensor& y, const Tensor& tokens, const Tensor& wte) noexcept;

}  // namespace llmt::kernels

#endif  // LLMT_HAS_CUDA
