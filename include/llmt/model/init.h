// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Masoud Jami
#pragma once

#ifdef LLMT_HAS_CUDA

#include <cuda_runtime.h>

#include <cstdint>

#include "llmt/model/param_store.h"

namespace llmt {

/**
 * Role-keyed weight initialization (GPT-2 scheme):
 *   Embedding, Matrix  → normal(0, 0.02)
 *   ResidualProj       → normal(0, 0.02 / sqrt(2·n_layer)) — keeps the
 *                        residual stream's variance flat across depth
 *   Norm               → ones
 * Values are drawn per parameter from (seed, stream_id=fnv1a(name), index),
 * so init is bitwise-reproducible and independent of registration order or
 * memory layout. Requires a finalized store.
 */
void init_params(const ParamStore& store, cudaStream_t s, uint64_t seed, int n_layer) noexcept;

}  // namespace llmt

#endif  // LLMT_HAS_CUDA
