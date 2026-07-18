// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Masoud Jami
#pragma once

#ifdef LLMT_HAS_CUDA

#include <cuda_runtime.h>

#include "llmt/core/tensor.h"

namespace llmt::kernels {

/**
 * Fills every element of dst with value (non-zero fills aren't memset-able).
 * T must be the C++ type of dst's dtype — mismatches abort. Instantiated
 * in fill.cu.
 */
template <TensorElement T>
void fill_value(cudaStream_t s, Tensor& dst, T value) noexcept;

}  // namespace llmt::kernels

#endif  // LLMT_HAS_CUDA
