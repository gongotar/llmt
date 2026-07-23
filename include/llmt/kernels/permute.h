// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Masoud Jami
#pragma once

#ifdef LLMT_HAS_CUDA

#include <cuda_runtime.h>

#include "llmt/core/tensor.h"

namespace llmt::kernels {

// Where these two kernels sit in an attention layer — split enters
// per-head land, merge exits it (running as attention_fwd's last stage):
//
//   x [B,T,C] → Linear(qkv) → split → rope → attention_fwd → Linear(Wo)
//               token rows   ┕━ per-head land [B,H,T,hd] ━┙  token rows
//
// The backward pass mirrors the roles: same kernels, opposite directions.

/**
 * Head split after the fused QKV projection. With H = n_head and
 * A = H · hd (the attention inner width; hd is derived from the tensors):
 *   input:   qkv  [B, T, 3A]           (per token: Q part | K part | V part)
 *   outputs: q, k, v  each [B, H, T, hd]
 * Element-wise — head h's slice of each part of the token row:
 *   q[b, h, t, d] = qkv[b, t,      h·hd + d]
 *   k[b, h, t, d] = qkv[b, t,  A + h·hd + d]
 *   v[b, h, t, d] = qkv[b, t, 2A + h·hd + d]
 * Re-shelves per-token rows into per-head [T, hd] blocks — the layout the
 * batched attention GEMMs consume. Pure data movement — no arithmetic, no
 * policy axis, so no ctx. Outputs must not alias qkv (physical reorder,
 * not a view).
 */
void permute_split(cudaStream_t s, Tensor& q, Tensor& k, Tensor& v, const Tensor& qkv,
                   int64_t n_head) noexcept;

/**
 * Head merge before the output projection — the reverse re-shelving of
 * permute_split, for attention's single output tensor.
 *   input:  x  [B, H, T, hd]
 *   output: y  [B, T, A],  A = H · hd (attention inner width)
 * Element-wise: y[b, t, h·hd + d] = x[b, h, t, d].
 * Pure data movement; y must not alias x.
 */
void permute_merge(cudaStream_t s, Tensor& y, const Tensor& x) noexcept;

}  // namespace llmt::kernels

#endif  // LLMT_HAS_CUDA
