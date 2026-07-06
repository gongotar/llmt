// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Masoud Jami
// Cross-compilation helpers.
#pragma once

/**
 * LLMT_HD marks functions compiled for both host and device: pure logic
 * (Shape math, Philox RNG) that kernels call inline and host tests verify.
 */
#if defined(__CUDACC__)
#define LLMT_HD __host__ __device__
#else
#define LLMT_HD
#endif
