// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Masoud Jami
// Fatal-error reporting and error-checking macros for CUDA / cuBLASLt calls
// (docs/DESIGN.md task 0.6). Every runtime API call goes through a macro;
// contract violations go through detail::fatal.
#pragma once

#include <cstdarg>
#include <cstdio>
#include <cstdlib>

namespace llmt::detail {

/** Prints "[llmt] <where>: <formatted message>" to stderr and aborts —
 *  the library's response to contract violations (nothing throws). */
[[noreturn]] inline void fatal(const char* where, const char* fmt, ...) noexcept {
    va_list args;
    va_start(args, fmt);
    std::fprintf(stderr, "[llmt] %s: ", where);
    std::vfprintf(stderr, fmt, args);
    std::fprintf(stderr, "\n");
    va_end(args);
    std::abort();
}

}  // namespace llmt::detail

#ifdef LLMT_HAS_CUDA

#include <cublasLt.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            std::fprintf(stderr, "[llmt] CUDA error %s:%d: %s (%s)\n", __FILE__,  \
                         __LINE__, cudaGetErrorString(err_), #call);              \
            std::abort();                                                         \
        }                                                                         \
    } while (0)

#define CUBLAS_CHECK(call)                                                        \
    do {                                                                          \
        cublasStatus_t st_ = (call);                                              \
        if (st_ != CUBLAS_STATUS_SUCCESS) {                                       \
            std::fprintf(stderr, "[llmt] cuBLAS error %s:%d: status %d (%s)\n",   \
                         __FILE__, __LINE__, static_cast<int>(st_), #call);       \
            std::abort();                                                         \
        }                                                                         \
    } while (0)

// Call after kernel launches in debug paths.
#define CUDA_CHECK_LAST() CUDA_CHECK(cudaGetLastError())

#endif  // LLMT_HAS_CUDA
