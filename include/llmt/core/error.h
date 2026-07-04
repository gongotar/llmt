// Error-checking macros for CUDA / cuBLASLt calls (DESIGN.md task 0.6).
// Every runtime API call in the library goes through one of these.
#pragma once

#ifdef LLMT_HAS_CUDA

#include <cublasLt.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

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
