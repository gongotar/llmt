// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Masoud Jami

#include "llmt/core/device.h"
#include "llmt/core/error.h"
#include "llmt/kernels/algo_cache.h"

namespace llmt {

namespace {

// fp32 CUDA cores per SM by architecture (for peak-FLOPs estimation).
int cores_per_sm(int major, int minor) noexcept {
    if (major == 8 && minor == 0) return 64;   // A100
    if (major == 8) return 128;                // Ampere consumer / Ada (8.6, 8.9)
    if (major == 9) return 128;                // Hopper
    if (major >= 10) return 128;               // Blackwell (verify when relevant)
    return 64;                                 // conservative fallback
}

}  // namespace

Device::Device(int index) noexcept : m_index(index) {
    CUDA_CHECK(cudaSetDevice(m_index));

    cudaDeviceProp p{};
    CUDA_CHECK(cudaGetDeviceProperties(&p, m_index));
    m_props.name = p.name;
    m_props.sm_major = p.major;
    m_props.sm_minor = p.minor;
    m_props.sm_count = p.multiProcessorCount;
    m_props.vram_bytes = p.totalGlobalMem;
    m_props.l2_bytes = static_cast<size_t>(p.l2CacheSize);

    // Attribute API (the cudaDeviceProp clock fields are deprecated in CUDA 12).
    int clock_khz = 0, mem_clock_khz = 0, bus_width_bits = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, m_index));
    CUDA_CHECK(cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, m_index));
    CUDA_CHECK(cudaDeviceGetAttribute(&bus_width_bits, cudaDevAttrGlobalMemoryBusWidth, m_index));

    // peak fp32 = 2 (FMA) × cores × clock ; peak BW = 2 (DDR) × mem clock × bus bytes
    const double cores = static_cast<double>(m_props.sm_count) * cores_per_sm(p.major, p.minor);
    m_props.peak_fp32_tflops = 2.0 * cores * (clock_khz * 1e3) / 1e12;
    m_props.peak_bw_gbs = 2.0 * (mem_clock_khz * 1e3) * (bus_width_bits / 8.0) / 1e9;

    CUDA_CHECK(cudaStreamCreate(&m_stream));
    CUBLAS_CHECK(cublasLtCreate(&m_blas));
    CUDA_CHECK(cudaMalloc(&m_blas_workspace, kBlasWorkspaceBytes));
    m_algo_cache = std::make_unique<kernels::AlgoCache>();
}

Device::~Device() {
    if (m_blas_workspace) cudaFree(m_blas_workspace);
    if (m_blas) cublasLtDestroy(m_blas);
    if (m_stream) cudaStreamDestroy(m_stream);
}

RunCtx Device::make_ctx(uint64_t seed, KernelBackend backend,
                        PrecisionPolicy precision) const noexcept {
    RunCtx ctx;
    ctx.stream = m_stream;
    ctx.blas = m_blas;
    ctx.blas_workspace = m_blas_workspace;
    ctx.blas_workspace_bytes = kBlasWorkspaceBytes;
    ctx.algo_cache = m_algo_cache.get();
    ctx.precision = precision;
    ctx.backend = backend;
    ctx.seed = seed;
    return ctx;
}

void Device::synchronize() const noexcept { CUDA_CHECK(cudaStreamSynchronize(m_stream)); }

}  // namespace llmt
