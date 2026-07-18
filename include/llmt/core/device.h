// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Masoud Jami
#pragma once

#ifdef LLMT_HAS_CUDA

#include <cublasLt.h>
#include <cuda_runtime.h>

#include <memory>
#include <string>

#include "llmt/core/run_ctx.h"

namespace llmt {

/**
 * Snapshot of what benchmarks and MFU math need to know about the GPU.
 * Peaks are computed from device attributes (docs/hardware.md holds the
 * datasheet cross-check; Phase 5 replaces both with measured numbers).
 */
struct DeviceProps {
    std::string name;
    int sm_major = 0, sm_minor = 0;
    int sm_count = 0;
    size_t vram_bytes = 0;
    size_t l2_bytes = 0;  // bandwidth benches must exceed this or they measure L2, not DRAM
    double peak_fp32_tflops = 0.0;
    double peak_bw_gbs = 0.0;
};

/**
 * Owns the per-GPU runtime state: the stream all M1 work runs on
 * (DESIGN invariant 7: explicit, single), the cuBLASLt handle and its
 * workspace buffer.
 */
class Device {
   public:
    static constexpr size_t kBlasWorkspaceBytes = 32ull << 20;  // 32 MiB

    explicit Device(int index = 0) noexcept;
    ~Device();
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    const DeviceProps& props() const noexcept { return m_props; }
    cudaStream_t stream() const noexcept { return m_stream; }
    cublasLtHandle_t blas() const noexcept { return m_blas; }
    int index() const noexcept { return m_index; }

    /**
     * Builds a RunCtx around this device's stream and blas handle. The
     * activation arena stays null — it is owned by the model and attached
     * once planning has run (DESIGN §8.2). Parameters deliberately have no
     * defaults: seed and precision come from TrainConfig, backend from
     * RunConfig — wiring should be visible at the call site.
     */
    RunCtx make_ctx(uint64_t seed, KernelBackend backend,
                    PrecisionPolicy precision) const noexcept;

    void synchronize() const noexcept;  // blocks until all work on stream() is done

   private:
    int m_index = 0;
    DeviceProps m_props;
    cudaStream_t m_stream = nullptr;
    cublasLtHandle_t m_blas = nullptr;
    void* m_blas_workspace = nullptr;
    std::unique_ptr<kernels::AlgoCache> m_algo_cache;
};

}  // namespace llmt

#endif  // LLMT_HAS_CUDA
