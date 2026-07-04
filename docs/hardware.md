# Target hardware (task 0.1)

Pods are disposable and the GPU model can vary per session (RunPod Secure
Cloud, subject to host availability); `CMAKE_CUDA_ARCHITECTURES` defaults to
`native`, and benchmark
results are recorded per GPU model. Current + candidate cards:

## NVIDIA RTX A4000 (current pod, $0.25/hr)

| Property | Value | Status |
|---|---|---|
| Architecture | Ampere, `sm_86` | |
| VRAM | 16 GB GDDR6 | per RunPod listing |
| Peak fp32 | ~19.2 TFLOPs (6144 cores × 2 × ~1.56 GHz boost) | ⚠ verify |
| Peak HBM bandwidth | ~448 GB/s (256-bit GDDR6) | ⚠ verify (Phase 5 bench) |
| Pod CPU | 16 vCPU, 62 GB RAM | per RunPod listing |
| CUDA | 12.4.1 (`-devel` image) | |

## NVIDIA RTX 4000 Ada (preferred when available, $0.26/hr)

| Property | Value | Status |
|---|---|---|
| Architecture | Ada Lovelace, `sm_89` (FP8-capable — useful from M2) | |
| VRAM | 20 GB GDDR6 | |
| Peak fp32 | ~26.7 TFLOPs | ⚠ verify |
| Peak HBM bandwidth | ~360 GB/s | ⚠ verify |

⚠ rows are datasheet values; Phase 5's bandwidth micro-benchmark and the GEMM
benchmark (task 4.5) replace them with measured numbers, which are what MFU and
%-of-peak reports use.

## Notes
- Bitwise-determinism guarantees (DESIGN invariant 5) are **per-machine**;
  switching GPU models changes results within tolerance, not correctness.
- Upgrade path: rent H100 (`sm_90`) by the hour for Hopper-specific benchmarks
  when M3 justifies it; code is portable (recompile only).

## Dev host

MacBook Pro M3 Pro (12 cores, 36 GB) — no CUDA; host-only builds + Python
tooling. See `docs/dev.md` for the remote workflow.
