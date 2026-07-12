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

## Measured GEMM ceiling (task 4.5 — `llmt_bench_matmul`, strict FP32)

RTX A5000 (Ampere `sm_86`, computed peak 27.8 TFLOPs), 2026-07-09:

| shape (M1 config) | TFLOPs | % peak |
|---|---|---|
| proj `[8192,384]·[384,384]ᵀ` | 14.9 | 53.8% |
| mlp_up `[8192,384]·[1536,384]ᵀ` | 15.7 | 56.7% |
| mlp_dn `[8192,1536]·[384,1536]ᵀ` | 15.5 | 55.7% |
| lm_head `[8192,384]·[50304,384]ᵀ` | 13.3 | 47.7% |
| attn QKᵀ `192×[256,64]·[256,64]ᵀ` | 9.6 | 34.4% |

RTX 4000 Ada (`sm_89`, nominal 26.7 TFLOPs, 360 GB/s), 2026-07-12 — with the
CEILING reference (4096³ square GEMM = empirical arch max):

| shape | TFLOPs | % nominal | % of ceiling |
|---|---|---|---|
| CEILING ref | 12.83 | 48.0% | — |
| proj | 11.86 | 44.4% | 92% |
| mlp_up / mlp_dn | 12.4 / 12.1 | ~46% | 94–97% |
| lm_head | 11.89 | 44.5% | 93% |
| attn QKᵀ | 6.72 | 25.2% | ~87% of its memory roofline (360 GB/s · AI≈21 ≈ 7.7 T) |

Model GEMMs run at 92–97% of the achievable ceiling: cuBLASLt-level
orchestration leaves nothing on the table; headroom is precision-policy
territory (TF32/BF16), not tuning.

This is the "cuBLAS-bound" ceiling MFU is judged against: at these shapes,
~50–57% of nominal peak is what perfect orchestration could reach for the
big GEMMs; small batched attention shapes cap far lower (undersized per-SM
work — the motivation for fused attention later). Re-measure per GPU model.

## Notes
- Bitwise-determinism guarantees (DESIGN invariant 5) are **per-machine**;
  switching GPU models changes results within tolerance, not correctness.
- Upgrade path: rent H100 (`sm_90`) by the hour for Hopper-specific benchmarks
  when M3 justifies it; code is portable (recompile only).

## Dev host

MacBook Pro M3 Pro (12 cores, 36 GB) — no CUDA; host-only builds + Python
tooling. See `docs/dev.md` for the remote workflow.
