# M1 Implementation Tracker

Task list for **Milestone 1: it trains** (see `DESIGN.md`). Mark items `[x]` as
they land. A phase is *done* only when its **exit criteria** box is checked.
Phases are ordered by dependency — later phases assume earlier ones are green.

**Legend:** `[ ]` open · `[x]` done · `[-]` skipped/obsolete (add a note why)
· **(S)** stretch, not required for M1 exit.

## Current status

> **Phase:** 5 ✅ done (reviewed) → starting Phase 6 (embedding, RoPE,
> attention forward, fused cross-entropy)
> **Last updated:** 2026-07-18
> **Notes:** Pod workflow proven end-to-end on RunPod Secure Cloud
> (current pod: RTX A4000 `sm_86` 16 GB; preferred when available:
> RTX 4000 Ada `sm_89`). `CMAKE_CUDA_ARCHITECTURES=native` since the GPU
> model varies per session. Per-session loop: deploy pod → update
> `scripts/pod.env` → `./scripts/remote.sh setup` (fresh pod) →
> `./scripts/remote.sh test` to iterate → Terminate pod when done.

---

## Phase 0 — Environment & scaffolding

*No CUDA on the macOS dev host: the remote GPU workflow is a Phase-0
deliverable, not an afterthought.*

- [x] 0.1 Choose & document target GPU box (`docs/hardware.md`): GPU model, SM
      arch (`sm_XX`), peak fp32/tf32 TFLOPs, peak HBM GB/s, VRAM, driver/CUDA
      version. These numbers feed MFU & bandwidth benchmarks later.
- [x] 0.2 Remote dev workflow scripted: `scripts/remote.sh` (rsync/git push →
      cmake build → run tests on the box), documented in `docs/dev.md`.
- [x] 0.3 `git init`, `.gitignore` (build dirs, `*.bin` data, checkpoints —
      but **not** `tests/golden/`), `.clang-format`.
- [x] 0.4 CMake skeleton: `llmt` static library target, CUDA language enabled,
      `CMAKE_CUDA_ARCHITECTURES` configurable, Release `-O3`, optional
      `-lineinfo` flag for profiling builds.
- [x] 0.5 Repository tree from DESIGN §6 created with placeholder headers that
      compile (empty lib links).
- [x] 0.6 `src/core/error.h`: `CUDA_CHECK`, `CUBLAS_CHECK` macros (report
      file:line + error string, abort).
- [x] 0.7 Test harness: vendor doctest (single header, dev-only dependency);
      `llmt_tests` CMake target; one trivial passing test.
- [x] 0.8 Python tooling env: `tools/requirements.txt` (torch, numpy, tiktoken,
      pinned versions); venv setup documented. (local `.venv`: torch 2.12.1)
- [x] 0.9 `scripts/ci.sh`: build + run all tests on the GPU box in one command
      (this *is* CI for now).
- [x] **Exit: one command on the mac syncs, builds, and runs a passing GPU test
      remotely.** (`./scripts/remote.sh test` → GPU smoke test green on RTX A4000)

## Phase 1 — core/

- [x] 1.1 `DType` enum + `dtype_size()`; `Shape` (rank ≤ 4, `numel()`).
- [x] 1.2 `Tensor` non-owning view (`data`, `dtype`, `shape`, `bytes()`);
      contiguous row-major only — no strides, by design.
- [x] 1.3 `Device`: device selection, properties snapshot (SM count, peak
      FLOPs/BW from 0.1 constants keyed by arch), stream creation, cuBLASLt
      handle lifetime.
- [x] 1.4 `Arena`: single `cudaMalloc`, 256-byte-aligned bump allocation,
      `reset()`, high-water-mark report. Two planned instances: persistent
      (params/opt) and activations.
- [x] 1.5 `Rng`: Philox counter-based; `uniform(seed, stream_id, offset)` and
      `normal(...)` (Box–Muller) device functions + fill kernels.
- [x] 1.6 `RunCtx` struct wired (stream, blas, arena ref, precision, backend,
      seed, step).
- [x] 1.7 `PrecisionPolicy` (fp32-everything for M1, but the type exists and is
      queried, per invariant 3).
- [x] 1.8 Tests: Arena alignment/offsets/reset; Rng mean/variance sanity +
      bitwise reproducibility (same counters → same bits, twice); Device
      sanity (device found, SM count > 0, peak FLOPs/BW positive, stream +
      cuBLASLt handle created, `make_ctx` wiring).
- [x] **Exit: core unit tests green on GPU box.** (RTX 4000 Ada; incl. bitwise host↔device Philox match)

## Phase 2 — Oracle & test infrastructure (before any real kernel)

- [x] 2.1 Golden file format spec (`docs/golden.md`): raw little-endian `.bin`
      per tensor + `manifest.json` (name, shape, dtype, seed, torch version,
      tolerance overrides).
- [x] 2.2 `tools/gen_golden.py` framework: case registry, deterministic seeding,
      `dump(name, tensor)` helper; `--case X` regeneration of a single case.
- [x] 2.3 C++ side: `GoldenCase` loader (manifest + tensors → host buffers) and
      `assert_allclose(actual, expected, rtol, atol)` reporting max abs/rel
      diff and first offending index.
- [x] 2.4 Determinism harness: run a callable twice into two buffers, compare
      bitwise, report first diverging byte.
- [x] 2.5 Finite-difference gradient checker (host-side helper, micro shapes).
- [x] 2.6 First end-to-end proof: goldens for a trivial op (e.g. `y = 2x`)
      generated, committed, loaded, and diffed in a C++ test.
- [x] **Exit: adding a new golden-tested kernel is a ~15-line Python case + a
      ~20-line C++ test.** (scale2x proves the pipeline; template in docs/golden.md)

## Phase 3 — ParamStore

- [x] 3.1 Registration API: `add(name, shape, Role) → Param{weight, grad}`
      (views become valid only after finalize).
- [x] 3.2 `finalize()`: single allocation, four per-kind-packed spans (see DESIGN §8.2 as amended),
      alignment, fixed offsets; registration is closed afterwards.
- [x] 3.3 Named lookup + iteration by `Role` (Embedding / Matrix / Norm) +
      flat spans (`params_flat()`, `grads_flat()`, opt-state spans).
- [x] 3.4 **Aliasing** for weight tying: `alias("lm_head.w", "embed.wte")` —
      two names, one span; document the grad-accumulation contract
      (invariant 9); test that both names see the same memory.
- [x] 3.5 Init (`model/init.cu`): role-keyed — normal(0, 0.02) for
      matrices/embeddings, ones for norm weights, residual-projection scaling
      `0.02/√(2L)` (wo, mlp down-proj); uses Rng; per-tensor stream_id so init
      is layout-independent.
- [x] 3.6 Tests: offsets/alignment, lookup, role iteration, alias identity,
      init statistics (mean/std within tolerance), init bitwise-reproducible.
- [x] **Exit: a toy 2-tensor model registers, finalizes, inits reproducibly.** (22 cases green on RTX 4000 Ada; layout-independent bitwise init proven)

## Phase 4 — GEMM wrapper + Linear

- [x] 4.1 `matmul()` wrapper over cuBLASLt: row-major convention mapping
      (document the weight-layout convention here — this file is the single
      source of truth), transpose flags, `beta` ∈ {0,1} for grad accumulation,
      workspace from arena.
- [x] 4.2 Algo cache: heuristic resolved once per (shape, trans, dtype) key,
      cached for the run (determinism, invariant 5).
- [x] 4.3 Batched-strided variant (needed by attention in Phase 6).
- [x] 4.4 Correctness test vs golden (`torch.matmul`, several shapes incl.
      non-square, both transposes).
- [x] 4.5 Micro-benchmark: achieved TFLOPs vs peak for model-relevant shapes
      (records our "cuBLAS-bound" ceiling for MFU context).
- [x] 4.6 `Linear` layer (no bias): `forward` `y = x·Wᵀ`; `backward`
      `dx = dy·W`, `dW += dyᵀ·x` (beta=1 into flat grads).
- [x] 4.7 Golden test: Linear fwd + bwd (dx, dW) vs PyTorch.
- [x] **Exit: Linear fwd/bwd matches oracle at 1e-5 rel; GEMM ceiling
      documented.** (27 cases green; model GEMMs at 92–97% of measured ceiling)

## Phase 5 — Elementwise & norm kernels (forward)

*Each kernel: naive-but-correct first, warp-shuffle reductions where they
matter; golden test + bandwidth benchmark before moving on.*

- [x] 5.1 `rmsnorm_fwd` (saves `rstd` for backward), one warp/block per row.
- [x] 5.2 `gelu_fwd` — tanh approximation; oracle uses
      `gelu(approximate='tanh')` (recorded micro-decision).
- [x] 5.3 `residual_fwd` (add).
- [x] 5.4 `softmax_fwd` row-wise: max-subtract, exp, normalize, `MaskSpec`
      parameter (M1 implements `Causal` only; the seam exists).
- [x] 5.5 Golden tests for 5.1–5.4.
- [x] 5.6 Bandwidth benchmarks: % of peak HBM BW reported for each; recorded
      in `bench/results/` as JSON (A4500 + RTX 4000 Ada, L2-exceeding shapes).
- [x] 5.7 rmsnorm at 69% of peak vs 84–88% for its siblings: RESOLVED
      (bisection 2026-07-17 A4500 + occupancy sweep 2026-07-19 Ada; ncu
      blocked by RunPod counter perms). Cause: the write pass re-reads x,
      and the re-read is served by L2 only when L2 outsizes the inter-touch
      churn (~7 MB): A4000/A4500 (4/6 MB L2) → DRAM re-reads → 67–69%
      "accounting artifact" (bus near-saturated); RTX 4000 Ada (40 MB L2)
      → free → 86.2%, penalty gone. NOT the reduction barrier, NOT the
      two-phase structure, NOT L1 (zero gap at all occupancies). M2
      register-caching note now conditional: only pays on small-L2 GPUs.
      Bench shapes scaled 16× so tensors exceed L2 (unscaled shapes read
      327–570% "of peak" on Ada — L2 bandwidth, not DRAM). Full story in
      docs/knowledge.md "Counted vs actual bytes".
- [x] 5.8 `__stcg` L1-bypassing stores on rmsnorm/softmax: NO EFFECT
      (2026-07-18, A4000, same-pod A/B: 67.5% vs 67.7% — noise) → reverted.
      Implies sm_86 doesn't allocate stores in L1 by default; write-allocation
      theory of the rmsnorm gap dropped. Register caching (5.7 note) remains
      the M2 fix.
- [x] **Exit: all four green vs oracle; norms ≥ ~50% of peak BW (naive-kernel
      bar — refined in M2/M3).** (32 cases / 206,362 assertions green on
      A4500, A4000, RTX 4000 Ada; rmsnorm 70% A4500 / 86% Ada, gelu/residual
      84–89%, softmax 84%; the rmsnorm L2 story in knowledge.md)

## Phase 6 — Embedding, RoPE, attention forward, fused cross-entropy

- [ ] 6.1 `embedding_fwd`: gather `[B,T] → [B,T,C]`.
- [ ] 6.2 `permute` kernels: `[B,T,3C] → 3×[B,H,T,hd]` (post-QKV split) and
      `[B,H,T,hd] → [B,T,C]` (pre-out-proj merge). Golden-tested — permutes
      are a classic silent-bug site.
- [ ] 6.3 `rope_fwd` applied to Q and K views (`rope_theta` from config).
- [ ] 6.4 `MaskSpec` struct in the public attention op signature (Causal only).
- [ ] 6.5 Naive attention forward: batched-strided QKᵀ GEMM → scale → masked
      softmax (probs overwrite scores in place) → PV GEMM → merge heads.
- [ ] 6.6 Golden test: attention fwd vs
      `torch.nn.functional.scaled_dot_product_attention` (+ manual per-stage
      goldens: scores, probs — so a failure localizes itself).
- [ ] 6.7 `cross_entropy_fwd` fused: from logits `[B,T,V]` — row max,
      log-sum-exp, per-token NLL, `loss_mask`-weighted mean → single scalar.
      Numerically stable; golden test incl. a mask with zeros.
- [ ] 6.8 MLP layer (Linear → GELU → Linear) composed; golden fwd test.
- [ ] **Exit: every layer's forward matches oracle at 1e-5 rel.**

## Phase 7 — Planner, Block, GPT forward

- [ ] 7.1 `ActivationPlanner`: `plan()` pass (layers declare
      `need(name, shape)`), offset assignment (M1: simple stacking),
      `bind()` pass (resolved `Tensor` views handed back).
- [ ] 7.2 Startup memory report (per-buffer table + total); unit test asserts
      total against the analytic formula from DESIGN §8.4.
- [ ] 7.3 `Block`: pre-norm wiring norm→attn→res→norm→mlp→res; the single
      readable wiring function.
- [ ] 7.4 `GPT`: ParamStore registration walk, plan/bind walk, forward =
      embedding → blocks → final norm → lm_head GEMM (tied weight via alias)
      → fused CE. Returns host `float` loss (documented sync).
- [ ] 7.5 `forward_logits` (no activation retention, its own smaller plan) for
      eval/generation.
- [ ] 7.6 Golden test: 2-layer micro config — logits and loss vs full PyTorch
      reference model (this oracle model is written now and reused in 10.6
      and Phase 12).
- [ ] 7.7 Initial-loss sanity: fresh model ≈ `ln(vocab)` ± 0.1.
- [ ] **Exit: full-model forward matches PyTorch at 1e-4 rel on the micro
      config; memory report matches formula.**

## Phase 8 — Backward passes (the big one)

*Budget: as much time as Phases 1–7 combined. Implement in reverse model
order so each stage's `d_input` feeds the next test. Every kernel: golden
test + finite-difference spot check. After each landing: determinism check
(two identical calls → bitwise-equal grads).*

- [ ] 8.1 `cross_entropy_bwd` fused: `dlogits = (softmax − onehot)·mask/denom`,
      written **in place of logits** (§8.4). Golden test.
- [ ] 8.2 lm_head backward: `dx` GEMM + `dW` accumulation into the **tied**
      grad view (`beta=1`); test tied-grad correctness vs PyTorch tied model
      specifically.
- [ ] 8.3 `rmsnorm_bwd` (uses saved `rstd`; two reductions per row). Golden +
      finite-diff — the classic bug farm, take it slow.
- [ ] 8.4 `gelu_bwd`, `residual_bwd` (grad fan-out add). Golden tests.
- [ ] 8.5 Attention backward chain, per-stage golden tests:
  - [ ] 8.5a merge-heads bwd (permute), out-proj bwd (Linear pattern)
  - [ ] 8.5b `dV = PᵀdO`, `dP = dO·Vᵀ` (batched GEMMs)
  - [ ] 8.5c softmax bwd with mask: `dS = P ⊙ (dP − rowsum(dP ⊙ P))`
  - [ ] 8.5d `dQ = dS·K·scale`, `dK = dSᵀ·Q·scale`
  - [ ] 8.5e `rope_bwd` (inverse rotation on dQ, dK)
  - [ ] 8.5f split-heads bwd (permute), QKV-proj bwd
  - [ ] 8.5g full attention-layer golden: dx, dWq/k/v/o vs PyTorch
- [ ] 8.6 MLP backward composed; golden test.
- [ ] 8.7 `embedding_bwd` **deterministic scatter** (no atomics; e.g.
      channel-parallel accumulation); accumulates into the tied grad view;
      bitwise-determinism test is mandatory here.
- [ ] 8.8 `GPT::backward`: reverse walk; full-model golden — *every named
      gradient* vs PyTorch 2-layer micro model at 1e-4 rel.
- [ ] 8.9 Full determinism test: two identical fwd+bwd → bitwise-equal flat
      grad buffer.
- [ ] **Exit: all grads match oracle; bitwise determinism holds.**

## Phase 9 — Optimizer

- [ ] 9.1 `zero_grad`: one memset on the flat grad buffer.
- [ ] 9.2 `clip_grad_norm`: global L2 via single reduction kernel over flat
      grads + conditional scale; returns pre-clip norm to host.
- [ ] 9.3 Param groups by role: decay mask (no decay on norms/embeddings),
      optional per-group lr scale; represented as per-element or per-range
      metadata usable by one fused kernel.
- [ ] 9.4 Fused `adamw` kernel over flat buffers: bias-corrected moments,
      **decoupled** weight decay (match `torch.optim.AdamW` semantics exactly).
- [ ] 9.5 `lr_at`: warmup + cosine; unit test vs precomputed reference values.
- [ ] 9.6 Golden test: one optimizer step vs `torch.optim.AdamW` (same grads,
      same groups) at 1e-6.
- [ ] 9.7 Trajectory test: 3 steps of (fixed synthetic grads → step) vs
      PyTorch — catches state-update ordering bugs.
- [ ] **Exit: optimizer step matches PyTorch bit-for-tolerance across a
      multi-step trajectory.**

## Phase 10 — Data pipeline

- [ ] 10.1 `tools/prepare_data.py`: download/ingest corpus → token ids →
      `.bin` with header (magic, version, vocab_size, dtype u16/u32, count) +
      train/val split. Two modes:
  - [ ] 10.1a **char-level** (vocab ≈ 65; the dev-loop default, DESIGN §8.4)
  - [ ] 10.1b **GPT-2 BPE** via tiktoken (vocab 50257 padded → 50304)
- [ ] 10.2 `DataLoader`: mmap the `.bin`, reproducible random window sampling
      via Rng(seed, step), pinned staging buffer, H2D on `ctx.stream`;
      `targets` = tokens shifted by one; `loss_mask` = ones.
- [ ] 10.3 Header/vocab validation against `ModelConfig::vocab` at startup
      (fail loudly on mismatch).
- [ ] 10.4 Tests: determinism (same seed → same batch sequence), no window
      crosses EOF, header round-trip, u16/u32 both.
- [ ] **Exit: TinyShakespeare prepared in both modes; loader tests green.**

## Phase 11 — Training loop, observability, checkpointing, examples

- [ ] 11.1 `StepTimer`: CUDA events, tokens/sec, analytic FLOPs/token
      (≈ `6·N_params` + attention term — formula documented in the code),
      MFU vs Device peak.
- [ ] 11.2 `Logger`: step, loss, grad-norm, lr, tok/s, MFU → stdout +
      JSONL file (plottable); run dir contains the serialized configs.
- [ ] 11.3 NVTX ranges per layer, toggled by `RunConfig::nvtx`; verify named
      timeline in Nsight Systems once.
- [ ] 11.4 Config JSON serialization (Model/Train/Run) — load, save, dump into
      run dir and checkpoint header.
- [ ] 11.5 Checkpoint save/load: versioned header + embedded ModelConfig +
      tensor manifest + flat param blob; round-trip test
      (save → load → bitwise-equal params, same loss on a fixed batch).
- [ ] 11.6 PyTorch-side loader for our checkpoint format
      (`tools/reference_train.py` imports init from it) — required for the
      reference-run comparison.
- [ ] 11.7 `evaluate()` helper: mean val loss over N batches via `forward`
      (no backward).
- [ ] 11.8 `examples/hello_train.cpp` exactly as DESIGN §9; compiles and runs
      end-to-end on char-level data.
- [ ] 11.9 `examples/generate.cpp`: load checkpoint → autoregressive
      `forward_logits` (full recompute per token — no KV cache in M1, fine at
      T=256) → temperature + top-k sampling.
- [ ] 11.10 (S) Full training-state resume: optimizer state + step + data
      order round-trip (`--resume` flag).
- [ ] **Exit: `hello_train` runs 5000 char-level steps unattended; loss curve
      logged; checkpoints load back.**

## Phase 12 — M1 validation & exit

- [ ] 12.1 Convergence smoke test in `scripts/ci.sh`: 4-layer char-level micro
      model, 500 steps → assert loss < recorded threshold (~minutes).
- [ ] 12.2 End-to-end determinism test in CI: 50 steps twice → bitwise-equal
      loss sequence and final params.
- [ ] 12.3 **Reference-run comparison:** `tools/reference_train.py` (~100-line
      PyTorch, same init loaded from our checkpoint, same data order, same
      hyperparameters) → overlay loss curves; they must track within noise
      for hundreds of steps. Plot committed to `docs/`.
- [ ] 12.4 Full char-level TinyShakespeare run: val loss < 1.6; fixed-prompt
      generations saved per checkpoint and eyeballed.
- [ ] 12.5 Full GPT-2-BPE TinyShakespeare run: initial loss ≈ 10.8, train
      loss < 3.5 and falling; generations recognizably Shakespeare-shaped.
- [ ] 12.6 Final val loss within a few % of the PyTorch reference under the
      identical budget (DESIGN §10.4).
- [ ] 12.7 MFU measured and recorded (target ≥ 10% naive); bench JSONs
      committed; memory high-water mark vs §8.4 budget recorded.
- [ ] 12.8 `docs/M1_RESULTS.md`: configs, curves, generations, MFU, memory,
      known limitations (no dropout, no grad accum, sync per step, no KV
      cache). Tag `v0.1`.
- [ ] **Exit: DESIGN §9 exit criteria all met — M1 done.**

---

## Deferred (decided, not forgotten)

| Item | Milestone |
|---|---|
| Dropout kernels + replayable RNG usage | M2 |
| Gradient accumulation (`grad_accum > 1`, loss scaling, deferred sync) | M2 |
| bf16 mixed precision (PrecisionPolicy seam already in place) | M2 |
| Own BPE tokenizer in C++ | M2 |
| Custom pod Docker image (`docker/Dockerfile` running `setup_pod.sh` as build step, pushed to Docker Hub, used as RunPod template) — adopt once `setup_pod.sh` stabilizes | M2 |
| `scripts/pod_up.sh` / `pod_down.sh` via runpodctl/API (no web UI per session) | M2 |
| Fused kernels (add+norm, rope+qkv) + `KernelBackend::Fused` cross-checks | M2 |
| rmsnorm register-cached row (skip pass-2 re-read; only pays on small-L2 GPUs, see 5.7). Tiered by C: per-lane registers to C≈1K (compile-time C buckets — register arrays need static indexing or they spill); block-per-row + per-thread registers + two-stage smem reduction to C≈8K; full smem row staging beyond / for runtime C | M2 |
| KV cache for generation | M2/M3 |
| Flash-style attention, activation-buffer reuse/checkpointing | M3 |
| Cache cuBLASLt descriptors alongside the cached algo (entries become RAII owners of 4 Lt handles). Trigger: profiling shows host-bound gaps between kernels (attention GEMMs most likely). Superseded entirely if CUDA graphs land | M3 |
| NCCL data parallel, ZeRO-1 | M4 |
