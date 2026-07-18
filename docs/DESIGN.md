# LLM Training Library — Plan & Architecture

A C++/CUDA library for training transformer LLMs from scratch on a single GPU
(initially), depending only on small building blocks (CUDA runtime, cuBLASLt).
Working name: **`llmt`** (placeholder — everything lives in `namespace llmt`).

**Goal of milestone 1:** a complete, correct, reproducible local training run of a
small GPT-style model on limited data (e.g. TinyShakespeare / TinyStories), with
every layer — attention, MLP, embeddings, norms, loss, optimizer — implemented in
this library.

---

## 1. Guiding principle

Separate the three axes that change for different reasons:

| Axis | Changes because of… | Lives in |
|---|---|---|
| **Model definition** | research ideas (attention variants, norms, losses) | `layers/`, `model/` |
| **Execution machinery** | performance work (fusion, precision, streams) | `core/`, `kernels/` |
| **Training policy** | per-experiment tuning (lr, schedule, batch) | `optim/`, `train/` |

**Flexibility in the math, rigidity in the machinery.** Research seams are explicit
extension points; the execution machinery is deliberately constrained (static
graph, fixed shapes per run, one precision policy per run).

## 2. Dependency policy

| Dependency | Status | Rationale |
|---|---|---|
| CUDA runtime, own kernels | core of the project | this *is* the library |
| cuBLAS / cuBLASLt | allowed | competitive from-scratch GEMM is its own research project; matmul is ~90% of FLOPs, everything else is ours |
| CUTLASS | optional later | "almost from scratch" matmul if we want to own that layer too |
| cuDNN, Thrust | not used | defeats the purpose |
| NCCL | later (milestone 4) | multi-GPU data parallelism |
| PyTorch (Python only) | test oracle only | golden-file generation, never linked |

More dependencies may be adopted later, but each must be a *small building block*
replaceable behind an interface we own.

## 3. Milestones & expected size

| Milestone | Scope | LoC | Solo effort |
|---|---|---|---|
| **M1: it trains** | GPT-2-small-ish, fp32, naive attention, single GPU, loss ↓ on tiny corpus | 5–8k | 4–8 weeks part-time |
| **M2: it's respectable** | bf16 mixed precision, fused kernels, grad accumulation, own BPE tokenizer, sampling/generation | +4–6k | +4–8 weeks |
| **M3: it's fast** | flash-style fused attention, kernel tuning toward cuBLAS-bound MFU, activation checkpointing | +3–5k | +2–4 months |
| **M4: it scales** | NCCL data parallel, ZeRO-1 optimizer sharding | +2–4k | +1–2 months |
| **M5: it's a framework** | autograd, general tensors, dynamic graphs | 30k+ | only if the library earns it |

Calibration: llm.c trains GPT-2 in ~5k lines total; that is the M1–M2 band.

## 4. Architectural invariants (decided up front, expensive to retrofit)

1. **Flat parameter storage.** All parameters live in one contiguous device
   buffer; layers hold *views* (offset + shape) into it. Same for gradients and
   optimizer state (`m`, `v`). Consequences: single-kernel optimizer step,
   single-`write()` checkpoints, one-reduction global grad-norm clipping, and
   NCCL/ZeRO later operate on flat ranges without refactoring.

2. **Static activation memory plan.** Layers *declare* their activation and
   scratch needs for a given `(batch, seq)`; a planner assigns offsets in one
   pre-allocated arena. No `cudaMalloc` in the hot loop, ever. The planner is the
   future home of buffer-lifetime reuse and activation checkpointing.

3. **Precision policy is cross-cutting.** One `PrecisionPolicy` object (master
   weight dtype, GEMM compute dtype, reduction dtype, plus **per-layer and per-role
   overrides** — e.g. logits/loss pinned to fp32 while blocks run bf16, or one
   specific layer held at higher precision while debugging instability).
   Layers query it; no dtype is hard-coded in a signature. M1 sets everything
   to fp32 but the seam exists from commit 1.

4. **Counter-based RNG.** Philox-style `hash(seed, stream_id, offset)` — stateless,
   bitwise-reproducible, replayable (required later by activation checkpointing
   to recompute dropout identically). Dropout itself is deferred to M2
   (`ModelConfig::dropout` reserved, fixed at 0 in M1).

5. **Determinism by construction.** Bitwise run-to-run reproducibility is a hard
   requirement (it is what makes convergence bugs bisectable). Two concrete
   consequences: **no atomics in any gradient path** (embedding backward uses a
   deterministic scatter, not `atomicAdd`), and the cuBLASLt algorithm choice is
   resolved once per shape and cached, never re-heuristicated mid-run.

6. **Kernels are dumb.** Raw pointers + dims, no state, no knowledge of layers.
   Composition happens one level up. Every kernel is individually testable and
   swappable (naive ↔ fused). **The naive kernel is the permanent reference
   implementation and is never deleted.**

7. **Explicit stream.** Single stream in M1, but passed explicitly through every
   launch so multi-stream overlap later is a localized change.

8. **Versioned checkpoints.** Header = format version + embedded `ModelConfig` +
   tensor manifest. Loading never needs out-of-band knowledge.

9. **Weight tying is a ParamStore feature.** `tie_embeddings` means two names
   (`embed.wte`, `lm_head.w`) resolve to the *same* flat span. Its gradient
   receives two accumulations per step — from the lm_head GEMM (`beta=1`) and
   from the embedding backward scatter — into the same grad view; ordering and
   zeroing are owned by ParamStore's contract, not by the layers.

10. **Recorded micro-decisions** (anything the PyTorch oracle must match
    exactly): no bias terms anywhere (Llama-style, consistent with RMSNorm/RoPE);
    GELU is the tanh approximation (`torch gelu(approximate='tanh')`);
    pre-norm blocks; `grad_accum` is asserted `== 1` until M2;
    `GPT::forward` returning a host `float` implies one device sync per step —
    accepted for M1, revisited with M2 grad accumulation.

## 5. Research flexibility seams

Each seam is an enum in the config plus a `switch` in exactly one place — no
plugin machinery, just a guaranteed single decision point.

| Seam | Planned variants | Design hook |
|---|---|---|
| Attention score path | causal, sliding-window, doc-mask (packing), logit soft-cap, sinks | `MaskSpec` parameter of the attention op — never baked into a kernel |
| KV heads | MHA → GQA → MQA | `n_kv_head` distinct from `n_head` from day 1 |
| Positional encoding | RoPE (+ NTK/YaRN scaling), ALiBi, NoPE | pluggable transform applied to Q/K |
| Norm | RMSNorm / LayerNorm, pre/post, QK-norm | norm is a component the block composes |
| MLP | GELU-MLP, SwiGLU (different param shapes), MoE later | MLP owns its own parameter layout |
| Block wiring | parallel attn+MLP, residual scaling, per-layer configs | model = list of block configs, not `n_layer` copies of one |
| Loss | z-loss, label smoothing, masked tokens, MTP | loss takes per-token `loss_mask` weights from day 1 |
| Optimizer | AdamW → Lion/Muon; **param groups** | groups by parameter *role* (no decay on norms/embeddings, separate embedding lr) |
| Init | scaled residual init, muP-style | init scheme keyed on parameter role |
| Data | packing, curriculum, mixing | loader yields `(tokens, targets, loss_mask)` |

### Frozen (deliberately inflexible)

Decoder-only transformers · fixed `(batch, seq)` per run · static graph ·
one precision policy per run · training only (no serving) · contiguous
row-major tensors only.

## 6. Repository layout

```
llmt/
├── src/
│   ├── core/        device.cu, allocator.cu, tensor.h, dtype.h, rng.cu, error.h
│   ├── kernels/     embedding.cu, rmsnorm.cu, attention_naive.cu, softmax.cu,
│   │                gelu.cu, residual.cu, cross_entropy.cu, adamw.cu, rope.cu,
│   │                permute.cu (head split/merge), matmul.cpp (cuBLASLt wrapper)
│   ├── layers/      linear.cpp, attention.cpp, mlp.cpp, norm.cpp, block.cpp
│   ├── model/       config.h, gpt.cpp, init.cu, checkpoint.cpp
│   ├── optim/       adamw.cpp, param_groups.cpp, scheduler.h
│   ├── data/        dataloader.cpp            (tokenizer.cpp arrives in M2)
│   └── train/       trainer.cpp, logger.cpp, timer.cpp
├── include/llmt/    public headers mirroring the above
├── tests/
│   ├── unit/        per-kernel numerical tests vs golden files
│   ├── golden/      PyTorch-generated reference tensors (.bin + manifest)
│   └── convergence/ end-to-end smoke training runs
├── bench/           kernel micro-benchmarks, end-to-end MFU benchmark
├── tools/           prepare_data.py, gen_golden.py (PyTorch oracle scripts)
├── examples/        hello_train.cpp, generate.cpp
└── CMakeLists.txt
```

---

## 7. The interface

The public API is small: **configs in, a model, an optimizer, a data loader, and
a loop the user owns.** The library does not hide the training loop — that is
where the research happens.

### 7.1 Configuration (three structs, three meanings)

```cpp
// Defines the checkpoint. Serialized into every checkpoint header.
struct ModelConfig {
    int   n_layer   = 6;
    int   n_head    = 6;
    int   n_kv_head = 6;          // GQA-ready from day 1
    int   d_model   = 384;
    int   d_ff      = 4 * 384;    // MLP owns interpretation (SwiGLU differs)
    int   vocab     = 50304;      // padded for GEMM alignment
    int   seq_len   = 256;
    NormType norm   = NormType::RMSNorm;   // enum seams
    ActType  act    = ActType::GELU;       // tanh approximation, no biases anywhere
    PosType  pos    = PosType::RoPE;
    float rope_theta = 10000.f;
    float dropout   = 0.0f;      // reserved; fixed at 0 in M1, kernels arrive in M2
    bool  tie_embeddings = true; // two names, one flat span (invariant 9)
};

// Defines the experiment. (ModelConfig, TrainConfig, seed, data) ⇒ reproducible run.
struct TrainConfig {
    float lr = 3e-4f, weight_decay = 0.1f, grad_clip = 1.0f;
    float beta1 = 0.9f, beta2 = 0.95f;
    int   batch = 32, grad_accum = 1;
    int   steps = 5000, warmup = 100;
    LrScheduleType schedule = LrScheduleType::WarmupCosine;
    uint64_t seed = 1337;
    PrecisionPolicy precision = PrecisionPolicy::fp32();
};

// Machinery only. MUST NOT affect the math / convergence.
struct RunConfig {
    KernelBackend backend = KernelBackend::Naive;   // Naive | Fused
    std::string checkpoint_dir = "ckpt/";
    int  log_every = 100, ckpt_every = 1000;
    bool nvtx = false;
};
```

All three serialize to/from JSON (experiments get diffed within a week of the
library working).

### 7.2 Core vocabulary types

```cpp
enum class DType { F32, BF16, /* later: FP8 */ };

// Non-owning view. Contiguous, row-major, up to 4 dims. No autograd, no strides.
struct Tensor {
    void*  data;
    DType  dtype;
    Shape  shape;                 // small fixed array + rank
    int64_t numel() const;
    size_t  bytes() const;
};

// Everything a launch needs, threaded explicitly through all calls.
struct RunCtx {
    cudaStream_t     stream;
    cublasLtHandle_t blas;
    Arena&           activations;   // pre-planned arena (see §8)
    const PrecisionPolicy& precision;
    KernelBackend    backend;
    uint64_t         seed;          // counter-based RNG root
    int64_t          step;          // RNG offset component
};
```

### 7.3 The model interface

```cpp
class GPT {
public:
    GPT(const ModelConfig& cfg, Device& dev, const PrecisionPolicy& prec);

    // tokens: [B,T] int32 (device) · targets: [B,T] int32 · loss_mask: [B,T] f32
    // Returns mean masked loss. Activations retained for backward.
    float forward(RunCtx& ctx, const Tensor& tokens,
                  const Tensor& targets, const Tensor& loss_mask);

    void  backward(RunCtx& ctx);          // fills the flat grad buffer

    // Inference-only forward for sampling/eval (no activation retention).
    void  forward_logits(RunCtx& ctx, const Tensor& tokens, Tensor& logits);

    ParamStore&        params();          // flat buffers + named views + roles
    const ModelConfig& config() const;

    void save(const std::string& path) const;        // versioned, self-describing
    static GPT load(const std::string& path, Device& dev);
};
```

### 7.4 Optimizer, schedule, data

```cpp
class AdamW {
public:
    // Param groups are constructed by ROLE (embedding / matrix / norm / bias):
    // no weight decay on norms & embeddings, optional separate embedding lr.
    AdamW(ParamStore& params, const TrainConfig& cfg);
    void  step(RunCtx& ctx, float lr);    // fused kernel over flat buffers
    float clip_grad_norm(RunCtx& ctx, float max_norm);  // one reduction
    void  zero_grad(RunCtx& ctx);
};

float lr_at(const TrainConfig& cfg, int64_t step);   // pure function

struct Batch { Tensor tokens, targets, loss_mask; }; // device-resident

class DataLoader {
public:
    // Memory-mapped flat binary of token ids (produced by tools/prepare_data.py).
    DataLoader(const std::string& bin_path, int batch, int seq_len, uint64_t seed);
    Batch next(RunCtx& ctx);   // pinned-host → device copy on ctx.stream
};
```

---

## 8. Architecture & class overview

### 8.1 Layer diagram (who owns what)

```
┌────────────────────────────── examples/hello_train.cpp ─────────────────────┐
│  owns the training loop; composes everything below                          │
└──────────────────────────────────────────────────────────────────────────────┘
    │            │              │               │                │
    ▼            ▼              ▼               ▼                ▼
┌────────┐  ┌─────────┐  ┌────────────┐  ┌───────────┐  ┌───────────────┐
│  GPT   │  │  AdamW  │  │ DataLoader │  │ Checkpoint│  │ Logger/Timer  │
│(model/)│  │ (optim/)│  │  (data/)   │  │  (model/) │  │ MFU, NVTX     │
└───┬────┘  └────┬────┘  └────────────┘  └───────────┘  └───────────────┘
    │            │ operates on flat buffers
    ▼            ▼
┌──────────────────────────────┐     ┌──────────────────────────────┐
│ ParamStore                   │     │ ActivationPlanner + Arena    │
│ flat params/grads/m/v +      │     │ layers declare needs →       │
│ named views + roles          │     │ planner assigns offsets      │
└──────────────┬───────────────┘     └──────────────┬───────────────┘
               │ views                              │ buffers
               ▼                                    ▼
┌──────────────────────────────────────────────────────────────────┐
│ layers/: Embedding · Norm · Attention · MLP · Block · LMHead     │
│ concrete classes, each with plan() / forward() / backward()      │
└──────────────────────────────┬───────────────────────────────────┘
                               │ launches
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│ kernels/: stateless __global__ fns (naive + fused variants)      │
│ + cuBLASLt for GEMM                                              │
└──────────────────────────────┬───────────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│ core/: Device · Arena · Tensor · DType · Rng · error handling    │
└──────────────────────────────────────────────────────────────────┘
```

### 8.2 Class responsibilities

**core/**
- `Device` — device selection, cuBLASLt handle, stream creation, properties
  (SM count, peak FLOPs & bandwidth for MFU computation).
- `Arena` — one big `cudaMalloc`, bump allocation with alignment; `reset()`
  between phases. Two instances: persistent (params/opt state) and per-step
  (activations).
- `Tensor` — dumb view (§7.2). Creation helpers: `arena.alloc({B,T,C}, DType::F32)`.
- `Rng` — Philox counter-based; `rng(seed, stream_id, offset)`.

**ParamStore** (model/)
- Registration phase: each layer calls
  `store.add("blk3.attn.wq", {C, C}, Role::Matrix)` → returns a `Param`
  (weight view + grad view; views become valid at finalize). After
  registration, `finalize()` makes **one** device allocation (a private
  Arena) carved into four 256B-aligned spans — params, grads, Adam m,
  Adam v — each packing all parameters in registration order (spans of
  equal dtype share per-parameter offsets), zero-filled at creation.
  Alignment padding inside the spans stays zero for the store's lifetime,
  so flat-span reductions/updates read only inert zeros (invariant 5).
- Provides: named lookup (checkpoint I/O, tests), iteration by role
  (param groups, init), flat spans (fused optimizer step, clipping, future NCCL).
- Aliasing: `store.alias("lm_head.w", "embed.wte")` for weight tying — both
  names resolve to one span; the shared grad view accumulates from both
  backward contributions (invariant 9).

**ActivationPlanner** (core/)
- **Owned by `GPT`** (only the model knows its layers' needs), along with the
  activation arena it fills. Planning runs once, lazily, when the batch shape
  is first seen; `RunCtx` merely carries a reference to the arena.
- Pass 1 (`plan`): walk layers, each declares
  `need("attn_scores", {B,H,T,T})`; planner assigns arena offsets
  (M1: naive stacking; M3: lifetime-based reuse + checkpointing).
- Pass 2 (`bind`): hand each layer its resolved `Tensor` views.
- Forward/backward then run with **zero allocation**.

**layers/** — each concrete class follows the same contract. This is a
*convention, not an abstract base class*: composition is fully static (`Block`
holds concrete `Norm`/`Attention`/`MLP` members, `GPT` holds `vector<Block>`),
so virtual dispatch would only add indirection in the hot loop. Variation
(RMSNorm vs LayerNorm, GELU vs SwiGLU) is an enum switch *inside* the concrete
class, per the seams in §5. If we want the contract compiler-checked, a C++20
`concept` does it at zero runtime cost. A virtual `Layer` base earns its place
only if M5-style dynamic architectures ever happen.

```cpp
class Attention {
public:
    Attention(const BlockConfig& cfg, int layer_idx, ParamStore& ps);
    void plan(ActivationPlanner& p, int B, int T);
    void forward (RunCtx& ctx, const Tensor& x, Tensor& y);
    void backward(RunCtx& ctx, const Tensor& dy, Tensor& dx);  // accumulates dW
private:
    Param wq_, wk_, wv_, wo_;
    Tensor qkv_, scores_, attn_out_;   // planner-assigned activation views
    MaskSpec mask_;                    // seam: never baked into the kernel
};
```

`Block` composes Norm → Attention → residual → Norm → MLP → residual and is the
single place block wiring is readable. `GPT` owns Embedding → `vector<Block>` →
final Norm → LMHead → fused cross-entropy, and drives plan/bind/forward/backward
in order (backward in reverse).

**kernels/** — free functions only, e.g.
`void rmsnorm_fwd(cudaStream_t, float* y, float* rstd, const float* x, const float* w, int N, int C);`
Naive and fused variants share one signature; the layer picks via
`ctx.backend`. GEMM goes through one `matmul()` wrapper owning all cuBLASLt
setup — the only file that knows cuBLASLt exists.

### 8.3 Dataflow of one training step

```
loader.next() ─► tokens,targets,mask [B,T] device
model.forward: embed ─► block×N (norm→attn→res→norm→mlp→res) ─► norm ─► lm_head GEMM
               ─► fused softmax+CE (masked mean) ─► loss (one float to host)
model.backward: exact reverse; dW accumulates into flat grad buffer
opt: clip_grad_norm (one reduction) ─► adamw fused step (one kernel) ─► zero_grad
```

### 8.4 Memory budget (M1 defaults: B=32, T=256, C=384, L=6, V=50304, fp32)

| Buffer | Size | Note |
|---|---|---|
| logits `[B,T,V]` | **~1.65 GB** | the dominant buffer; `dlogits` is computed **in place of** logits by the fused CE backward — never two vocab-sized buffers |
| block activations | ~150 MB/layer × 6 ≈ 0.9 GB | attention probs `[B,H,T,T]` ≈ 50 MB/layer overwrite scores in place |
| params + grads + m + v | ~30 M × 16 B ≈ 0.5 GB | flat buffers |
| **Total** | **≈ 3–3.5 GB** | fits an 8 GB GPU; planner prints the real plan at startup |

For fast development iteration there is a **char-level data mode** (vocab ≈ 65):
it shrinks logits to ~2 MB, removes the tiktoken dependency from the dev loop,
and makes smoke tests run in seconds. GPT-2 BPE mode is the "real" M1 target.

---

## 9. Hello-world training application

The complete `examples/hello_train.cpp` — this is the acceptance test for the
whole M1 API design:

```cpp
#include <llmt/llmt.h>
using namespace llmt;

int main(int argc, char** argv) {
    // 1. Configs (could equally be loaded from JSON files)
    ModelConfig mc{.n_layer = 6, .n_head = 6, .n_kv_head = 6,
                   .d_model = 384, .vocab = 50304, .seq_len = 256};
    TrainConfig tc{.lr = 3e-4f, .batch = 32, .steps = 5000,
                   .warmup = 100, .seed = 1337};
    RunConfig   rc{.backend = KernelBackend::Naive,
                   .checkpoint_dir = "ckpt/", .log_every = 100};

    // 2. Wire up
    Device dev(0);
    GPT model(mc, dev, tc.precision);            // registers params, inits weights
    AdamW opt(model.params(), tc);               // role-based param groups
    DataLoader train("data/tiny_train.bin", tc.batch, mc.seq_len, tc.seed);
    DataLoader val  ("data/tiny_val.bin",   tc.batch, mc.seq_len, tc.seed);
    RunCtx ctx = dev.make_ctx(rc, tc);           // stream, rng, backend
    // (activation planning happens inside GPT at first forward — it owns
    //  the planner and its arena; see §8.2)

    // 3. The loop — owned by the user, on purpose
    StepTimer timer(dev, mc, tc);                // tokens/sec + MFU
    for (int64_t step = 1; step <= tc.steps; ++step) {
        ctx.step = step;
        Batch b = train.next(ctx);

        float loss = model.forward(ctx, b.tokens, b.targets, b.loss_mask);
        model.backward(ctx);
        float gnorm = opt.clip_grad_norm(ctx, tc.grad_clip);
        opt.step(ctx, lr_at(tc, step));
        opt.zero_grad(ctx);

        if (step % rc.log_every == 0)
            log_step(step, loss, gnorm, lr_at(tc, step), timer.report());
        if (step % rc.ckpt_every == 0) {
            float vloss = evaluate(ctx, model, val, /*iters=*/20);
            model.save(rc.checkpoint_dir + "step" + std::to_string(step) + ".llmt");
            log_eval(step, vloss);
        }
    }
    return 0;
}
```

**M1 exit criterion** (loss numbers are tokenizer-dependent):

- Initial loss ≈ `ln(vocab)` (≈ 10.8 for GPT-2 BPE, ≈ 4.2 for char-level) —
  a wrong initial loss is itself a bug signal.
- Char-level TinyShakespeare: val loss < 1.6 within a few thousand steps.
- GPT-2 BPE TinyShakespeare: train loss < 3.5 and steadily falling
  (the small corpus overfits — expected without dropout).
- Final val loss within a few % of the PyTorch reference script under an
  identical budget (§10.2).
- `examples/generate.cpp` (load checkpoint → `forward_logits` → top-k sample)
  produces recognizably Shakespeare-shaped text.

---

## 10. Testing & evaluation strategy

Four layers of testing, built **alongside** the code, not after. Rule of thumb:
the oracle infrastructure is written *before* the first real kernel.

### 10.1 Correctness: golden-file tests (PyTorch as oracle, never linked)

- `tools/gen_golden.py` builds each layer in PyTorch with **identical config and
  deterministic inputs**, dumps `input / params / output / d_output / d_input /
  d_params` as raw `.bin` + a small JSON manifest into `tests/golden/`.
  Files are committed (tiny shapes) so CI needs no Python.
- Each C++ unit test loads the golden files, runs our layer's `forward` and
  `backward`, and diffs: fp32 tolerance `~1e-5` relative, bf16 later `~1e-2`.
- Coverage per layer, both directions: embedding, norm, attention (per-head,
  masked), MLP, fused cross-entropy, RoPE, full block, full model 2-layer micro
  config, and **AdamW itself** (one step vs `torch.optim.AdamW`).
- **Backends are cross-checked:** every fused kernel runs against both the golden
  file *and* the naive kernel. The naive path is permanent.
- Supplement: finite-difference gradient check on micro shapes (catches oracle
  script bugs).

### 10.2 Correctness: end-to-end

- **Determinism test (CI):** same seed, two runs, 50 steps → bitwise-identical
  loss sequence and parameter buffers. Catches uninitialized memory, stray
  atomics, RNG misuse.
- **Convergence smoke test (CI, ~minutes):** 4-layer micro model, 500 steps on a
  fixed corpus → assert `loss < threshold`. Thresholds recorded per config.
- **Reference-run comparison (manual, per milestone):** identical model in a
  ~100-line PyTorch script, same data order, same init (loadable from our
  checkpoint format), same hyperparameters → overlay loss curves. They should
  track within noise for hundreds of steps. This is the strongest possible
  evidence the whole system is right, and the first thing to reach for when a
  run diverges.
- **RunConfig invariance:** switching `KernelBackend` must not change
  *convergence* (allowing rounding-level per-step differences).

### 10.3 Performance: benchmarks with roofline context

Raw ms numbers are meaningless without a bound to compare against; every
benchmark reports **% of a theoretical limit**:

- **Kernel micro-benchmarks** (`bench/kernels`): each kernel timed over a shape
  sweep via CUDA events; memory-bound kernels (norms, residual, softmax, adamw)
  report **achieved vs peak HBM bandwidth** — e.g. "rmsnorm_fwd: 78% of peak BW".
  Regression-tracked as JSON per commit.
- **End-to-end MFU** (`bench/train`): tokens/sec and
  `MFU = achieved model FLOPs / peak GPU FLOPs` printed by `StepTimer` in every
  run. The single number that says how far we are from cuBLAS-bound. Realistic
  targets: M1 naive ~10–20%, M2 fused ~25–35%, M3 flash-attention 40%+.
- **Memory report:** planner prints the activation plan (per-buffer sizes,
  total) at startup; asserted against an analytic formula in tests.
- **NVTX ranges** per layer from day 1 → named timelines in Nsight Systems;
  `RunConfig::nvtx` toggles them.

### 10.4 Model quality evaluation (is the *training* any good?)

- **Validation loss / perplexity** on a held-out split, logged every checkpoint.
- **Sampled generations** at fixed prompts and temperature per checkpoint —
  qualitative but catches whole classes of subtle bugs (e.g. off-by-one in the
  causal mask produces fluent-looking garbage with suspiciously low loss).
- **Baseline parity:** final val loss within a few % of the PyTorch reference
  script trained with the same budget. Deviation = bug until proven otherwise.

### 10.5 Development order (test-first where it pays)

1. core/ + ParamStore + Arena → unit tests, determinism harness
2. `gen_golden.py` + golden-file test harness  ← *before any real kernel*
3. matmul wrapper + Linear → golden test
4. Norm, GELU, residual → golden tests + bandwidth benchmarks
5. Embedding, RoPE, naive attention, fused CE → golden tests
6. Block, GPT forward → golden test vs PyTorch full-model
7. Backward passes, layer by layer, reverse order → golden tests
   *(budget: as much time as items 1–6 combined)*
8. AdamW + clipping → golden test; convergence smoke test
9. DataLoader, checkpointing, hello_train → reference-run comparison
10. Generation example → M1 exit

---

## 11. Risks / known hard parts

| Risk | Mitigation |
|---|---|
| Backward-pass bugs (norm & attention especially) | golden tests per layer, finite differences, reference-run overlay |
| Silent divergence hard to localize | determinism + bitwise reproducibility; bisect by layer with goldens |
| Fused kernels calcifying the architecture | `MaskSpec` seam, naive path permanent, backend cross-checks |
| cuBLASLt API friction (layouts, heuristics, workspaces) | quarantine in one `matmul()` wrapper file; pin/cache algo selection (determinism) |
| Nondeterminism creeping in via atomics / algo heuristics | invariant 5: no atomics in grad paths, cached GEMM algos, bitwise CI test |
| Logits memory (`B·T·V`) dwarfing everything | in-place `dlogits` in fused CE; char-level dev mode; memory budget in §8.4 |
| No local CUDA (macOS dev host) | remote Linux/NVIDIA box workflow scripted from day 0 (sync → build → test); Phase 0 deliverable |
| Scope creep toward "framework" (M5) | milestone gates; frozen list in §5 |
