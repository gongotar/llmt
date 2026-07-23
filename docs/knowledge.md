# Knowledge base

Working notes on concepts behind the code — grows as the project does.

## Tensor dimension notation (B, T, C, H, hd, V)

Standard letters used in every signature, comment, and shape in this library.
M1 default config as the running example:

| Letter | Name | Ours | Meaning |
|---|---|---|---|
| **B** | batch | 32 | sequences trained *simultaneously* — pure GPU parallelism; sequences never interact |
| **T** | time / sequence length | 256 | tokens per sequence (position = the sequence's time axis) |
| **C** | channels / d_model | 384 | width of the model: every token is a vector of C numbers ("the residual stream") |
| **V** | vocabulary | 50304 | number of distinct token ids |
| **H** | heads | 6 | attention splits into H independent heads, each learning different patterns |
| **hd** | head dim | 64 | = C/H; each head works on its own slice of the C channels (6 × 64 = 384) |

### Shape flow of one training step

```
tokens        [B,T]        integers — token ids
  ↓ embedding lookup (each id selects its row of the [V,C] table)
activations   [B,T,C]      every token is now a C-dim vector
  ↓ transformer blocks: always [B,T,C] in → [B,T,C] out
  ↓ inside attention only — split C into heads:
Q,K,V         [B,H,T,hd]   rank 4: batch × head × position × head-slice
attn scores   [B,H,T,T]    rank 4: per head, every position vs every position
  ↓ merge heads back
activations   [B,T,C]
  ↓ project against embedding table (weight tying)
logits        [B,T,V]      per token: one score per vocabulary word
loss          scalar
```

### Why rank ≤ 4 is enough (and not a compromise)

Rank 4 appears **only inside attention**, where two independent groupings
(batch, head) coexist with two interacting axes (position × position, or
position × head-slice). No transformer construct needs a fifth axis.
Everything else is rank ≤ 3 — the `[B,T,C]` residual stream is the highway;
attention temporarily reshapes to rank 4 and merges back.

PyTorch supports arbitrary rank because it can't know your model; we know
exactly what we're building, so `Shape` is a fixed `int64_t[4]` + rank —
a stack value type, copyable straight into kernel arguments, no heap.
("Flexibility in the math, rigidity in the machinery" applied to one struct.)

The `[B,T,C] ↔ [B,H,T,hd]` reshapes physically reorder memory (permute
kernels, task 6.2) — a classic silent-bug site, hence golden-tested.

## Mixed precision: why PrecisionPolicy has three members

The canonical M2 setting is `{master=FP32, compute=BF16, reduce=FP32}` — three
answers to three different questions about the same training step:

- **master = FP32** — parameters must stay fp32 because of the optimizer
  update: `w += lr · Δ` where `Δ/w ≈ 10⁻⁴…10⁻⁶`. BF16 has ~3 decimal digits of
  mantissa — most updates would round to exactly zero and training silently
  stalls. So weights live in fp32; a bf16 copy is made for compute.
- **compute = BF16** — GEMMs and activations. This is where ~99% of FLOPs and
  memory traffic are, tensor cores run bf16 at ~2× fp32-tensor throughput, and
  activations halve in size. The math tolerates it: individual products don't
  need many digits.
- **reduce = FP32** — anything that *sums many values*: norm variances (sum
  over 384 channels), softmax denominators, the loss mean (sum over B·T
  tokens), gradient accumulations. Adding thousands of bf16 values loses
  low-order bits catastrophically; the standard fix is: read bf16, accumulate
  in an fp32 register, write bf16. Costs nearly nothing (the accumulator is
  one register), saves the numerics.

Common per-layer/per-site overrides (M2, via a resolution API added when its
first consumer exists): logits/loss pinned to fp32 (softmax over a 50k vocab
is precision-hungry), and debug-pinning suspect layers to fp32 to bisect
instabilities. Related design note: dtype dispatch is a one-branch runtime
`switch` at each kernel launch site selecting a template instantiation —
kernels are compile-time typed inside; dtype is static per run (frozen list),
so CUDA graphs / fusion / JIT capture concrete kernels and the switch cost is
erased on replay.

## Parameters: roles, init, weight decay

**Parameters** (= weights) are the adjustable numbers training nudges; the
model is ~30M of them, all registered in ParamStore. **Activations** are
computed per forward pass from inputs × parameters — parameters are stored
and initialized; activations are transient and never "initialized".

**Residual stream**: each block amends its input rather than replacing it —
`output = x + f(x)`. The flowing `x` `[B,T,C]` is a per-token "whiteboard"
every block adds a suggestion onto. `wo` (attention output projection) and
the MLP down-projection are the *last matrices inside* `f` — their product
is what lands on the whiteboard, so they're tagged `Role::ResidualProj`.
At init, shrinking those matrices (σ = 0.02/√(2L)) shrinks the 2L random
additions and keeps the stream's variance flat across depth. (The matrices
are directly random-initialized; `f(x)` inherits randomness through them.)

**Norms** are volume regulators: RMSNorm divides each token vector by its
own rms (standard "volume" regardless of stream drift — softmax hates wild
scales), then multiplies channel-wise by a learnable `g[C]`. Init `g = 1` =
"no exceptions yet, change nothing" — the multiplicative neutral.

**Weight decay**: `w ← w − lr·grad − lr·λ·w` — every weight shrinks by
`lr·λ` (≈0.003%/step) unless gradients keep re-funding it. Rent economy:
recurring patterns earn their rent, one-off memorizations (large, finely
tuned weights = jagged, overfit functions) dissolve. Exemptions where
"smaller ≠ simpler": Norm `g` (neutral is 1, not 0) and embeddings (rare
tokens pay rent every step but earn income rarely).

### Parameter reference (M1 model: L=6, C=384, V=50304, d_ff=1536)

| Name | Shape | Role | Meaning |
|---|---|---|---|
| `embed.wte` | [V,C] | Embedding | **w**ord **t**oken **e**mbeddings — token lookup table (GPT-2 name) |
| `blk{b}.norm1.g` / `norm2.g` | [C] | Norm | RMSNorm scales before attention / MLP |
| `blk{b}.attn.wq/wk/wv` | [C,C] | Matrix | query ("what am I looking for") / key ("what do I offer") / value ("what do I contribute") projections |
| `blk{b}.attn.wo` | [C,C] | ResidualProj | attention output projection → residual stream |
| `blk{b}.mlp.w_up` | [d_ff,C] | Matrix | widen C → d_ff |
| `blk{b}.mlp.w_down` | [C,d_ff] | ResidualProj | narrow back, → residual stream |
| `final_norm.g` | [C] | Norm | last norm before scoring |
| `lm_head.w` | alias→wte | — | scoring matrix (weight tying) |

50 tensors, ~30M elements (embedding ≈ 19M). No `wpe` (GPT-2's positional
embeddings) — RoPE computes positions instead of storing them.

**normal(0, 0.02)**: theory (Xavier/He) says σ ≈ 1/√C to keep variance
steady through `y = xW`; GPT-2 fixed 0.02 for all sizes and it became the
convention every reproduction matches (incl. us — oracle comparability).

**C = 384, 6 heads, 6 layers**: the classic tiny-GPT config (nanoGPT's
TinyShakespeare model) — head dim 64 is the field-wide standard, ~30M
params, minutes to train. `lr` (3e-4) = global step size; `λ` (0.1) =
weight-decay rent coefficient; effective shrink per step is `lr·λ`.

## CUDA fundamentals (as encountered in this codebase)

- **Kernel launch** `k<<<blocks, threads>>>(args)`: grid of blocks × threads;
  `(n + 255)/256` = ceiling division so every element gets a thread; kernels
  guard with `if (i < n)`. Launches are **asynchronous** for the CPU.
- **Thread identity**: `i = blockIdx.x * blockDim.x + threadIdx.x` — global
  index = block number × block size + position in block. `.x` because grids
  can be 3-D; we mostly use 1-D.
- **`__global__`** = kernel (runs on GPU, launched from CPU, returns void);
  **`__device__`** = GPU-only helper; **`__host__ __device__`** = compiled for
  both (our `LLMT_HD` macro — lets pure functions like Philox run in host
  tests *and* kernels).
- **Streams**: per-stream FIFO ordering is the synchronization model — a
  memcpy issued after a kernel in the same stream waits automatically.
  Overlap requires multiple streams (M2+); `RunCtx` carries the stream
  explicitly from day 1.
- **`cudaMalloc`** allocates GPU VRAM and returns a *device* virtual address —
  not CPU-dereferenceable, not mmap, no DMA at allocation time.
- **DMA & pinned memory**: transfers are done by DMA engines that read
  physical addresses and cannot page-fault, so pageable host memory must be
  staged through a pinned buffer (even for blocking copies). Allocating
  pinned (`cudaMallocHost`) skips the staging and enables true async —
  the reason the DataLoader uses a pinned staging buffer.

## Linear layers: backward is just more matmuls

For `y = x·Wᵀ` (W stored [out, in], PyTorch convention), the gradients are
computed *literally* by matrix multiplication — no new kind of operation:

    forward:   y  = x · Wᵀ
    backward:  dx = dy · W        (grad w.r.t. the input)
               dW += dyᵀ · x      (grad w.r.t. the weight; accumulates)

Why: x[i][k] touched every output y[i][j] with coefficient W[j][k], so its
gradient collects each output's gradient weighted by the same coefficient —
dx[i][k] = Σⱼ dy[i][j]·W[j][k], which is by definition the product dy·W.
`dy` is the gradient arriving from the layers above (in isolated layer tests:
fabricated random noise). Three GEMMs, one layer — Linear is ~15 lines.

PyTorch parallel: `requires_grad=True` gives a tensor a same-shaped `.grad`
companion (≙ our Param{weight, grad}); `y.backward(dy)` chain-rules through
the recorded graph and deposits results into every `.grad` (accumulating,
like our invariant 9).

### The two streams meeting in backward

    FORWARD (saved):   x₁ ─▶ [norm] ─▶ x₂ ─▶ [linear] ─▶ x₃ ─▶ [head] ─▶ loss
    BACKWARD (flows):     ◀─dx₁─      ◀─dx₂─         ◀─dx₃─          ◀─ dloss

Backward at layer k needs two ingredients and yields two products:
- needs `dy` — the gradient arriving from the layer above (= that layer's
  `dx`; the baton is transformed at every hop, never "the same dy"), and
  `x` — the layer's own forward input, REMEMBERED (an activation, not a
  gradient — why the planner retains buffers after forward).
- yields `dW` (kept for its own weights) and `dx` (becomes the next layer
  down's `dy`; without it the chain snaps and everything below learns
  nothing). Exception: the embedding passes no dx — no gradient into
  integer token ids.

`dW` never touches weights inside backward: the optimizer step (a separate
loop phase, after ALL grads are complete) applies `w -= lr·(…)` over
flat(Grad) in one fused kernel.

## The three non-GEMM ops, one line each

- **RMSNorm** standardizes what enters a block: `y = x/√(mean(x²)+eps) · g` —
  volume regulation per token vector (rstd saved for backward).
- **GELU** makes depth nonlinear: smooth "pass positive, squash negative"
  between the MLP's matrices — without it, stacked Linears collapse into one.
- **Softmax** converts "how relevant" scores into "how much attention"
  weights: positive, sum to 1, soft-winner-take-most; the max-subtraction is
  purely overflow safety; causal-masked entries get exactly 0.

## Attention's shape story (per sequence; full mechanics in Phase 6)

1. **Projections** (weights involved, ordinary Linears): x `[T,C]` →
   Q, K, V, each `[T,C]`, via `x·Wq/k/vᵀ`. Tokens are rows.
2. **Head split** (no math, slicing + permute): head h owns its own 64 of
   the 384 channels → `Q_h [T,hd]`, `K_h [T,hd]`. Different slices,
   different numbers per head — nothing repeated.
3. **Scores** (NO weights — activation × activation):
   `scores_h = Q_h · K_hᵀ` → `[T,hd]·[hd,T] = [T,T]` — every token's query
   dotted with every token's key. B·H = 192 private little GEMMs → one
   rank-3 batched matmul call.

## Strided-batched GEMM

One launch = batch-many INDEPENDENT 2-D GEMMs in parallel (each with full
`α·A_i·B_i + β·C_i` semantics). The batch axis of a `[batch, rows, cols]`
tensor is only iterated, never multiplied along — slice i of A meets only
slice i of B. "Strided": all slices live in one contiguous buffer, item i at
`base + i·stride` (stride = rows·cols) — one pointer + one stride per tensor
instead of a pointer list. This is attention's shape: B·H = 192 heads, each
its own private QKᵀ, one launch instead of 192.

## GEMM efficiency: why "% of nominal peak" is measured against a fantasy

Two structural discounts sit between a GPU's nominal FP32 peak and what any
GEMM can achieve — compute both before judging a benchmark number:

**1. The roofline.** A GEMM must move its operands; if the bytes take longer
than the math, bandwidth binds. Arithmetic intensity AI = FLOPs / bytes
(minimum traffic: read A, B; write C); the ridge point is
`peak_flops / mem_bandwidth` (≈36 FLOP/B on RTX A5000). Shapes with AI below
the ridge are memory-bound: their true ceiling is `bandwidth · AI`, not the
compute peak. Our batched attention scores (AI ≈ 21) cap at ~59% of nominal
*by physics* — small per-batch matrices additionally waste tiles/launches,
which is exactly what flash-attention-style fusion (M3) recovers.

**2. The Ampere-consumer shared pipe.** `sm_86`'s "128 FP32 cores/SM" are
64 dedicated FP32 + 64 shared FP32-or-INT32 units; nominal peak assumes all
128 do FMAs every cycle, but real kernels must issue integer work (addresses,
counters, predicates) which steals shared-half slots. Empirically, cuBLAS
SGEMM on GA10x tops out at ~55–65% of nominal even for huge square matrices.
(A100-class SMs are balanced differently — MFU numbers don't compare across
GPU classes.)

Methodology: the bench's CEILING case (4096³ square GEMM — maximal AI, no
quantization) measures the empirical arch ceiling once; judge every shape
against `min(ceiling, its roofline)`. Our M1 projections at 54–57% of nominal
are at/near that practical ceiling — nothing is "missing".

Escape hatch deliberately declined for M1: TF32 tensor cores would lift the
compute roof ~4× at 10-bit mantissa cost — a precision-policy decision
(would break 1e-5 golden tolerances), not a wrapper default.

## Blocks, SMs, and choosing a block size

A block runs *entirely on one SM* (never split), but an SM hosts **multiple
blocks concurrently** — packed until one per-SM budget runs out: resident
threads (1536 on `sm_86`/`sm_89`), resident blocks (24 on Ada), 64K registers,
~100 KB shared memory. Whatever exhausts first caps co-residency; the warp
scheduler interleaves all resident warps to hide memory stalls.

Why 1024-thread blocks are suboptimal on consumer GPUs (÷1536/SM):

| block | blocks/SM | resident threads | occupancy ceiling |
|---|---|---|---|
| 256 | 6 | 1536 | 100% |
| 512 | 3 | 1536 | 100% |
| 1024 | 1 (512 slots wasted) | 1024 | 67% |

Additional strikes against big blocks: register stiffness (64K ÷ 1024 = 64
regs/thread before one block even fits; smaller blocks degrade gracefully
instead), and coarse tail granularity (the last wave of a launch parcels work
in block units; block-wide `__syncthreads()` waits on 32 warps vs 8).
Datacenter parts (A100/H100) allow 2048 threads/SM, which 1024 divides — why
HPC code sometimes uses it. Block size is per-kernel, per-arch tuning → a
file-local constant in each kernel family, never a global.

**Rule of thumb**: smallest multiple of 32 that satisfies the kernel's
cooperation needs — elementwise kernels need none (256 is generous),
reductions want the row to fit, only shared-memory tiling argues for large.

## Precision: which type governs what, per kernel

Storage dtype always comes from the tensors (template `T`, honored at
load/store). Arithmetic precision is a *policy* axis only where the hardware
has a real dial: `gemm_compute` selects the tensor-core multiply mode
(FP32/TF32/BF16/FP16 — the only place narrow arithmetic is faster), and
`reduce` selects accumulator precision (GEMM accumulators, norm/softmax sums
— kept separate because the standard bf16 mode is *multiply in bf16,
accumulate in fp32*). All other non-GEMM arithmetic is fp32 **by design**,
not by policy: those kernels are memory-bound, so narrower math saves
nothing and only adds rounding — expressed once as the
`kernel_compute_t = float` alias, never as bare literals. A kernel's
interface carries exactly the policy axes that apply to it:

| Kernel | Storage | Elementwise arithmetic | Accumulation / reduction | Takes |
|---|---|---|---|---|
| `matmul` | T (uniform A/B/C) | `policy.gemm_compute` | `policy.reduce` (Lt derives scale type from the pair) | `RunCtx` |
| `rmsnorm_fwd` | T | `kernel_compute_t` | `policy.reduce` — sum-of-squares and `rstd` dtype | `RunCtx` |
| `softmax_fwd` | T | `kernel_compute_t` (exp, scale) | `policy.reduce` — max/sum | `RunCtx` |
| `gelu_fwd` | T | `kernel_compute_t` | — | stream |
| `residual_fwd` | T | `kernel_compute_t` | — | stream |
| `fill_value` | T (value checked) | — | — | stream |
| `cross_entropy_fwd` | T (logits/mask) | `kernel_compute_t` (exp, log) | `policy.reduce` — max/lse/mean accumulators; but loss & row_nll STORE as `host_result_t` | `RunCtx` |

A second fixed alias, `host_result_t = float` (precision.h): results
published to the host (the loss scalar, per-token NLL) never inherit the
reduce dtype — those tensors are tiny, so narrowing buys no bandwidth and
only quantizes the numbers training is steered by (an fp8 reduce would
turn the loss curve into a ~0.5-step staircase). reduce governs how sums
are CARRIED; what the host reads is a publication contract. (Same reason
PyTorch AMP returns fp32 losses.) rstd is different: device-internal,
consumed by backward under the same policy — it keeps the reduce dtype.

Capability today: every axis FP32, each enforced by its own labeled guard.

## The attention layout sandwich (why permute exists)

Token-row land is where Linears live (one [C] row per token); per-head land
[B, H, T, hd] is where the batched attention GEMMs and softmax live (each
head's [T, hd] block contiguous = one batch entry). permute_split /
permute_merge are the border crossings — pure copies, no arithmetic:

  x [B,T,C] → Linear(qkv) → split → rope → attention_fwd → Linear(Wo)
              token rows   ┕━ per-head land [B,H,T,hd] ━┙  token rows

Merge runs as attention_fwd's last stage (settled: the op takes
per-head q/k/v and returns token rows — the future fused kernel writes
token rows natively, so the shared-signature seam wants merge inside; the
naive-only scratch tensors are simply invalid under the fused backend).
The backward pass mirrors the roles (grads get split where activations were
merged and vice versa). llmt Tensors are deliberately stride-less, so the
per-head "view" is materialized by a copy instead of stride tricks; the two
copies are memory-bound noise next to the GEMMs, and M3's fused attention
absorbs the whole sandwich.

## Grid math: threads launched = work units × threads per unit

Elementwise kernels assign **1 thread per element**, so blocks =
ceil(n / block_size) and thread count ≈ element count. Cooperative kernels
assign a **team per work unit** — rmsnorm/softmax give each row a 32-thread
warp (coalesced loads + register-shuffle reduction) — so the grid is counted
in units per block (`ceil(n_rows / kWarpsPerBlock)`, 8 rows per 256-thread
block), and total threads = 32 × rows *by design*, not by accident. When
checking a launch, always ask "what is the work unit and how many threads
own one?" — comparing raw thread count to element count only works in the
1-thread-per-unit case.

## Attention scores: Tq, Tk, and the causal triangle

The score tensor is [batch, Tq, Tk]. Both trailing axes count token
positions, in different roles: row q = "token q asking" (query), column
k = "token k being looked at" (key); scores[b][q][k] = how much q attends
to k. In self-attention both roles are played by the same T tokens
(T = sequence length), so Tq = Tk = T and the matrix is square — and only
then does "column q" mean "my own position", which is what the causal rule
cuts at. Causal = keep yourself and earlier (k ≤ q), mask the future
(--- entries get probability exactly 0):

|            | k=the | k=cat | k=sat | k=down |
|------------|-------|-------|-------|--------|
| **q=the**  | s00   | ---   | ---   | ---    |
| **q=cat**  | s10   | s11   | ---   | ---    |
| **q=sat**  | s20   | s21   | s22   | ---    |
| **q=down** | s30   | s31   | s32   | s33    |

When the axes differ (KV-cache decode: Tq=1 against Tk=500 cached keys;
cross-attention), rows and columns index different sequences, "column q"
is meaningless, and the kernel's Tq == Tk check refuses — the mask must
then come from an explicit alignment, not from squareness.

## Counted vs actual bytes: the rmsnorm 69% post-mortem

The bandwidth bench counts *compulsory* traffic (each algorithm input read
once, each output written once) and trusts caches to absorb re-reads. That
trust failed for rmsnorm: its write pass re-reads the row, and a
kernel-variant bisection (A4500, 2026-07-17) showed the re-read largely
misses and returns to DRAM:

| variant (same [8192,384] shapes) | % peak | verdict |
|---|---|---|
| elementwise single-pass | 85.2 | baseline = residual |
| warp-per-row single-pass | 78.7 | row mapping: −6 pts |
| two passes, no re-read | 82.0 | reduction + phase gap ≈ free |
| two passes with re-read (= rmsnorm) | 67.8 | **re-read: −14 pts** |
| rmsnorm with warp_sum removed | 69.4 | shuffle barrier: 0 pts |

At a full miss the kernel moves 12 B/elem vs 8 counted → 434 GB/s counted
≈ 650 GB/s actual: the DRAM bus is near-saturated. Lessons: (1) a low
%-of-peak can mean "accounting undercounts real traffic", not "bus idle" —
check that before hunting kernel inefficiencies; (2) don't assume L1 holds
a row across two loop passes — with ~48 resident warps streaming, it
doesn't (store hints were a no-op, so read-stream churn/policy suffices to
evict; exact mechanism unpinned without counters); (3) cheap fix when it
matters (M2): keep the row in
registers across passes (12 floats/lane at C=384) — the no-reread variant
shows ≈ +13 pts. Method note: when ncu is unavailable (RunPod blocks GPU
perf counters — ERR_NVGPUCTRPERM), single-variable kernel variants answer
"why slow" questions almost as well.

**Epilogue — mechanism closed (RTX 4000 Ada, 2026-07-18).** An occupancy
sweep (persistent grid-stride warps, 48→8 warps/SM, no shared-memory
carveout confound) showed a ZERO re-read/no-reread gap at every occupancy
on Ada — so L1 never held the rows anywhere, and the L1 reuse-distance
theory was wrong too. The real story is one level down: **the re-read is
served by L2 iff L2 outsizes the GPU-wide churn between a line's touches**
(~7 MB reads+writes at these shapes). A4000 (4 MB L2): re-reads fall to
DRAM → 67.7%. A4500 (6 MB): partial → 69%. Ada (40 MB): free → rmsnorm
86.2%, best of the four kernels. Consequences: (a) the M2 register-caching
fix only pays on small-L2 parts — on workstation Ada the naive kernel is
already at the DRAM ceiling; (b) a bandwidth bench whose tensors fit L2
measures L2, not DRAM — the unscaled [8192,·] shapes hit 327–570% "of
peak" on Ada; bench shapes are now 16× in the batch axes (≥ 200 MB per
tensor) and the header prints the device's L2 size as a tripwire.

**Confirmation (A4500, 5 MiB L2, 2026-07-18, scaled shapes).** Predictions
held on one device: rmsnorm 70.0% depressed WHILE softmax held 84.1%
(differential test); gelu/residual 89%. The occupancy sweep in the DRAM
regime showed the gap collapsing monotonically 24 → 3 pts as warps/SM drop
48 → 8, with the re-read kernel reaching 83.3% at 1 block/SM — so
L2-aware occupancy capping genuinely recovers ~20 pts on small-L2 parts
(threshold lower than the naive ~7 MB churn estimate: effective L2 reuse
capacity < nominal). Still dominated by the register fix for production
(no occupancy dependence, no device model), but validated as a technique.

## Cache-hint intrinsics (per-access, since replacement policy is fixed)

L1/L2 replacement policy is hardware; what CUDA exposes is a cache *hint*
per load/store. The governing rule: **a hint only pays when it protects a
reuse from eviction** — allocating a line costs nothing by itself; the cost
is always the victim it evicts. Also: L1 does not survive kernel
boundaries, so L1-allocating a store that isn't re-read *in-kernel* is pure
cost; L2 does persist across kernels (producer → consumer).

| intrinsic | meaning | use when |
|---|---|---|
| default / `__ldca` | cache normally in L1+L2 | data re-read in-kernel (rmsnorm pass-1 rows, softmax rows, g) |
| `__ldcg` | load via L2 only, skip L1 | reused across blocks but not within (rare here) |
| `__ldcs` | load, evict-first in L1+L2 | streamed input never revisited |
| `__ldlu` | load, "last use" — drop after | final re-read of data nothing else needs |
| `__stcg` | store via L2 only, skip L1 | output not re-read in-kernel but consumed by the NEXT kernel (rmsnorm y, softmax probs) |
| `__stcs` | store, evict-first in L1+L2 | output nobody reads soon (also downgrades the L2 copy — beware) |

Tried and reverted (5.8, A4000, same-pod A/B): `__stcg` on rmsnorm/softmax
output stores changed nothing (67.5% vs 67.7% — noise). Conclusion: sm_86
evidently doesn't allocate stores in L1 by default, so there was nothing to
bypass — and the "write allocations push the working set past L1" theory of
the rmsnorm gap is out (the actual mechanism — L2 vs churn — is closed in
the post-mortem epilogue above). NOT applied to gelu/residual: single-pass
kernels have no reuse, hence no victim worth protecting — hints there
change nothing.

## Scaling the register-cached-row fix with C

Per-thread burden is C ÷ (threads per row) — scale the TEAM, not the
per-thread array. Tiers: (1) C ≤ ~1 K: warp-per-row, C/32 floats per lane
in registers (needs compile-time C buckets: register arrays must be
statically indexed, else they silently spill to "local memory" = L1/L2/DRAM
— the re-read rebuilt); (2) C ~1–8 K: block-per-row, C/256 per thread,
two-stage reduction (xor-shuffle within warps, warp partials meet in a few
bytes of shared memory + one __syncthreads); (3) larger / runtime C: stage
the whole row in shared memory (~100 KB/block on sm_86 ≈ C 25K, explicitly
managed, no spill cliff). Register pressure trades against occupancy
(64 K regs/SM), so tier 1 dies by ~48 regs/lane, not at 255.

## Reading 4-D index math: flatten to 2-D first

Row-major [B, H, T, hd] is byte-identical to a 2-D matrix [B·H·T, hd]:
B·H·T rows of hd numbers. Any element access is then just
`row_index · hd + column`. The row index is NOT a product of b, h, t — it
is the count of rows before yours, odometer-style (t ticks fastest):
every earlier b contributes H·T rows, every earlier h contributes T,
plus t:

  i = b·(H·T) + h·T + t = (b·H + h)·T + t
  offset = i·hd + d          ← the kernels' ((b·H + h)·T + t)·hd + d

Every nested-parentheses address in the permute/rope/attention kernels is
this one pattern applied to that tensor's own axis order.
