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
