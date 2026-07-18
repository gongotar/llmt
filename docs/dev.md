# Development workflow

The dev host is a CUDA-less Mac; all GPU work runs on a disposable RunPod pod
(see `docs/hardware.md`). Everything is reproducible from this repo — pods are
terminated at the end of each session without losing anything.

## Per-session loop

1. Start (or deploy) the pod on RunPod: Secure Cloud · RTX 4000 Ada ·
   On-Demand · RunPod PyTorch `-devel` template · ~30 GB container disk ·
   no network volume.
2. Put its endpoint in `scripts/pod.env` (gitignored):
   ```
   LLMT_POD_HOST=157.157.221.29
   LLMT_POD_PORT=23676
   ```
3. First time on a fresh pod: `scripts/remote.sh setup`
   (rsync + apt/pip bootstrap, ~2 min, idempotent).
4. Iterate: `scripts/remote.sh test` — rsyncs the tree, builds with Ninja,
   runs ctest. `scripts/remote.sh shell` for an interactive session.
5. End of session: **Terminate** the pod in the RunPod console. Stopping
   instead spares nothing for us: a stopped pod's container disk is wiped
   anyway (we run without a volume disk), and its GPU may be taken by another
   user while stopped — treat pods as cattle.

## Code style (beyond .clang-format)

- Every source file starts with the two-line SPDX header
  (`AGPL-3.0-only`, copyright) — except `third_party/` (own licenses),
  golden binaries, and markdown docs. See `LICENSING.md` (dual licensing).
- Classes/functions in library headers get Doxygen `/** … */` comments:
  prose-first, 1–3 sentences of behavior; `@param`/`@return` only where the
  signature isn't self-explanatory (e.g. the RNG triple). No ceremony
  ("@brief Gets the stream." is noise). Tests and scripts use plain comments.
- Private/encapsulated class members are `m_` prefixed (`m_used`, `m_stream`);
  transparent POD structs (Shape, Tensor, U32x4) keep plain field names.
- `noexcept` on every function that cannot throw (our error paths `abort()`,
  they never throw — so nearly everything qualifies; keep it truthful).
- `const` on every local that is never reassigned.
- Headers never use unnamed namespaces (ODR trap) — internal-but-in-header
  code goes in `namespace detail`; `.cu`/`.cpp` internals use unnamed
  namespaces.
- C++ standard is pinned to 20: nvcc 12.4's ceiling. Do not raise it until
  the CUDA toolchain does.

## Git workflow

Remote: `git@github.com:gongotar/llmt.git` (GPL-3.0). Commit and push after
each meaningful chunk — typically at phase exit and after notable mid-phase
landings. To keep GPU spend low, write code Mac-side (host-only builds catch
C++ errors), then start a pod and run the full GPU test batch when a phase's
tests are ready; commit only green states.

## Local (Mac) builds

`cmake -S . -B build && cmake --build build` works host-side: CUDA is
auto-detected and skipped, building only host targets + host tests. Useful for
fast compile checks of non-CUDA code; anything real runs on the pod.

## Costs

RTX 4000 Ada: $0.26/hr, billed per second. No storage costs after Terminate.

## Profiling (Nsight Compute, on the pod)

`ncu` ships with the CUDA toolkit image. Profile one kernel at a time,
skipping the bench's warmup launches (`-s` = launch skip, `-c` = launch
count):

```sh
ncu --set full -k regex:rmsnorm_fwd  -s 3 -c 5 -o rmsnorm  ./llmt_bench_bandwidth
ncu --set full -k regex:residual_fwd -s 3 -c 5 -o residual ./llmt_bench_bandwidth
```

Read the reports on the pod (no GUI needed):

```sh
ncu --import rmsnorm.ncu-rep --page details | less
```

To answer "why is kernel X below peak", read the sections in this order:

1. **GPU Speed Of Light → DRAM Throughput %.** Well above the bench's
   %-of-peak → our byte accounting undercounts (the kernel really moves
   more DRAM bytes, e.g. cache re-reads that miss). Matches the bench →
   the bus is genuinely idle part of the time; go on.
2. **Memory Workload Analysis.** Actual DRAM read/write bytes vs the
   bench's counted bytes; L1/L2 hit rates (are the row re-reads served
   by cache, as assumed?).
3. **Warp State Statistics.** Top stall reasons: Long Scoreboard = waiting
   on DRAM (expected for memory-bound); Barrier/Membar = sync cost;
   MIO/LG Throttle = the L1/load-store pipe is the bottleneck (too many
   load instructions per byte of DRAM traffic).
4. **Scheduler Statistics.** Eligible warps/cycle far below theoretical =
   occupancy or dependency problem rather than bandwidth.

ncu serializes and replays each profiled launch; wall time balloons but the
per-launch numbers are sound. Add `-lineinfo` to nvcc flags if per-source-line
attribution is wanted (TODO 0.x note).

Counter access is a host driver policy (`RmProfilingAdminOnly`) that
containers cannot change — RunPod pods report BLOCKED (`ERR_NVGPUCTRPERM`).
`setup_pod.sh` probes and prints this at bootstrap. When counters are needed,
use a VM-class GPU (Lambda / cloud GPU VM); otherwise `nsys` (works without
counters) or single-variable kernel-variant bisection answer most questions —
see knowledge.md "Counted vs actual bytes" for a worked example.
