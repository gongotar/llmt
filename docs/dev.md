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
