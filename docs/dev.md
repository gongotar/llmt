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

## Local (Mac) builds

`cmake -S . -B build && cmake --build build` works host-side: CUDA is
auto-detected and skipped, building only host targets + host tests. Useful for
fast compile checks of non-CUDA code; anything real runs on the pod.

## Costs

RTX 4000 Ada: $0.26/hr, billed per second. No storage costs after Terminate.
