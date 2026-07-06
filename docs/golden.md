# Golden-file format & workflow (task 2.1)

PyTorch is the **oracle, never linked**: `tools/gen_golden.py` builds each case
in PyTorch (CPU, deterministic), dumps tensors to `tests/golden/<case>/`, and
the files are committed (tiny shapes only). C++ tests load and diff against
them — CI needs no Python.

## Tensor file: `<name>.bin`

Self-describing little-endian binary — no JSON parsing needed in C++:

| offset | type | field |
|---|---|---|
| 0 | u32 | magic `0x544D4C4C` ("LLMT") |
| 4 | u32 | format version (1) |
| 8 | u32 | dtype code (0=fp32, 1=bf16, 2=i32 — matches `DType`) |
| 12 | u32 | rank (≤ 4) |
| 16 | i64 × rank | dims |
| … | payload | row-major contiguous, little-endian |

Scalars (e.g. losses) are dumped as shape `[1]`.

## `manifest.json`

Provenance for humans (case name, torch version, tensor table). **Parsed by
nothing** — the binary headers carry all machine-read facts. Tolerances live
in the C++ tests, next to the per-kernel knowledge they encode.

## Workflow

- Regenerate one case: `.venv/bin/python tools/gen_golden.py --case scale2x`
- Regenerate all: `.venv/bin/python tools/gen_golden.py`
- List cases: `--list`
- Generation is CPU-only and seeded per case → identical bytes on Mac and pod.
- Commit regenerated files together with the change that motivated them;
  record the torch version bump if that's what changed.

## Adding a golden-tested kernel (the exit criterion)

Python (~15 lines):
```python
@case
def rmsnorm_fwd(w):
    torch.manual_seed(42)
    x = torch.randn(4, 64)
    g = torch.randn(64)
    w.dump("x", x); w.dump("g", g)
    w.dump("y", torch.nn.functional.rms_norm(x, (64,), g))
```

C++ (~20 lines): load `x`,`g`,`y` via `GoldenCase`, upload, run kernel,
download, `allclose` + `CHECK_MESSAGE`.
