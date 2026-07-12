#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Masoud Jami
"""Golden-file generator (task 2.2). PyTorch is the oracle — never linked.

Usage:
  python tools/gen_golden.py            # regenerate all cases
  python tools/gen_golden.py --case X   # regenerate one case
  python tools/gen_golden.py --list     # list registered cases

Format: docs/golden.md. Generation is CPU-only and per-case seeded, so the
produced bytes are identical on any machine.
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
GOLDEN_DIR = ROOT / "tests" / "golden"

MAGIC = 0x544D4C4C  # "LLMT" little-endian
VERSION = 1
DTYPE_CODE = {torch.float32: 0, torch.bfloat16: 1, torch.int32: 2}
NUMPY_FMT = {torch.float32: "<f4", torch.int32: "<i4"}  # bf16 arrives in M2

CASES = {}


def case(fn):
    """Register a golden case. The function name is the case name."""
    CASES[fn.__name__] = fn
    return fn


class CaseWriter:
    def __init__(self, name: str):
        self.name = name
        self.dir = GOLDEN_DIR / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.tensors = {}

    def dump(self, name: str, t: torch.Tensor) -> None:
        t = t.detach().contiguous().cpu()
        if t.dim() == 0:
            t = t.reshape(1)  # scalars as [1] (Shape rank 0 means "no shape")
        assert t.dim() <= 4, f"{self.name}/{name}: rank {t.dim()} > 4"
        assert t.dtype in NUMPY_FMT, f"{self.name}/{name}: unsupported {t.dtype}"

        header = struct.pack("<IIII", MAGIC, VERSION, DTYPE_CODE[t.dtype], t.dim())
        header += struct.pack(f"<{t.dim()}q", *t.shape)
        payload = t.numpy().astype(NUMPY_FMT[t.dtype], copy=False).tobytes()
        (self.dir / f"{name}.bin").write_bytes(header + payload)
        self.tensors[name] = {"shape": list(t.shape), "dtype": str(t.dtype).split(".")[-1]}

    def write_manifest(self) -> None:
        manifest = {
            "case": self.name,
            "torch_version": torch.__version__,
            "tensors": self.tensors,  # provenance only — C++ reads the .bin headers
        }
        (self.dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


# ---------------------------------------------------------------- cases ----


@case
def scale2x(w: CaseWriter):
    """End-to-end pipeline proof (task 2.6): y = 2x."""
    torch.manual_seed(1234)
    x = torch.randn(8, 16)
    w.dump("x", x)
    w.dump("y", x * 2)


@case
def matmul(w: CaseWriter):
    """GEMM wrapper (task 4.4): all four transpose combos + beta accumulation.
    Deliberately non-square with distinct dims (7,5,9) so any row/col-major
    or operand-swap mistake changes the result shape or values."""
    torch.manual_seed(41)
    a = torch.randn(7, 5)    # [M, K]
    at = torch.randn(5, 7)   # [K, M] — used transposed
    b = torch.randn(5, 9)    # [K, N]
    bt = torch.randn(9, 5)   # [N, K] — used transposed
    c0 = torch.randn(7, 9)   # pre-existing C for the beta=1 case
    for name, t in [("a", a), ("at", at), ("b", b), ("bt", bt), ("c0", c0)]:
        w.dump(name, t)
    w.dump("c_nn", a @ b)
    w.dump("c_nt", a @ bt.T)
    w.dump("c_tn", at.T @ b)
    w.dump("c_tt", at.T @ bt.T)
    w.dump("c_beta", c0 + a @ b)


@case
def matmul_rank3(w: CaseWriter):
    """matmul() with rank-3 tensors (strided-batched, task 4.3): plain batch
    + the QK^T pattern attention will use in Phase 6."""
    torch.manual_seed(43)
    a = torch.randn(3, 4, 6)
    b = torch.randn(3, 6, 5)
    w.dump("a", a)
    w.dump("b", b)
    w.dump("c", a @ b)
    q = torch.randn(3, 4, 8)
    k = torch.randn(3, 4, 8)
    w.dump("q", q)
    w.dump("k", k)
    w.dump("scores", q @ k.transpose(-2, -1))


@case
def linear(w: CaseWriter):
    """Linear layer fwd+bwd (task 4.7): y = x·W^T, grads via autograd."""
    torch.manual_seed(47)
    x = torch.randn(8, 12, requires_grad=True)
    weight = torch.randn(10, 12, requires_grad=True)  # [out, in]
    y = x @ weight.T
    dy = torch.randn(8, 10)
    y.backward(dy)
    for name, t in [("x", x), ("weight", weight), ("dy", dy), ("y", y),
                    ("dx", x.grad), ("dw", weight.grad)]:
        w.dump(name, t)


# ------------------------------------------------------------------ main ----


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", help="regenerate a single case")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        print("\n".join(sorted(CASES)))
        return 0

    names = [args.case] if args.case else sorted(CASES)
    for name in names:
        if name not in CASES:
            print(f"unknown case '{name}' (see --list)", file=sys.stderr)
            return 1
        w = CaseWriter(name)
        CASES[name](w)
        w.write_manifest()
        print(f"wrote {w.dir.relative_to(ROOT)}: {', '.join(sorted(w.tensors))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
