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
