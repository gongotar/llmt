# llmt

A C++/CUDA library for training transformer LLMs **from scratch**: own
kernels for attention, MLP, norms, embeddings, loss and optimizer — cuBLASLt
for GEMM is the only compute dependency. PyTorch is used strictly as a test
oracle (golden files), never linked.

**Status:** work in progress — building toward Milestone 1, a full training
run of a small GPT-style model on a single GPU. Progress: [docs/TODO.md](docs/TODO.md).

## Documentation

- [docs/DESIGN.md](docs/DESIGN.md) — architecture, interfaces, testing strategy
- [docs/TODO.md](docs/TODO.md) — milestone tracker (current status at the top)
- [docs/dev.md](docs/dev.md) — build & remote-GPU development workflow
- [docs/knowledge.md](docs/knowledge.md) — background notes (CUDA, tensors, precision)

## Build

CMake ≥ 3.24, C++20, CUDA 12.x (GPU parts auto-skip on CUDA-less hosts):

```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build && ctest --test-dir build
```

## License

Dual-licensed: **AGPL-3.0-only** or a commercial license — see
[LICENSING.md](LICENSING.md).
