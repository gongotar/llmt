#!/usr/bin/env bash
# Build + test in one command (task 0.9). Runs on the GPU pod (or any machine).
#   scripts/ci.sh         # configure + build + run all tests
#   scripts/ci.sh build   # configure + build only
set -euo pipefail

# Non-interactive SSH shells miss the CUDA toolkit PATH on pod images.
[[ -d /usr/local/cuda/bin ]] && export PATH="/usr/local/cuda/bin:$PATH"

cd "$(dirname "$0")/.."
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

if [[ "${1:-test}" != "build" ]]; then
    ctest --test-dir build --output-on-failure
fi
