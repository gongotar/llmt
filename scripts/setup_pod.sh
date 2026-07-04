#!/usr/bin/env bash
# Idempotent pod environment bootstrap (task 0.2). Runs ON the pod.
# Assumes the RunPod PyTorch -devel image (CUDA toolkit + torch preinstalled).
set -euo pipefail

# CUDA toolkit lives outside the default non-interactive PATH on pod images.
[[ -d /usr/local/cuda/bin ]] && export PATH="/usr/local/cuda/bin:$PATH"

echo "== apt packages =="
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends rsync git ccache

echo "== build tools (pip: apt's cmake 3.22 is below our 3.24 minimum) =="
pip install --quiet --root-user-action=ignore "cmake>=3.28" ninja

echo "== python packages =="
pip install --quiet --root-user-action=ignore -r tools/requirements.txt

echo "== environment report =="
nvcc --version | tail -1
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader
cmake --version | head -1
ninja --version
python -c "import torch; print('torch', torch.__version__, 'cuda_ok', torch.cuda.is_available())"
echo "== setup done =="
