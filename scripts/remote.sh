#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Masoud Jami
# Dev loop against the GPU pod (task 0.2). Usage:
#   scripts/remote.sh sync|setup|build|test|shell|all
# Pod endpoint comes from scripts/pod.env (gitignored, changes per pod):
#   LLMT_POD_HOST=1.2.3.4
#   LLMT_POD_PORT=12345
set -euo pipefail

cd "$(dirname "$0")/.."
[[ -f scripts/pod.env ]] && source scripts/pod.env
: "${LLMT_POD_HOST:?set LLMT_POD_HOST (or create scripts/pod.env)}"
: "${LLMT_POD_PORT:?set LLMT_POD_PORT (or create scripts/pod.env)}"
SSH_KEY="${LLMT_SSH_KEY:-$HOME/.ssh/id_ed25519}"
REMOTE="root@${LLMT_POD_HOST}"
REMOTE_DIR="/root/llmt"
SSH=(ssh -p "$LLMT_POD_PORT" -i "$SSH_KEY" -o StrictHostKeyChecking=accept-new)

do_sync() {
    rsync -az --delete \
        --exclude '.git' --exclude 'build*' --exclude '.venv' \
        --exclude 'ckpt' --exclude 'data/*.bin' --exclude '__pycache__' \
        -e "${SSH[*]}" ./ "${REMOTE}:${REMOTE_DIR}/"
}

case "${1:-all}" in
    sync)  do_sync ;;
    setup)
        # Fresh pod images lack rsync; bootstrap it before the first sync.
        "${SSH[@]}" "$REMOTE" "command -v rsync >/dev/null || (apt-get update -qq && apt-get install -y -qq rsync)"
        do_sync
        "${SSH[@]}" "$REMOTE" "cd $REMOTE_DIR && bash scripts/setup_pod.sh"
        ;;
    build) do_sync; "${SSH[@]}" "$REMOTE" "cd $REMOTE_DIR && bash scripts/ci.sh build" ;;
    test)  do_sync; "${SSH[@]}" "$REMOTE" "cd $REMOTE_DIR && VERBOSE=${VERBOSE:-} bash scripts/ci.sh" ;;
    shell) "${SSH[@]}" -t "$REMOTE" "cd $REMOTE_DIR 2>/dev/null; exec bash -l" ;;
    all)   do_sync; "${SSH[@]}" "$REMOTE" "cd $REMOTE_DIR && VERBOSE=${VERBOSE:-} bash scripts/ci.sh" ;;
    *)     echo "usage: $0 sync|setup|build|test|shell|all" >&2; exit 1 ;;
esac
