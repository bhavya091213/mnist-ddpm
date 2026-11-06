#!/usr/bin/env bash
# Usage: cloud/sync.sh <IP> push|pull|status   (run from the project root)
set -euo pipefail

IP="${1:?usage: sync.sh IP push|pull|status}"
CMD="${2:?usage: sync.sh IP push|pull|status}"
KEY="$HOME/.ssh/lambda-mnist"
SSH_OPTS=(-i "$KEY" -o StrictHostKeyChecking=accept-new)
REMOTE="ubuntu@$IP"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

rs() { rsync -az -e "ssh ${SSH_OPTS[*]}" "$@"; }

case "$CMD" in
  push)
    ssh "${SSH_OPTS[@]}" "$REMOTE" "mkdir -p ~/mnist-flow"
    EXCL=(--exclude .venv --exclude /data --exclude /checkpoints --exclude /results --exclude /runs
          --exclude __pycache__ --exclude .git --exclude .env --exclude .orchestrate)
    if command -v rsync >/dev/null 2>&1; then
      rs "${EXCL[@]}" pyproject.toml README.md src tests cloud "$REMOTE:~/mnist-flow/"
    else
      scp "${SSH_OPTS[@]}" -r pyproject.toml README.md src tests cloud "$REMOTE:~/mnist-flow/"
    fi
    ;;
  pull)
    mkdir -p checkpoints/ddpm checkpoints/rf results runs
    for m in ddpm rf; do
      rs "$REMOTE:~/mnist-flow/checkpoints/$m/final.pt" "checkpoints/$m/"
    done
    rs "$REMOTE:~/mnist-flow/results/" results/
    rs "$REMOTE:~/mnist-flow/runs/" --include '*.log' --include STATUS --exclude '*' runs/
    ;;
  status)
    ssh "${SSH_OPTS[@]}" "$REMOTE" \
      'cd ~/mnist-flow; cat runs/STATUS 2>/dev/null || echo "(no STATUS)"; f=$(ls -t runs/*.log 2>/dev/null | head -1); [ -n "$f" ] && { echo "--- $f"; tail -n 5 "$f"; } || true'
    ;;
  *) echo "unknown subcommand: $CMD" >&2; exit 2 ;;
esac
