#!/usr/bin/env bash
# Runs ON the instance, intended under nohup:
#   cd ~/mnist-flow && nohup bash cloud/remote_run.sh > runs/remote_run.out 2>&1 &
# Env overrides: EPOCHS (default 20), N (default 10000), EVAL_BS (default 2500),
# SAMPLE_EVERY (default 0 = off; N saves per-epoch sample grids/GIFs every N epochs).
set -euo pipefail

cd "$HOME/mnist-flow"
EPOCHS="${EPOCHS:-20}"
N="${N:-10000}"
PY=".venv/bin/python"
mkdir -p runs checkpoints results/figures
STATUS=runs/STATUS
STEP="init"
: > "$STATUS"

trap 'echo "FAILED $STEP" >> "$STATUS"' ERR

for m in ddpm rf; do
  STEP="train_$m"
  "$PY" -u -m mnist_flow.train --model "$m" --epochs "$EPOCHS" --device cuda --out checkpoints/ \
    --sample-every "${SAMPLE_EVERY:-0}" \
    > "runs/train_$m.log" 2>&1
  echo "train_$m done" >> "$STATUS"
done

if [ "${SAMPLE_EVERY:-0}" -gt 0 ]; then
  STEP="epoch_viz"
  "$PY" -u -m mnist_flow.epoch_viz --runs runs > runs/epoch_viz.log 2>&1
  echo "epoch_viz done" >> "$STATUS"
fi

STEP="evaluate"
"$PY" -u -m mnist_flow.evaluate --ckpt-dir checkpoints --out results --n "$N" --batch-size "${EVAL_BS:-2500}" --device cuda > runs/evaluate.log 2>&1
echo "evaluate done" >> "$STATUS"

STEP="trajectory"
"$PY" -u -m mnist_flow.trajectory --ckpt-dir checkpoints --out results/figures --device cuda > runs/trajectory.log 2>&1
echo "trajectory done" >> "$STATUS"

echo "all done" >> "$STATUS"
