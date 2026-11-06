#!/usr/bin/env bash
# Runs ON the instance. Idempotent. Usage: bash ~/mnist-flow/cloud/remote_setup.sh
set -euo pipefail

cd "$HOME/mnist-flow"
nvidia-smi
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

if [ ! -d .venv ]; then
  if command -v uv >/dev/null 2>&1; then
    uv venv --system-site-packages .venv
  else
    python3 -m venv --system-site-packages .venv
  fi
fi

if command -v uv >/dev/null 2>&1; then
  uv pip install --python .venv/bin/python -e ".[dev]"
else
  .venv/bin/python -m pip install -q -U pip setuptools wheel
  .venv/bin/python -m pip install -q -e ".[dev]"
fi

.venv/bin/python -c "from mnist_flow.data import load_mnist; load_mnist(); print('mnist ready')"
.venv/bin/python -m pytest -q -x
echo "setup done"
