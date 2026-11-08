#!/bin/bash
# One-time setup: the conda env at CONDA_ENV from environment.yml, then the real datasets into data/.
# Usage: bash scripts/hyak/setup_env.sh [config file] [KEY=VALUE ...]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
load_config "$@"
if [[ ! -x $PY ]]; then
  command -v conda > /dev/null || { echo "conda not found: install Miniforge first (see README.md)" >&2; exit 1; }
  conda env create -p "$CONDA_ENV" -f "$REPO/environment.yml"
fi
"$PY" - "$REPO" << 'EOF'
import sys
sys.path.insert(0, sys.argv[1] + "/src")
from datasets import load_split
for name in ("wine", "airfoil", "breast_cancer", "mnist1"):    # downloads once into data/ (mnist: test images too)
    X, _, Xt, _ = load_split(name)
    print(name, X.shape, "held-out", Xt.shape)
EOF
"$PY" -c "import gurobipy; print('gurobipy', gurobipy.gurobi.version())"
