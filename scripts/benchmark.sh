#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
echo "=== the error model, counted from the aligned copies ==="
$PY -I scripts/web_survey.py catalogue
echo "=== build the benchmark ==="
$PY scripts/make_perturbed.py --splits test
echo "benchmark COMPLETED"
