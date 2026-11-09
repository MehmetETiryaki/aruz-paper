#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
books=$(ls $STORE/datasets/ottoman_poetry/latoc/pdf/*.pdf | xargs -n1 -d '\n' basename | sed 's/\.pdf$//')
after="$PY scripts/build_corpus.py && echo \"$books\" | tr '\n' '\0' | xargs -0 $PY scripts/review_book.py && $PY scripts/label_check.py && $PY scripts/source_check.py && $PY scripts/fit_check.py && $PY scripts/macron_check.py && $PY scripts/export_splits.py && $PY scripts/audit_final.py"
if [ "${1:-}" = "--no-layout" ]; then
    ${LATOC_PYTHON:-python} scripts/run_latoc.py --steps 2 3
    bash -c "$after"
else
    bash scripts/run_latoc.sh
    bash -c "$after"
fi
