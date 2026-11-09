#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
$PY scripts/build_corpus.py
ls $STORE/datasets/ottoman_poetry/latoc/pdf/*.pdf | xargs -n1 -d '\n' basename | sed 's/\.pdf$//' | tr '\n' '\0' | xargs -0 $PY scripts/review_book.py
$PY scripts/label_check.py
$PY scripts/source_check.py
$PY scripts/fit_check.py
$PY scripts/macron_check.py
$PY scripts/export_splits.py
$PY scripts/audit_final.py
echo "rebuild COMPLETED"
