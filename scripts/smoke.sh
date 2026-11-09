#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
export PYTHONPATH=src
M=$STORE/projects/aruz/models
mkdir -p "$M" "$STORE/datasets/ottoman_poetry/perturbed"
echo "== book table"
$PY scripts/book_table.py
echo "== gold from labels"
$PY scripts/join_gold.py
echo "== normalizer"
$PY scripts/build_normalizer.py
echo "== restorer (one order, one weight)"
$PY scripts/train_restorer.py --orders 4 --weights .8 --out "$M/smoke_restorer.pkl"
echo "== meter HMM (20 poems, one epoch)"
$PY scripts/train_hmm.py --limit 20 --epochs 1 --out "$M/smoke_hmm.pkl"
echo "== systems on three dev poems"
$PY scripts/paper_eval.py --split dev --limit 3 --workers 2 --systems "HMM + restorer" "Kurt & Kara + restorer" "aligner" "HMM (no indels) + restorer"
echo "== tagger (two poems, one epoch)"
$PY scripts/neural_tagger.py --split dev --limit 2 --seed 13
echo "== robustness benchmark"
$PY scripts/make_perturbed.py --splits test
echo "== detector (five poems, one file)"
$PY scripts/detect_eval.py --calibrate --evaluate --poems 5 --limit-files 1 --workers 2
echo "== language-model rows from the shipped caches"
$PY scripts/llm_baseline.py score --label claude-opus-5.5 --split test --allow-test --all-poems
$PY scripts/llm_baseline.py score --label deepseek-v4.1-flash-test --split test --allow-test --all-poems
echo "smoke OK"
