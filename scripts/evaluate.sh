#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
SPLIT=${1:-dev}
ONLY=${2:-}
PY=${PYTHON:-python}
M=$STORE/projects/aruz/models
P=$STORE/datasets/ottoman_poetry/perturbed/test
T=""
[ $SPLIT = test ] && T="--allow-test"
if [ "$ONLY" != "--tagger-only" ] && [ "$ONLY" != "--from-tagger" ]; then
    export OMP_NUM_THREADS=1
    echo "=== $SPLIT: every system, and the Task 1 baselines ==="
    $PY scripts/paper_eval.py --split $SPLIT $T --workers ${WORKERS:-$(nproc)}
fi
export OMP_NUM_THREADS=${WORKERS:-$(nproc)}
if [ $SPLIT = dev ] && [ "$ONLY" = "--tagger-only" ]; then
    echo "=== dev: BiLSTM tagger (seed 13) ==="
    $PY scripts/neural_tagger.py --seed 13 --model $M/bilstm_tagger.pt
fi
if [ $SPLIT = test ] && [ "$ONLY" != "--tagger-only" ]; then
    export OMP_NUM_THREADS=1
    echo "=== test: robustness, the clean sample (the poem sample is written here, before the tagger reads it) ==="
    rm -f data/results/test/perturbed/poems.json
    $PY scripts/paper_eval.py --split test --allow-test --poem-sample 300 --workers ${WORKERS:-$(nproc)} \
        --systems "HMM + restorer" pipeline "Kurt & Kara + restorer"
    export OMP_NUM_THREADS=${WORKERS:-$(nproc)}
fi
if [ $SPLIT = test ]; then
    echo "=== test: BiLSTM tagger, five seeds, on the clean text and the robustness files ==="
    for s in 13 14 15 16 17; do
        if [ $s = 13 ]; then MODEL=$M/bilstm_tagger.pt; SUB=""; else MODEL=$M/bilstm_tagger_seed$s.pt; SUB=tagger_seeds/seed$s; fi
        [ -e $MODEL ] || { echo "$MODEL is missing: run scripts/train.sh first"; exit 1; }
        $PY scripts/neural_tagger.py --split test --allow-test --seed $s --model $MODEL --results-subdir "$SUB"
        $PY scripts/neural_tagger.py --split test --allow-test --seed $s --model $MODEL --results-subdir "$SUB" \
            --poem-sample 300 --perturbed $P/*.jsonl
    done
fi
if [ $SPLIT = test ] && [ "$ONLY" != "--tagger-only" ]; then
    export OMP_NUM_THREADS=1
    echo "=== test: robustness, every file ==="
    for f in $P/*.jsonl; do
        echo "--- $(basename $f .jsonl)"
        LOG=$STORE/projects/aruz/logs/evaluate-$(date +%Y%m%d-%H%M%S)-$(basename $f .jsonl).out
        $PY scripts/paper_eval.py --split test --allow-test --poem-sample 300 --workers ${WORKERS:-$(nproc)} \
            --systems "HMM + restorer" pipeline "Kurt & Kara + restorer" --perturbed $f > $LOG 2>&1 || { echo "FAILED: see $LOG"; exit 1; }
        grep -E "line exact" $LOG | head -2
    done
    echo "=== test: the detector, calibrated on clean dev ==="
    $PY scripts/detect_eval.py --calibrate --poems 300
    $PY scripts/detect_eval.py --evaluate --sample data/results/test/perturbed/poems.json
fi
echo "evaluate $SPLIT COMPLETED"
