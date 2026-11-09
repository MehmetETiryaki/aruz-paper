#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
M=$STORE/projects/aruz/models
OLD=$M/replaced_$(date +%F)
mkdir -p $OLD
echo "=== tokenization and vocabulary consistency ==="
$PY scripts/check_vocabulary.py
export OMP_NUM_THREADS=4
if [ "${1:-}" != "--hmm-only" ] && [ "${1:-}" != "--tagger-only" ]; then
    echo "=== restorer ==="
    [ -e $M/restorer.pkl ] && mv $M/restorer.pkl $OLD/
    $PY scripts/train_restorer.py --out $M/restorer.pkl
fi
if [ "${1:-}" != "--tagger-only" ]; then
    for f in hmm_em hmm_em_untied hmm_em_noindels hmm_em_nosymbols; do
        [ -e $M/$f.pkl ] && mv $M/$f.pkl $OLD/
    done
    EM="--epochs 60 --converge 1e-6"
    S1=$M/experiments/stage1; mkdir -p $S1
    echo "=== HMM, EM (the innate model: no medd and no ulama are emissions, no reading priors), two-stage ==="
    $PY scripts/train_hmm.py --no-indels $EM --out $M/hmm_em_noindels.pkl
    $PY scripts/train_hmm.py --emissions-from $M/hmm_em_noindels.pkl $EM --out $M/hmm_em.pkl
    echo "=== HMM ablations, each two-stage: emissions untied; no medd and no ulama symbols ==="
    $PY scripts/train_hmm.py --untied --no-indels $EM --out $S1/untied.pkl
    $PY scripts/train_hmm.py --untied --emissions-from $S1/untied.pkl $EM --out $M/hmm_em_untied.pkl
    $PY scripts/train_hmm.py --no-symbols --no-indels $EM --out $S1/nosymbols.pkl
    $PY scripts/train_hmm.py --no-symbols --emissions-from $S1/nosymbols.pkl $EM --out $M/hmm_em_nosymbols.pkl
fi
if [ "${1:-}" != "--hmm-only" ]; then
    echo "=== BiLSTM tagger, five seeds ==="
    export OMP_NUM_THREADS=${WORKERS:-$(nproc)}
    ls $M/bilstm_tagger*.pt >/dev/null 2>&1 && mv $M/bilstm_tagger*.pt $OLD/
    for s in 13 14 15 16 17; do
        if [ $s = 13 ]; then MODEL=$M/bilstm_tagger.pt; SUB=""; else MODEL=$M/bilstm_tagger_seed$s.pt; SUB=tagger_seeds/seed$s; fi
        $PY scripts/neural_tagger.py --seed $s --model $MODEL --results-subdir "$SUB"
    done
fi
echo "train COMPLETED"
