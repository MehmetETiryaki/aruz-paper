#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
S=$STORE/projects/aruz/logs/scripts-$(date +%Y%m%d-%H%M%S)
mkdir -p $S && cp scripts/*.sh $S/
bash $S/train.sh "$@"
ONLY=""; [ "${1:-}" = "--tagger-only" ] && ONLY="--tagger-only"
bash $S/evaluate.sh dev $ONLY
bash $S/benchmark.sh
bash $S/evaluate.sh test $ONLY
echo "run COMPLETED"
