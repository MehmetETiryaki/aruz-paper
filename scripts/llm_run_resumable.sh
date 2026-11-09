#!/bin/bash
set -u
LABEL=$1; PROVIDER=$2; MODEL=$3; SPLIT=$4; shift 4
cd "$(dirname "$0")/.."
ALLOW=""; [ "$SPLIT" = test ] && ALLOW="--allow-test"
SCORE_ARGS=(); skip=0
for a in "$@"; do
    if [ $skip = 1 ]; then skip=0; continue; fi
    case "$a" in --workers|--effort|--max-calls|--prompt|--budget-usd) skip=1 ;; --stop-on-error) ;; *) SCORE_ARGS+=("$a") ;; esac
done
for pass in 1 2 3 4 5 6 7 8; do
    echo "=== pass $pass $(date -Is)"
    python scripts/llm_baseline.py run --provider "$PROVIDER" --model "$MODEL" --label "$LABEL" --split "$SPLIT" $ALLOW "$@" 2>&1 | tee /tmp/llm_pass_$$.log
    LAST=$(grep "^cached answers:" /tmp/llm_pass_$$.log | tail -1)
    HAVE=$(echo "$LAST" | sed -E 's/cached answers: ([0-9]+) of ([0-9]+).*/\1/'); WANT=$(echo "$LAST" | sed -E 's/cached answers: ([0-9]+) of ([0-9]+).*/\2/')
    echo "=== pass $pass done: $HAVE of $WANT answered"
    [ -n "$HAVE" ] && [ "$HAVE" = "$WANT" ] && break
    grep -q "BUDGET REACHED" /tmp/llm_pass_$$.log && { echo "=== budget reached: no further passes"; break; }
    sleep 60
done
rm -f /tmp/llm_pass_$$.log
echo "=== scoring $(date -Is)"
python scripts/llm_baseline.py score --label "$LABEL" --split "$SPLIT" $ALLOW "${SCORE_ARGS[@]}"
echo "=== LLM_RUN_DONE $LABEL $(date -Is)"
