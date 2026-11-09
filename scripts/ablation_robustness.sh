#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
P=$STORE/datasets/ottoman_poetry/perturbed/test
SYS="HMM (no indels) + restorer"
export OMP_NUM_THREADS=1
echo "=== $(date): $SYS on the clean sample and every robustness file ==="
$PY scripts/paper_eval.py --split test --allow-test --poem-sample 300 --workers ${WORKERS:-$(nproc)} --systems "$SYS"
for f in $P/*.jsonl; do
    echo "--- $(basename $f .jsonl)"
    LOG=$STORE/projects/aruz/logs/ablation-robustness-$(date +%Y%m%d-%H%M%S)-$(basename $f .jsonl).out
    $PY scripts/paper_eval.py --split test --allow-test --poem-sample 300 --workers ${WORKERS:-$(nproc)} \
        --systems "$SYS" --perturbed $f > $LOG 2>&1 || { echo "FAILED: see $LOG"; exit 1; }
    grep -E "full lines" $LOG | head -1
done
echo "=== full model vs no indels, per input (meter found; meter acc. per line, full lines, lines with no reading) ==="
$PY - <<'PYEOF'
import json, pathlib
d = pathlib.Path("data/results/test/perturbed")
for inp in sorted(p.name for p in d.iterdir() if p.is_dir()):
    cells = []
    for name in ("HMM + restorer", "HMM (no indels) + restorer"):
        p = d / inp / f"{name}.json"
        if not p.exists():
            cells.append("missing"); continue
        r = json.loads(p.read_text(encoding="utf-8")); t = r["found"]["task2"]
        cells.append(f"{100 * r['task1']['line']['accuracy']:5.1f} {100 * t['exact']:5.1f} {sum(1 for x in t['output per line'] if not x):3d}")
    print(f"{inp:14s} full: {cells[0]}   no indels: {cells[1]}")
PYEOF
echo "ablation robustness COMPLETED $(date)"
