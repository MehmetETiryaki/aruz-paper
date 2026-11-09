import importlib.util
import json
import os
from collections import Counter
from pathlib import Path

from aruz.editions import apply_orthography

ROOT = Path(__file__).resolve().parent.parent
STORE = Path(os.environ["STORE"])
spec = importlib.util.spec_from_file_location("eval_meter", ROOT / "scripts/eval_meter.py")
em = importlib.util.module_from_spec(spec)
spec.loader.exec_module(em)

def lower(w):
    return w.replace("İ", "i").replace("I", "ı").lower()

def lines(split):
    for line in open(STORE / f"datasets/ottoman_poetry/final/{split}.jsonl", encoding="utf-8"):
        r = json.loads(line)
        for u in r["units"]:
            for m in u["misras"]:
                yield apply_orthography(m, r["silent"], r["context"])

def main() -> None:
    counts = Counter()
    for t in lines("train"):
        counts.update(lower(w.strip(".,;:!?()")) for w in em.popular(t, capital_i=False).split())
    forms = sorted(w for w in counts if w[:1] == "i" and counts[w] > counts["ı" + w[1:]])
    em.CAPITAL_I_TABLE.parent.mkdir(parents=True, exist_ok=True)
    em.CAPITAL_I_TABLE.write_text(json.dumps(forms, ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"{len(forms)} i-initial words -> {em.CAPITAL_I_TABLE}")
    for split in ("train", "dev", "test"):
        changed = [(a, b) for a, b in ((em.popular(t, capital_i=False), em.popular(t)) for t in lines(split)) if a != b]
        print(f"{split}: {len(changed)} clean lines change with the capital-I reading", changed[:5])

if __name__ == "__main__":
    main()
