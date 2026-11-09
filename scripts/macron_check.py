import argparse
import json
import os
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from fit_check import INVENTORY, fit, slots

ROOT = Path(__file__).resolve().parent.parent
MACRON = "̄"
MIN_LONG, TIMES = 5, 10
SPLIT = re.compile(r"(\s+|-)")
STRIP = ".,;:!?“”\"()«»"

def words(m):
    return [t.strip(STRIP).lower() for t in SPLIT.split(unicodedata.normalize("NFC", m)) if t.strip(STRIP)]

def bare(w):
    return unicodedata.normalize("NFC", unicodedata.normalize("NFD", w).replace(MACRON, ""))

def macrons(w):
    return unicodedata.normalize("NFD", w).count(MACRON)

def cost(m, r, patterns):
    ways = tuple(map(tuple, slots(m, r)))
    return min([c for c in (fit(ways, p) for p in patterns) if c is not None], default=None)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path(os.environ["STORE"]) / "datasets/ottoman_poetry/processed/latoc.jsonl")
    args = parser.parse_args()
    recs = [json.loads(l) for l in open(args.corpus, encoding="utf-8")]
    count = Counter()
    for r in recs:
        for u in r["units"]:
            if u["usable"]:
                for m in u["misras"]:
                    count.update(words(m))
    forms = defaultdict(list)
    for w in count:
        forms[bare(w)].append(w)
    long_form = {}
    for ws in forms.values():
        for w in ws:
            better = [v for v in ws if macrons(v) > macrons(w) and count[v] >= MIN_LONG and count[v] >= TIMES * count[w]]
            if better:
                long_form[w] = max(better, key=count.get)
    out, stats = {}, Counter()
    for r in recs:
        patterns = r["variants"]
        for u in r["units"]:
            if not u["usable"]:
                continue
            for m in u["misras"]:
                toks = SPLIT.split(unicodedata.normalize("NFC", m))
                hit = {i: long_form[t.strip(STRIP).lower()] for i, t in enumerate(toks) if t.strip(STRIP).lower() in long_form}
                if not hit:
                    continue
                stats[r["split"], "mısras with a rarer short form"] += 1
                fixed = "".join(t.replace(t.strip(STRIP), hit[i] if t.strip(STRIP).islower() else hit[i][0].upper() + hit[i][1:])
                                if i in hit else t for i, t in enumerate(toks))
                pats = patterns or [p for _, ps in INVENTORY for p in ps]
                a, b = cost(m, r, pats), cost(fixed, r, pats)
                if b is not None and (a is None or b < a):
                    why = "spelling: " + ", ".join(f"{toks[i].strip(STRIP)} for {v}" for i, v in hit.items()) + \
                          f" (the corpus's long form fits the meter with {b} licences, as printed {a})"
                    out.setdefault(r["id"], {}).setdefault(str(u["n"]), []).append([m, why])
                    stats[r["split"], "mısras: the meter wants the long form (misprint)"] += 1
    (ROOT / "data/macron_check.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    for k, v in sorted(stats.items()):
        print(f"{k[0]:5s} {k[1]:50s} {v}")

if __name__ == "__main__":
    main()
