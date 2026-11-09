import argparse
import json
import os
from collections import Counter
from functools import lru_cache
from pathlib import Path

from aruz.editions import apply_orthography
from aruz.meter import METERS
from aruz.scan import VOWELS, syllabify, weight

ROOT = Path(__file__).resolve().parent.parent
MAX_IMALE = 4
POEM_SHARE = .3
INVENTORY = [(m.name, [p.replace(" ", "") for p in m.patterns]) for m in METERS]

def slots(line, r):
    text = apply_orthography(line, r["silent"], r["context"])
    words = [w for w in (x.strip(".,;:!?“”\"()«»") for x in text.split()) if any(c in VOWELS for c in w)]
    out = []
    for i, w in enumerate(words):
        sylls = syllabify(w)
        for j, s in enumerate(sylls):
            wt = weight(*s)
            final = j == len(sylls) - 1
            ulama = final and i + 1 < len(words) and words[i + 1][0] in VOWELS and s[2] > 0
            mono = len(sylls) == 1
            if wt == "S":
                ways = [(".", 0), ("-", 0 if mono else 1)]
            elif wt == "LH":
                ways = [("-.", 0), ("-", 0)] if ulama else [("-.", 0)]
            else:
                ways = [("-", 0)] + ([(".", 0)] if ulama or mono else [(".", 1)] if s[1] and not s[2] else [])
            out.append(ways)
    return out

def fit(ways, pattern):
    n = len(pattern)

    @lru_cache(maxsize=None)
    def go(i, k):
        if i == len(ways):
            return 0 if k == n else None
        best = None
        last = i == len(ways) - 1
        for slot, cost in ways[i]:
            if last and k == n - 1:
                return 0 if best is None else min(best, 0)
            if k + len(slot) > n:
                continue
            if all(p == "x" or p == c for p, c in zip(pattern[k:k + len(slot)], slot)):
                rest = go(i + 1, k + len(slot))
                if rest is not None and (best is None or rest + cost < best):
                    best = rest + cost
        return best

    return go(0, 0)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path(os.environ["STORE"]) / "datasets/ottoman_poetry/processed/latoc.jsonl")
    args = parser.parse_args()
    out, stats = {}, Counter()
    for line in open(args.corpus, encoding="utf-8"):
        r = json.loads(line)
        variants = r["variants"]
        if not variants:
            lines = [tuple(map(tuple, slots(m, r))) for u in r["units"]
                     if u["usable"] and not any("ziyade" in i for i in u["issues"]) for m in u["misras"]]
            if not lines:
                continue
            def score(patterns):
                cs = [min([c for c in (fit(w, p) for p in patterns) if c is not None], default=None) for w in lines]
                ok = [c for c in cs if c is not None and c <= MAX_IMALE]
                return len(ok), -sum(ok)
            variants = max((p for _, p in INVENTORY), key=score)
        misfit, total, costs_by = {}, 0, []
        for u in r["units"]:
            if not u["usable"] or any("ziyade" in i for i in u["issues"]):
                continue
            for m in u["misras"]:
                total += 1
                ways = tuple(map(tuple, slots(m, r)))
                costs = [c for c in (fit(ways, v) for v in variants) if c is not None]
                costs_by.append((min(costs) if costs else None, ways))
                why = None if costs and min(costs) <= MAX_IMALE else \
                    ("needs more than %d licences" % MAX_IMALE if costs else "does not scan under its vezin")
                if why:
                    misfit.setdefault(str(u["n"]), []).append([m, why])
                    stats[r["split"], why + ("" if r["variants"] else " (no label: best inventory meter)")] += 1
                stats[r["split"], "mısras"] += 1
        bad = sum(len(v) for v in misfit.values())
        better = None
        if r["variants"] and len(costs_by) >= 2 and all(c is not None for c, _ in costs_by):
            label = sum(c for c, _ in costs_by)
            for name, patterns in INVENTORY:
                if name == r["meter"]:
                    continue
                other = [min([c for c in (fit(ways, p) for p in patterns) if c is not None], default=None) for _, ways in costs_by]
                if all(c is not None for c in other) and sum(other) + max(2, .5 * len(other)) <= label:
                    better = f"{name} fits every mısra with {sum(other)} licences, the label needs {label}"
                    break
        if total and bad > POEM_SHARE * total or better:
            out[r["id"]] = {"poem": better or f"{bad}/{total} mısras do not fit the label" if r["variants"] else
                            f"no label, and {bad}/{total} mısras fit no single inventory meter", "units": misfit}
            stats[r["split"], "poems: label does not fit"] += 1
        elif misfit:
            out[r["id"]] = {"units": misfit}
    (ROOT / "data/fit_check.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    for k, v in sorted(stats.items()):
        print(f"{k[0]:5s} {k[1]:40s} {v}")

if __name__ == "__main__":
    main()
