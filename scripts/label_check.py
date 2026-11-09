import argparse
import json
import os
from collections import Counter
from pathlib import Path

from aruz.editions import apply_orthography
from aruz.scan import VOWELS, syllabify, weight

ROOT = Path(__file__).resolve().parent.parent
LONG_FOR_SHORT, SHORT_FOR_LONG = .6, .8
MIN_LINES = 5

def weight_sets(line, r):
    text = apply_orthography(line, r["silent"], r["context"])
    words = [w for w in (x.strip(".,;:!?“”\"()«»") for x in text.split()) if any(c in VOWELS for c in w)]
    slots = []
    for i, w in enumerate(words):
        sylls = syllabify(w)
        for j, s in enumerate(sylls):
            wt = weight(*s)
            last = i == len(words) - 1 and j == len(sylls) - 1
            ulama = j == len(sylls) - 1 and i + 1 < len(words) and words[i + 1][0] in VOWELS and s[2] > 0
            if wt == "LH" and not last:
                slots += [{"L"}, {"S", "L"} if ulama else {"S"}]
            elif wt == "S":
                slots.append({"S", "L"} if len(sylls) == 1 else {"S"})
            else:
                slots.append({"L", "S"} if ulama else {"L"})
    return slots

def check(r):
    by_variant = {len(v): v for v in r["variants"]}
    aligned = {}
    for u in r["units"]:
        if not u["usable"]:
            continue
        for m in u["misras"]:
            ws = weight_sets(m, r)
            if len(ws) in by_variant:
                aligned.setdefault(len(ws), []).append(ws)
    for n, lines in aligned.items():
        pattern = by_variant[n]
        if 2 <= len(lines) < MIN_LINES:
            for j, p in enumerate(pattern[:-1]):
                if p == "-" and all(w[j] == {"S"} for w in lines) or p == "." and all("S" not in w[j] for w in lines):
                    return f"label {'long' if p == '-' else 'short'}, text the other at syllable {j + 1} in all {len(lines)} lines"
            continue
        if len(lines) < MIN_LINES:
            continue
        for j, p in enumerate(pattern[:-1]):
            if p == "-" and sum("S" in w[j] for w in lines) >= .9 * len(lines) and \
                    sum(w[j] == {"S"} for w in lines) >= .5 * len(lines):
                return f"label long, text short or open at syllable {j + 1} in most of {len(lines)} lines"
        long_for_short, short_for_long = Counter(), Counter()
        for ws in lines:
            for j, (w, p) in enumerate(zip(ws[:-1], pattern[:-1])):
                if p == "." and "S" not in w:
                    long_for_short[j] += 1
                elif p == "-" and "L" not in w:
                    short_for_long[j] += 1
        for j, k in long_for_short.items():
            if k >= LONG_FOR_SHORT * len(lines):
                return f"label short, text long at syllable {j + 1} in {k}/{len(lines)} aligned lines"
        for j, k in short_for_long.items():
            if k >= SHORT_FOR_LONG * len(lines):
                return f"label long, text short at syllable {j + 1} in {k}/{len(lines)} aligned lines"
    return None

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path(os.environ["STORE"]) / "datasets/ottoman_poetry/processed/latoc.jsonl")
    args = parser.parse_args()
    recs = [json.loads(l) for l in open(args.corpus, encoding="utf-8")]
    suspect = {r["id"]: why for r in recs if r["variants"] for why in [check(r)] if why}
    (ROOT / "data/label_check.json").write_text(json.dumps(suspect, ensure_ascii=False, indent=1, sort_keys=True),
                                                encoding="utf-8")
    per_book = Counter(k.split("#")[0] for k in suspect)
    print(f"{len(suspect)}/{sum(1 for r in recs if r['variants'])} labels suspect: {dict(per_book)}")

if __name__ == "__main__":
    main()
