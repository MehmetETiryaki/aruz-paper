import argparse
import importlib.util
import json
import os
import pickle
import random
import re
import unicodedata
from collections import Counter, defaultdict
from math import inf, log
from pathlib import Path

from aruz.editions import apply_orthography
from aruz.align import align
from aruz.hmm.lattice import LatticeHMM
from aruz.lexicon import Lexicon
from aruz.meter import METERS, build
from aruz.scan import VOWELS, lattice, syllabify

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("fit_check", ROOT / "scripts/fit_check.py")
fit_check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fit_check)
NAMES = [m.name for m in METERS]
NO_FIT = 10.

CAPITAL_I_TABLE = ROOT / "data/normalizer/capital_i.json"
_capital_i = None

def capital_i_forms():
    global _capital_i
    if _capital_i is None:
        if not CAPITAL_I_TABLE.exists():
            raise SystemExit(f"{CAPITAL_I_TABLE} is missing: run scripts/build_normalizer.py")
        _capital_i = set(json.loads(CAPITAL_I_TABLE.read_text(encoding="utf-8")))
    return _capital_i

def popular(line: str, capital_i: bool = True) -> str:
    line = line.replace("Ī", "İ").replace("Î", "İ")
    line = unicodedata.normalize("NFD", line.replace("ı", "\0").replace("ñ", "n").replace("Ñ", "N"))
    line = "".join(c for i, c in enumerate(line) if not unicodedata.combining(c) or c in "\u0308\u0327"
                   or c == "\u0307" and line[i - 1:i] == "I")
    line = re.sub(r"[ʿʾʼ’‘'`´ˇ]", "", unicodedata.normalize("NFC", line)).replace("\0", "ı")
    line = re.sub(r"[“”„\"«»]", "", line)
    line = re.sub(r"(?<=\w)[-–](?=\w)", "", line)
    if capital_i and "I" in line:
        forms = capital_i_forms()
        line = re.sub(r"(?<![\w])I(\w*)", lambda m: ("İ" if "i" + m.group(1).replace("I", "ı").replace("İ", "i").lower()
                                                     in forms else "I") + m.group(1), line)
    return line

INPUT = "A"
HMM_NAME = "HMM (untrained)"

PERTURBED = {}

QUOTES = re.compile(r"[“”„\"«»]")

def drop_skipped(recs, perturbed):
    out = []
    for r in recs:
        units = [{**u, "misras": [m for m in u["misras"] if perturbed.get((r["id"], m), "") is not None]}
                 for u in r["units"]]
        units = [u for u in units if u["misras"]]
        if units:
            out.append({**r, "units": units})
    return out

def words(m, r):
    t = QUOTES.sub("", apply_orthography(m, r["silent"], r["context"]))
    if INPUT == "B":
        t = popular(t)
    elif INPUT == "P":
        t = popular(PERTURBED[r["id"], m])
    return [w for w in (x.strip(".,;:!?“”\"()«»") for x in t.split()) if any(c in VOWELS for c in w)]

def hmm_scores(model, starts, recs, alternatives=None):
    prior = {m: log(model.A[model.bos_t, st].sum().item()) for m, st in starts.items()}
    out = {}
    for r in recs:
        for u in r["units"]:
            for m in u["misras"]:
                lat = lattice(model, words(m, r), **({"alternatives": alternatives} if alternatives else {}))
                row = {}
                for name, st in starts.items():
                    lat.starts = st
                    row[name] = model.lattice_forward(lat).item() - prior[name]
                lat.starts = None
                out[m] = row
    return prior, out

def fit_scores(recs):
    out = {}
    for r in recs:
        for u in r["units"]:
            for m in u["misras"]:
                ways = tuple(map(tuple, fit_check.slots(popular(m) if INPUT == "B" else m, r)))
                out[m] = {name: min([c for c in (fit_check.fit(ways, p) for p in patterns) if c is not None],
                                    default=NO_FIT) for name, patterns in fit_check.INVENTORY}
    return out

def length_scores(recs):
    lengths = {name: {len(p) for p in patterns} for name, patterns in fit_check.INVENTORY}
    out = {}
    for r in recs:
        for u in r["units"]:
            for m in u["misras"]:
                n = sum(len(syllabify(w)) for w in words(m, r))
                out[m] = {name: float(n in ls) for name, ls in lengths.items()}
    return out

def predict(rows, score, prior):
    total = {name: sum(row[name] for row in rows) for name in NAMES}
    return max(NAMES, key=lambda name: (score(total[name]), prior.get(name, 0)))

def report(name, gold, pred):
    acc = sum(g == p for g, p in zip(gold, pred)) / len(gold)
    fam = sum(g.split("_")[0] == p.split("_")[0] for g, p in zip(gold, pred)) / len(gold)
    f1s = []
    for m in set(gold):
        tp = sum(g == p == m for g, p in zip(gold, pred))
        fp = sum(p == m != g for g, p in zip(gold, pred))
        fn = sum(g == m != p for g, p in zip(gold, pred))
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.)
    print(f"  {name:22s} accuracy {acc:6.1%}   macro-F1 {sum(f1s) / len(f1s):6.1%}   family {fam:6.1%}   (n={len(gold)})")
    return Counter((g, p) for g, p in zip(gold, pred) if g != p)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--limit", type=int, default=0, help="score a random sample of this many poems")
    parser.add_argument("--input", default="A", choices=["A", "B"], help="A: the edition's text; B: popular (no diacritics)")
    parser.add_argument("--no-hmm", action="store_true", help="score the baselines only")
    parser.add_argument("--model", type=Path, help="a trained HMM (scripts/train_hmm.py) instead of the untrained one")
    parser.add_argument("--restorer", type=Path, help="a trained restorer (scripts/train_restorer.py): adds its systems")
    args = parser.parse_args()
    global INPUT
    INPUT = args.input
    final = Path(os.environ["STORE"]) / "datasets/ottoman_poetry/final"
    train = [json.loads(l) for l in open(final / "train.jsonl", encoding="utf-8")]
    recs = [r for r in map(json.loads, open(final / f"{args.split}.jsonl", encoding="utf-8")) if r["meter"] in NAMES]
    if args.limit:
        recs = random.Random(0).sample(recs, args.limit)
    freq = Counter(r["meter"] for r in train if r["meter"])
    print(f"{args.split}, input {args.input}: {len(recs)} poems; train meter counts {dict(freq.most_common())}")

    model, starts, _ = build()
    if args.model:
        model = LatticeHMM.load(args.model)
    global HMM_NAME
    HMM_NAME = "HMM (EM)" if args.model else "HMM (untrained)"
    hmm_prior, hmm = ({n: 0. for n in NAMES}, None) if args.no_hmm else hmm_scores(model, starts, recs)
    systems = {
        "majority": ({m: {n: 0. for n in NAMES} for r in recs for u in r["units"] for m in u["misras"]}, lambda x: 0),
        "syllable count": (length_scores(recs), lambda x: x),
        "aligner (fit_check)": (fit_scores(recs), lambda x: -x),
        HMM_NAME: (hmm, lambda x: x),
    }
    if args.no_hmm:
        del systems[HMM_NAME]
    elif args.input == "B":
        lexicon = Lexicon()
        INPUT = "A"
        for r in train:
            for u in r["units"]:
                for m in u["misras"]:
                    lexicon.update(words(m, r))
        INPUT = "B"
        systems[HMM_NAME + " + lexicon"] = (hmm_scores(model, starts, recs, lexicon.alternatives)[1], lambda x: x)
    if args.input == "B":
        lexicon = lexicon if not args.no_hmm else Lexicon()
        if args.no_hmm:
            INPUT = "A"
            for r in train:
                for u in r["units"]:
                    for m in u["misras"]:
                        lexicon.update(words(m, r))
            INPUT = "B"
        rows = {}
        for r in recs:
            for u in r["units"]:
                for m in u["misras"]:
                    cands = [lexicon.alternatives(w) for w in words(m, r)]
                    rows[m] = {name: min([x[0] for x in (align(cands, p) for p in patterns) if x], default=NO_FIT)
                               for name, patterns in fit_check.INVENTORY}
        systems["aligner + lexicon"] = (rows, lambda x: -x)
    if args.input == "B" and args.restorer:
        with open(args.restorer, "rb") as f:
            restorer = pickle.load(f)
        if not args.no_hmm:
            systems[HMM_NAME + " + restorer"] = (hmm_scores(model, starts, recs, restorer.alternatives)[1], lambda x: x)
        rows = {}
        for r in recs:
            for u in r["units"]:
                for m in u["misras"]:
                    cands = [restorer.alternatives(w) for w in words(m, r)]
                    rows[m] = {name: min([x[0] for x in (align(cands, p) for p in patterns) if x], default=NO_FIT)
                               for name, patterns in fit_check.INVENTORY}
        systems["aligner + restorer"] = (rows, lambda x: -x)
    errors = defaultdict(Counter)
    for level in ("poem", "beyit", "mısra"):
        print(f"\n{level}:")
        if level == "poem":
            items = [(r["meter"], [m for u in r["units"] for m in u["misras"]]) for r in recs]
        elif level == "beyit":
            items = [(r["meter"], u["misras"]) for r in recs for u in r["units"]]
        else:
            items = [(r["meter"], [m]) for r in recs for u in r["units"] for m in u["misras"]]
        gold = [g for g, _ in items]
        for name, (rows, score) in systems.items():
            prior = freq if not name.startswith("HMM") else {n: hmm_prior[n] + 1e-9 * freq.get(n, 0) for n in NAMES}
            pred = [predict([rows[m] for m in ms], score, prior) for _, ms in items]
            errors[level, name] = report(name, gold, pred)
    for name in systems:
        if name.startswith(("HMM", "aligner")):
            print(f"\ncommonest mısra-level confusions, {name}: {errors['mısra', name].most_common(6)}")

if __name__ == "__main__":
    main()
