import importlib.util
import json
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from aruz.editions import apply_orthography
from aruz.lexicon import lower, normalize, reduce
from aruz.restore import units
from aruz.scan import VOWELS, origin, syllabify, weight

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("eval_meter", ROOT / "scripts/eval_meter.py")
em = importlib.util.module_from_spec(spec)
spec.loader.exec_module(em)
TURKISH = set("abcçdefgğhıijklmnoöprsştuüvyzABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ")
EDGE = set(".,;:!?“”\"()«»")

def weights(f):
    return [weight(*s) for s in syllabify(f)]

def main() -> None:
    final = Path(os.environ["STORE"]) / "datasets/ottoman_poetry/final"
    bad = defaultdict(Counter)
    n = Counter()
    for split in ("train", "dev", "test"):
        for line in open(final / f"{split}.jsonl", encoding="utf-8"):
            r = json.loads(line)
            for u in r["units"]:
                for m in u["misras"]:
                    n["mısras"] += 1
                    plain_line = em.popular(apply_orthography(m, r["silent"], r["context"]))
                    for w in plain_line.split():
                        core = w.strip("".join(EDGE))
                        odd = {c for c in unicodedata.normalize("NFC", core) if c not in TURKISH}
                        if odd:
                            bad["input"][f"{split}: {w!r} has {''.join(sorted(odd))!r}"] += 1
                    em.INPUT = "A"
                    ed = em.words(m, r)
                    em.INPUT = "B"
                    pl = em.words(m, r)
                    if len(ed) != len(pl):
                        bad["tokens"][f"{split}: {' '.join(ed)!r} vs {' '.join(pl)!r}"] += 1
                        continue
                    for e, p in zip(ed, pl):
                        n["words"] += 1
                        if p[:1] == "İ" and re.sub(r"^[\Wʿʾʼ’']*", "", e)[:1] == "I":
                            n["capital I read as İ (alphabet convention, not compared)"] += 1
                            continue
                        if normalize(e) != normalize(p):
                            bad["key"][f"{e!r} -> {normalize(e)!r} but {p!r} -> {normalize(p)!r}"] += 1
                        letters = [unicodedata.normalize("NFC", c) for c in unicodedata.normalize("NFC", lower(p))]
                        uplain = [a for a, _ in units(e)]
                        if uplain != letters:
                            bad["units"][f"{e!r}: units {''.join(uplain)!r}, input {''.join(letters)!r}"] += 1
                        rf = reduce(e)
                        try:
                            if weights(lower(e)) != weights(rf) or origin(lower(e)) != origin(rf):
                                bad["reduce"][f"{e!r} -> {rf!r}: {weights(lower(e))} {origin(lower(e))} vs "
                                              f"{weights(rf)} {origin(rf)}"] += 1
                        except ValueError:
                            bad["reduce"][f"{e!r}: no vowel"] += 1
                        got = lower(em.popular(rf))
                        if got != lower(p) and not (p[:1] == "İ" and got[1:] == lower(p)[1:]):
                            bad["round"][f"{e!r} -> {rf!r} -> {em.popular(rf)!r}, input {p!r}"] += 1
    sys.path.insert(0, str(ROOT / "src"))
    from aruz.reading import from_annotation, normalized, render
    recs = {}
    for split in ("dev", "test"):
        for line in open(final / f"{split}.jsonl", encoding="utf-8"):
            r = json.loads(line)
            recs[r["id"]] = r
    for split in ("dev", "test"):
        path = ROOT / f"data/annotation/{split}_gold.jsonl"
        if not path.exists():
            continue
        for g in map(json.loads, open(path, encoding="utf-8")):
            n["gold lines"] += 1
            r = recs.get(g["id"])
            if r is None:
                bad["gold"][f"{g['id']}: not in the final sets"] += 1
                continue
            em.INPUT = "A"
            forms = [reduce(w) for w in em.words(g["text"], r)]
            pat = next((v for v in g["variants"] if len(v) == len("".join(s[1] for s in g["syllables"]))), "")
            got = render(forms, normalized(from_annotation(g["syllables"]), pat))
            plain = lambda s: normalize(re.sub(r"[‿ˑ\-]", "", s))
            if got is None:
                bad["gold"][f"{g['id']} {g['unit']}/{g['misra']}: cannot be rendered"] += 1
            elif plain(got[0]) != plain(" ".join(forms)):
                bad["gold"][f"{g['id']}: {got[0]!r} vs {' '.join(forms)!r}"] += 1
    print(", ".join(f"{k} {v}" for k, v in n.items()))
    for check in ("input", "tokens", "key", "units", "reduce", "round", "gold"):
        c = bad[check]
        print(f"{check:7s} {'OK' if not c else f'{sum(c.values())} failures, {len(c)} kinds'}")
        for k, v in c.most_common(8):
            print(f"          {v:5d}  {k}")
    sys.exit(1 if any(bad.values()) else 0)

if __name__ == "__main__":
    main()
