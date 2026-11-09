import importlib.util
from collections import defaultdict
from pathlib import Path

from aruz.meter import METERS
from aruz.reading import KUSUR
from aruz.scan import VOWELS, moved, syllabify, weight

ROOT = Path(__file__).resolve().parent.parent
INVENTORY = [(m.name, [p.replace(" ", "") for p in m.patterns]) for m in METERS]

def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

meter_eval = module("eval_meter")
task2 = module("task2_metrics")

def tokens(m, r, inp):
    meter_eval.INPUT = inp
    return meter_eval.words(m, r)

def from_hmm(sc):
    groups = defaultdict(list)
    for arc, slot in sc.path:
        groups[arc.info.word, arc.info.syllable].append((arc.info, slot))
    items = list(groups.items())
    out = []
    for n, ((w, j), group) in enumerate(items):
        info = group[0][0]
        last = n == len(items) - 1
        slots = "".join({"L": "-", "S": ".", "I": "?"}.get(s.kind, "-" if i.natural != "S" else ".") for i, s in group)
        kusur = []
        if not last:
            if info.medd is False:
                kusur.append("no medd")
            elif info.natural == "S" and slots == "-":
                kusur.append("imâle")
            elif info.natural == "L" and slots == ".":
                kusur.append("zihaf")
        sylls = syllabify(info.form)
        nxt = items[n + 1][1][0][0] if not last else None
        gunne = sylls[-1][1] and sylls[-1][2] == 1 and sylls[-1][0][-1:].lower() in "nñ"
        if j == len(sylls) - 1 and nxt is not None and nxt.word != info.word and not info.ulama and not gunne \
                and sylls[-1][2] > 0 and nxt.form[:1].lower() in VOWELS:
            kusur.append("no ulama")
        out.append([info.text, slots, kusur])
    for n, ((w, j), group) in enumerate(items[:-1]):
        info = group[0][0]
        sylls = syllabify(info.form)
        if info.ulama and j == len(sylls) - 1:
            out[n + 1][0] = moved(info) + out[n + 1][0]
    return out

def weights(form):
    return [weight(*s) for s in syllabify(form)] if form else None

def distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]

def score_scansion(name, gold, outputs, meter_ok):
    t = task2.score(gold, outputs, meter_ok)
    inserted = [x for out in outputs if out for x in out if "?" in x[1]]
    if inserted:
        print(f"  {name:28s} syllables in an insert state: {len(inserted)}, e.g. {inserted[:3]}")
    m = t["measures"]
    print(f"  {name:28s} meter found {sum(meter_ok) / len(gold):6.1%}   full lines {t['exact']:6.1%}   syllables "
          f"{m['syllable spans']['f1'] or 0:6.1%}   kusur {m['kusur pooled']['f1'] or 0:6.1%}   "
          + "  ".join(f"{k} {m['kusur'][k]['f1'] or 0:.2f}" for k in KUSUR))
    return {"lines": len(gold), "meter found": sum(meter_ok) / len(gold), "exact": t["exact"],
            "exact per line": t["exact per line"], "output per line": outputs, "meter ok per line": list(meter_ok),
            "measures": m}
