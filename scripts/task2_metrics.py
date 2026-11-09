import unicodedata
from collections import Counter
from difflib import SequenceMatcher

from aruz.reading import KUSUR, from_annotation
from aruz.scan import syllabify

VOWELS = "aeıiou"

def stream(s):
    s = unicodedata.normalize("NFD", s.replace("İ", "i").replace("I", "ı").lower())
    return "".join(c for c in s if c.isalpha() or c in "ʿʾʼ")

def spans(syllables):
    out, at = [], 0
    for text, _, _ in syllables:
        t = stream(text)
        v = next((i for i, c in enumerate(t) if c in VOWELS), None)
        out.append((at, at + len(t), None if v is None else at + v))
        at += len(t)
    return out

def boundaries(gold_text, gold_syllables):
    out, at = [], 0
    words = [w for w in gold_text.split() if stream(w)]
    for w, nxt in zip(words, words[1:]):
        at += len(stream(w))
        first = unicodedata.normalize("NFD", nxt.lower())
        first = next(c for c in first if c.isalpha())
        if syllabify(w)[-1][2] > 0 and first in VOWELS + "öü":
            out.append(at)
    return out

def score_line(g, out, ok):
    c = Counter()
    gs = from_annotation(g["syllables"])
    pat = next((v for v in g["variants"] if len(v) == len("".join(s[1] for s in gs))), "")
    gsp = spans(gs)
    gold_spans = {(a, b) for a, b, _ in gsp}
    gold_ends = {b for _, b, _ in gsp}
    links = boundaries(g["text"], gs)
    offset = [sum(len(s[1]) for s in gs[:i]) for i in range(len(gs))]

    def bits(i, kusur):
        if i == len(gs) - 1:
            return set()
        return set(kusur) - {"imâle", "zihaf"} if pat[offset[i]:offset[i] + 1] == "x" else set(kusur)

    c["span gold"] += len(gsp)
    for i in range(len(gs)):
        for k in bits(i, gs[i][2]):
            c[f"{k} gold"] += 1
    c["medd gold"] += sum(s[1] == "-." for s in gs)
    for p in links:
        kind = "link" if p - 1 in gold_ends else "stay" if p in gold_ends else None
        if kind is None:
            raise ValueError(f"{g['item']}: no gold boundary at an eligible word boundary")
        c[f"{kind} gold"] += 1
    if out is None or not ok:
        return c

    letters_sys = stream("".join(s[0] for s in out))
    letters_gold = stream("".join(s[0] for s in gs))
    m = {}
    for a, b, n in SequenceMatcher(None, letters_sys, letters_gold, autojunk=False).get_matching_blocks():
        m.update({a + i: b + i for i in range(n)})
    place = lambda p: 0 if p == 0 else (m[p - 1] + 1 if p - 1 in m else None)
    osp = spans(out)
    ends = {place(b) for _, b, _ in osp} - {None}
    holder = {i: j for j, (a, b, _) in enumerate(gsp) for i in range(a, b)}
    c["span sys"] += len(osp)
    aligned = {}
    for j, (a, b, v) in enumerate(osp):
        c["span match"] += (place(a), place(b)) in gold_spans
        if v is not None and v in m and m[v] in holder:
            aligned.setdefault(holder[m[v]], j)
    same_boundaries = True
    for p in links:
        kind = "link" if p - 1 in gold_ends else "stay"
        said = "link" if p - 1 in ends else "stay" if p in ends else None
        if said:
            c[f"{said} sys"] += 1
            c[f"{said} match"] += said == kind
            same_boundaries &= said == kind
    matched = set(aligned.values())
    for i in range(len(gs)):
        j = aligned.get(i)
        if j is None:
            continue
        a, b, _ = osp[j]
        same = (place(a), place(b)) == gsp[i][:2]
        for k in bits(i, out[j][2]):
            c[f"{k} sys"] += 1
            c[f"{k} match"] += k in bits(i, gs[i][2])
            c[f"{k} strict"] += k in bits(i, gs[i][2]) and same
        if gs[i][1] == "-." and out[j][1] == "-.":
            c["medd match"] += 1
        c["medd sys"] += out[j][1] == "-."
    for j, s in enumerate(out):
        if j not in matched:
            for k in s[2]:
                c[f"{k} sys"] += 1
            c["medd sys"] += s[1] == "-."

    def same_slots(i):
        free = i == len(gs) - 1 or pat[offset[i]:offset[i] + 1] == "x"
        return gs[i][1] == out[i][1] or (free and "-." not in (gs[i][1], out[i][1]))
    c["exact"] = int(same_boundaries and len(out) == len(gs) and
                     all(same_slots(i) and bits(i, out[i][2]) == bits(i, gs[i][2]) for i in range(len(gs))))
    return c

def prf(tp, sys_, gold):
    p = tp / sys_ if sys_ else None
    r = tp / gold if gold else None
    return {"precision": p, "recall": r, "f1": 2 * tp / (sys_ + gold) if sys_ + gold else None, "gold": gold}

def summarize(c):
    pool = lambda field: sum(c[f"{k} {field}"] for k in KUSUR)
    return {"syllable spans": prf(c["span match"], c["span sys"], c["span gold"]),
            "links": prf(c["link match"], c["link sys"], c["link gold"]),
            "stays": prf(c["stay match"], c["stay sys"], c["stay gold"]),
            "boundary accuracy": (c["link match"] + c["stay match"]) / max(c["link gold"] + c["stay gold"], 1),
            "always link accuracy": c["link gold"] / max(c["link gold"] + c["stay gold"], 1),
            "kusur": {k: prf(c[f"{k} match"], c[f"{k} sys"], c[f"{k} gold"]) for k in KUSUR},
            "kusur, span too": {k: prf(c[f"{k} strict"], c[f"{k} sys"], c[f"{k} gold"]) for k in KUSUR},
            "kusur pooled": prf(pool("match"), pool("sys"), pool("gold")),
            "kusur pooled, span too": prf(pool("strict"), pool("sys"), pool("gold")),
            "medd": prf(c["medd match"], c["medd sys"], c["medd gold"])}

def score(gold, outputs, meter_ok):
    total, per_line = Counter(), []
    for g, out, ok in zip(gold, outputs, meter_ok):
        c = score_line(g, out, ok)
        per_line.append(bool(c["exact"]))
        total.update(c)
    return {"exact": sum(per_line) / len(gold), "exact per line": per_line, "measures": summarize(total)}
