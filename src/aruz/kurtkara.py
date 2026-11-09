from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Tuple

from .meter import METERS
from .scan import VOWELS, syllabify

PRIORITY = ["cVc", "cvcc", "Vc", "vcc", "Vcc", "cVcc", "cvc", "vc"]

def written(form: str) -> List[dict]:
    out = []
    sylls = syllabify(form)
    for j, (text, long_vowel, coda) in enumerate(sylls):
        onset = text[:1] not in VOWELS
        kind = ("c" if onset else "") + ("V" if long_vowel else "v") + "c" * coda
        out.append({"text": text, "weight": "S" if coda == 0 and not long_vowel else "L",
                    "overlong": (long_vowel and coda >= 1) or coda >= 2, "type": kind, "coda": coda,
                    "long_vowel": long_vowel, "last": j == len(sylls) - 1})
    return out

def line_syllables(words: Sequence[str]) -> List[dict]:
    out = []
    for i, w in enumerate(words):
        for s in written(w):
            s["word"] = i
            s["vowel_next"] = s["last"] and i + 1 < len(words) and words[i + 1][:1] in VOWELS
            out.append(s)
    return out

ANCEPS_FEET = {"-.--", "..--", "-.-", "..-"}

def anceps(pattern: str) -> str:
    feet = [("x" + f[1:]) if f in ANCEPS_FEET else f for f in pattern.split()]
    p = "".join(feet)
    return p[:-1] + "x"

def shortened(pattern: str, free: str) -> Tuple[str, str]:
    if pattern[-3:] in ("-.-", "..-", "x.-"):
        return pattern[:-3] + "--", free[:-3] + "-x"
    return pattern[:-2] + pattern[-1], free[:-2] + free[-1]

def mismatches(weights: Sequence[str], pattern: str) -> int:
    return sum(1 for w, p in zip(weights, pattern) if p != "x" and (p == "-") != (w == "L"))

def candidates(lengths: Sequence[int], templates: Sequence[Tuple[str, str]]) -> List[Tuple[str, str]]:
    longest = max(lengths)
    return [(n, p) for n, p in templates if len(p) in (longest, longest + 1)]

def choose(verses: Sequence[Sequence[str]], templates: Sequence[Tuple[str, str]]) -> Optional[Tuple[str, str]]:
    cands = candidates([len(v) for v in verses], templates)
    if not cands:
        return None
    return min(cands, key=lambda np: sum(mismatches(v, np[1]) for v in verses))

def flaws(syls: List[dict], pattern: str, free: Optional[str] = None) -> Optional[List[list]]:
    syls = [dict(s) for s in syls]
    free = free or pattern
    n, m = len(pattern), len(syls)
    if m > n:
        return None
    weights = [s["weight"] for s in syls]
    medd = set()
    def place(meds):
        return [i + sum(1 for j in meds if j < i) for i in range(m)]

    def possible(meds):
        at = place(meds)
        return [i for i, s in enumerate(syls)
                if s["weight"] == "L" and s["coda"] >= 1 and s["overlong"] and i not in meds
                and at[i] + 1 < n and pattern[at[i] + 1] == "."
                and not (i + 1 < m and syls[i + 1]["text"].lower() in ("i", "ı"))]

    def no_zihaf(meds, after):
        at = place(meds)
        return not any(s["weight"] == "L" and i not in meds and at[i] < n and pattern[at[i]] == "."
                       and not (s["vowel_next"] and s["coda"] == 1)
                       for i, s in enumerate(syls) if i > after)

    def shifted(meds):
        out = []
        for j, w in enumerate(weights):
            out.append(w)
            if j in meds:
                out.append("S")
        return out

    for _ in range(2):
        need = n - m - len(medd)
        if need <= 0:
            break
        pos = []
        for i in possible(medd):
            if need == 1:
                ok = no_zihaf(medd | {i}, i)
            else:
                ok = any(no_zihaf(medd | {i, j}, i) for j in possible(medd | {i}) if j > i)
            if ok:
                pos.append(i)
        if not pos:
            break

        def reduction(i):
            return mismatches(shifted(medd), pattern) - mismatches(shifted(medd | {i}), pattern)
        if len(pos) == 1:
            medd.add(pos[0])
            continue
        best = max(reduction(i) for i in pos)
        tied = [i for i in pos if reduction(i) == best]
        rank = lambda i: PRIORITY.index(syls[i]["type"]) if syls[i]["type"] in PRIORITY else len(PRIORITY)
        medd.add(min(tied, key=lambda i: (rank(i), i)))
    if m + len(medd) < n:
        pattern, free = shortened(pattern, free)
        n = len(pattern)
    pattern = free
    out = []
    k = 0
    for i, s in enumerate(syls):
        if k >= n:
            break
        slots = pattern[k:k + 2] if i in medd else pattern[k]
        k += len(slots)
        kusur: List[str] = []
        text = s["text"]
        linked = False
        if i in medd:
            slots = "-."
        elif slots == "." and s["weight"] == "L":
            if s["vowel_next"] and s["coda"] == 1:
                linked = True
                text = text[:-1]
            else:
                kusur.append("zihaf")
        elif slots == "x":
            slots = "-" if s["weight"] == "L" or i == m - 1 else "."
        elif slots == "-" and s["weight"] == "S":
            kusur.append("imâle")
        if s["vowel_next"] and s["coda"] >= 1 and not linked and i != m - 1:
            kusur.append("no ulama")
        if s["overlong"] and i not in medd and i != m - 1:
            kusur.append("no medd")
        out.append([text, slots, kusur, linked])
    for i in range(len(out) - 1):
        if out[i][3]:
            out[i + 1][0] = syls[i]["text"][-1] + out[i + 1][0]
    return [[t, sl, ku] for t, sl, ku, _ in out]

def read_poem(lines: Sequence[Sequence[str]], inventory: Sequence[Tuple[str, Sequence[str]]]):
    templates = [(m.name, p.replace(" ", "")) for m in METERS for p in m.patterns]
    free = {p.replace(" ", ""): anceps(p) for m in METERS for p in m.patterns}
    syls = [line_syllables(ws) for ws in lines]
    weights = [[s["weight"] for s in ls] for ls in syls]
    found = choose(weights, templates)
    per_line = [choose([w], templates) for w in weights]
    by_meter: Dict[str, str] = {}
    for name in {n for n, _ in templates}:
        own = [(n, p) for n, p in templates if n == name]
        fit = [(n, p) for n, p in own if len(p) >= max(len(w) for w in weights)]
        by_meter[name] = min(fit or own, key=lambda np: (sum(mismatches(w, np[1]) for w in weights), len(np[1])))[1]

    def under(name: str, i: int) -> Optional[List[list]]:
        p = found[1] if found and name == found[0] else by_meter[name]
        return flaws(syls[i], p, free[p])

    return found, per_line, under
