from __future__ import annotations
from functools import lru_cache
from typing import List, Optional, Sequence, Tuple

from .scan import VOWELS, syllabify, weight

Syllable = Tuple[str, str, Tuple[str, ...]]
Way = Tuple[str, str, Tuple[str, ...], float, Optional[str]]
STANDARD = 1e-6

def readings(text: str, long_vowel: bool, coda: int, kusur: Tuple[str, ...] = (), extra: float = 0.,
             begins: Optional[str] = None) -> List[Way]:
    wt = weight(text, long_vowel, coda)
    base = len(kusur) + extra
    if wt == "S":
        return [(text, ".", kusur, base, begins), (text, "-", kusur + ("imâle",), base + 1, begins)]
    if wt == "LH":
        return [(text, "-.", kusur, base, begins), (text, "-", kusur + ("no medd",), base + 1 + STANDARD, begins)]
    ways = [(text, "-", kusur, base, begins)]
    if long_vowel and not coda:
        ways.append((text, ".", kusur + ("zihaf",), base + 1, begins))
    return ways

def begins(form: str) -> Optional[str]:
    return "vowel" if form[:1].lower() in VOWELS else None

def options(form: str, vowel_next: bool, last_word: bool, standard: bool = False) -> List[List[Way]]:
    sylls = syllabify(form)
    out = []
    for j, (text, long_vowel, coda) in enumerate(sylls):
        final = j == len(sylls) - 1
        if final and last_word:
            ways = [(text, "x", (), 0., None)]
        elif final and coda > 0 and vowel_next:
            gunne = long_vowel and coda == 1 and text[-1:].lower() in "nñ"
            ways = readings(text[:-1], long_vowel, coda - 1, begins="vowel")
            ways += readings(text, long_vowel, coda, () if gunne else ("no ulama",), 0. if gunne else STANDARD)
        else:
            ways = readings(text, long_vowel, coda)
        if standard:
            ways = [w for w in ways if not {"no medd", "no ulama"} & set(w[2])]
        out.append(ways)
    return out

def align(words: Sequence[Sequence[Tuple[str, float]]], pattern: str,
          max_cost: float = 99., standard: bool = False) -> Optional[Tuple[float, List[str], List[list]]]:
    n = len(pattern)
    starts = [{begins(f) for f, _ in cands} for cands in words]

    @lru_cache(maxsize=None)
    def word(i: int, k: int, need: Optional[str]):
        if i == len(words):
            return (0., (), ()) if k == n else None
        best = None
        nxt = starts[i + 1] if i + 1 < len(words) else set()
        for form, logp in words[i]:
            if need is not None and begins(form) != need:
                continue
            for cost, kk, then, syls in syllables(form, k, "vowel" in nxt, i == len(words) - 1):
                rest = word(i + 1, kk, then)
                if rest is None:
                    continue
                total = cost + rest[0] - 1e-3 * logp
                if best is None or total < best[0]:
                    best = (total, (form,) + rest[1], tuple(syls) + rest[2])
        return best

    def syllables(form, k, vowel_next, last_word):
        ways = options(form, vowel_next, last_word, standard)
        found = []

        def go(j, k, cost, acc, then):
            if j == len(ways):
                found.append((cost, k, then, acc))
                return
            for text, slots, kusur, c, begins_next in ways[j]:
                if slots == "x":
                    if k == n - 1:
                        go(j + 1, k + 1, cost, acc + [(text, pattern[k] if pattern[k] != "x" else "-", kusur)], None)
                    continue
                if k + len(slots) <= n and all(p in ("x", s) for p, s in zip(pattern[k:k + len(slots)], slots)):
                    go(j + 1, k + len(slots), cost + c, acc + [(text, slots, kusur)], begins_next)
        go(0, k, 0., [], None)
        best = {}
        for cost, kk, then, acc in found:
            if (kk, then) not in best or cost < best[kk, then][0]:
                best[kk, then] = (cost, kk, then, acc)
        return list(best.values())

    result = word(0, 0, None)
    if result is None or result[0] > max_cost:
        return None
    forms = list(result[1])
    syls = [[t, sl, list(k)] for t, sl, k in result[2]]
    k = 0
    for form in forms[:-1]:
        ss = syllabify(form)
        k += len(ss)
        if len(ss[-1][0]) > len(syls[k - 1][0]):
            syls[k][0] = ss[-1][0][-1] + syls[k][0]
    return round(result[0], 2), forms, syls
