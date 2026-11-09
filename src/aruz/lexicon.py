from __future__ import annotations
import unicodedata
from collections import Counter, defaultdict
from math import exp, log
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from .scan import LOAN_MARKS, VOWELS, syllabify

KEEP = {("s", "̧"), ("c", "̧"), ("o", "̈"), ("u", "̈")}
DROP = set("ʿʾʼ'’ˇ")
PUNCT = ".,;:!?«»\"()[]"

HEMZE = str.maketrans({"’": "ʾ", "'": "ʾ", "ʼ": "ʾ"})
CONSONANT_MARKS = {"\u0323", "\u0331", "\u0320", "\u032e", "\u0307"}

def reduce(form: str, keep_case: bool = False) -> str:
    out = []
    for ch in unicodedata.normalize("NFD", (form if keep_case else lower(form)).translate(HEMZE).replace("ˇ", "")):
        if ch in CONSONANT_MARKS and not (keep_case and ch == "\u0307" and out and out[-1] == "I"):
            continue
        out.append("\u0304" if ch == "\u0302" else ch)
    return unicodedata.normalize("NFC", "".join(out))

def lower(form: str) -> str:
    return form.replace("İ", "i").replace("I", "ı").lower()

def normalize(form: str, keep_case: bool = False) -> str:
    out = []
    for ch in unicodedata.normalize("NFD", form if keep_case else lower(form)):
        if ch in DROP or (not ch.isalpha() and not unicodedata.combining(ch)):
            continue
        if unicodedata.combining(ch):
            if out and ((out[-1].lower(), ch) in KEEP or (keep_case and out[-1] == "I" and ch == "\u0307")):
                out[-1] = unicodedata.normalize("NFC", out[-1] + ch)
            continue
        out.append(ch)
    return "".join(out)

def edits(key: str, alphabet: Set[str]) -> Iterable[str]:
    for i in range(len(key)):
        yield key[:i] + key[i+1:]
    for i in range(len(key) + 1):
        for ch in alphabet:
            yield key[:i] + ch + key[i:]

def split(popular: str, scholarly: str) -> Tuple[str, str]:
    i = 0
    while i < min(len(popular), len(scholarly)) and popular[i] == scholarly[i]:
        i += 1
    return popular[i:], scholarly[i:]

class Lexicon:
    def __init__(self, p_edit: float = .01, top: int = 3, min_rule: int = 2):
        self.forms: Dict[str, Counter] = defaultdict(Counter)
        self.cased: Dict[str, Counter] = defaultdict(Counter)
        self.subs: Dict[str, Counter] = defaultdict(Counter)
        self.rules: Dict[str, Counter] = defaultdict(Counter)
        self.tails: Counter = Counter()
        self.p_edit = p_edit
        self.top = top
        self.min_rule = min_rule

    def update(self, tokens: Iterable[str]) -> None:
        for t in tokens:
            t = t.strip(PUNCT)
            parts = [t] + (t.split("-") if "-" in t else [])
            for p in parts:
                if p and normalize(p):
                    self.forms[normalize(p)][reduce(p)] += 1
                    self.cased[normalize(p, keep_case=True)][reduce(p)] += 1

    def update_pairs(self, popular: str, scholarly: str) -> None:
        self.update(scholarly.split())
        ps = [normalize(w.strip(PUNCT)) for w in popular.split()]
        ss = [normalize(w.strip(PUNCT)) for w in scholarly.split()]
        if len(ps) != len(ss):
            return
        for p, s in zip(ps, ss):
            self.subs[p][s] += 1
            for n in range(1, len(p) + 1):
                self.tails[p[-n:]] += 1
            if p != s:
                pt, st = split(p, s)
                self.rules[pt][st] += 1

    def function_words(self, n: int = 20) -> List[str]:
        counts = Counter()
        for forms in self.forms.values():
            for f, c in forms.items():
                if not any(ch in LOAN_MARKS for ch in f) and len(syllabify(f)) <= 2:
                    counts[f] += c
        return [f for f, _ in counts.most_common(n)]

    def _keys(self, key: str) -> Dict[str, float]:
        keys: Dict[str, float] = {}
        def add(k: str, lp: float):
            if k in self.forms:
                keys[k] = max(keys.get(k, -float("inf")), lp)

        subs = self.subs.get(key)
        if subs:
            total = sum(subs.values())
            for k, c in subs.items():
                add(k, log(c / total))
        else:
            add(key, 0.)
            for pt, targets in self.rules.items():
                if pt and key.endswith(pt):
                    total = self.tails[pt]
                    for st, c in targets.items():
                        if c >= self.min_rule:
                            add(key[:len(key) - len(pt)] + st, log(c / total))
        if not keys:
            alphabet = {ch for k in self.forms for ch in k}
            for e in edits(key, alphabet):
                add(e, log(self.p_edit))
        return keys

    def alternatives(self, word: str) -> List[Tuple[str, float]]:
        scored: Dict[str, float] = {}
        for k, lp in self._keys(normalize(word)).items():
            counts = self.forms[k]
            total = sum(counts.values())
            for form, c in counts.items():
                scored[form] = max(scored.get(form, -float("inf")), lp + log(c / total))
        if not scored:
            return [(lower(word.strip(PUNCT)), 0.)]
        best = sorted(scored.items(), key=lambda x: -x[1])[:self.top]
        z = log(sum(exp(lp) for _, lp in best))
        return [(f, lp - z) for f, lp in best]
