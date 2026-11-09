from __future__ import annotations

import random
import re
from bisect import bisect_left
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Sequence, Tuple

VOWELS = "aeıioöuü"
LEFT, RIGHT = 1, 2
MAX_SPAN = 3

def tlower(s: str) -> str:
    return s.replace("İ", "i").replace("I", "ı").lower()

def tupper_first(w: str) -> str:
    return {"i": "İ", "ı": "I"}.get(w[:1], w[:1].upper()) + w[1:]

def letters(w: str) -> str:
    return re.sub(r"[^a-zçğıöşü]", "", tlower(w))

def word_weights(w: str) -> str:
    w = letters(w)
    vs = [i for i, c in enumerate(w) if c in VOWELS]
    out = []
    for k, i in enumerate(vs):
        if k + 1 < len(vs):
            out.append("-" if vs[k + 1] - i - 1 >= 2 else ".")
        else:
            out.append("-" if i < len(w) - 1 else ".")
    return "".join(out)

def line_weights(words: Sequence[str]) -> str:
    return " ".join(word_weights(w) for w in words)

def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b, autojunk=False).ratio()

def pad(w: str) -> str:
    return "^" * LEFT + w + "$" * RIGHT

def edits_of(e: str, w: str):
    p = pad(e)
    out = []
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, e, w, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        frm, to = e[i1:i2], w[j1:j2]
        if len(frm) > MAX_SPAN or len(to) > MAX_SPAN:
            return None
        out.append((p[i1:i1 + LEFT], frm, to, p[i2 + LEFT:i2 + LEFT + RIGHT], i1))
    return out

def sites(w: str):
    p = pad(w)
    for i in range(len(w) + 1):
        for n in range(0, MAX_SPAN + 1):
            if i + n > len(w):
                break
            yield p[i:i + LEFT], w[i:i + n], p[i + LEFT + n:i + LEFT + n + RIGHT], i, n

def apply_edit(w: str, i: int, n: int, to: str) -> str:
    return w[:i] + to + w[i + n:]

class WordEdits:

    def __init__(self, table: List[dict], scale: Dict[str, float], groups: Sequence[str]):
        self.by_site: Dict[tuple, List[Tuple[str, float, str]]] = defaultdict(list)
        for row in table:
            if row["group"] not in groups:
                continue
            rate = min(.95, row["applied"] / row["eligible"] * scale[row["group"]])
            self.by_site[row["left"], row["from"], row["right"]].append((row["to"], rate, row["group"]))

    def __call__(self, w: str, rng: random.Random, intensity: float):
        pre, core, post = re.match(r"^(\W*)(.*?)(\W*)$", w).groups()
        base = letters(core)
        if not base or base != tlower(core):
            return w, []
        out, applied, i = base, [], 0
        cands = defaultdict(list)
        for left, frm, right, at, n in sites(base):
            for to, rate, group in self.by_site.get((left, frm, right), ()):
                cands[at].append((n, to, rate, group))
        shift, last_end = 0, -1
        for at in sorted(cands):
            if at < last_end:
                continue
            for n, to, rate, group in cands[at]:
                if rng.random() < min(.95, rate * intensity):
                    out = apply_edit(out, at + shift, n, to)
                    applied.append({"kind": "edit", "group": group, "from": base[at:at + n], "to": to, "at": at})
                    shift += len(to) - n
                    last_end = at + max(n, 1)
                    break
        if not applied:
            return w, []
        return pre + (tupper_first(out) if core[:1].isupper() and out else out) + post, applied

class Substitutes:

    def __init__(self, vocab: Dict[str, int], observed: List[dict]):
        self.observed = observed
        self.by_syl: Dict[int, List[str]] = defaultdict(list)
        self.freq = vocab
        for w, n in vocab.items():
            if n >= 2 and len(w) >= 2:
                self.by_syl[len(word_weights(w))].append(w)
        for k in self.by_syl:
            self.by_syl[k].sort(key=lambda w: -vocab[w])

    def __call__(self, w: str, rng: random.Random, same_weights: Optional[bool]) -> Optional[str]:
        rows = [o for o in self.observed if same_weights is None or o["same weights"] == same_weights]
        if not rows:
            return None
        o = rng.choice(rows)
        base = letters(w)
        target = len(word_weights(base)) + o["syllable difference"]
        pool = self.by_syl.get(target, [])[:4000]
        if not pool:
            return None
        lo, hi = o["similarity band"]
        sample = rng.sample(pool, min(400, len(pool)))
        fits = [c for c in sample if c != base and lo <= similarity(base, c) < hi
                and (word_weights(c) == word_weights(base)) == o["same weights"]]
        if not fits:
            return None
        c = rng.choices(fits, weights=[self.freq[x] for x in fits])[0]
        return tupper_first(c) if w[:1].isupper() else c

IZAFET = re.compile(r"^(.+)-(?:y|ʾ)?[ıiuü]$")
ELISION = re.compile(r"^(k|n|ç)[’'ʼ]([^\W\d_].*)$", re.I)
EXPAND = {"k": "ki", "n": "ne", "ç": "çü"}
GROUPS = ("forms", "damage", "clutter", "structure")

KEY_GROUP = {"substitution, weights same": "forms", "line number": "clutter", "citation": "clutter",
             "footnote digit": "clutter", "beyit left out": "structure"}

def rate(profile: dict, key: str, scale: float, intensity: float = 1.) -> float:
    r = profile["rates"][key]
    return min(.95, r["applied"] / r["eligible"] * scale * intensity) if r["eligible"] else 0.

class Generator:
    def __init__(self, profile: dict, scale, groups: Sequence[str], vocab: Dict[str, int], popular):
        if not isinstance(scale, dict):
            scale = {g: scale for g in GROUPS}
        self.p, self.scale, self.groups, self.popular = profile, scale, tuple(groups), popular
        self.edits = WordEdits(profile["word edits"], scale, self.groups)
        self.subst = Substitutes(vocab, profile["substitutions"])
        self.short = sorted((w for w, n in vocab.items() if n >= 5), key=lambda w: -vocab[w])
        self.vocab = vocab
        self.lengths = profile["added word lengths"]

    def on(self, group):
        return group in self.groups

    def r(self, key, intensity=1.):
        return rate(self.p, key, self.scale[KEY_GROUP.get(key, "damage")], intensity)

    def line(self, raw: List[str], sch: List[str], meta: dict, rng: random.Random, intensity: float):
        words, edits = [], []
        for r_, s in zip(raw, sch):
            plain = self.popular(s)
            m, e = IZAFET.match(s), ELISION.match(r_)
            if m and self.on("damage") and rng.random() < self.r("izafet dropped", intensity):
                new = self.popular(m.group(1))
                edits.append({"kind": "izafet dropped", "group": "damage", "from": plain, "to": new})
                words.append(new)
                continue
            if e and self.on("damage") and rng.random() < self.r("elision written out", intensity):
                new = [EXPAND[e.group(1).lower()], self.popular(e.group(2))]
                if r_[:1].isupper():
                    new[0] = tupper_first(new[0])
                edits.append({"kind": "elision written out", "group": "damage", "from": plain, "to": " ".join(new)})
                words.extend(new)
                continue
            words.append(plain)
        out = []
        for w in words:
            if self.on("damage") and rng.random() < self.r("word dropped", intensity):
                edits.append({"kind": "word dropped", "group": "damage", "from": w, "to": ""})
                continue
            subst = None
            for group, same in (("forms", True), ("damage", False)):
                if subst is None and self.on(group) and rng.random() < self.r(f"substitution, weights {'same' if same else 'changed'}", intensity):
                    subst = self.subst(w, rng, same)
                    if subst:
                        edits.append({"kind": "substitution", "group": group, "from": w, "to": subst})
            if subst:
                out.append(subst)
            else:
                new, applied = self.edits(w, rng, intensity)
                for a in applied:
                    edits.append({**a, "word": w})
                out.append(new)
            if self.on("damage") and rng.random() < self.r("word added", intensity):
                n = rng.choice(self.lengths)
                pool = [x for x in self.short[:3000] if len(x) == n] or self.short[:50]
                add = rng.choices(pool, weights=[self.vocab[x] for x in pool])[0]
                edits.append({"kind": "word added", "group": "damage", "from": "", "to": add})
                out.append(add)
        if self.on("damage"):
            joined = []
            for w in out:
                if joined and rng.random() < self.r("words joined", intensity):
                    edits.append({"kind": "words joined", "group": "damage", "from": joined[-1] + " " + w,
                                  "to": joined[-1] + w})
                    joined[-1] += w
                elif len(w) >= 4 and rng.random() < self.r("word split", intensity):
                    k = rng.randrange(2, len(w) - 1)
                    edits.append({"kind": "word split", "group": "damage", "from": w, "to": w[:k] + " " + w[k:]})
                    joined.extend([w[:k], w[k:]])
                else:
                    joined.append(w)
            out = joined
        text = " ".join(out)
        if self.on("clutter"):
            if out and rng.random() < self.r("footnote digit", intensity):
                j = rng.randrange(len(out))
                d = str(rng.randint(1, 9 if rng.random() < self.p["footnote one digit"] else 40))
                edits.append({"kind": "footnote digit", "group": "clutter", "from": out[j], "to": out[j] + d})
                out[j] += d
                text = " ".join(out)
            if rng.random() < self.r("line number", intensity):
                style = rng.choices(list(self.p["line number styles"]), weights=list(self.p["line number styles"].values()))[0]
                num = style.replace("N", str(meta["beyit"]))
                edits.append({"kind": "line number", "group": "clutter", "from": "", "to": num})
                text = num + " " + text
            if rng.random() < self.r("citation", intensity):
                cite = f"({meta['poet']}; G. {meta['number']}/{meta['beyit']})"
                edits.append({"kind": "citation", "group": "clutter", "from": "", "to": cite})
                text = text + " " + cite
        return text, edits

    def skipped(self, units: List[dict], keep: set, rng: random.Random, intensity: float) -> set:
        if not self.on("structure"):
            return set()
        out = {k for k in range(len(units)) if k not in keep and rng.random() < self.r("beyit left out", intensity)}
        return out if len(out) < len(units) else set()

    def intensity(self, rng: random.Random) -> float:
        return rng.choice(self.p["source intensities"])
