from __future__ import annotations
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, replace
from math import inf, log
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from .hmm import Integerizer, Word
from .hmm.lattice import Arc, Lattice, LatticeHMM
from .meter import Slot, symbol

SHORT_VOWELS = set("aeıioöuüėAEIİOÖUÜĖ")
LONG_VOWELS = set("âîûāēīōūÂÎÛĀĒĪŌŪ")
VOWELS = SHORT_VOWELS | LONG_VOWELS
STOPS = "ʿʾʼ'’"
LOAN_MARKS = LONG_VOWELS | set(STOPS)

Alternatives = Callable[[str], List[Tuple[str, float]]]

def weight(text: str, long_vowel: bool, coda: int) -> str:
    if coda == 0:
        return "L" if long_vowel else "S"
    if long_vowel and coda == 1 and text[-1] in "nN":
        return "L"
    if long_vowel or coda >= 2:
        return "LH"
    return "L"

def origin(word: str) -> str:
    return "ar" if any(c in LOAN_MARKS for c in word) else "tr"

def syllabify(word: str) -> List[Tuple[str, bool, int]]:
    chars = [c for c in unicodedata.normalize("NFC", word)
             if c in "ʿʾʼ'’" or (c.isalpha() and unicodedata.category(c) != "Lm")]
    nuclei = [i for i, c in enumerate(chars) if c in VOWELS]
    if not nuclei:
        raise ValueError(f"no vowel in {word!r}")
    bounds = [0]
    for a, b in zip(nuclei, nuclei[1:]):
        bounds.append(b if b - a == 1 else b - 1)
    bounds.append(len(chars))
    out = []
    for (start, end), v in zip(zip(bounds, bounds[1:]), nuclei):
        out.append(("".join(chars[start:end]), chars[v] in LONG_VOWELS, end - v - 1))
    return out

@dataclass(frozen=True)
class ArcInfo:
    word: int
    form: str
    syllable: int
    text: str
    weight: str
    natural: str
    ulama: bool
    medd: Optional[bool]
    first: bool
    emit: str = ""

def moved(info: ArcInfo) -> str:
    return syllabify(info.form)[-1][0][-1]

def build_lattice(words: Sequence[str], vocab: Integerizer[Word],
                  alternatives: Alternatives = lambda w: [(w, 0.)],
                  function_words: Sequence[str] = (), innate: bool = True) -> Lattice:
    lat = Lattice()
    ends: Dict[object, int] = {False: 0}
    alts = list(alternatives) if isinstance(alternatives, list) else [alternatives(w) for w in words]
    next_vowel = [i + 1 < len(words) and any(f[:1] in VOWELS for f, _ in alts[i + 1]) for i in range(len(words))]
    for i, word in enumerate(words):
        last = i == len(words) - 1
        chains = []
        for form, logp in alts[i]:
            cls = form if form in function_words else origin(form)
            sylls = syllabify(form)
            end_text, end_long, end_coda = sylls[-1]
            can_ulama = not last and next_vowel[i] and end_coda > 0
            ways = [(u, k) for u in ([False, True] if can_ulama else [False])
                    for k in ((False, True) if innate and can_ulama and not u else (False,))]
            for ulama, kept in ways:
                steps = []
                for j, (text, long_vowel, coda) in enumerate(sylls):
                    final = j == len(sylls) - 1
                    if ulama and final:
                        text, coda = text[:-1], coda - 1
                    nat = weight(text, long_vowel, coda)

                    def arc(w, n, medd=None):
                        e = ""
                        if innate and w != "H":
                            e = "K" if kept and final else ("O" if medd is False else "")
                        return ArcInfo(i, form, j, text, w, n, ulama, medd, False, e)
                    if last and final:
                        steps.append([[arc(nat[0], nat[0])]])
                    elif nat == "LH":
                        steps.append([[arc("L", "L", True), arc("H", "H", True)], [arc("L", "L", False)]])
                    else:
                        steps.append([[arc(nat, nat)]])
                steps[0] = [[replace(way[0], first=True)] + way[1:] for way in steps[0]]
                need: object = ulama
                if innate and can_ulama and not ulama:
                    need = "kept" if kept else "cons"
                for begins, src in ends.items():
                    if begins in (True, "kept") and form[0] not in VOWELS:
                        continue
                    if begins == "cons" and form[0] in VOWELS:
                        continue
                    chains.append((src, steps, need, logp, cls))

        pending = []
        for src, steps, need, logp, cls in chains:
            layout, start = [], src
            for k, step in enumerate(steps):
                inner = [[lat.node() for _ in way[:-1]] for way in step]
                end = None if k == len(steps) - 1 else lat.node()
                layout.append((start, step, inner, end))
                start = end
            pending.append((layout, need, logp, cls))
        new_ends = {u: lat.node() for u in sorted({c[2] for c in chains}, key=str)}
        for layout, need, logp, cls in pending:
            for start, step, inner, end in layout:
                for way, nodes in zip(step, inner):
                    nodes = [start] + nodes + [new_ends[need] if end is None else end]
                    for info, s_, d in zip(way, nodes, nodes[1:]):
                        sym = symbol(info.emit or info.weight, cls)
                        w = vocab.index(Word(sym))
                        if w is None:
                            raise KeyError(f"symbol {sym} is not in the model's vocab")
                        lat.add(s_, d, w, logp if info.first else 0., info)
        ends = new_ends
    lat.final = ends[False]
    return lat

def lattice(model: LatticeHMM, words: Sequence[str], **kw) -> Lattice:
    innate = model.vocab.index(Word(symbol("K", "tr"))) is not None
    return build_lattice(words, model.vocab, innate=innate, **kw)

def identify(model: LatticeHMM, starts: Dict[str, List[int]], poem: List[Lattice]) -> Dict[str, float]:
    total = {}
    for m, st in starts.items():
        prior = model.A[model.bos_t, st].sum().item()
        total[m] = log(prior) if prior > 0 else -inf
        for lat in poem:
            lat.starts = st
            total[m] += model.lattice_forward(lat).item() - log(prior)
            lat.starts = None
    scores = torch.tensor(list(total.values()))
    if torch.isinf(scores).all():
        raise ValueError("no meter can produce this poem")
    return dict(zip(total, (scores - torch.logsumexp(scores, 0)).exp().tolist()))

@dataclass
class Scansion:
    path: List[Tuple[Arc, Slot]]
    logp: float
    words: List[str]
    forms: Dict[int, List[Tuple[str, float]]]
    deleted: List[int]

    def __str__(self) -> str:
        marks = {"L": "–", "S": "·", "H": "·"}
        parts = []
        groups: Dict[Tuple[int, int], List[Tuple[Arc, Slot]]] = defaultdict(list)
        for arc, slot in self.path:
            groups[arc.info.word, arc.info.syllable].append((arc, slot))
        carry = ""
        for (i, j), group in groups.items():
            info = group[0][0].info
            last = j == len(syllabify(info.form)) - 1
            text = carry + info.text + ("‿" if info.ulama and last else "")
            carry = moved(info) if info.ulama and last else ""
            mark = "".join(marks[a.info.natural] for a, _ in group)
            notes = "".join(note(a.info.natural, s.kind) for a, s in group)
            parts.append(f"{text}[{mark}{notes}]")
        out = " ".join(parts)
        for i, forms in self.forms.items():
            if len(forms) > 1 or forms[0][0] != self.words[i].lower():
                out += f"\n  {self.words[i]}: " + " | ".join(f"{f} {p:.2f}" for f, p in forms)
        if self.deleted:
            out += f"\n  missing syllable at {', '.join(map(str, self.deleted))}"
        return out

def note(natural: str, kind: str) -> str:
    if kind == "I":
        return "+"
    if kind == "L" and natural == "S":
        return "ⁱ"
    if kind == "S" and natural == "L":
        return "ᶻ"
    return ""

def scan(model: LatticeHMM, starts: List[int], slots: List[Slot], lat: Lattice,
         words: Sequence[str], min_posterior: float = .05) -> Scansion:
    lat.starts = starts
    try:
        logp, path = model.lattice_viterbi(lat)
        posteriors = model.arc_marginals(lat)
    finally:
        lat.starts = None
    forms: Dict[int, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for i, a in enumerate(lat.arcs):
        if a.info.first:
            forms[a.info.word][a.info.form] += posteriors.get(i, 0.)
    report = {i: sorted(((f, p) for f, p in fs.items() if p >= min_posterior), key=lambda x: -x[1])
              for i, fs in forms.items()}
    tagged = [(a, slots[t]) for a, t in path]
    chain = tagged[0][1]
    n = 1 + max(s.j for s in slots if (s.meter, s.chain) == (chain.meter, chain.chain))
    return Scansion(tagged, logp, list(words), report, deleted([s for _, s in tagged], n))

def deleted(tags: List[Slot], n: int) -> List[int]:
    pos = [-1] + [t.j for t in tags if t.kind != "I"] + [n]
    return [j for a, b in zip(pos, pos[1:]) for j in range(a + 1, b)]
