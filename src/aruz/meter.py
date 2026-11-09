from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

from .hmm import BOS_TAG, BOS_WORD, EOS_TAG, EOS_WORD, Integerizer, Tag, Word
from .hmm.lattice import LatticeHMM

WEIGHTS = ["L", "S", "H"]
INNATE_WEIGHTS = ["O", "K"]
ORIGINS = ["tr", "ar"]

@dataclass
class Meter:
    name: str
    display: str
    patterns: List[str]

METERS = [
    Meter("remel_8_mahzuf", "fâilâtün fâilâtün fâilâtün fâilün",
          ["x.-- -.-- -.-- -.-"]),
    Meter("remel_6_mahzuf", "fâilâtün fâilâtün fâilün",
          ["x.-- -.-- -.-"]),
    Meter("remel_8_mahbun_mahzuf", "fâilâtün feilâtün feilâtün fa'lün",
          ["x.-- ..-- ..-- --", "x.-- ..-- ..-- ..-"]),
    Meter("hezec_8_salim", "mefâîlün mefâîlün mefâîlün mefâîlün",
          [".--- .--- .--- .---"]),
    Meter("hezec_6_mahzuf", "mefâîlün mefâîlün feûlün",
          [".--- .--- .--"]),
    Meter("hezec_8_ahreb", "mef'ûlü mefâîlü mefâîlü feûlün",
          ["--. .--. .--. .--"]),
    Meter("hezec_6_ahreb_makbuz_mahzuf", "mef'ûlü mefâilün feûlün",
          ["--. .-.- .--"]),
    Meter("muctes_8_mahbun_mahzuf", "mefâilün feilâtün mefâilün fa'lün",
          [".-.- ..-- .-.- --", ".-.- ..-- .-.- ..-"]),
    Meter("muzari_8_ahreb", "mef'ûlü fâilâtü mefâîlü fâilün",
          ["--. -.-. .--. -.-"]),
    Meter("muzari_8_ahreb_salim", "mef'ûlü fâilâtün mef'ûlü fâilâtün",
          ["--. -.-- --. -.--"]),
    Meter("mutekarib_8_mahzuf", "feûlün feûlün feûlün feûl",
          [".-- .-- .-- .-"]),
    Meter("recez_8_salim", "müstef'ilün müstef'ilün müstef'ilün müstef'ilün",
          ["--.- --.- --.- --.-"]),
    Meter("hafif_mahbun_mahzuf", "fâilâtün mefâilün fa'lün",
          ["x.-- .-.- --", "x.-- .-.- ..-"]),
    Meter("seri_matvi_mevkuf", "müfteilün müfteilün fâilün",
          ["-..- -..- -.-"]),
]

@dataclass
class Slot:
    meter: str
    chain: int
    j: int
    kind: str

WEIGHT_PRIORS: Dict[str, Dict[str, Dict[str, float]]] = {
    "L": {"tr": {"L": .90, "S": .09, "H": .01}, "ar": {"L": .975, "S": .02, "H": .005}},
    "S": {"tr": {"S": .80, "H": .15, "L": .05}, "ar": {"S": .72, "H": .25, "L": .03}},
    "X": {"tr": {"L": .90, "S": .10, "H": 0.}, "ar": {"L": .90, "S": .10, "H": 0.}},
    "I": {"tr": {"L": 1/3, "S": 1/3, "H": 1/3}, "ar": {"L": 1/3, "S": 1/3, "H": 1/3}},
}

def symbol(weight: str, cls: str) -> str:
    return f"{weight}:{cls}"

def kinds(pattern: str) -> List[str]:
    weights = [{"-": "L", ".": "S", "u": "S", "x": "X"}[c] for c in pattern if not c.isspace()]
    weights[-1] = "X"
    return weights

def build(meters: List[Meter] = METERS,
          function_words: List[str] = [],
          p_delete: float = .01,
          p_insert: float = .01,
          tie: bool = True,
          innate: bool = True) -> Tuple[LatticeHMM, Dict[str, List[int]], List[Slot]]:

    tagset: Integerizer[Tag] = Integerizer()
    slots: List[Slot] = []
    for m in meters:
        for c, pattern in enumerate(m.patterns):
            for j, kind in enumerate(kinds(pattern)):
                for state in (kind, "I"):
                    tagset.add(Tag(f"{m.name}#{c}:{'I' if state == 'I' else ''}{j}"))
                    slots.append(Slot(m.name, c, j, state))
    tagset.add(EOS_TAG)
    tagset.add(BOS_TAG)
    classes = ORIGINS + list(function_words)
    weights = WEIGHTS + (INNATE_WEIGHTS if innate else [])
    vocab: Integerizer[Word] = Integerizer([Word(symbol(w, c)) for c in classes for w in weights]
                                           + [EOS_WORD, BOS_WORD])

    k, V = len(tagset), len(vocab) - 2
    bos, eos = tagset.index(BOS_TAG), tagset.index(EOS_TAG)
    A = torch.zeros(k, k)
    starts: Dict[str, List[int]] = {m.name: [] for m in meters}

    for m in meters:
        for c, pattern in enumerate(m.patterns):
            n = len(kinds(pattern))
            match = [tagset.index(Tag(f"{m.name}#{c}:{j}")) for j in range(n)] + [eos]
            insert = [tagset.index(Tag(f"{m.name}#{c}:I{j}")) for j in range(n)]
            prior = 1 / (len(meters) * len(m.patterns))
            A[bos, match[0]] = prior * (1 - p_delete)
            A[bos, match[1]] = prior * p_delete
            starts[m.name] += [match[0], match[1]]
            for j in range(n):
                A[match[j], match[j+1]] = 1 - p_delete - p_insert
                A[match[j], insert[j]] = p_insert
                if j + 2 <= n:
                    A[match[j], match[j+2]] = p_delete
                A[insert[j], match[j+1]] = 1
            for row in match[:n] + insert:
                A[row] /= A[row].sum()

    group = {"L": 0, "S": 1, "X": 2, "I": 3}
    B = torch.zeros(k, V)
    B_tie = torch.tensor([group[s.kind] for s in slots] + [4, 5])
    for t, s in enumerate(slots):
        for c in classes:
            for w, p in WEIGHT_PRIORS[s.kind]["ar" if c == "ar" else "tr"].items():
                B[t, vocab.index(Word(symbol(w, c)))] = p / len(classes)
                if innate and w == "L":
                    for x in INNATE_WEIGHTS:
                        B[t, vocab.index(Word(symbol(x, c)))] = p / len(classes)

    model = LatticeHMM(tagset, vocab, A_mask=A > 0, B_mask=B > 0, B_tie=B_tie if tie else None)
    model.A, model.B = A, B / B.sum(dim=1, keepdim=True).clamp(min=1e-30)
    return model, starts, slots
