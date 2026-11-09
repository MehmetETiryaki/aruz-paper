from dataclasses import dataclass, field
from itertools import product
from math import exp, inf, log
from typing import Dict, List, Optional

import torch
import unicodedata

from .lexicon import normalize as lex_normalize
from .scan import lattice

CAP = 30.
LINE_SIGNALS = ("ll", "licences", "inserts", "max costly", "max loo")
WORD_SIGNALS = ("costly", "loo")
INFO_SIGNALS = ("surprise", "oov")
COSTLY = {"imâle", "zihaf", "no medd"}
WILDCARD = [("".join("tā" if w == "L" else "ta" for w in ws), 0.) for n in range(1, 5) for ws in product("SL", repeat=n)]
WILDCARD = [(f, -log(len(WILDCARD))) for f, _ in WILDCARD]

def tag_posteriors(model, lat):
    Z = model.lattice_forward(lat).item()
    if Z == -inf:
        return Z, {}
    A = model._A(lat)
    beta = [torch.zeros(model.k) for _ in range(lat.n)]
    lk = [-inf] * lat.n
    beta[lat.final], lk[lat.final] = A[:, model.eos_t].clone(), 0.
    out = lat.out()
    for u in range(lat.final - 1, -1, -1):
        arcs = [a for a in out[u] if lk[a.dst] > -inf]
        if not arcs:
            continue
        base = max(lk[a.dst] + a.logp for a in arcs)
        for a in arcs:
            beta[u] += exp(lk[a.dst] + a.logp - base) * (A @ (model.B[:, a.w] * beta[a.dst]))
        if u > 0:
            beta[u][model.bos_t] = 0
        beta[u][model.eos_t] = 0
        k = torch.sum(beta[u])
        if k > 0:
            beta[u] /= k
            lk[u] = base + torch.log(k).item()
    post = {}
    for i, a in enumerate(lat.arcs):
        if model.k_alphas[a.src] == -inf or lk[a.dst] == -inf:
            continue
        scale = exp(model.k_alphas[a.src] + a.logp + lk[a.dst] - Z)
        post[i] = (model.alphas[a.src] @ A) * model.B[:, a.w] * beta[a.dst] * scale
    return Z, post

@dataclass
class Detector:
    model: object
    starts: Dict[str, List[int]]
    slots: list
    restorer: object
    from_hmm: object
    stats: Dict[str, tuple] = field(default_factory=dict)
    threshold: Optional[float] = None
    names: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.names = self.names or list(self.starts)
        self.prior = {m: log(self.model.A[self.model.bos_t, st].sum().item()) for m, st in self.starts.items()}
        kinds = [s.kind for s in self.slots]
        self.kind = kinds + ["X"] * (self.model.k - len(kinds))

    def lattice(self, words, wildcard=None):
        table = {j: WILDCARD if j == wildcard else self.restorer.alternatives(w) for j, w in enumerate(words)}
        return lattice(self.model, words, alternatives=_Alternatives(words, table))

    def line_ll(self, lat, meter):
        lat.starts = self.starts[meter]
        try:
            return self.model.lattice_forward(lat).item() - self.prior[meter]
        finally:
            lat.starts = None

    def find_meter(self, lines_words):
        lats = [self.lattice(ws) for ws in lines_words]
        total = {m: self.prior[m] + sum(self.line_ll(l, m) for l in lats) for m in self.names}
        return max(self.names, key=lambda m: total[m]), lats

    def raw_signals(self, words, lat, meter):
        lat.starts = self.starts[meter]
        try:
            Z, post = tag_posteriors(self.model, lat)
            path = None
            if Z > -inf:
                _, path = self.model.lattice_viterbi(lat)
        finally:
            lat.starts = None
        Z -= self.prior[meter] if Z > -inf else 0.
        n_syl = max(1, len(path) if path else sum(len(_syllables(w)) for w in words))
        line = {"ll": min(CAP, -Z / n_syl) if Z > -inf else CAP}
        if path:
            syl = self.from_hmm(_Path([(a, self.slots[t]) for a, t in path]))
            line["licences"] = sum(bool(COSTLY & set(s[2])) for s in syl)
            line["inserts"] = sum(self.slots[t].kind == "I" for _, t in path)
        else:
            line["licences"], line["inserts"] = CAP, CAP
        word = [{"costly": 0.} for _ in words]
        for i, a in enumerate(lat.arcs):
            if i not in post:
                continue
            p = post[i]
            w = a.info.weight
            costly = sum(p[t].item() for t in range(len(self.slots))
                         if (self.kind[t] == "I") or (self.kind[t] == "L" and w == "S")
                         or (self.kind[t] == "S" and w != "S"))
            word[a.info.word]["costly"] += costly
        for j, ws in enumerate(words):
            wl = self.lattice(words, wildcard=j)
            gain_ll = self.line_ll(wl, meter)
            if Z == -inf:
                word[j]["loo"] = CAP if gain_ll > -inf else 0.
            else:
                word[j]["loo"] = min(CAP, gain_ll - Z) if gain_ll > -inf else -CAP
            word[j]["surprise"] = min(CAP, -best_logp(self.restorer.model, ws) / max(1, len(ws)))
            word[j]["oov"] = float(lex_normalize(ws) not in self.restorer.lexicon.forms)
        for s in WORD_SIGNALS + INFO_SIGNALS:
            line["max " + s] = max(w[s] for w in word) if word else 0.
        return line, word, (Z, path)

    def z(self, name, value):
        mean, sd = self.stats[name]
        return (value - mean) / sd if sd > 0 else 0.

    def poem(self, lines_words):
        meter, lats = self.find_meter(lines_words)
        out = []
        for words, lat in zip(lines_words, lats):
            line, word, (Z, path) = self.raw_signals(words, lat, meter)
            score = sum(self.z(s, line[s]) for s in LINE_SIGNALS) / len(LINE_SIGNALS)
            wscores = [sum(self.z(s, w[s]) for s in WORD_SIGNALS) / len(WORD_SIGNALS) for w in word]
            order = sorted(range(len(words)), key=lambda j: -wscores[j])
            flagged = self.threshold is not None and score > self.threshold
            warning = self.warning(words, order[0], meter, word[order[0]]) if flagged and order else None
            reasons = []
            if Z == -inf:
                reasons.append(f"does not scan in {meter}")
            elif line["licences"] >= 3:
                reasons.append(f"needs {int(line['licences'])} licences in {meter}")
            if line["inserts"] and line["inserts"] < CAP:
                reasons.append(f"{int(line['inserts'])} syllable(s) outside the meter's positions")
            if order:
                j = order[0]
                if word[j]["oov"]:
                    reasons.append(f"'{words[j]}' is unknown to the restorer")
                if word[j]["loo"] >= 5 or (Z == -inf and word[j]["loo"] == CAP):
                    reasons.append(f"the line scans if '{words[j]}' is replaced")
                elif word[j]["costly"] >= 1:
                    reasons.append(f"'{words[j]}' needs a costly reading (imâle, shortening or an insert)")
            out.append({"words": words, "score": score, "flagged": flagged, "warning": warning,
                        "suspects": [(words[j], wscores[j]) for j in order], "reasons": reasons,
                        "signals": line, "word signals": word})
        return {"meter": meter, "lines": out}

    def wanted(self, words, j, meter):
        lat = self.lattice(words, wildcard=j)
        lat.starts = self.starts[meter]
        try:
            _, path = self.model.lattice_viterbi(lat)
        except ValueError:
            return None
        finally:
            lat.starts = None
        return "".join("L" if a.info.weight == "L" else "S" for a, _ in path if a.info.word == j)

    def warning(self, words, j, meter, signals):
        from difflib import get_close_matches
        from .scan import syllabify, weight
        w = words[j]

        def weights_of(plain):
            alts = self.restorer.alternatives(plain)
            if not alts:
                return None
            try:
                return "".join("L" if weight(*s) != "S" else "S" for s in syllabify(alts[0][0]))
            except ValueError:
                return None

        want, have = self.wanted(words, j, meter), weights_of(w)
        if want is None or have is None:
            kind = "the line does not scan around this word"
        elif len(want) > len(have):
            kind = "a syllable is missing here (a dropped vowel, izafet or word?)"
        elif len(want) < len(have):
            kind = "a syllable too many here (an added vowel, a doubled vowel or a word run in?)"
        else:
            kind = "a long and a short syllable do not fit here (a wrong letter or word?)"
        if signals.get("oov"):
            kind = "an unknown word (a typo or an OCR error?); " + kind
        fitting = []
        if want:
            if not hasattr(self, "_keys"):
                self._keys = list(self.restorer.lexicon.forms)
            for c in get_close_matches(lex_normalize(w), self._keys, n=40, cutoff=.7):
                if c != lex_normalize(w) and weights_of(c) == want:
                    fitting.append(c)
                if len(fitting) == 3:
                    break
        return {"word": w, "position": j, "kind": kind, "meter wants": want, "word gives": have, "fitting": fitting}

def best_logp(ngram, plain, beam=20):
    from .restore import BOS, EOS, lower
    letters = [unicodedata.normalize("NFC", c) for c in unicodedata.normalize("NFC", lower(plain))]
    hyps = [((BOS,) * (ngram.n - 1), 0.)]
    for ch in letters:
        options = [u for u, _ in ngram.by_plain[ch].most_common(8)] or [(ch, ch)]
        hyps = sorted(((h[1:] + (u,), lp + ngram.logp(h, u)) for h, lp in hyps for u in options), key=lambda x: -x[1])[:beam]
    return max(lp + ngram.logp(h, EOS) for h, lp in hyps)

class _Alternatives:

    def __init__(self, words, table):
        self.queue = {}
        for j, w in enumerate(words):
            self.queue.setdefault(w, []).append(table[j])

    def __call__(self, w):
        return self.queue[w].pop(0)

class _Path:
    def __init__(self, path):
        self.path = path

def _syllables(w):
    from .scan import syllabify
    return syllabify(w)
