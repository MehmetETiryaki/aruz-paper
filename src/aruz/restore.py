from __future__ import annotations
import unicodedata
from collections import Counter, defaultdict
from functools import lru_cache
from math import exp, inf, log
from itertools import product
from typing import Dict, Iterable, List, Tuple

from .lexicon import Lexicon, lower, normalize, reduce

KEEP = {"̈", "̧"}
SILENT = set("ʿʾʼ’'ˇ-")
BOS, EOS = ("<s>", "<s>"), ("</s>", "</s>")
Unit = Tuple[str, str]

def upper(s: str) -> str:
    return s.replace("i", "İ").replace("ı", "I").upper()

def capitalized(word: str) -> bool:
    first = next((c for c in word if c.isalpha() and c not in SILENT), "")
    return first.isupper()

def units(form: str, cased: bool = False) -> List[Unit]:
    cap = cased and capitalized(form)
    glyphs = []
    for c in unicodedata.normalize("NFD", lower(form)):
        if unicodedata.combining(c) and glyphs:
            glyphs[-1] += c
        else:
            glyphs.append(c)
    out, pending = [], ""
    for g in glyphs:
        if g[0] in SILENT:
            pending += g
            continue
        plain = unicodedata.normalize("NFC", g[0] + "".join(c for c in g[1:] if c in KEEP))
        out.append((upper(plain) if cap else plain, unicodedata.normalize("NFC", pending + g)))
        pending = ""
    if pending and out:
        out[-1] = (out[-1][0], out[-1][1] + pending)
    return out

class JointNgram:
    def __init__(self, n: int = 5, cased: bool = False):
        self.n, self.cased = n, cased
        self.counts: List[Dict[tuple, Counter]] = [defaultdict(Counter) for _ in range(n)]
        self.by_plain: Dict[str, Counter] = defaultdict(Counter)

    def train(self, words: Iterable[Tuple[str, int]]) -> None:
        for form, c in words:
            seq = [BOS] * (self.n - 1) + units(form, getattr(self, "cased", False)) + [EOS]
            for i in range(self.n - 1, len(seq)):
                for k in range(self.n):
                    self.counts[k][tuple(seq[i - k:i])][seq[i]] += c
                if seq[i] != EOS:
                    self.by_plain[seq[i][0]][seq[i]] += c
        self.vocab = len(self.counts[0][()]) + 1
        self.totals = [{h: (sum(c.values()), len(c)) for h, c in level.items()} for level in self.counts]

    def logp(self, history: tuple, unit: Unit) -> float:
        p = 1 / self.vocab
        for k in range(self.n):
            h = history[len(history) - k:] if k else ()
            if h not in self.totals[k]:
                break
            total, types = self.totals[k][h]
            p = (self.counts[k][h][unit] + types * p) / (total + types)
        return log(p)

    def restore(self, plain: str, beam: int = 20, top: int = 5) -> List[Tuple[str, float]]:
        cap = getattr(self, "cased", False) and capitalized(plain)
        letters = [unicodedata.normalize("NFC", c) for c in unicodedata.normalize("NFC", lower(plain))]
        letters = [upper(c) for c in letters] if cap else letters
        hyps = [((BOS,) * (self.n - 1), "", 0.)]
        for ch in letters:
            options = [u for u, _ in self.by_plain[ch].most_common(8)] or \
                ([u for u, _ in self.by_plain[lower(ch)].most_common(8)] if cap else []) or [(ch, lower(ch))]
            grown = []
            for hist, text, lp in hyps:
                for u in options:
                    grown.append((hist[1:] + (u,), text + u[1], lp + self.logp(hist, u)))
            hyps = sorted(grown, key=lambda h: -h[2])[:beam]
        done: Dict[str, float] = {}
        for hist, text, lp in hyps:
            score = lp + self.logp(hist, EOS)
            done[text] = max(done.get(text, -float("inf")), score)
        best = sorted(done.items(), key=lambda x: -x[1])[:top]
        z = log(sum(exp(s - best[0][1]) for _, s in best)) + best[0][1]
        return [(f, s - z) for f, s in best]

class Restorer:

    def __init__(self, lexicon: Lexicon, model: JointNgram, weight: float = .8, top: int = 5, cased: bool = False):
        self.lexicon, self.model, self.weight, self.top, self.cased = lexicon, model, weight, top, cased
        self.alternatives = lru_cache(maxsize=None)(self._alternatives)

    def __getstate__(self):
        state = dict(self.__dict__)
        del state["alternatives"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        forms = defaultdict(Counter)
        for k, c in self.lexicon.forms.items():
            forms[normalize(k)].update(c)
        self.lexicon.forms = forms
        self.alternatives = lru_cache(maxsize=None)(self._alternatives)

    def _alternatives(self, word: str) -> List[Tuple[str, float]]:
        key = normalize(word)
        model = {f: exp(lp) for f, lp in self.model.restore(word, top=self.top)}
        if key in self.lexicon.forms:
            counts = self.lexicon.forms[key]
            total = sum(counts.values())
            lex = {f: c / total for f, c in counts.items()}
            seen = self.lexicon.cased.get(normalize(word, keep_case=True)) if getattr(self, "cased", False) else None
            if seen and normalize(word, keep_case=True) != key:
                n, types = sum(seen.values()), len(seen)
                lex = {f: (seen.get(f, 0) + types * lex.get(f, 0.)) / (n + types) for f in set(lex) | set(seen)}
            mixed = {f: self.weight * lex.get(f, 0.) + (1 - self.weight) * model.get(f, 0.) for f in set(lex) | set(model)}
        else:
            mixed = model
        merged: Dict[str, float] = {}
        for f, p in mixed.items():
            merged[reduce(f)] = merged.get(reduce(f), 0.) + p
        best = sorted(merged.items(), key=lambda x: -x[1])[:self.top]
        z = sum(p for _, p in best)
        return [(f, log(p / z)) for f, p in best]

def _lse(values):
    m = max(values)
    return m if m == -inf else m + log(sum(exp(v - m) for v in values))

class WordLM:
    BOS, EOS = "<s>", "</s>"

    def __init__(self, n: int = 2):
        self.n = n
        self.counts: List[Dict[tuple, Counter]] = [defaultdict(Counter) for _ in range(n)]

    def train(self, lines: Iterable[List[str]]) -> None:
        for forms in lines:
            seq = [self.BOS] * (self.n - 1) + list(forms) + [self.EOS]
            for i in range(self.n - 1, len(seq)):
                for k in range(self.n):
                    self.counts[k][tuple(seq[i - k:i])][seq[i]] += 1
        self.vocab = len(self.counts[0][()]) + 1
        self.totals = [{h: (sum(c.values()), len(c)) for h, c in level.items()} for level in self.counts]

    def logp(self, history: tuple, form: str) -> float:
        p = 1 / self.vocab
        for k in range(self.n):
            h = history[len(history) - k:] if k else ()
            if h not in self.totals[k]:
                break
            total, types = self.totals[k][h]
            p = (self.counts[k][h][form] + types * p) / (total + types)
        return log(p)

class ContextRestorer:

    def __init__(self, restorer: Restorer, lm: WordLM, alpha: float = 1.):
        self.restorer, self.lm, self.alpha = restorer, lm, alpha

    def alternatives(self, word: str) -> List[Tuple[str, float]]:
        return self.restorer.alternatives(word)

    def _pass(self, words, viterbi):
        cands = [self.restorer.alternatives(w) for w in words]
        m = self.lm.n - 1
        start = (WordLM.BOS,) * m
        score = lambda s, f, lp: self.alpha * lp + self.lm.logp(s, f)
        combine = max if viterbi else _lse
        F = [{start: 0.}]
        back = []
        for cs in cands:
            cur, bp = {}, {}
            for st, v in F[-1].items():
                for j, (f, lp) in enumerate(cs):
                    ns = (st + (f,))[len(st) + 1 - m:] if m else ()
                    x = v + score(st, f, lp)
                    if ns not in cur or (viterbi and x > cur[ns]):
                        if viterbi or ns not in cur:
                            cur[ns] = x
                            bp[ns] = (st, j)
                    if not viterbi:
                        cur[ns] = _lse([cur[ns], x]) if (st, j) != bp[ns] else cur[ns]
            F.append(cur)
            back.append(bp)
        return cands, F, back

    def best_line(self, words: Sequence[str]) -> List[str]:
        cands, F, back = self._pass(words, viterbi=True)
        end = {st: v + self.lm.logp(st, WordLM.EOS) for st, v in F[-1].items()}
        st = max(end, key=end.get)
        out = []
        for i in range(len(words), 0, -1):
            st, j = back[i - 1][st]
            out.append(cands[i - 1][j][0])
        return out[::-1]

    def alternatives_line(self, words: Sequence[str]) -> List[List[Tuple[str, float]]]:
        cands = [self.restorer.alternatives(w) for w in words]
        m = self.lm.n - 1
        n = len(words)
        start = (WordLM.BOS,) * m
        nxt = lambda st, f: (st + (f,))[len(st) + 1 - m:] if m else ()
        score = lambda st, f, lp: self.alpha * lp + self.lm.logp(st, f)
        F = [{start: 0.}]
        for cs in cands:
            cur = defaultdict(list)
            for st, v in F[-1].items():
                for f, lp in cs:
                    cur[nxt(st, f)].append(v + score(st, f, lp))
            F.append({st: _lse(vs) for st, vs in cur.items()})
        B = [None] * (n + 1)
        B[n] = {st: self.lm.logp(st, WordLM.EOS) for st in F[n]}
        for i in range(n - 1, -1, -1):
            B[i] = {st: _lse([score(st, f, lp) + B[i + 1][nxt(st, f)] for f, lp in cands[i]]) for st in F[i]}
        z = _lse([F[n][st] + B[n][st] for st in F[n]])
        out = []
        for i, cs in enumerate(cands):
            marg = [(f, _lse([F[i][st] + score(st, f, lp) + B[i + 1][nxt(st, f)] for st in F[i]]) - z) for f, lp in cs]
            out.append(sorted(marg, key=lambda x: -x[1]))
        return out
