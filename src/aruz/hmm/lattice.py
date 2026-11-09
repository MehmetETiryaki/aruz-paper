

from __future__ import annotations
from dataclasses import dataclass, field
from math import exp, inf
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor

from .corpus import Tag, Word
from .hmm import HiddenMarkovModel
from .integerize import Integerizer

@dataclass
class Arc:
    src: int
    dst: int
    w: int
    logp: float = 0.
    info: Any = None

@dataclass
class Lattice:
    arcs: List[Arc] = field(default_factory=list)
    n: int = 1
    final: int = 0
    starts: Optional[List[int]] = None

    def node(self) -> int:
        self.n += 1
        return self.n - 1

    def add(self, src: int, dst: int, w: int, logp: float = 0., info: Any = None) -> Arc:
        if not src < dst:
            raise ValueError(f"arc {src}->{dst} breaks the topological order")
        arc = Arc(src, dst, w, logp, info)
        self.arcs.append(arc)
        return arc

    def into(self) -> List[List[Arc]]:
        into: List[List[Arc]] = [[] for _ in range(self.n)]
        for a in self.arcs:
            into[a.dst].append(a)
        return into

    def out(self) -> List[List[Arc]]:
        out: List[List[Arc]] = [[] for _ in range(self.n)]
        for a in self.arcs:
            out[a.src].append(a)
        return out

class LatticeCorpus(list):
    def __init__(self, lattices: List[Lattice], tagset: Integerizer[Tag], vocab: Integerizer[Word]):
        super().__init__(lattices)
        self.tagset = tagset
        self.vocab = vocab

class LatticeHMM(HiddenMarkovModel):

    def _integerize_sentence(self, sentence, corpus):
        if isinstance(sentence, Lattice):
            if corpus.tagset != self.tagset or corpus.vocab != self.vocab:
                raise TypeError("The corpus that this lattice came from uses a different tagset or vocab")
            return sentence
        return super()._integerize_sentence(sentence, corpus)

    def logprob(self, sentence, corpus):
        if isinstance(sentence, Lattice):
            return self.lattice_forward(sentence)
        return super().logprob(sentence, corpus)

    def E_step(self, isent, mult: float = 1) -> None:
        if not isinstance(isent, Lattice):
            return super().E_step(isent, mult)
        log_Z_forward = self.lattice_forward(isent)
        if log_Z_forward == -inf:
            raise ValueError("E_step on a lattice with probability 0 under the current model; "
                             "its expected counts are undefined (0/0)")
        log_Z_backward = self.lattice_backward(isent, mult=mult)
        assert torch.isclose(log_Z_forward, log_Z_backward), f"backward log-probability {log_Z_backward} doesn't match forward log-probability {log_Z_forward}!"

    def _A(self, lat: Lattice) -> Tensor:
        if lat.starts is None:
            return self.A
        A = self.A.clone()
        keep = torch.zeros(self.k, dtype=torch.bool)
        keep[lat.starts] = True
        A[self.bos_t, ~keep] = 0
        return A

    def lattice_forward(self, lat: Lattice) -> Tensor:
        alpha: List[Tensor] = [torch.zeros(self.k) for _ in range(lat.n)]
        k_list: List[float] = [-inf] * lat.n
        alpha[0], k_list[0] = self.eye[self.bos_t], 0.
        A = self._A(lat)
        into = lat.into()

        for v in range(1, lat.n):
            arcs = [a for a in into[v] if k_list[a.src] > -inf]
            if not arcs:
                continue
            base = max(k_list[a.src] + a.logp for a in arcs)
            for a in arcs:
                alpha[v] += exp(k_list[a.src] + a.logp - base) * (alpha[a.src] @ A) * self.B[:, a.w]
            k = torch.sum(alpha[v])
            if k > 0:
                alpha[v] /= k
                k_list[v] = base + torch.log(k).item()

        self.alphas, self.k_alphas = alpha, k_list
        end = (alpha[lat.final] @ A)[self.eos_t]
        if k_list[lat.final] == -inf or end == 0:
            self.Z = torch.tensor(-inf)
            return self.Z
        self.Z = torch.log(end) + k_list[lat.final]
        return self.Z

    def lattice_backward(self, lat: Lattice, mult: float = 1) -> Tensor:
        beta: List[Tensor] = [torch.zeros(self.k) for _ in range(lat.n)]
        total_log_k: List[float] = [-inf] * lat.n
        A = self._A(lat)
        beta[lat.final], total_log_k[lat.final] = A[:, self.eos_t].clone(), 0.
        out = lat.out()

        for u in range(lat.final - 1, -1, -1):
            arcs = [a for a in out[u] if total_log_k[a.dst] > -inf]
            if not arcs:
                continue
            base = max(total_log_k[a.dst] + a.logp for a in arcs)
            for a in arcs:
                beta[u] += exp(total_log_k[a.dst] + a.logp - base) * (A @ (self.B[:, a.w] * beta[a.dst]))
            if u > 0:
                beta[u][self.bos_t] = 0
            beta[u][self.eos_t] = 0
            k = torch.sum(beta[u])
            if k > 0:
                beta[u] /= k
                total_log_k[u] = base + torch.log(k).item()

        self.arc_posteriors: Dict[int, float] = {}
        for i, a in enumerate(lat.arcs):
            if self.k_alphas[a.src] == -inf or total_log_k[a.dst] == -inf:
                continue
            scale = exp(self.k_alphas[a.src] + a.logp + total_log_k[a.dst] - self.Z.item())
            p = torch.outer(self.alphas[a.src], self.B[:, a.w] * beta[a.dst]) * A * scale
            self.A_counts += p * mult
            self.B_counts[:, a.w] += p.sum(dim=0) * mult
            self.arc_posteriors[i] = p.sum().item()
        self.A_counts[:, self.eos_t] += self.alphas[lat.final] * A[:, self.eos_t] \
            * exp(self.k_alphas[lat.final] - self.Z.item()) * mult

        log_Z_backward = torch.log(beta[0][self.bos_t]) + total_log_k[0]
        self.betas = beta
        return log_Z_backward

    def arc_marginals(self, lat: Lattice) -> Dict[int, float]:
        counts = (self.A_counts, self.B_counts) if hasattr(self, "A_counts") else None
        self._zero_counts()
        self.E_step(lat, mult=0)
        if counts:
            self.A_counts, self.B_counts = counts
        return self.arc_posteriors

    def lattice_viterbi(self, lat: Lattice) -> Tuple[float, List[Tuple[Arc, int]]]:
        lA, lB = self._A(lat).log(), self.B.log()
        delta: List[Tensor] = [torch.full((self.k,), -inf) for _ in range(lat.n)]
        backpointers: List[List[Optional[Tuple[Arc, int]]]] = [[None] * self.k for _ in range(lat.n)]
        delta[0][self.bos_t] = 0
        into = lat.into()

        for v in range(1, lat.n):
            for a in into[v]:
                road_probs = delta[a.src].unsqueeze(1) + lA
                max_road_probs, max_road_idx = road_probs.max(dim=0)
                score = max_road_probs + lB[:, a.w] + a.logp
                for t in torch.nonzero(score > delta[v]).flatten().tolist():
                    delta[v][t] = score[t]
                    backpointers[v][t] = (a, int(max_road_idx[t]))

        end = delta[lat.final] + lA[:, self.eos_t]
        best, t = end.max(dim=0)
        if best == -inf:
            raise ValueError("No path through this lattice has nonzero probability under this model")
        path: List[Tuple[Arc, int]] = []
        v, t = lat.final, int(t)
        while v != 0:
            a, s = backpointers[v][t]
            path.append((a, t))
            v, t = a.src, s
        return best.item(), path[::-1]
