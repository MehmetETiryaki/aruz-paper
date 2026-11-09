

from __future__ import annotations
import logging
from math import inf, log, exp
from pathlib import Path
import os, time
from typing import Callable, List, Optional, cast
from typeguard import typechecked

import torch
from torch import Tensor, cuda, nn
from jaxtyping import Float

from tqdm import tqdm
import pickle

from .integerize import Integerizer
from .corpus import BOS_TAG, BOS_WORD, EOS_TAG, EOS_WORD, Sentence, Tag, TaggedCorpus, IntegerizedSentence, Word

TorchScalar = Float[Tensor, ""]

logger = logging.getLogger(Path(__file__).stem)

torch.manual_seed(1337)
cuda.manual_seed(69_420)

class HiddenMarkovModel:
    

    def __init__(self, 
                 tagset: Integerizer[Tag],
                 vocab: Integerizer[Word],
                 unigram: bool = False,
                 A_mask: Optional[Tensor] = None,
                 B_mask: Optional[Tensor] = None,
                 B_tie: Optional[Tensor] = None):

        if vocab[-2:] != [EOS_WORD, BOS_WORD]:
            raise ValueError("final two types of vocab should be EOS_WORD, BOS_WORD")

        self.k = len(tagset)
        self.V = len(vocab) - 2
        self.unigram = unigram

        self.tagset = tagset
        self.vocab = vocab

        self.bos_t: Optional[int] = tagset.index(BOS_TAG)
        self.eos_t: Optional[int] = tagset.index(EOS_TAG)
        if self.bos_t is None or self.eos_t is None:
            raise ValueError("tagset should contain both BOS_TAG and EOS_TAG")
        assert self.eos_t is not None
        self.eye: Tensor = torch.eye(self.k)

        if unigram and A_mask is not None:
            raise ValueError("A_mask is not supported for a unigram HMM")
        self.A_mask = torch.ones(self.k, self.k, dtype=torch.bool) if A_mask is None else A_mask.clone()
        self.A_mask[:, self.bos_t] = False
        self.A_mask[self.eos_t, :] = False
        self.B_mask = torch.ones(self.k, self.V, dtype=torch.bool) if B_mask is None else B_mask.clone()
        self.B_mask[[self.eos_t, self.bos_t], :] = False
        self.B_tie = B_tie
        if B_tie is not None:
            if B_tie.shape != (self.k,):
                raise ValueError(f"B_tie should have shape ({self.k},), got {tuple(B_tie.shape)}")
            if not torch.equal(self._tie(self.B_mask.float()) > 0, self.B_mask):
                raise ValueError("tied tags must have identical B_mask rows")

        self.init_params()
 
    def init_params(self) -> None:

        WB = 0.01*torch.rand(self.k, self.V)
        WB[~self.B_mask] = -inf
        self.B = WB.softmax(dim=1).nan_to_num(0)
        if self.B_tie is not None:
            self.B = self._tie(self.B)
            self.B = (self.B / self.B.sum(dim=1, keepdim=True)).nan_to_num(0)
        self.B[self.eos_t, :] = 0
        self.B[self.bos_t, :] = 0
        
        rows = 1 if self.unigram else self.k
        WA = 0.01*torch.rand(rows, self.k)
        WA[:, self.bos_t] = -inf
        if not self.unigram:
            WA[~self.A_mask] = -inf
        self.A = WA.softmax(dim=1).nan_to_num(0)
        if self.unigram:
            self.A = self.A.repeat(self.k, 1)
        self.A[self.eos_t, :] = 0

    def printAB(self) -> None:
        print("Transition matrix A:")
        col_headers = [""] + [str(self.tagset[t]) for t in range(self.A.size(1))]
        print("\t".join(col_headers))
        for s in range(self.A.size(0)):
            row = [str(self.tagset[s])] + [f"{self.A[s,t]:.3f}" for t in range(self.A.size(1))]
            print("\t".join(row))
        print("\nEmission matrix B:")        
        col_headers = [""] + [str(self.vocab[w]) for w in range(self.B.size(1))]
        print("\t".join(col_headers))
        for t in range(self.A.size(0)):
            row = [str(self.tagset[t])] + [f"{self.B[t,w]:.3f}" for w in range(self.B.size(1))]
            print("\t".join(row))
        print("\n")

    def M_step(self, λ: float) -> None:

        assert not self.B_counts[[self.eos_t, self.bos_t], :].any(), 'Your expected emission counts ' \
                'from EOS and BOS are not all zero, meaning you\'ve accumulated them incorrectly!'

        assert not self.B_counts[~self.B_mask].any(), 'expected emission counts in masked cells'
        assert not self.A_counts[~self.A_mask].any(), 'expected transition counts in masked cells'

        if self.B_tie is not None:
            self.B_counts = self._tie(self.B_counts)
        self.B_counts += λ
        self.B_counts[~self.B_mask] = 0
        self.B = (self.B_counts / self.B_counts.sum(dim=1, keepdim=True)).nan_to_num(0)
        self.B[self.eos_t, :] = 0
        self.B[self.bos_t, :] = 0

        assert self.A_counts[:, self.bos_t].any() == 0, 'Your expected transition counts ' \
                'to BOS are not all zero, meaning you\'ve accumulated them incorrectly!'
        assert self.A_counts[self.eos_t, :].any() == 0, 'Your expected transition counts ' \
                'from EOS are not all zero, meaning you\'ve accumulated them incorrectly!'
                

        if self.unigram:
            self.A_counts = self.A_counts.sum(dim=0, keepdim=True)
            self.A_counts += λ
            self.A_counts[:, self.bos_t] = 0
            self.A = self.A_counts / self.A_counts.sum(dim=1, keepdim=True)
            self.A = self.A.repeat(self.k, 1)
        else:
            self.A_counts += λ
            self.A_counts[~self.A_mask] = 0
            self.A = (self.A_counts / self.A_counts.sum(dim=1, keepdim=True)).nan_to_num(0)
        self.A[self.eos_t, :] = 0

    def _tie(self, M: Tensor) -> Tensor:
        groups = torch.zeros(int(self.B_tie.max()) + 1, M.size(1)).index_add_(0, self.B_tie, M)
        return groups[self.B_tie]

    def _zero_counts(self):
        self.A_counts = torch.zeros((self.k, self.k), requires_grad=False)
        self.B_counts = torch.zeros((self.k, self.V), requires_grad=False)

    def train(self,
              corpus: TaggedCorpus,
              loss: Callable[[HiddenMarkovModel], float],
              λ: float = 0,
              tolerance: float = 0.001,
              max_steps: int = 50000,
              save_path: Optional[Path|str] = "my_hmm.pkl") -> None:
        
        if λ < 0:
            raise ValueError(f"{λ=} but should be >= 0")
        elif λ == 0:
            λ = 1e-20
      
        dev_loss = loss(self)
        
        old_dev_loss: float = dev_loss
        steps: int = 0
        while steps < max_steps:

            

            self._zero_counts()
            for sentence in tqdm(corpus, total=len(corpus), leave=True):
                isent = self._integerize_sentence(sentence, corpus)
                self.E_step(isent)
                steps += 1

            self.M_step(λ)
            if save_path: self.save(save_path, temp=steps)
            
            dev_loss = loss(self)
            if dev_loss >= old_dev_loss * (1-tolerance):
                break
            old_dev_loss = dev_loss
        
        if save_path: self.save(save_path)
  
    def _integerize_sentence(self, sentence: Sentence, corpus: TaggedCorpus) -> IntegerizedSentence:

        if corpus.tagset != self.tagset or corpus.vocab != self.vocab:
            raise TypeError("The corpus that this sentence came from uses a different tagset or vocab")

        return corpus.integerize_sentence(sentence)

    @typechecked
    def logprob(self, sentence: Sentence, corpus: TaggedCorpus) -> TorchScalar:

        isent = self._integerize_sentence(sentence, corpus)
        return self.forward_pass(isent)

    def E_step(self, isent: IntegerizedSentence, mult: float = 1) -> None:

        log_Z_forward = self.forward_pass(isent)
        if log_Z_forward == -inf:
            raise ValueError("E_step on a sentence with probability 0 under the current model; "
                             "its expected counts are undefined (0/0)")
        log_Z_backward = self.backward_pass(isent, mult=mult)

        assert torch.isclose(log_Z_forward, log_Z_backward), f"backward log-probability {log_Z_backward} doesn't match forward log-probability {log_Z_forward}!"

    @typechecked
    def forward_pass(self, isent: IntegerizedSentence) -> TorchScalar:
        
        alpha = [torch.empty(self.k) for _ in isent]    
        alpha[0] = self.eye[self.bos_t]
        k_list = [torch.tensor(0.0)]

        
        for j in range(1, len(isent)):
            word_index = isent[j][0]
            if isent[j][1] is None:
                if j == len(isent) - 1:
                    alpha[j] = (alpha[j-1] @ self.A)
                else:
                    alpha[j] = (alpha[j-1] @ self.A) * self.B[:, word_index]
            else:
                tag_index = isent[j][1]
                if j == len(isent) - 1:
                    alpha[j] = ((alpha[j-1] @ self.A)) * self.eye[tag_index]
                else:
                    alpha[j] = ((alpha[j-1] @ self.A) * self.B[:, word_index]) * self.eye[tag_index]

            k = torch.sum(alpha[j])
            if k == 0:
                self.alphas, self.k_alphas = None, None
                self.Z = torch.tensor(-inf)
                return self.Z
            alpha[j] /= k
            k_list.append(k_list[-1] + torch.log(k))

        log_Z = torch.log(alpha[-1][self.eos_t]) + k_list[-1]

        self.alphas = alpha
        self.Z = log_Z
        self.k_alphas = k_list

        return log_Z

    @typechecked
    def backward_pass(self, isent: IntegerizedSentence, mult: float = 1) -> TorchScalar:

        beta = [torch.empty(self.k) for _ in isent]
        beta[-1] = self.eye[self.eos_t]
        total_log_k = 0

        for j in range(len(isent)-1, 0, -1):
            cur_word_index = isent[j][0]
            if j == len(isent)-1:
                p = self.A
            else:
                p = (self.A * self.B[:, cur_word_index])
            beta[j-1] = p @ beta[j]
            if isent[j-1][1] is not None:
                beta[j-1] = beta[j-1] * self.eye[isent[j-1][1]]
            if j > 1:
                beta[j-1][self.bos_t] = 0
            beta[j-1][self.eos_t] = 0

            if j == len(isent) - 1:
                self.A_counts += torch.exp(torch.log(((torch.outer(self.alphas[j-1], beta[j]))) * p) - self.Z + total_log_k + self.k_alphas[j-1]) * mult
            else:
                self.A_counts += torch.exp(torch.log(((torch.outer(self.alphas[j-1], beta[j]))) * p) - self.Z + total_log_k + self.k_alphas[j-1]) * mult
                self.B_counts[:, cur_word_index] += torch.exp(torch.log((self.alphas[j] * beta[j])) - self.Z + total_log_k + self.k_alphas[j]) * mult

            k = torch.sum(beta[j-1])
            beta[j-1] /= k
            total_log_k += torch.log(k)

        log_Z_backward = torch.log(beta[0][self.bos_t]) + total_log_k

        self.betas = beta
        self.log_Z_backward = log_Z_backward

        return log_Z_backward

    def viterbi_tagging(self, sentence: Sentence, corpus: TaggedCorpus) -> Sentence:

        isent = self._integerize_sentence(sentence, corpus)

        alpha        = [torch.empty(self.k)                  for _ in isent]  
        backpointers = [torch.empty(self.k, dtype=torch.int) for _ in isent]
        tags: List[int]
        tags = [0] * len(isent)
        alpha[0] = self.eye[self.bos_t]

        for j in range(1, len(isent)):
            word_index, tag_index = isent[j]
            road_probs = alpha[j-1].unsqueeze(1) * self.A
            max_road_probs, max_road_idx = road_probs.max(dim=0)
            backpointers[j] = max_road_idx
            if j == len(isent) - 1:
                alpha[j] = max_road_probs
            else:
                alpha[j] = max_road_probs * self.B[:, word_index]
            if tag_index is not None:
                alpha[j] = alpha[j] * self.eye[tag_index]

            k = torch.sum(alpha[j])
            if k == 0:
                raise ValueError(f"No tagging has nonzero probability under this model: {sentence}")
            alpha[j] /= k

        tags[-1] = self.eos_t
        for idx in range(len(isent) - 2, -1, -1):
            tags[idx] = int(backpointers[idx + 1][tags[idx + 1]])
        assert tags[0] == self.bos_t, "Viterbi backtrace did not end at BOS_TAG"

        return Sentence([(word, self.tagset[tags[j]]) for j, (word, tag) in enumerate(sentence)])

    def save(self, path: Path|str, temp=None, interval: int = 300) -> None:

        if isinstance(path, str): path = Path(path)

        now = time.time()
        if temp is not None:
            path = path.with_name(f"{path.stem}-{temp}{path.suffix}")
            try: 
                if now < self._save_time + interval: 
                    return
            except AttributeError: pass

        try: old_temp_path, old_save_time = self._temp_path, self._save_time
        except AttributeError: old_temp_path = None
        
        self._temp_path, self._save_time = path, now
        if temp is None: del self._temp_path
        
        try:
            torch.save(self, path, pickle_protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Saved model to {path}")
        except Exception as e:   
            self._temp_path, self._save_time = old_temp_path, old_save_time
            raise e
        
        if old_temp_path: 
            try: os.remove(old_temp_path)
            except FileNotFoundError: pass

    @classmethod
    def load(cls, path: Path|str, device: str = 'cpu') -> HiddenMarkovModel:
        if isinstance(path, str): path = Path(path)
            
        model = torch.load(path, map_location=device, weights_only=False)

        if model.__class__ != cls:
            raise ValueError(f"Type Error: expected object of type {cls.__name__} but got {model.__class__.__name__} " \
                             f"from saved file {path}.")

        logger.info(f"Loaded model from {path}")
        return model
