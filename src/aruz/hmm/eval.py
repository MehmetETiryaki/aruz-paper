
import logging
from pathlib import Path
from math import nan, exp
from typing import Counter, Tuple, Optional, Callable, Union

import torch
from torch import nn as nn
from tqdm import tqdm

from .corpus import Sentence, Word, EOS_WORD, BOS_WORD, OOV_WORD, TaggedCorpus
from .hmm import HiddenMarkovModel
from .integerize import Integerizer

log = logging.getLogger(Path(__file__).stem)

def viterbi_tagger(model: HiddenMarkovModel, eval_corpus: TaggedCorpus) -> Callable[[Sentence], Sentence]:
    def tagger(input:Sentence) -> Sentence:
        return model.viterbi_tagging(input, eval_corpus)
    return tagger

def model_cross_entropy(model: HiddenMarkovModel,
                        eval_corpus: TaggedCorpus) -> float:
    logprob = 0.0
    token_count = 0
    for gold in tqdm(eval_corpus, total=len(eval_corpus)):
        logprob += model.logprob(gold, eval_corpus).item()
        token_count += len(gold) - 1
    cross_entropy = -logprob / token_count
    log.info(f"Cross-entropy: {cross_entropy:.4f} nats (= perplexity {exp(cross_entropy):.3f})")
    return cross_entropy

def viterbi_error_rate(model: HiddenMarkovModel,
                     eval_corpus: TaggedCorpus,
                     known_vocab: Optional[Integerizer[Word]] = None,
                     show_cross_entropy = True) -> float:

    if show_cross_entropy:
        model_cross_entropy(model, eval_corpus)
    return tagger_error_rate(viterbi_tagger(model, eval_corpus),
                             eval_corpus,
                             known_vocab=known_vocab)

def tagger_error_rate(tagger: Callable[[Sentence], Sentence],
                     eval_corpus: TaggedCorpus,
                     known_vocab: Optional[Integerizer[Word]] = None) -> float:

    counts: Counter[Tuple[str, str]] = Counter()
    for gold in tqdm(eval_corpus, total=len(eval_corpus)):
        predicted = tagger(gold.desupervise())
        counts += eval_tagging(predicted, gold, known_vocab)

    def fraction(c:str) -> float:
        num = counts['NUM',c]
        denom = counts['DENOM',c]
        return nan if denom==0 else num / denom

    categories = ['ALL', 'KNOWN', 'SEEN', 'NOVEL']
    if known_vocab is None:
        categories.remove('KNOWN')
    results = [f"{c.lower()}: {(fraction(c)):.3%}" for c in categories]            
    log.info(f"Tagging accuracy: {', '.join(results)}")

    return 1 - fraction('ALL')

def eval_tagging(predicted: Sentence, 
                 gold: Sentence, 
                 known_vocab: Optional[Integerizer[Word]]) -> Counter[Tuple[str, str]]:

    counts: Counter[Tuple[str, str]] = Counter()
    for ((word, tag), (goldword, goldtag)) in zip(predicted, gold):
        assert word == goldword or word == OOV_WORD
        if word == BOS_WORD or word == EOS_WORD:
            continue
        if goldtag is None:
            continue
        
        if word == OOV_WORD:                      category = 'NOVEL'
        elif known_vocab and word in known_vocab: category = 'KNOWN'
        else:                                     category = 'SEEN'    

        for c in (category, 'ALL'):
            counts['DENOM', c] += 1
            if tag == goldtag:
                counts['NUM', c] += 1

    return counts

def write_tagging(model_or_tagger: Union[HiddenMarkovModel, Callable[[Sentence], Sentence]],
                        eval_corpus: TaggedCorpus,
                        output_path: Path) -> None:
    if isinstance(model_or_tagger, HiddenMarkovModel):
        tagger = viterbi_tagger(model_or_tagger, eval_corpus)
    else:
        tagger = model_or_tagger
    with open(output_path, 'w') as f:
        for gold in tqdm(eval_corpus, total=len(eval_corpus)):
            predicted = tagger(gold.desupervise())
            f.write(str(predicted)+"\n")
