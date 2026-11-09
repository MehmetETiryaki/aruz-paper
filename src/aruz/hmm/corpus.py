

import logging
from pathlib import Path
from typing import Counter, Iterable, Iterator, List, NewType, Optional, Tuple
from more_itertools import peekable
from .integerize import Integerizer

log = logging.getLogger(Path(__file__).stem)

Word = NewType('Word', str)
Tag = NewType('Tag', str)
TWord = Tuple[Word, Optional[Tag]]

IntegerizedTWord = Tuple[int, Optional[int]]
IntegerizedSentence = List[IntegerizedTWord]

OOV_WORD: Word = Word("_OOV_")
BOS_WORD: Word = Word("_BOS_WORD_")
EOS_WORD: Word = Word("_EOS_WORD_")
BOS_TAG: Tag = Tag("_BOS_TAG_")
EOS_TAG: Tag = Tag("_EOS_TAG_")

import random
random.seed(1234)

class Sentence(List[TWord]):
    def __init__(self, sentence: Optional[List[TWord]] = None):
        if sentence:
            super().__init__(tword for tword in sentence)
        else:
            super().__init__()

    def __str__(self) -> str:
        return " ".join([word if tag is None else f"{word}/{tag}" for (word, tag) in self[1:-1]])

    def desupervise(self):
        sentence = Sentence()
        [sentence.append(
            (word, tag if tag == BOS_TAG or tag == EOS_TAG else None)) for word, tag in self]
        return sentence

    def is_supervised(self) -> bool:
        return all(tag is not None for _, tag in self)

class TaggedCorpus:

    def __init__(self, *files: Path,
                 tagset: Optional[Integerizer[Tag]] = None, 
                 vocab: Optional[Integerizer[Word]] = None,
                 vocab_threshold: int = 1, 
                 add_oov: bool = True):

        super().__init__()
        self.files = files

        if tagset is None or vocab is None:
            self.tagset: Integerizer[Tag] = Integerizer()
            self.vocab: Integerizer[Word] = Integerizer()
        
            word_counts: Counter[Word] = Counter()
            for word, tag in self.get_tokens(oovs=False):
                if word == EOS_WORD:
                    continue
                word_counts[word] += 1
                if tag is not None:
                    self.tagset.add(tag)
            log.info(f"Read {sum(word_counts.values())} tokens from {', '.join(file.name for file in files)}")

            for word, count in word_counts.items():
                if count >= vocab_threshold:
                    self.vocab.add(word)
                else:
                    self.vocab.add(OOV_WORD)

            if add_oov:
                self.vocab.add(OOV_WORD)
            self.tagset.add(EOS_TAG)
            self.tagset.add(BOS_TAG)
            self.vocab.add(EOS_WORD)
            self.vocab.add(BOS_WORD)

        if tagset is None:
            log.info(f"Created {len(self.tagset)} tag types")
        else:
            self.tagset = tagset

        if vocab is None:
            log.info(f"Created {len(self.vocab)} word types")
        else:
            self.vocab = vocab

        self.oov_w = self.vocab.index(OOV_WORD)

        if tagset is not None or vocab is not None:
            for word, tag in self.get_tokens(oovs=False):
                self.integerize_word(word)
                if tag is not None:
                    self.integerize_tag(tag)

    def __str__(self) -> str:
        return "\n".join(str(sentence) for sentence in self)

    def __iter__(self) -> Iterator[Sentence]:
        return iter(self.get_sentences())

    def __len__(self) -> int:
        self._num_sentences: int
        try:
            return self._num_sentences
        except AttributeError:
            self._num_sentences = sum(1 for _ in self)
            return self._num_sentences

    def num_tokens(self) -> int: 
        self._num_tokens: int
        try:
            return self._num_tokens
        except AttributeError:
            self._num_tokens = sum(1 for _ in self.get_tokens())
            return self._num_tokens

    def get_tokens(self, oovs: bool = True) -> Iterable[TWord]:
        for file in self.files:
            with open(file) as f:
                for line in f:
                    for token in line.split():
                        word: Word
                        tag: Optional[Tag]
                        if "/" in token:
                            w, t = token.rsplit("/", 1)
                            word, tag = Word(w), Tag(t)
                        else:
                            word, tag = Word(token), None
                        if (not oovs) or word in self.vocab:
                            yield word, tag
                        else:
                            yield OOV_WORD, tag
                    yield EOS_WORD, EOS_TAG

    def get_sentences(self) -> Iterable[Sentence]:

        sentence = Sentence([(BOS_WORD, BOS_TAG)])
        for word, tag in self.get_tokens():
            sentence.append((word, tag))
            if word == EOS_WORD:
                yield sentence
                sentence = Sentence([(BOS_WORD, BOS_TAG)])

    def draw_sentences_forever(self, randomize: bool = True) -> Iterable[Sentence]:
        sentences = peekable(self.get_sentences())
        assert sentences
        if not randomize:
            import itertools
            return itertools.cycle(sentences)
        else:
            pool = tuple(sentences)
            while True:
                for sentence in random.sample(pool, len(pool)):
                    yield sentence

    def integerize_tag(self, tag: Tag) -> int:
        t = self.tagset.index(tag)
        if t is None:
            raise KeyError(tag, self, "This tag is not in the tagset of this corpus, and we don't support OOV tags")
        return t

    def integerize_word(self, word: Word) -> int:
        w = self.vocab.index(word)
        if w is None: 
            w = self.oov_w
            if w is None:
                raise KeyError(word, self, "This word is not in the vocab of this corpus, and can't be converted to OOV since OOV is also missing from the vocab")
        return w

    def integerize_tword(self, tword: TWord) -> IntegerizedTWord:
        word, tag = tword
        return self.integerize_word(word), (None if tag is None else self.integerize_tag(tag))

    def integerize_sentence(self, sentence: Sentence) -> IntegerizedSentence:
        return [self.integerize_tword(tword) for tword in sentence]
