from __future__ import annotations
from typing import List, Optional, Sequence, Tuple

from .scan import LONG_VOWELS, STOPS, VOWELS, syllabify

LENGTHEN = {"a": "ā", "e": "ē", "i": "ī", "o": "ō", "u": "ū", "ı": "ı̄", "ö": "ȫ", "ü": "ǖ"}
SHORTEN = {"ā": "a", "ē": "e", "ī": "i", "ō": "o", "ū": "u", "â": "a", "î": "i", "û": "u"}
Mark = Tuple[bool, bool, int]
Syllable = Tuple[str, str, Sequence[str]]
KUSUR = ("imâle", "zihaf", "no medd", "no ulama")
ANNOTATED = {"none": (), "free": (), "ulama": (), "medd": (), "imâle": ("imâle",), "ulama+imâle": ("imâle",),
             "shortening": ("zihaf",), "no medd": ("no medd",), "no ulama": ("no ulama",), "no ulama+medd": ("no ulama",),
             "no ulama+no medd": ("no medd", "no ulama")}

def from_annotation(syllables) -> List[list]:
    return [[text, slots, list(ANNOTATED[label])] for text, slots, label in syllables]

def _in_stream(c: str) -> bool:
    import unicodedata
    return c in "ʿʾʼ'’" or (c.isalpha() and unicodedata.category(c) != "Lm")

def normalized(syllables: Sequence[Syllable], pattern: str) -> List[Syllable]:
    out, k = [], 0
    for text, slots, kusur in syllables:
        out.append((text, "x" if pattern[k:k + 1] == "x" and slots != "-." else slots, kusur))
        k += len(slots)
    return out

def _letters(s: str) -> int:
    return sum(1 for c in s if _in_stream(c) and c not in STOPS)

def linked(forms: Sequence[str], syllables: Sequence[Syllable]) -> List[int]:
    out: List[int] = []
    read = spelled = 0
    for wi, form in enumerate(forms):
        ss = syllabify(form)
        read += sum(_letters(s[0]) for s in syllables[len(out):len(out) + len(ss)])
        spelled += _letters(form)
        out += [0] * (len(ss) - 1) + [max(0, spelled - read) if wi < len(forms) - 1 else 0]
    return out

def render(forms: Sequence[str], syllables: Sequence[Syllable]) -> Optional[Tuple[str, List[Mark]]]:
    try:
        per_word = [syllabify(f) for f in forms]
    except ValueError:
        return None
    if sum(map(len, per_word)) != len(syllables):
        return None
    moved = linked(forms, syllables)
    out_words, marks, k = [], [], 0
    for wi, (form, ss) in enumerate(zip(forms, per_word)):
        last_word = wi == len(forms) - 1
        decisions = []
        for j, (text, long_vowel, coda) in enumerate(ss):
            slots = syllables[k + j][1]
            final = j == len(ss) - 1
            ulama = moved[k + j]
            free = slots == "x" or (last_word and final)
            if free:
                held = long_vowel
            elif slots.startswith("-"):
                spoken = syllables[k + j][0]
                v = next((p for p, c in enumerate(spoken) if c in VOWELS), -1)
                held = long_vowel or not any(c.isalpha() and c not in STOPS for c in spoken[v + 1:])
            else:
                held = False
            medd = slots == "-."
            decisions.append((held, medd, ulama))
        bounds, pos = [], 0
        for text, _, _ in ss:
            pos += sum(1 for c in text if _in_stream(c))
            bounds.append(pos)
        out, n, j, vowel_done = [], 0, 0, False
        for c in form:
            if not _in_stream(c):
                out.append(c)
                continue
            if c in VOWELS and not vowel_done:
                held = decisions[j][0]
                if c in LONG_VOWELS and not held:
                    c = SHORTEN.get(c, c)
                elif c not in LONG_VOWELS and held:
                    c = LENGTHEN.get(c, c)
                vowel_done = True
            out.append(c)
            n += 1
            if j < len(bounds) and n == bounds[j]:
                if decisions[j][1]:
                    out.append("ˑ")
                j, vowel_done = j + 1, False
        out_words.append("".join(out))
        marks += decisions
        k += len(ss)
    text = ""
    out_words = [w.replace("ñ", "n").translate(str.maketrans("", "", STOPS)) for w in out_words]
    for wi, w in enumerate(out_words):
        text += w
        if wi < len(out_words) - 1:
            text += "‿" if marks[sum(map(len, per_word[:wi + 1])) - 1][2] else " "
    return text, marks
