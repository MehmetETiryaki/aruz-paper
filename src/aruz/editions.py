from __future__ import annotations
import re
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .meter import METERS, kinds
from .scan import VOWELS, syllabify

TEFILE = {
    "fAilAtün": "-.--", "feilAtün": "..--", "fAilAtü": "-.-.", "feilAtü": "..-.",
    "fAilün": "-.-", "feilün": "..-", "falün": "--", "fAilAn": "-.-", "feilAn": "..-", "falAn": "--",
    "mefAIlün": ".---", "mefAilün": ".-.-", "mefAIlü": ".--.", "mefAilü": ".-..", "mefAIlAn": ".---",
    "mefUlü": "--.", "mefUlün": "---", "mefUlAtü": "---.",
    "feUlün": ".--", "feUlü": ".-.", "feUl": ".-", "fal": "-",
    "faUlün": ".--", "faUlü": ".-.", "faUl": ".-",
    "fa": "-", "fA": "-", "feal": ".-",
    "müstefilün": "--.-", "müfteilün": "-..-", "mütefAilün": "..-.-", "müfAaletün": ".-..-",
}
LONG = {"̂": True, "̄": True}
ARABIC = re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")
NUMBER = re.compile(r"^[-–\[]?(\d+)(?:[.*]+|=\d+)?[-–\]]?\s*(\d*)\s*(\(.*\))?$")
TITLED = re.compile(r"^(\d+)\.?\s*([^\W\d_].*)$")
SIGLUM = r"[A-ZÇĞİÖŞÜ]{1,3}\d?"
APPARATUS = re.compile(r"^\d+[a-f]-|^\d+[a-z]$|\|\||^\(?([A-ZÜ]\d?:\s*\d+[ab],?\s*)+\)?$"
                       rf"|^{SIGLUM}\s+\d+[ab]$|^\[{SIGLUM}-\d+[ab]\]$"
                       rf"|^\d+[ab]\s.*:\s|^\d+:\s*{SIGLUM}\s+\d+[ab]"
                       rf"|:.*\b{SIGLUM}[.,;]|^(\S+\s)?({SIGLUM}[.,;]\s?)+$"
                       rf"|[:\]].*\b{SIGLUM}\.?$"
                       rf"|^({SIGLUM}\s+\d+[ab],?\s*)+$|^\*\s*\d+\s+{SIGLUM}\s"
                       r"|Derkenar|^Sene\s+\d+$|(,\s*\d+){2,}$|^_+[;.]?$"
                       r"|^\[?[IVXLC]+\]?\.?$|^[A-Z]{1,3}$"
                       r"|(\b\d+[ab]\b.*){2,}|^\([a-z]\)$")
LABEL = re.compile(r"^\w+:\s*")
HEADING = re.compile(r"^[ḤḪHḥḫh]arf|^(ahreb|ahrem)\d*$|^\[.*\]$|^\([^()]*\)$"
                     r"|^(recez|remel|hezec|müctes|muzāriʿ|muzari|ḫafīf|hafif|serīʿ|münseriḥ|mütekārib|kāmil|vāfir|"
                     r"basīṭ|medīd)\s+[–\-+/ ]+", re.I)
VELEHU = re.compile(r"^ve\s?lehū?\b", re.I)
GENRE_TITLE = re.compile(r"^(taḳrīż|takrîz|taḳrîż|tārīḫ|tarih|ḳaṣīde|kaside|ġazel|gazel|ḳıṭʿa|kıt.a|rubāʿī|rubai|müfred|mesnevī|terkīb|tercīʿ|naẓīre|nazire|"
                         r"murabbaʿ|muḫammes|müseddes|müsebbaʿ|şarḳı|naʿt|münācāt|tevḥīd|mersiye|dīger|diğer|ʿarūż|"
                         r"der-|berāy-ı)", re.I)
ROMAN = re.compile(r"^\[?[IVXLC]+\]?\.?$")
BACK_MATTER = re.compile(r"^(KAYNAKÇA|KAYNAKLAR|BİBLİYOGRAFYA|DİZİN|SÖZLÜK|SÖZLÜKÇE|İNDEKS|GENEL DİZİN|EKLER|"
                         r"İLAVELER|ŞÜPHELİ ŞİİRLER|NİSBETİ ŞÜPHELİ ŞİİRLER)$"
                         r"|^EK\s*\d*\s*:|ŞÜPHELİ")
STANZAIC = re.compile(r"(?i)murabbaʿ|murabba|muḫammes|muhammes|müseddes|müsebbaʿ|şarḳı|şarkı|terkīb|terkib|tercīʿ|terci|musammaṭ")
NONSTANZAIC = re.compile(r"(?i)ġazel|gazel|ḳaṣīde|kaside|ḳıṭʿa|kıt.a|rubāʿī|rubai|müfred|tārīḫ|tarih|mesnevī")
WEIGHTS_LINE = re.compile(r"^[\s/–\-.·]+$")
TEFILE_LIKE = re.compile(r"(?i)^[\[(]?(mef|fā|fa|fe|müs|müf|müt)\S{0,7}(lün|lü|tün|tü|ūl|ul|ʿ|lā|l)[\])]?$")
RUNNING_HEAD = re.compile(r"^D[İIĪ]V[AĀ]N$")

def tefile_key(token: str) -> str:
    out = []
    for ch in unicodedata.normalize("NFD", token.lower()):
        if ch in LONG and out and out[-1] in "aiu":
            out[-1] = out[-1].upper()
        elif ch == "̈" and out:
            out[-1] = unicodedata.normalize("NFC", out[-1] + ch)
        elif ch.isalpha() and not unicodedata.combining(ch) and unicodedata.category(ch) != "Lm":
            out.append(ch)
    return "".join(out)

FIRST_FREE = {"fAilAtün", "feilAtün"}
LAST_FREE = {"feilün", "falün", "feilAn", "falAn"}

def tefiles(line: str) -> Optional[List[str]]:
    tokens = re.sub(r"[()\[\]\d/]", " ", LABEL.sub("", line)).split()
    if len(tokens) < 2:
        return None
    keys = [tefile_key(t) for t in tokens]
    joined: List[str] = []
    for k in keys:
        if joined and (joined[-1] not in TEFILE or k not in TEFILE) and joined[-1] + k in TEFILE:
            joined[-1] += k
        else:
            joined.append(k)
    keys = joined
    if len(keys) < 2:
        return None
    if keys[-1] not in TEFILE and keys[-1][:-1] in TEFILE:
        keys[-1] = keys[-1][:-1]
    return keys if all(k in TEFILE for k in keys) else None

def restore_long(keys: List[str]) -> Optional[List[str]]:
    swap = {"mefAilün": "mefAIlün", "mefAilü": "mefAIlü"}
    if meter_id("".join(TEFILE[k] for k in keys)):
        return None
    fixed = [swap.get(k, k) for k in keys]
    return fixed if fixed != keys and meter_id("".join(TEFILE[k] for k in fixed)) else None

def vezin(line: str) -> Optional[Tuple[str, List[str], str]]:
    keys = tefiles(line)
    if keys:
        fixed = restore_long(keys)
        note = "long ī restored in the vezin (unmarked or misprinted in the edition)" if fixed else ""
        keys = fixed or keys
        return "".join(TEFILE[k] for k in keys), variants(keys), note
    if WEIGHTS_LINE.match(line) and sum(c in "-–.·" for c in line) >= 6:
        p = "".join("-" if c in "-–" else "." for c in line if c in "-–.·")
        return p, [p], "vezin printed as weights"
    return None

def meter_pattern(line: str) -> Optional[str]:
    v = vezin(line)
    return v[0] if v else None

def variants(keys: List[str]) -> List[str]:
    first = TEFILE[keys[0]]
    if keys[0] in FIRST_FREE:
        first = "x" + first[1:]
    middle = "".join(TEFILE[k] for k in keys[1:-1]) if len(keys) > 1 else ""
    lasts = ["..-", "--"] if keys[-1] in LAST_FREE else [TEFILE[keys[-1]]]
    if len(keys) == 1:
        return [first]
    return [first + middle + last for last in lasts]

def meter_id(pattern: str) -> Optional[str]:
    for m in METERS:
        for p in m.patterns:
            ks = kinds(p)
            if len(ks) == len(pattern) and all(
                    k == "X" or (k == "L") == (c == "-") for k, c in zip(ks[:-1], pattern[:-1])):
                return m.name
    return None

def vezin_like(line: str) -> bool:
    words = LABEL.sub("", line).split()
    return len(words) >= 2 and sum(1 for w in words if TEFILE_LIKE.match(w)) >= .6 * len(words)

def heading(line: str) -> bool:
    if meter_pattern(line):
        return False
    letters = [c for c in line if c.isalpha()]
    return bool(HEADING.match(line)) or (len(letters) >= 3 and all(c.isupper() for c in letters))

SUFFIXES = {a + b for a in ("", "y", "n", "d", "t", "l", "s") for b in
            ("a", "e", "ı", "i", "u", "ü", "ın", "in", "un", "ün", "a", "da", "de", "dan", "den", "ta", "te", "tan",
             "ten", "la", "le", "dur", "dür", "dır", "dir", "tur", "tür", "uñ", "üñ", "ıñ", "iñ", "dı", "di", "du", "dü")}
SUFFIXES |= {"nıñ", "niñ", "nuñ", "nüñ", "nın", "nin", "nun", "nün", "ya", "ye", "yı", "yi", "ile", "ise", "çün", "içün"}
TRAIL = ".,;:!?“”\"()«»—–"
ELIDING = {"n", "ne", "no", "ni", "nic", "niç", "k", "ki", "z", "zi", "gib", "nit", "nite"}

def silent_mark(word: str, i: int) -> bool:
    return mark_reason(word, i) is not None

def mark_reason(word: str, i: int) -> Optional[str]:
    after = word[i + 1:]
    if re.match(r"[ltdḍrzsşṣżṭẓẕn]\u0331?-", after) and not IZAFET.match(after) or after.startswith(("ll", "all", "ol")) or \
            re.match(r"l[^aeıioöuüāīūâîû\W]", after):
        return "article"
    before = word[:i].lstrip("ʿʾ")
    if before.lower() in ELIDING and after[:1].isalpha():
        return "elision"
    return "proper noun" if proper_suffix(word, i) else None

IZAFET = re.compile(rf"[^\W\d_]\u0331?-y?[ıiuü][{re.escape(TRAIL)}]*$")

def proper_suffix(word: str, i: int) -> bool:
    before, suffix = word[:i].lstrip("ʿʾ"), word[i + 1:].rstrip(TRAIL)
    plain = unicodedata.normalize("NFD", suffix)
    return before[:1].isupper() and sum(c.isalpha() for c in before) >= 3 and len(suffix) <= 4 and suffix.isalpha() \
        and suffix.islower() and not re.search("[āīūâîûēō\u0304\u0302]", suffix + plain)

def clean_title(lines: List[str]) -> List[str]:
    out = []
    for l in lines:
        l = re.sub(r"^\d+[.=*\d]*\s+", "", l.strip()).rstrip("*").strip()
        if not l or len(l.split()) > 10 or re.search(r"[\d=]", l) or ARABIC.search(l) or \
                sum(c.isalpha() for c in l) < 3:
            continue
        if re.search(r"(?i)d[iī]v[aā]n[ıi]$", l) and len(l.split()) <= 4:
            continue
        l = " ".join(w for w in l.split() if not RUNNING_HEAD.match(w))
        l = re.sub(r"^\[[A-Z]{1,3}\d*-?\d*[ab]?\]\s*", "", l)
        l = re.sub(r"\s+[a-f]$", "", l)
        if sum(c.isalpha() for c in l) < 3 or ":" in l or re.search(r"[.,]\s|\.$", l):
            continue
        if 7 <= n_syllables(l) <= 17 and not (GENRE_TITLE.match(l) or VELEHU.match(l)) or \
                7 <= n_syllables(l) <= 17 and len([w for w in l.split()[1:] if "-" not in w and w[0].islower()
                                                    and w not in ("ve", "u", "ü", "ile", "der")]) >= 2:
            continue
        out.append(l)
    return out[-3:]

def apply_orthography(line: str, silent: str, context: str = "") -> str:
    words = []
    for word in line.split(" "):
        out = []
        for i, c in enumerate(word):
            if c in silent or (c in context and silent_mark(word, i)) or (c in "’'ʼʾ" and word[i + 1:i + 5] == "llāh"):
                continue
            out.append(c)
        words.append("".join(out))
    return " ".join(words)

RESTORED = re.compile(r"[\[(]([^\W\d_][^\[\]()\d?]*)[\])]")
FOLIO_PREFIX = re.compile(rf"^(({SIGLUM}\s+\d+[ab],?\s*)+)([IVX]+\s+)?")

def clean_verse(line: str) -> str:
    line = line.replace("\xad", "").replace("\xa0", " ")
    line = FOLIO_PREFIX.sub("", line)
    line = re.sub(r"\s*\(\d+\)|\s*\(\s*[*∗\uf02a]+\s*\)", "", line)
    line = RESTORED.sub(r"\1", line)
    line = re.sub(r"(?<=[^\W\d_’'])–(?=[^\W\d_])", "-", line)
    line = re.sub(r"(?:(?<=[-\s])|^|(?<=üstü))([ḫḪ])v(?=[āīaeıiouü])", r"\1ˇ", line)
    line = re.sub(r"(?<!\S)\d+=(?!\S)|[*∗\uf02a]", "", line)
    line = line.replace("\uf020", " ")
    line = re.sub(r"\s+\d+$", "", line)
    line = re.sub("[ʻ῾ʽ˓]", "ʿ", line)
    line = line.replace("\u0320", "\u0331")
    line = re.sub(r"(?<=[^\s\d])\d+", "", line)
    line = re.sub(r"(?<=[^\W\d_])[xX](?![^\W\d_])", "", line)
    line = re.sub(r"(?<=[^\W\d_]) +-(?=[^\W\d_])", "-", line)
    return line.strip()

@dataclass
class Poem:
    number: Optional[int]
    page: int
    meter_raw: str
    pattern: str
    variants: List[str]
    title: List[str] = field(default_factory=list)
    units: List[Tuple[int, List[str]]] = field(default_factory=list)
    damaged: int = 0
    inherited: bool = False
    paired: bool = False
    realigned: bool = False
    stanzas: List[int] = field(default_factory=list)
    regrouped: str = ""
    vezin_note: str = ""
    stanzaic_section: bool = False
    notes: Dict[str, str] = field(default_factory=dict)

def page_lines(page: ET.Element) -> List[str]:
    regions: List[Tuple[float, str, str]] = []
    for tag in ("title", "paragraph"):
        for el in page.iter(tag):
            if el.text and el.get("bbox"):
                y = float(el.get("bbox").split(",")[1])
                regions.append((y, " ".join(el.text.split()), el.text))
    norms = {r[1] for r in regions}
    drop = set()
    for norm in norms:
        parts = [o for o in norms if o != norm and o in norm]
        if not parts:
            continue
        if sum(len(o) for o in parts) >= .9 * len(norm):
            drop.add(norm)
        else:
            drop.update(parts)
    keep, seen = [], []
    for y, norm, text in sorted(regions, key=lambda r: r[0]):
        if norm in drop or any(norm == n and abs(y - y0) < 20 for n, y0 in seen):
            continue
        seen.append((norm, y))
        keep.append(text)
    return [l.strip() for text in keep for l in text.splitlines() if l.strip()]

def join_split_words(lines: List[Tuple[int, str]]) -> List[Tuple[int, str]]:
    out: List[Tuple[int, str]] = []
    run: List[Tuple[int, str]] = []
    for pg, line in lines + [(0, "a b")]:
        if len(line.split()) == 1 and not NUMBER.match(line):
            run.append((pg, line))
            continue
        if len(run) >= 3:
            out.append((run[0][0], " ".join(l for _, l in run)))
        else:
            out += run
        run = []
        out.append((pg, line))
    return out[:-1]

def rhyme(line: str) -> str:
    t = "".join(c for c in unicodedata.normalize("NFD", line.lower()) if c.isalpha() and not unicodedata.combining(c))
    return t[-2:]

def realign(poem: Poem) -> bool:
    lines = [m for _, ms in poem.units for m in ms]
    if len(lines) < 6:
        return False
    rhymes = Counter(rhyme(l) for l in lines)
    r, n = rhymes.most_common(1)[0]
    if n < .4 * len(lines):
        return False
    def score(off):
        seconds = lines[off + 1::2]
        return sum(rhyme(l) == r for l in seconds) / max(len(seconds), 1)
    if score(1) - score(0) < .3:
        return False
    poem.units = [(1, [lines[0]])] + [(i + 2, lines[1 + 2 * i:3 + 2 * i]) for i in range((len(lines) - 1 + 1) // 2)]
    poem.units = [u for u in poem.units if u[1]]
    poem.realigned = True
    return True

def regroup(poem: Poem) -> bool:
    units = poem.units
    if len(units) < 3 or sum(len(ms) == 1 for _, ms in units) < .8 * len(units):
        return False
    restarts = [i for i, (n, _) in enumerate(units) if n == 1 and i > 0]
    if poem.stanzas or restarts:
        bounds = sorted(set([0] + poem.stanzas + restarts + [len(units)]))
        poem.units = [(i + 1, [m for _, ms in units[a:b] for m in ms]) for i, (a, b) in enumerate(zip(bounds, bounds[1:])) if b > a]
        poem.regrouped = "stanzas"
    else:
        lines = [m for _, ms in units for m in ms]
        poem.units = [(i + 1, lines[2 * i:2 * i + 2]) for i in range((len(lines) + 1) // 2)]
        poem.regrouped = "mısra pairs"
    return True

def n_syllables(line: str) -> int:
    words = [w for w in line.split() if any(c in VOWELS for c in w)]
    return sum(len(syllabify(w)) for w in words)

def verse_like(lines: List[str]) -> bool:
    ok = sum(1 for l in lines if 7 <= n_syllables(l) <= 17 and not re.search(r"[.:;]\s*$|\d", l))
    return len(lines) >= 6 and ok >= .85 * len(lines)

def repair_numbers(poems: List[Poem], stats: Counter) -> None:
    numbered = [p for p in poems if p.number is not None]
    cands = [[c for c in (p.number, p.number // 10, p.number // 100) if c > 0] for p in numbered]
    for i, p in enumerate(numbered):
        near1 = (cands[i - 1] if i > 0 else []) + (cands[i + 1] if i + 1 < len(cands) else [])
        near2 = (cands[i - 2] if i > 1 else []) + (cands[i + 2] if i + 2 < len(cands) else [])
        def fits(c):
            return c == 1 or any(1 <= abs(c - o) <= 3 for o in near1) or any(2 <= abs(c - o) <= 6 for o in near2)
        choice = next((c for c in cands[i] if fits(c)), None)
        if choice is None:
            stats["poem numbers out of sequence (set to none)"] += 1
        elif choice != p.number:
            stats["poem numbers with a footnote digit cut"] += 1
        p.number = choice

def title_line(line: str) -> bool:
    words = line.split()
    return 1 <= len(words) <= 3 and line[0].isupper() and not re.search(r"\d", line)

def letters(line: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", line.lower()) if c.isalpha() and not unicodedata.combining(c))[:25]

def glued(n: int, line: str) -> bool:
    return bool(re.search(rf"(?<=[^\W\d_]){n}(?!\d)", line))

def parse(xml_path: Path, glyphs: List[Tuple[str, str]] = (),
          notes: Dict[int, List[Tuple[str, Optional[int], str]]] = None) -> Tuple[List[Poem], Counter]:
    root = ET.parse(xml_path).getroot()
    stats: Counter = Counter()
    lines: List[Tuple[int, str]] = []
    for page in root.findall("page"):
        for line in page_lines(page):
            for pattern, repl in glyphs:
                line = re.sub(pattern, repl, line)
            lines.append((int(page.get("number")), line))
    for i, (pg, line) in enumerate(lines):
        if BACK_MATTER.match(line.strip()):
            stats["lines after back matter heading"] = len(lines) - i
            lines = lines[:i]
            break
    unwrapped: List[Tuple[int, str]] = []
    for pg, line in lines:
        if unwrapped and unwrapped[-1][1].endswith("-") and not NUMBER.match(line):
            unwrapped[-1] = (unwrapped[-1][0], unwrapped[-1][1] + line)
            stats["wrapped lines joined"] += 1
        else:
            unwrapped.append((pg, line))
    lines = unwrapped
    joined = join_split_words(lines)
    stats["split words joined"] = len(lines) - len(joined)
    lines = joined
    notes = {pg: [(k, n, why) if k != "marker" or any(p == pg and glued(n, l) for p, l in lines) else ("page", None, why)
                  for k, n, why in ns] for pg, ns in (notes or {}).items()}
    unlabeled = not any(meter_pattern(l) for _, l in lines)
    numbered = sum(1 for _, l in lines if NUMBER.match(l) or TITLED.match(l)) > .2 * len(lines)
    stats["beyits numbered" if numbered else "beyits unnumbered (paired)"] += 1

    poems: List[Poem] = []
    poem: Optional[Poem] = None
    pending: List[Tuple[int, str]] = []
    unit: Optional[Tuple[int, List[str]]] = None
    auto = False

    stanza_break = False
    stanzaic = False
    unassigned: List[Tuple[int, str]] = []

    def new_poem(*args, **kw) -> Poem:
        p = Poem(*args, **kw)
        p.stanzaic_section = stanzaic
        poems.append(p)
        return p

    def note(pg: int, line: str):
        if RESTORED.search(FOLIO_PREFIX.sub("", line)):
            poem.notes[unit[1][-1]] = "editor's note: words supplied by the editor"
            stats["mısras with words supplied by the editor"] += 1
        for kind, n, why in notes.get(pg, []):
            if kind == "page" or kind == "marker" and glued(n, line) or kind == "beyit" and unit[0] == n:
                poem.notes[unit[1][-1]] = why
                stats["mısras with an editor's note: " + why] += 1

    def close(next_number: Optional[int] = None):
        nonlocal unit
        if unit and poem and unit[1]:
            k = (next_number - unit[0]) if next_number else 1
            if k > 1 and len(unit[1]) == 2 * k:
                poem.units += [(unit[0] + i, unit[1][2 * i:2 * i + 2]) for i in range(k)]
                stats["sparsely numbered beyits split"] += k
            elif k > 1 and len(unit[1]) > 2 and not poem.stanzas:
                ls = unit[1]
                poem.units += [(unit[0] + i, ls[2 * i:2 * i + 2]) for i in range((len(ls) + 1) // 2)]
                stats["sparse blocks paired despite a line count mismatch"] += 1
            else:
                poem.units.append(unit)
        unit = None

    def ahead(i: int) -> Optional[int]:
        for _, l in lines[i + 1:i + 5]:
            m = NUMBER.match(l) or TITLED.match(l)
            if m:
                return int(m.group(1))
        return None

    for i, (pg, line) in enumerate(lines):
        nxt = lines[i + 1][1] if i + 1 < len(lines) else ""
        if NUMBER.match(line) and re.match(rf"^{SIGLUM}\.\s*\d+:", nxt):
            stats["footnote numbers before a source note"] += 1
            continue
        if poem and ROMAN.match(line):
            close()
            poem.stanzas.append(len(poem.units))
            stanza_break = True
            stats["stanza marks"] += 1
            continue
        if APPARATUS.search(line) and not NUMBER.match(line):
            stats["apparatus lines"] += 1
            continue
        if heading(line):
            stats["headings"] += 1
            stanzaic = bool(STANZAIC.search(line)) or (stanzaic and not NONSTANZAIC.search(line))
            pending.append((pg, line))
            continue
        if poem and GENRE_TITLE.match(line) and not unlabeled and not meter_pattern(line) \
                and (unit is None or len(unit[1]) >= 2) \
                and not (poem.variants and min(abs(n_syllables(line) - len(v)) for v in poem.variants) <= 2):
            close()
            stanzaic = bool(STANZAIC.search(line)) or (stanzaic and not NONSTANZAIC.search(line))
            poem = new_poem(None, pg, "", "", [], [line])
            stats["genre titles"] += 1
            pending = [(pg, line)]
            continue
        if unlabeled and title_line(line):
            close()
            stanzaic = bool(STANZAIC.search(line)) or (stanzaic and not NONSTANZAIC.search(line))
            poem = new_poem(None, pg, "", "", [], [line])
            stats["titles (no vezin in the edition)"] += 1
            pending = []
            continue
        t = TITLED.match(line)
        if t and not NUMBER.match(line) and poem:
            last = unit[0] if unit else (poem.units[-1][0] if poem.units else 0)
            n = int(t.group(1))
            seen = {letters(l) for _, ms in poem.units[-3:] for l in ms} | {letters(l) for l in (unit[1] if unit else [])}
            twice = unit and n == last > 1 and len(unit[1]) >= 2 and letters(t.group(2)) not in seen
            if twice:
                stats["beyit numbers printed twice by the edition"] += 1
            couplets = len(poem.units) >= 2 and sum(len(ms) == 2 for _, ms in poem.units) >= .8 * len(poem.units)
            if n == 1 and last > 1 and couplets and not stanza_break:
                close()
                poem = new_poem(None, pg, "", "", [], [])
                stats["new poem at a numbering restart"] += 1
                last = 0
            if (n == 1 and last == 0) or twice or (0 < n - last <= 50 and not GENRE_TITLE.match(t.group(2))):
                close(n)
                unit, auto = (n, [clean_verse(t.group(2))]), False
                note(pg, line)
                stats["numbered on the line"] += 1
                continue
        if t and not NUMBER.match(line):
            stats["numbered titles"] += 1
            close()
            pending = [(pg, line)]
            continue
        v = vezin(line)
        pattern = v[0] if v else None
        unreadable = not pattern and vezin_like(line)
        if unreadable:
            v, pattern = ("", [], "vezin line unreadable (typo or cut off): no label"), ""
            stats["vezin lines unreadable"] += 1
        last = unit[0] if unit else (poem.units[-1][0] if poem and poem.units else 0)
        if v and poem and last and ahead(i) == last + 1 and (not poem.variants or pattern == poem.pattern) \
                and not any(heading(l) or GENRE_TITLE.match(l) for _, l in pending):
            if pattern and not poem.variants:
                poem.meter_raw, poem.pattern, poem.variants, poem.vezin_note = line, pattern, v[1], "vezin read from its repeat"
            stats["vezin repeated inside a poem"] += 1
            continue
        if v and poem and not poem.units and unit is None:
            if unreadable:
                poem.pattern, poem.variants, poem.vezin_note = "", [], v[2]
                poem.meter_raw = poem.meter_raw + " | " + line if poem.meter_raw else line
                continue
            if poem.inherited or not poem.meter_raw:
                poem.meter_raw, poem.pattern, poem.variants, poem.vezin_note = line, pattern, v[1], v[2]
                poem.inherited = False
            elif " | " not in poem.meter_raw and len(line) > len(poem.meter_raw) and \
                    " ".join(line.split()).startswith(" ".join(poem.meter_raw.split())) and pattern:
                poem.meter_raw, poem.pattern, poem.variants, poem.vezin_note = line, pattern, v[1], v[2]
                stats["vezin printed cut off, then whole"] += 1
            elif poem.vezin_note == "vezin line unreadable (typo or cut off): no label":
                poem.meter_raw += " | " + line
            elif any(x not in poem.variants for x in v[1]):
                poem.meter_raw += " | " + line
                poem.variants += [x for x in v[1] if x not in poem.variants]
            continue
        if VELEHU.match(line) and poem:
            close()
            header = [l for _, l in pending if NUMBER.match(l) or TITLED.match(l)]
            nums = [int((NUMBER.match(l) or TITLED.match(l)).group(1)) for l in header]
            poem = new_poem(nums[-1] if nums else None, pg, poem.meter_raw, poem.pattern, list(poem.variants),
                            [line], inherited=True)
            pending = []
            stats["velehū titles"] += 1
            continue
        if v:
            misra_numbered = poem and len(poem.units) >= 3 and \
                sum(len(ms) == 1 for _, ms in poem.units) >= .8 * len(poem.units)
            if unit and poem and (len(unit[1]) >= 2 or (misra_numbered and unit[1])):
                poem.units.append(unit)
                pending = [x for x in pending if heading(x[1])]
            header = [l for _, l in pending] + (unit[1] if unit and len(unit[1]) < 2 else [])
            unit = None
            nums = [int((NUMBER.match(l) or TITLED.match(l)).group(1)) for l in header
                    if NUMBER.match(l) or TITLED.match(l)]
            fused = re.match(r"^[-–](\d+)[-–]", line)
            number = nums[-1] if nums else (int(fused.group(1)) if fused else None)
            poem = new_poem(number, pg, line, pattern, v[1], [l for l in header if not NUMBER.match(l)])
            poem.vezin_note = v[2]
            pending = []
            continue
        m = NUMBER.match(line)
        if m and poem and unit and len(unit[1]) == 1 and int(m.group(1)) not in (unit[0] + 1, 1) and not stanza_break \
                and not (meter_pattern(nxt) or vezin_like(nxt)):
            stats["footnote markers"] += 1
            continue
        if m and poem and int(m.group(1)) == 1 and not stanza_break and len(poem.units) >= 2 and \
                (unit is None or unit[0] > 1) and sum(len(ms) == 2 for _, ms in poem.units) >= .8 * len(poem.units):
            close()
            poem = new_poem(None, pg, "", "", [], [])
            stats["new poem at a numbering restart"] += 1
        if m and poem:
            n = int(m.group(1))
            close(n)
            unit, auto = (n, []), False
            stanza_break = False
            pending = [(pg, line)]
            continue
        if ARABIC.search(line):
            stats["arabic-script lines"] += 1
            pending.append((pg, line))
            continue
        if poem and numbered and unit is None and poem.variants and not unlabeled:
            unit, auto = ((poem.units[-1][0] + 1 if poem.units else 1), []), False
            stats["verse lines without a number adopted"] += 1
        if poem and not numbered and (unit is None or (auto and len(unit[1]) == 2)):
            close()
            unit, auto = ((poem.units[-1][0] + 1 if poem.units else 1), []), True
            poem.paired = True
            stats["unnumbered units"] += 1
        if unit is not None:
            if unicodedata.combining(line[0]):
                poem.damaged += 1
            unit[1].append(clean_verse(line))
            note(pg, line)
        else:
            stats["lines unassigned"] += 1
            unassigned.append((pg, line))
            pending.append((pg, line))
    close()

    blocks: List[List[Tuple[int, str]]] = []
    for pg, line in unassigned:
        if blocks and pg - blocks[-1][-1][0] <= 1:
            blocks[-1].append((pg, line))
        else:
            blocks.append([(pg, line)])
    kept = []
    for block in blocks:
        texts = [clean_verse(l) for _, l in block]
        if verse_like(texts):
            recovered = Poem(None, block[0][0], "", "", [], ["(verse that no poem took)"], paired=True)
            recovered.stanzaic_section = True
            recovered.units = [(i + 1, texts[2 * i:2 * i + 2]) for i in range((len(texts) + 1) // 2)]
            poems.append(recovered)
            stats["verse blocks recovered as poems without vezin"] += 1
            stats["lines recovered from unassigned"] += len(texts)
        else:
            kept += block
    unassigned = kept
    stats["lines unassigned"] = len(unassigned)
    poems.sort(key=lambda p: p.page)
    for p in poems:
        if regroup(p):
            stats["poems regrouped: " + p.regrouped] += 1
    for p in poems:
        if p.paired and realign(p):
            stats["pairing realigned by rhyme"] += 1
    repair_numbers(poems, stats)
    for p in poems:
        stats["poems"] += 1
        stats["units"] += len(p.units)
    stats.unassigned = unassigned
    return poems, stats
