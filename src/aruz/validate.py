from __future__ import annotations
import re
import unicodedata
from collections import Counter
from typing import Dict, List, Set, Tuple

from .editions import apply_orthography, letters, mark_reason, meter_pattern, silent_mark, vezin_like
from .scan import VOWELS, syllabify, weight

LETTERS = set("abcçdefgğhıijklmnoöprsştuüvyz"
              "āīūōēâîûėḳḥḫṣṭẓżẕġñẟḍʿʾʼ’'ˇ") | {"̱", "̠"}
PUNCT = set(" -.,;:!?“”\"()«»—–\t‿")
CORRUPT = {"bad character", "damaged", "no vowel", "arabic script", "length", "empty", "fragment", "vezin line as verse",
           "truncated word", "ends on an izafet (truncated)", "empty brackets (text lost)", "garbled word",
           "editor's gap (text lost)", "impossible word (three final consonants)"}
POEM_RISKS = {"vezin label suspect", "numbering resets inside the poem", "pairing suspect (rhyme)",
              "recovered verse block (attribution unchecked)", "pairing realigned by rhyme (first line alone)",
              "pairing not confirmed by rhyme", "unnumbered stanzaic poem (stanzas unknown)",
              "vezin label suspect (one position misfits in most lines)", "pairing shifted (first mısra rhymes, second not)",
              "vezin line inside the poem (boundary missed)", "editor's note: attribution doubtful",
              "unit 1 then a jump (lost digit or merged poem)", "poem printed twice (most units are duplicates)",
              "continuation of another poem (starts late, no vezin)", "refrain in an unnumbered poem (stanzaic)",
              "fragment of a longer poem (starts after unit 5)", "head of a poem continued in the next record",
              "continuation of another poem (vezin printed again)",
              "another poet's taḳrīż", "vezin labels for other languages"}
UNIT_RISKS = {"shared with another book", "duplicate", "editor's doubtful reading", "breaks the poem's rhyme (truncated?)",
              "editor's note: meter broken", "editor's note: attribution doubtful", "apostrophe of unclear value",
              "letter uncertain (C may be ʿ)", "editor's note: variant text", "editor's note: struck out in the manuscript",
              "editor's note: words supplied by the editor", "editor's note: text changed by the editor"}
HARMONY = str.maketrans("aıou", "eiöü")
COPY = str.maketrans("aıuüo", "eiiiö")

def rhyme(line: str) -> str:
    t = "".join(c for c in unicodedata.normalize("NFD", line.lower()) if c.isalpha() and not unicodedata.combining(c))
    return t[-2:].translate(HARMONY)

ARABIC = re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")

def tokens(line: str) -> List[str]:
    return [w.strip(".,;:!?“”\"()«»—–") for w in line.split() if w.strip(".,;:!?“”\"()«»—–")]

def misra_issues(line: str, lengths: Set[int]) -> List[str]:
    issues = []
    if not line.strip():
        return ["empty"]
    if sum(len(syllabify(w)) for w in tokens(line) if any(c in VOWELS for c in w)) < 5:
        issues.append("fragment")
    if unicodedata.combining(line[0]):
        issues.append("damaged")
    if any(seg and not any(c in VOWELS for c in seg) for w in tokens(line) for seg in w.split("-")):
        issues.append("truncated word")
    if re.search(r"[a-zçğıöşüāīūḳḥḫṣżṭẓẕġñ][A-ZÇĞİÖŞÜĀĪŪḲḤḪṢŻṬẒẔĠÑ]", unicodedata.normalize("NFC", line)) or \
            re.search(r"[^\W\d_]—[^\W\d_]|[^\W\d_]-(\s|$)", line):
        issues.append("garbled word")
    if "…" in line or ".." in line:
        issues.append("editor's gap (text lost)")
    if re.search(r"-y?[ıiuü][?!.,;:]", line):
        issues.append("ends on an izafet (truncated)")
    if "()" in line or "[]" in line:
        issues.append("empty brackets (text lost)")
    if "(?)" in line or "[?]" in line or "??" in line:
        issues.append("editor's doubtful reading")
    last = tokens(line)[-1] if tokens(line) else ""
    if re.search(r"-y?[ıiuü]$", last):
        issues.append("ends on an izafet (truncated)")
    for w in tokens(line):
        tail = 0
        for c in reversed([c for c in w.split("-")[-1] if c.isalpha()]):
            if c.lower() in VOWELS:
                break
            tail += 1
        if tail >= 3 and tail < len([c for c in w if c.isalpha()]):
            issues.append("impossible word (three final consonants)")
    if meter_pattern(line) or vezin_like(line):
        issues.append("vezin line as verse")
    if ARABIC.search(line):
        issues.append("arabic script")
    if any(c not in LETTERS and c.lower() not in LETTERS and c not in PUNCT and c not in "İI"
           for c in unicodedata.normalize("NFC", line)):
        issues.append("bad character")
    sylls = []
    for w in tokens(line):
        if not any(c in VOWELS for c in w):
            issues.append("no vowel")
            continue
        sylls += syllabify(w)
    if sylls and lengths:
        n = len(sylls)
        extra = sum(1 for t, lv, coda in sylls[:-1] if weight(t, lv, coda) == "LH")
        off = min(0 if n <= L <= n + extra else min(abs(L - n), abs(L - n - extra)) for L in lengths)
        if off >= 3:
            issues.append("length")
        elif off:
            issues.append(f"length±{off}")
    return issues

def check(poems: List[dict]) -> Counter:
    stats: Counter = Counter()
    seen: Set[str] = set()
    for p in poems:
        lengths = {len(v) for v in p["variants"]}
        sizes = Counter(len(u["misras"]) for u in p["units"])
        usual = sizes.most_common(1)[0][0] if sizes else 2
        p["issues"] = []
        if not p["units"]:
            p["issues"].append("no units")
        if p["inherited"]:
            p["issues"].append("inherited vezin")
        if not p["variants"]:
            p["issues"].append("no vezin printed")
        if p["title"] == ["(verse that no poem took)"]:
            p["issues"].append("recovered verse block (attribution unchecked)")
        if p.get("realigned"):
            p["issues"].append("pairing realigned by rhyme (first line alone)")
        numbers = [u["n"] for u in p["units"]]
        gaps = sum(b - a - 1 for a, b in zip(numbers, numbers[1:]) if b > a + 1)
        if gaps:
            p["issues"].append(f"{gaps} units missing")
        if len(numbers) >= 2 and numbers[0] == 1 and numbers[1] - numbers[0] >= 5 or \
                any(b - a >= 30 for a, b in zip(numbers, numbers[1:])):
            p["issues"].append("unit 1 then a jump (lost digit or merged poem)")
        if numbers and numbers[0] > 5 and not p.get("regrouped"):
            p["issues"].append("fragment of a longer poem (starts after unit 5)")
        if any(re.search(r"(?i)ta[ḳk][rṛ][iīî][żz]", t) for t in p["title"]):
            p["issues"].append("another poet's taḳrīż")
        if re.search(r"ʿArabī:|Fārsī:|Arabî:|Farsî:", p["meter_raw"]):
            p["issues"].append("vezin labels for other languages")
        if numbers and numbers[0] > 1 and not p["variants"]:
            p["issues"].append("continuation of another poem (starts late, no vezin)")
        if numbers and numbers[0] != 1 and numbers[0] < 50:
            p["issues"].append(f"starts at unit {numbers[0]}")
        for u in p["units"]:
            issues: List[str] = []
            if len(u["misras"]) != usual:
                issues.append(f"{len(u['misras'])} mısras where the poem has {usual}")
            n_syll = [sum(len(syllabify(w)) for w in tokens(apply_orthography(m, p["silent"], p["context"]))
                          if any(c in VOWELS for c in w)) for m in u["misras"]]
            ziyade = len(n_syll) >= 4 and all(n_syll[i] * 2 <= n_syll[i - 1] for i in range(1, len(n_syll), 2))
            for i, m in enumerate(u["misras"]):
                text = apply_orthography(m, p["silent"], p["context"])
                if ziyade and i % 2:
                    issues.append("müstezâd ziyade (length unchecked)")
                    issues += [x for x in misra_issues(text, set()) if not x.startswith("length")]
                    continue
                issues += misra_issues(text, lengths)
                if any(c in p["silent"] and 0 < i < len(w) - 1 and (w[i + 1].isalpha() or w[i + 1] == "-") and not silent_mark(w, i)
                       for w in m.split() for i, c in enumerate(w)):
                    issues.append("apostrophe of unclear value")
                first = m.split()[0] if m.split() else ""
                if any(c in p["context"] and 0 < i < len(first) - 1 and mark_reason(first, i) == "proper noun"
                       for i, c in enumerate(first)):
                    issues.append("apostrophe of unclear value")
                key = "".join(c for c in unicodedata.normalize("NFD", text.lower()) if c.isalpha()
                              and not unicodedata.combining(c)).translate(COPY)
                if key in seen:
                    issues.append("duplicate")
                seen.add(key)
            issues += [p.get("notes", {})[m] for m in u["misras"] if m in p.get("notes", {})]
            u["issues"] = sorted(set(issues))
            u["usable"] = not (set(u["issues"]) & CORRUPT)
            if p.get("regrouped") == "stanzas":
                u["issues"] = [i for i in u["issues"] if i != "duplicate"]
            stats["units"] += 1
            stats["usable units"] += u["usable"]
            for i in u["issues"]:
                stats["unit: " + re.sub(r"\d+ mısras where the poem has \d+", "wrong mısra count", i)] += 1
        def key(u):
            return letters(u["misras"][0]) if u["misras"] else ""
        if any(v["n"] < u["n"] or v["n"] == u["n"] and (key(u) == key(v) or u["n"] == 1) for u, v in zip(p["units"], p["units"][1:])) \
                and not p.get("regrouped"):
            p["issues"].append("numbering resets inside the poem")
        couplets = [u for u in p["units"] if len(u["misras"]) == 2]
        if len(couplets) >= 4 and len(couplets) == len(p["units"]):
            a = Counter(rhyme(u["misras"][0]) for u in couplets[1:]).most_common(1)[0][1] / (len(couplets) - 1)
            b = Counter(rhyme(u["misras"][1]) for u in couplets).most_common(1)[0][1] / len(couplets)
            if a >= .6 and b < .6:
                p["issues"].append("pairing suspect (rhyme)")
        seconds = [u["misras"][-1] for u in p["units"] if len(u["misras"]) >= 2]
        ends = Counter(rhyme(m) for m in seconds)
        lastw = Counter(m.split()[-1].lower() if m.split() else "" for m in seconds)
        if len(seconds) >= 4:
            (r, rn), (w, wn) = ends.most_common(1)[0], lastw.most_common(1)[0]
            if max(rn, wn) >= .7 * len(seconds):
                for u in p["units"]:
                    m = u["misras"][-1] if len(u["misras"]) >= 2 else ""
                    if m and rhyme(m) != r and (m.split() or [""])[-1].lower() != w:
                        u["issues"].append("breaks the poem's rhyme (truncated?)")
        if p.get("paired") and not p.get("realigned"):
            if p.get("stanzaic_section"):
                p["issues"].append("unnumbered stanzaic poem (stanzas unknown)")
            r = ends.most_common(1)[0][0] if ends else ""
            ok = [len(u["misras"]) == 2 and (rhyme(u["misras"][1]) == r or rhyme(u["misras"][0]) == rhyme(u["misras"][1]))
                  for u in p["units"]]
            if len(p["units"]) < 3 or not all(ok[:-1]):
                p["issues"].append("pairing not confirmed by rhyme")
        if p.get("paired") and any("duplicate" in u["issues"] for u in p["units"]):
            p["issues"].append("refrain in an unnumbered poem (stanzaic)")
        if p["units"] and sum("duplicate" in u["issues"] for u in p["units"]) >= .5 * len(p["units"]) \
                and not p.get("regrouped"):
            p["issues"].append("poem printed twice (most units are duplicates)")
        if any(i == "editor's note: attribution doubtful" for u in p["units"] for i in u["issues"]):
            p["issues"].append("editor's note: attribution doubtful")
        if any("vezin line as verse" in u["issues"] for u in p["units"][:-1]):
            p["issues"].append("vezin line inside the poem (boundary missed)")
        lines = [m for u in p["units"] if u["usable"] for m in u["misras"]]
        if len(lines) >= 6:
            r, n = Counter(rhyme(m) for m in lines).most_common(1)[0]
            broken = any(len(u["misras"]) != 2 for u in p["units"])
            if n >= .4 * len(lines) and any(len(u["misras"]) == 2 and u["usable"] and rhyme(u["misras"][0]) == r
                                            and rhyme(u["misras"][1]) != r for u in p["units"][0 if broken else 1:]):
                p["issues"].append("pairing shifted (first mısra rhymes, second not)")
        misfit = sum(1 for u in p["units"] if any(i.startswith("length") for i in u["issues"]))
        if p["variants"] and p["units"] and misfit > .5 * len(p["units"]):
            p["issues"].append("vezin label suspect")
        for i in p["issues"]:
            stats["poem: " + re.sub(r"^\d+ units missing|starts at unit \d+", lambda m: "units missing"
                                    if "missing" in m.group(0) else "starts late", i)] += 1
    for a, b in zip(poems, poems[1:]):
        if a["book"] == b["book"] and "continuation of another poem (starts late, no vezin)" in b["issues"] \
                and a["units"] and b["units"][0]["n"] > a["units"][-1]["n"]:
            a["issues"].append("head of a poem continued in the next record")
            stats["poem: head of a poem continued in the next record"] += 1
        elif a["book"] == b["book"] and a["units"] and b["units"] and 1 < b["units"][0]["n"] and b["page"] - a["page"] <= 1 \
                and a["meter_raw"] and b["meter_raw"].lower().startswith(a["meter_raw"].lower()) \
                and (b["number"] is None or a["number"] is not None and b["number"] != a["number"] + 1):
            b["issues"].append("continuation of another poem (vezin printed again)")
            stats["poem: continuation of another poem (vezin printed again)"] += 1
    return stats
