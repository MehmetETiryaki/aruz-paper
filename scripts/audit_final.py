import json
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict

from aruz.editions import apply_orthography, meter_pattern
from aruz.scan import VOWELS, syllabify
from aruz.editions import vezin_like
from aruz.validate import LETTERS, PUNCT, tokens

F = os.environ["STORE"] + "/datasets/ottoman_poetry/final/"
problems = defaultdict(list)
seen = defaultdict(set)
counts = Counter()
for split in ("train", "dev", "test"):
    for line in open(F + split + ".jsonl", encoding="utf-8"):
        r = json.loads(line)
        counts[split, "poems"] += 1
        if not r["units"]:
            problems["poem without units"].append(r["id"])
        if split != "train" and not r["variants"]:
            problems["dev/test poem without vezin"].append(r["id"])
        sizes = Counter(len(u["misras"]) for u in r["units"])
        if re.search(r"[^\W\d_]\d", r["meter_raw"]):
            problems["footnote number in meter_raw"].append((r["id"], r["meter_raw"]))
        for u in r["units"]:
            if len(u["misras"]) != sizes.most_common(1)[0][0] and len(r["units"]) > 1:
                problems["unit shape differs from its poem"].append((r["id"], u["n"], len(u["misras"])))
            for m in u["misras"]:
                counts[split, "mısras"] += 1
                t = apply_orthography(m, r["silent"], r["context"])
                if not m.strip():
                    problems["empty mısra"].append(r["id"])
                if re.search(r"\d", m):
                    problems["digit in mısra"].append((r["id"], m))
                if any(c not in LETTERS and c.lower() not in LETTERS and c not in PUNCT and c not in "İI"
                       for c in unicodedata.normalize("NFC", t)):
                    problems["character outside the alphabet"].append((r["id"], m))
                if meter_pattern(m) or vezin_like(m):
                    problems["vezin line as verse"].append((r["id"], m))
                if re.search(r"\?\?|[(\[\])]|…|\.\.", m):
                    problems["editor's mark in mısra (doubt, brackets, gap dots)"].append((r["id"], m))
                if re.search(r"[a-zçğıöşüāīūḳḥḫṣżṭẓẕġñ][A-ZÇĞİÖŞÜĀĪŪḲḤḪṢŻṬẒẔĠÑ]|[^\W\d_]—[^\W\d_]|–", unicodedata.normalize("NFC", m)):
                    problems["garbled word (glued capital, dash inside a word)"].append((r["id"], m))
                if re.search(r"(?:^|[\s-]|üstü)[ḫḪ]v[āīaeıiouü]", m):
                    problems["ḫv not written ḫˇ"].append((r["id"], m))
                if re.search(r"-y?[ıiuü][.,;:!?]*$", m):
                    problems["mısra ends on an izafet (truncated)"].append((r["id"], m))
                n = sum(len(syllabify(w)) for w in tokens(t) if any(c in VOWELS for c in w))
                if not 5 <= n <= 22:
                    problems["implausible mısra length"].append((r["id"], n, m))
                key = "".join(c for c in unicodedata.normalize("NFD", m.lower()) if c.isalpha() and not unicodedata.combining(c))
                if len(key) >= 20:
                    seen[key].add(split)
leak = [k for k, s in seen.items() if len(s) > 1]
if leak:
    problems["mısra in more than one split"] = leak
print({f"{s} {k}": v for (s, k), v in sorted(counts.items())})
for k, v in problems.items():
    print(f"{k}: {len(v)}   e.g. {v[:4]}")
if not problems:
    print("final sets: no problems found")
sys.exit(1 if problems else 0)
