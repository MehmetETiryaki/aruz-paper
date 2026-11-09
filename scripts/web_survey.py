import argparse
import html
import json
import re
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

STORE = Path("$STORE/datasets/ottoman_poetry")
WS = STORE / "web_survey"
ROOT = Path(__file__).resolve().parent.parent

OCR_KINDS = {
    "divansiirindeucbuyukler1fuzuli": ("archive.org: Cevdet Kudret 1976, Divan Şiirinde Üç Büyükler: Fuzûlî", "ocr-popular-book"),
    "FuzuliDivanv_201707": ("archive.org: Fuzûlî Dîvânı (Latin edition, scan)", "ocr-edition"),
    "FuzuliDivanv": ("archive.org: Fuzûlî Dîvânı (edition with apparatus, scan)", "ocr-edition"),
    "FuzuliDivanaSerhi-Prof.Dr.AliNihatTarlan-Turki": ("archive.org: Tarlan, Fuzûlî Divanı Şerhi", "ocr-commentary"),
    "FuzlVeBkDivanlarnnKarlatrlmas": ("archive.org: thesis, Fuzûlî ve Bâkî Divanlarının Karşılaştırılması", "ocr-thesis"),
    "fuzuli_202510": ("archive.org: FUZÛLÎ (anthology booklet)", "ocr-popular-book"),
}
MISC_KINDS = {"turkedebiyati": "educational-site", "listelist": "listicle-news", "news": "listicle-news",
              "encyclopedia": "listicle-news", "school": "school-exam-site", "blog": "blog-column",
              "religious": "blog-column", "column": "blog-column", "wikisource": "wikisource"}

def page_text(t: str) -> str:
    t = re.sub(r"<script.*?</script>|<style.*?</style>|<!--.*?-->", " ", t, flags=re.S | re.I)
    t = re.sub(r"<br\s*/?>|</p>|</div>|</li>|</h\d>|</tr>|</td>|</blockquote>", "\n", t, flags=re.I)
    return html.unescape(re.sub(r"<[^>]+>", "", t))

def extract(args):
    raw, docs = WS / "raw", []
    for f in sorted((raw / "antoloji").glob("*.html")):
        t = f.read_text(encoding="utf-8", errors="replace")
        m = re.search(r'<div class="pd-text">(.*?)</div>', t, re.S)
        if m:
            docs.append({"doc": f"antoloji/{f.stem}", "site": "antoloji.com", "kind": "user-anthology",
                         "url": f"https://www.antoloji.com/{f.stem}/", "text": page_text(m.group(1))})
    t = (raw / "projetsm/fuzuli_eserleri.html").read_text(encoding="utf-8", errors="replace")
    for r in re.findall(r"<tr.*?</tr>", t, re.S):
        href = re.search(r'href="(/eserler/[^"]+)"', r)
        if href:
            cells = [c for c in (re.sub(r"\s+", " ", html.unescape(re.sub("<[^>]+>", "", c))).strip()
                                 for c in re.findall(r"<td.*?</td>", r, re.S)) if c]
            docs.append({"doc": f"projetsm/{href.group(1).rsplit('/', 1)[1]}", "site": "projetsm.com",
                         "kind": "music-repertoire", "url": "http://projetsm.com" + href.group(1),
                         "text": cells[1], "note": "first line of the güfte only (the score page)"})
    urls = [line.split("\t") for line in (raw / "misc/urls.tsv").read_text().splitlines() if line.strip()]
    for i, (tag, url) in enumerate(urls, 1):
        f = raw / f"misc/{i:02d}_{tag}.html"
        t = f.read_text(encoding="utf-8", errors="replace") if f.exists() else ""
        if tag not in MISC_KINDS or len(t) < 20000 or "Just a moment" in t[:5000]:
            continue
        docs.append({"doc": f"misc/{f.stem}", "site": re.sub(r"^https?://(www\.)?([^/]+).*", r"\2", url),
                     "kind": MISC_KINDS[tag], "url": url, "text": page_text(t)})
    for f in sorted((raw / "archive_org").glob("*_djvu.txt")):
        ident = f.name.split("__")[0]
        site, kind = OCR_KINDS[ident]
        docs.append({"doc": f"archive_org/{ident}", "site": site, "kind": kind,
                     "url": f"https://archive.org/details/{ident}", "text": f.read_text(encoding="utf-8", errors="replace")})
    with open(WS / "documents.jsonl", "w", encoding="utf-8") as out:
        out.writelines(json.dumps(d, ensure_ascii=False) + "\n" for d in docs)
    print(Counter(d["kind"] for d in docs), f"{len(docs)} documents -> {WS / 'documents.jsonl'}")

def lower(s: str) -> str:
    return s.replace("İ", "i").replace("I", "ı").lower()

def fold(s: str) -> str:
    s = unicodedata.normalize("NFD", lower(s).replace("ı", "i"))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[ʿʾʼ’'`´ˇ]", "", s)
    s = re.sub(r"[^a-z]+", " ", s)
    return s.strip()

def corpus():
    misras = []
    for split in ("train", "dev", "test"):
        for line in open(STORE / f"final/{split}.jsonl", encoding="utf-8"):
            r = json.loads(line)
            for ui, u in enumerate(r["units"]):
                for mi, m in enumerate(u["misras"]):
                    misras.append((r["id"], split, r["poet"], ui, mi, m))
    return misras

RESIDUE = [("line number", re.compile(r"^\s*(\(?\d{1,3}[).]?|[IVX]+\.)\s+")),
           ("citation", re.compile(r"\s*\((?=[^)]*\d)[^)]*\)\s*$")),
           ("footnote digit", re.compile(r"(?<=[^\W\d_])\d{1,2}(?=\s|$|[.,;:!?])")),
           ("trailing dash", re.compile(r"\s+[-–—]\s*$"))]

def strip_residue(x: str):
    found = []
    for name, rx in RESIDUE:
        if rx.search(x):
            found.append(name)
            x = rx.sub(" " if name == "footnote digit" else "", x)
    return re.sub(r"\s+", " ", x).strip(), found

def match(args):
    misras = corpus()
    keys = [fold(m[5]) for m in misras]
    index = defaultdict(set)
    df = Counter()
    for i, k in enumerate(keys):
        for w in set(k.split()):
            df[w] += 1
            index[w].add(i)
    by_poem = defaultdict(list)
    for i, m in enumerate(misras):
        by_poem[m[0]].append(i)

    def candidates(k, n=4):
        ws = sorted(set(w for w in k.split() if len(w) > 2 and w in df), key=lambda w: df[w])[:n]
        c = Counter()
        for w in ws:
            if df[w] < 3000:
                c.update(index[w])
        return [i for i, h in c.most_common(30) if h >= min(2, len(ws))]

    def score(a, b):
        return SequenceMatcher(None, a, b, autojunk=False).ratio()

    pairs, structure = [], []
    for d in map(json.loads, open(WS / "documents.jsonl", encoding="utf-8")):
        lines = [re.sub(r"\s+", " ", x).strip() for x in d["text"].split("\n")]
        lines = [x for x in lines if len(fold(x)) >= 12]
        hits = []
        clean = [strip_residue(x) for x in lines]
        for li, (x, _) in enumerate(clean):
            k = fold(x)
            if len(k) > 160 or x.startswith("("):
                continue
            best = (0, None, None)
            for i in candidates(k):
                s = score(k, keys[i])
                if s > best[0]:
                    best = (s, i, "misra")
                m = misras[i]
                for j in (i - 1, i + 1):
                    if 0 <= j < len(misras) and misras[j][0] == m[0] and misras[j][3] == m[3]:
                        a, b = sorted((i, j))
                        s2 = score(k, keys[a] + " " + keys[b])
                        if s2 > best[0] + .05 and len(k) > 1.4 * len(keys[i]):
                            best = (s2, (a, b), "beyit-on-one-line")
            if best[0] >= .75:
                hits.append((li, best[1], best[0], best[2]))
        per_poem = defaultdict(list)
        for h in hits:
            i = h[1][0] if isinstance(h[1], tuple) else h[1]
            per_poem[misras[i][0]].append(h)
        for pid, hs in per_poem.items():
            if len(hs) < 2 and hs[0][2] < .8:
                continue
            best = {}
            for h in hs:
                key = h[1]
                if key not in best or h[2] > best[key][2]:
                    best[key] = h
            hs = sorted(best.values())
            order = []
            for li, i, s, kind in hs:
                idx = list(i) if isinstance(i, tuple) else [i]
                text, residue = clean[li]
                pairs.append({"doc": d["doc"], "site": d["site"], "kind": d["kind"], "poem": pid,
                              "split": misras[idx[0]][1], "poet": misras[idx[0]][2],
                              "unit": misras[idx[0]][3], "misra": misras[idx[0]][4], "layout": kind,
                              "edition": " ".join(misras[j][5] for j in idx), "web": text, "web raw": lines[li],
                              "residue": residue, "similarity": round(s, 3)})
                order.extend(idx)
            n_poem = len(by_poem[pid])
            first, last = min(order), max(order)
            gaps = [lines[a + 1:b] for (a, *_), (b, *_) in zip(hs, hs[1:]) if b > a + 1]
            structure.append({"doc": d["doc"], "kind": d["kind"], "poem": pid, "split": misras[order[0]][1],
                              "poem misras": n_poem, "copied misras": len(set(order)),
                              "web lines": len(hs), "beyit on one line": sum(h[3] != "misra" for h in hs),
                              "out of order": sum(b < a for a, b in zip(order, order[1:])),
                              "skipped inside": sum(1 for i in by_poem[pid] if first < i < last and i not in order),
                              "lines between copied lines": sum(len(g) for g in gaps)})
    with open(WS / "pairs.jsonl", "w", encoding="utf-8") as out:
        out.writelines(json.dumps(p, ensure_ascii=False) + "\n" for p in pairs)
    with open(WS / "structure.jsonl", "w", encoding="utf-8") as out:
        out.writelines(json.dumps(p, ensure_ascii=False) + "\n" for p in structure)
    print(f"{len(pairs)} pairs from {len({p['doc'] for p in pairs})} documents, {len(structure)} poem copies")
    print(Counter(p["kind"] for p in pairs))

from aruz.perturb import (ELISION, IZAFET, edits_of, letters, line_weights, similarity, sites, word_weights,
                          apply_edit, VOWELS)

PROFILE = {"ocr-edition": "ocr", "ocr-commentary": "ocr", "ocr-thesis": "ocr", "ocr-popular-book": "ocr"}
BANDS = [(0., .25), (.25, .5), (.5, .75), (.75, 1.01)]

def module(name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

def folded(text, popular, flags=False):
    out = []
    for raw in text.split():
        w = letters(popular(raw))
        if w:
            out.append((w, bool(IZAFET.match(raw)), bool(ELISION.match(raw))) if flags else w)
    return out

def align(ef, wf):
    n, m = len(ef), len(wf)
    INF = 1e9
    D = [[INF] * (m + 1) for _ in range(n + 1)]
    B = [[None] * (m + 1) for _ in range(n + 1)]
    D[0][0] = 0

    def sub(a, b):
        return 0. if a == b else 1 - similarity(a, b)

    for i in range(n + 1):
        for j in range(m + 1):
            if D[i][j] >= INF:
                continue
            for di, dj in ((1, 1), (1, 0), (0, 1), (2, 1), (1, 2)):
                a, b = i + di, j + dj
                if a > n or b > m:
                    continue
                if (di, dj) == (1, 1):
                    c = sub(ef[i], wf[j])
                elif di + dj == 1:
                    c = .9
                elif (di, dj) == (2, 1):
                    s = sub(ef[i] + ef[i + 1], wf[j])
                    c = s + .1 if s < .15 else INF
                else:
                    s = sub(ef[i], wf[j] + wf[j + 1])
                    c = s + .1 if s < .15 else INF
                if D[i][j] + c < D[a][b]:
                    D[a][b], B[a][b] = D[i][j] + c, (i, j)
    out, i, j = [], n, m
    while (i, j) != (0, 0):
        pi, pj = B[i][j]
        out.append((list(range(pi, i)), list(range(pj, j))))
        i, j = pi, pj
    return out[::-1]

def kind_of(e, w, vocab):
    asc = str.maketrans("çğıöşü", "cgiosu")
    if e.translate(asc) == w.translate(asc):
        return "Turkish letter lost or swapped (ı -> i, ü -> u ...)"
    if re.sub(r"([aeıioöuü])\1", r"\1", w) == e:
        return "long vowel typed double (efgan -> efgaan)"
    sk = lambda x: "".join(c for c in x if c not in VOWELS)
    if sk(e) == sk(w):
        if len(e) == len(w):
            diffs = [i for i, (a, b) in enumerate(zip(e, w)) if a != b]
            first = next((i for i, c in enumerate(e) if c in VOWELS), 0)
            if all(i <= first for i in diffs):
                return "vowel changed in the stem (it- -> et-, ışk -> aşk, milk -> mülk)"
            if all(i >= len(e) - 3 for i in diffs) and len(e) > 3:
                return "vowel changed in the ending (-dur -> -dır, -uñ -> -in, olmış -> olmuş)"
            return "vowel changed"
        return "vowel inserted (aşna -> aşina)" if len(w) > len(e) else "vowel dropped"
    if e[:1] == "m" and w[:1] == "b" and e[1:] == w[1:]:
        return "men/maña written ben/bana"
    if len(e) == len(w) and all(a == b or (a, b) == ("h", "k") for a, b in zip(e, w)):
        return "h written k (yoh -> yok, çoh -> çok)"
    if e[:-1] == w[:-1] and (e[-1:], w[-1:]) in {("d", "t"), ("b", "p"), ("c", "ç"), ("g", "k")}:
        return "final devoicing (derd -> dert)"
    if re.sub(r"(.)\1", r"\1", e) == re.sub(r"(.)\1", r"\1", w):
        return "double consonant " + ("dropped" if len(e) > len(w) else "added")
    if similarity(e, w) >= .5 and edits_of(e, w) is not None:
        return "letter edit, non-word (typo or OCR)" if w not in vocab else "letter edit, another word"
    return "word substituted"

def levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]

def catalogue(args):
    popular = module("eval_meter").popular
    vocab, train_vocab = set(), Counter()
    for split in ("train", "dev", "test"):
        for line in open(STORE / f"final/{split}.jsonl", encoding="utf-8"):
            for u in json.loads(line)["units"]:
                for m in u["misras"]:
                    ws = folded(m, popular)
                    vocab.update(ws)
                    if split == "train":
                        train_vocab.update(ws)
    pairs = [json.loads(x) for x in open(WS / "pairs.jsonl", encoding="utf-8")]
    structure = [json.loads(x) for x in open(WS / "structure.jsonl", encoding="utf-8")]

    per_kind = defaultdict(Counter)
    weight_kind = defaultdict(Counter)
    line_stats = defaultdict(Counter)
    examples = defaultdict(list)
    prof = {k: {"edits": Counter(), "edit weights": Counter(), "sites": Counter(), "rates": defaultdict(Counter),
                "subst": [], "added": [], "doc": defaultdict(Counter), "styles": Counter(), "digits": Counter()}
            for k in ("web", "ocr")}
    keys_seen = set()
    rows_out = []
    for p in pairs:
        src = p["kind"]
        P = prof[PROFILE.get(src, "web")]
        et = folded(p["edition"], popular, flags=True)
        ef, wf = [w for w, _, _ in et], folded(p["web"], popular)
        L = line_stats[src]
        L["lines"] += 1
        L["words"] += len(ef)
        L["letters differ"] += ef != wf
        L["syllable count differs"] += sum(map(len, map(word_weights, ef))) != sum(map(len, map(word_weights, wf)))
        L["weights differ"] += line_weights(ef) != line_weights(wf)
        L["cer num"] += levenshtein(" ".join(ef), " ".join(wf))
        L["cer den"] += len(" ".join(ef))
        for r in p.get("residue", []):
            L["residue: " + r] += 1
        R = P["rates"]
        R["line number"]["eligible"] += 1
        R["citation"]["eligible"] += 1
        R["footnote digit"]["eligible"] += 1
        for r in p.get("residue", []):
            R[{"line number": "line number", "citation": "citation", "footnote digit": "footnote digit"}.get(r, "other")]["applied"] += 1
        raw = p.get("web raw", "")
        mnum = re.match(r"^\s*(\(?\d{1,3}[).]?)\s+", raw)
        if mnum:
            P["styles"][re.sub(r"\d+", "N", mnum.group(1))] += 1
        for d in re.findall(r"(?<=[^\W\d_])(\d{1,2})(?=\s|$|[.,;:!?])", raw):
            P["digits"][len(d)] += 1
        doc = P["doc"][p["doc"]]
        doc["words"] += len(ef)
        found = []
        for ei, wi in align(ef, wf):
            e = " ".join(ef[i] for i in ei)
            w = " ".join(wf[j] for j in wi)
            changed_w = line_weights([ef[i] for i in ei]).replace(" ", "") != line_weights([wf[j] for j in wi]).replace(" ", "")
            if len(ei) == 1 and len(wi) == 1:
                e1, w1 = ef[ei[0]], wf[wi[0]]
                _, iz, el = et[ei[0]]
                R["izafet dropped"]["eligible"] += iz
                R["substitution, weights same"]["eligible"] += 1
                R["substitution, weights changed"]["eligible"] += 1
                R["word split"]["eligible"] += len(e1) >= 4
                for left, frm, right, at, n in sites(e1):
                    P["sites"][left, frm, right] += 1
                if e1 == w1:
                    continue
                k = kind_of(e1, w1, vocab)
                ed = edits_of(e1, w1)
                if iz and ed and all(frm and not to and right.startswith("$") for _, frm, to, right, _ in ed) \
                        and len(e1) - len(w1) <= 2:
                    k = "izafet dropped"
                    R["izafet dropped"]["applied"] += 1
                elif k == "word substituted" or ed is None or similarity(e1, w1) < .5:
                    k = "word substituted"
                    same = word_weights(e1) == word_weights(w1)
                    R[f"substitution, weights {'same' if same else 'changed'}"]["applied"] += 1
                    sim = similarity(e1, w1)
                    P["subst"].append({"similarity band": next(b for b in BANDS if b[0] <= sim < b[1]),
                                       "syllable difference": len(word_weights(w1)) - len(word_weights(e1)),
                                       "same weights": same})
                else:
                    for left, frm, to, right, at in ed:
                        P["edits"][left, frm, to, right] += 1
                        single = apply_edit(e1, at, len(frm), to)
                        P["edit weights"][left, frm, to, right] += word_weights(single) != word_weights(e1)
                found.append((k, e1, w1, changed_w))
            elif not wi:
                R["word dropped"]["applied"] += 1
                found.append(("word dropped", e, "", changed_w))
            elif not ei:
                R["word added"]["applied"] += 1
                P["added"].append(len(w))
                found.append(("word added", "", w, changed_w))
            elif len(ei) == 2:
                R["words joined"]["applied"] += 1
                found.append(("words joined", e, w, changed_w))
            else:
                _, iz, el = et[ei[0]]
                if el and wf[wi[0]] in ("ki", "ne", "çü"):
                    R["elision written out"]["applied"] += 1
                    found.append(("elision written out", e, w, changed_w))
                else:
                    R["word split"]["applied"] += 1
                    found.append(("word split", e, w, changed_w))
        R["word dropped"]["eligible"] += len(ef)
        R["word added"]["eligible"] += len(ef)
        R["words joined"]["eligible"] += max(0, len(ef) - 1)
        R["elision written out"]["eligible"] += sum(el for _, _, el in et)
        doc["deviations"] += len(found)
        for k, e, w, ch in found:
            per_kind[src][k] += 1
            weight_kind[src][k] += ch
            if len(examples[k]) < 6 and (e, w) not in [x[:2] for x in examples[k]]:
                examples[k].append((e, w))
        rows_out.append({**p, "deviations": [list(x) for x in found]})
    for s in structure:
        P = prof[PROFILE.get(s["kind"], "web")]
        P["rates"]["beyit left out"]["applied"] += s["skipped inside"] / 2
        P["rates"]["beyit left out"]["eligible"] += (s["skipped inside"] + s["copied misras"]) / 2
    with open(WS / "pairs_classified.jsonl", "w", encoding="utf-8") as out:
        out.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows_out)

    profiles = {}
    for name, P in prof.items():
        table = []
        for (left, frm, to, right), n in P["edits"].items():
            elig = P["sites"][left, frm, right]
            if n < 2 or not elig:
                continue
            table.append({"left": left, "from": frm, "to": to, "right": right, "applied": n, "eligible": elig,
                          "group": "damage" if P["edit weights"][left, frm, to, right] * 2 >= n else "forms"})
        rates = {k: {"applied": v["applied"], "eligible": v["eligible"]} for k, v in P["rates"].items() if k != "other"}
        if name == "ocr":
            rates["beyit left out"] = {"applied": 0, "eligible": 0}
        total = sum(d["deviations"] for d in P["doc"].values()) / max(1, sum(d["words"] for d in P["doc"].values()))
        intens = sorted(round(d["deviations"] / d["words"] / total, 3) for d in P["doc"].values() if d["words"] >= 50)
        digits = P["digits"]
        profiles[name] = {"word edits": sorted(table, key=lambda r: -r["applied"]), "rates": rates,
                          "substitutions": P["subst"], "added word lengths": P["added"] or [2],
                          "source intensities": intens or [1.],
                          "line number styles": dict(P["styles"]) or {"N": 1},
                          "footnote one digit": digits[1] / max(1, digits[1] + digits[2]),
                          "measured": {}}
    for src, L in line_stats.items():
        M = profiles[PROFILE.get(src, "web")]["measured"]
        for k in ("lines", "words", "letters differ", "syllable count differs", "weights differ", "cer num", "cer den"):
            M[k] = M.get(k, 0) + L[k]
        M["deviations"] = M.get("deviations", 0) + sum(per_kind[src].values())
    for name in profiles:
        M = profiles[name]["measured"]
        profiles[name]["measured"] = {"mısras": M["lines"], "edition words": M["words"],
                                      "word deviations per word": round(M["deviations"] / M["words"], 4),
                                      "mısras whose letters differ": round(M["letters differ"] / M["lines"], 4),
                                      "mısras whose syllable count differs": round(M["syllable count differs"] / M["lines"], 4),
                                      "mısras whose weights differ": round(M["weights differ"] / M["lines"], 4),
                                      "character error rate": round(M["cer num"] / M["cer den"], 4)}
    out = ROOT / "data/robustness/profiles.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"made by": "scripts/web_survey.py catalogue", "profiles": profiles},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"profiles -> {out}: " + ", ".join(f"{k}: {len(v['word edits'])} word edits" for k, v in profiles.items()))

    kinds = sorted(line_stats, key=lambda k: -line_stats[k]["words"])
    devs = sorted({d for v in per_kind.values() for d in v}, key=lambda d: -sum(per_kind[k][d] for k in kinds))
    md = ["| deviation (after the alphabet fold) | " + " | ".join(kinds) + " | changes weights (%) | example (edition → copy) |",
          "|---|" + "---|" * len(kinds) + "---|---|",
          "| *mısras / edition words* | " + " | ".join(f"{line_stats[k]['lines']} / {line_stats[k]['words']}" for k in kinds) + " | | |"]
    for d in devs:
        tot = sum(per_kind[k][d] for k in kinds)
        ch = sum(weight_kind[k][d] for k in kinds)
        ex = "; ".join(f"{a} → {b}" for a, b in examples[d][:3])
        md.append(f"| {d} | " + " | ".join(f"{100 * per_kind[k][d] / line_stats[k]['words']:.2f}" for k in kinds)
                  + f" | {100 * ch / max(1, tot):.0f} | {ex} |")
    md += ["", "| line level (% of mısras) | " + " | ".join(kinds) + " |", "|---|" + "---|" * len(kinds)]
    for s in ("letters differ", "syllable count differs", "weights differ", "residue: line number",
              "residue: citation", "residue: footnote digit"):
        md.append(f"| {s} | " + " | ".join(f"{100 * line_stats[k][s] / line_stats[k]['lines']:.0f}" for k in kinds) + " |")
    md.append("| character error rate | " + " | ".join(
        f"{100 * line_stats[k]['cer num'] / max(1, line_stats[k]['cer den']):.1f}" for k in kinds) + " |")
    st = defaultdict(Counter)
    for s in structure:
        k = s["kind"]
        st[k]["poem copies"] += 1
        st[k]["whole poem"] += s["copied misras"] == s["poem misras"]
        st[k]["beyits left out inside the copy"] += s["skipped inside"] > 0
        st[k]["beyits out of order"] += s["out of order"] > 0
        st[k]["other lines between"] += s["lines between copied lines"] > 0
    md += ["", "| poem copies | " + " | ".join(kinds) + " |", "|---|" + "---|" * len(kinds)]
    for s in ("poem copies", "whole poem", "beyits left out inside the copy", "beyits out of order", "other lines between"):
        md.append(f"| {s} | " + " | ".join(str(st[k][s]) for k in kinds) + " |")
    (ROOT / "docs/web_survey_tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (WS / "catalogue.json").write_text(json.dumps({"deviations": {k: dict(v) for k, v in per_kind.items()},
                                                   "changed weights": {k: dict(v) for k, v in weight_kind.items()},
                                                   "lines": {k: dict(v) for k, v in line_stats.items()},
                                                   "examples": dict(examples)}, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
    print("\n".join(md))

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=["extract", "match", "catalogue"])
    args = parser.parse_args()
    {"extract": extract, "match": match, "catalogue": catalogue}[args.step](args)

if __name__ == "__main__":
    main()
