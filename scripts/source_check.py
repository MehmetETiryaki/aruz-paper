import argparse
import importlib.util
import json
import os
import re
import unicodedata
from collections import Counter
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parent.parent
TEFILE = re.compile(r"(?<!\S)(?:mef|fa|fā|fe|müs|müf|müt)\S{0,8}(?:lün|lü|tün|tü)(?!\S)")
MARKS = re.compile(r"[ʾ’'`ˇ´΄ʼ]")

def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)

def latoc(work: Path):
    cwd = os.getcwd()
    os.chdir(work)
    spec = importlib.util.spec_from_file_location("pre", ROOT / "third_party/latoc/2_preprocessing.py")
    pre = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pre)
    pre.load_priority_rules()
    pre.load_controversy_cache()
    os.chdir(cwd)
    return pre

def norm(s: str) -> str:
    s = nfc(s).replace("\xad", "").replace("\uf020", " ").replace("\u0320", "\u0331").replace("C", "c")
    s = re.sub("[ʿʻ῾ʽ˓‘]", "c", s)
    s = re.sub(r"(?<=[^\W\d_])\d+", "", s)
    s = MARKS.sub("", s)
    s = re.sub(r"[\[\]()]", "", s)
    s = re.sub(r"[\d*∗“”\"«»?]", " ", s)
    s = re.sub(r"(?<=[^\W\d_])\s*-\s*(?=[^\W\d_])", "-", s)
    s = re.sub(r"(?<![^\W\d_])-|-(?![^\W\d_])|[–—/+]", " ", s)
    s = re.sub(r"[.,;:!]", " ", s)
    s = " ".join(s.lower().split())
    s = re.sub(r"(?:(?<=[-\s])|^|(?<=üstü))ḫv(?=[āīaeıiouü])", "ḫ", s)
    return " " + s + " "

def main() -> None:
    store = Path(os.environ["STORE"])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=store / "datasets/ottoman_poetry/processed/latoc.jsonl")
    parser.add_argument("--pdfs", type=Path, default=store / "datasets/ottoman_poetry/latoc/pdf")
    parser.add_argument("--work", type=Path, default=store / "projects/aruz/latoc/YOUR_PATH")
    args = parser.parse_args()
    pre = latoc(args.work.parent)
    glyphs = {}
    for line in (ROOT / "configs/glyph_context.tsv").read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            b, pattern, repl = [nfc(f) for f in line.split("\t")]
            glyphs.setdefault(b, []).append((pattern, repl))
    independent = {}
    for line in (ROOT / "configs/independent_glyphs.tsv").read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            b, raw, letter = [nfc(f) for f in line.split("\t")]
            independent.setdefault(b, []).append((raw, letter))
    pdfs = {nfc(p.stem): p for p in args.pdfs.glob("*.pdf")}
    recs = [json.loads(l) for l in open(args.corpus, encoding="utf-8")]
    out, stats = {}, Counter()
    for book in sorted({r["book"] for r in recs}):
        doc, key, pages = fitz.open(pdfs[book]), pre.get_file_key(book), {}
        own = sorted(independent.get(book, []), key=lambda x: -len(x[0]))
        table = dict(own)

        def page(i, reading):
            if (i, reading) not in pages:
                text = nfc(doc[i - 1].get_text()) if 1 <= i <= len(doc) else ""
                if reading == "latoc":
                    text = pre.apply_certain_rules(pre.apply_controversy_cache(pre.apply_priority_rules(text, key), key))
                elif own:
                    text = re.sub("|".join(re.escape(raw) for raw, _ in own), lambda m: table[m.group(0)], text)
                lines = []
                for l in text.splitlines():
                    for pattern, repl in glyphs.get(book, []) if reading == "latoc" or book.startswith("sünbülzade") else []:
                        l = re.sub(pattern, repl, l)
                    lines.append(l)
                pages[i, reading] = " ".join(norm(l) + "¶" for l in lines)
            return pages[i, reading]

        def verdict(k, text):
            flat = re.sub(r"\s*¶\s*", " ", text)
            if len(k.strip()) < 8 or re.search(re.escape(k[:-1]) + r"\s*¶", text):
                return None
            if k in flat:
                return "does not end where a source line ends (a word cut off?)"
            if re.search(re.escape(k[:-1]) + r"[^\W\d_]", flat):
                return "cut off (the source line goes on)"
            return "not found in the source pages"

        for r in (r for r in recs if r["book"] == book):
            span = range(r["page"] - 1, r["page"] + len(r["units"]) // 5 + 4)
            texts = {reading: " ⁂ ".join(page(i, reading) for i in span) for reading in ("latoc", "independent")}
            text, cursor, where = texts["latoc"], 0, []
            for u in r["units"]:
                spots = []
                for m in u["misras"]:
                    k = norm(m)
                    why = verdict(k, texts["latoc"])
                    if why is None and verdict(k, texts["independent"]):
                        why = "not confirmed by the independent glyph reading"
                    if why:
                        out.setdefault(r["id"], {}).setdefault(str(u["n"]), []).append([m, why])
                    stats[book, why or "found"] += 1
                    hit = None if why or len(k.strip()) < 8 else \
                        re.compile(re.escape(k[:-1]) + r"\s*¶").search(text, cursor) or re.search(re.escape(k[:-1]) + r"\s*¶", text)
                    spots.append(hit.span() if hit else None)
                    if hit:
                        cursor = hit.end()
                for (a, b), m in zip(zip(spots, spots[1:]), u["misras"][1:]):
                    gap = text[a[1]:b[0]] if a and b else ""
                    if a and b and (b[0] < a[1] or "⁂" not in gap and any(len(w) > 2 for w in re.findall(r"[^\W\d_]+", gap))):
                        why = "mısras not consecutive in the PDF"
                        out.setdefault(r["id"], {}).setdefault(str(u["n"]), []).append([m, why])
                        stats[book, why] += 1
                where.append((spots[0], spots[-1]))
            for (_, a), (b, _) in zip(where, where[1:]):
                if a and b and b[0] > a[1] and len(TEFILE.findall(text[a[1]:b[0]])) >= 2:
                    out.setdefault(r["id"], {})["poem"] = "a vezin printed between two units (two poems merged)"
                    stats[book, "poems with a vezin printed inside"] += 1
                    break
    (ROOT / "data/source_check.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, sort_keys=True),
                                                 encoding="utf-8")
    for book in sorted({b for b, _ in stats}):
        n = {w: stats[book, w] for w in ("found", "cut off (the source line goes on)", "does not end where a source line ends (a word cut off?)",
                                         "not found in the source pages", "not confirmed by the independent glyph reading",
                                         "mısras not consecutive in the PDF", "poems with a vezin printed inside")}
        print(f"{book}: {n}")

if __name__ == "__main__":
    main()
