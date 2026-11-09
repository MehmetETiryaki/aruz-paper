import argparse
import hashlib
import re
import json
import os
import subprocess
import unicodedata
from collections import Counter

import fitz
from dataclasses import asdict
from pathlib import Path

from aruz.editions import clean_title, meter_id, parse
from aruz.validate import check

ROOT = Path(__file__).resolve().parent.parent

def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)

def read_tsv(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            yield [nfc(f) for f in line.split("\t")]

NOTES = {"editor's note: meter broken": re.compile(r"(?i)vezn[iı]\s+bozuk|vezin\s+bozuk|vezne\s+uymuyor|vezniyle\s+yazılmıştır"
                                                   r"|vez(?:in|ni)\s+aksa|vezin\s+(?:ve\s+\S+\s+)?çıkma|vezin\s+hata"
                                                   r"|vezin\s+uyumsuz|makb[ûu]z|zihaf"),
         "editor's note: attribution doubtful": re.compile(r"(?i)bulunmaması\s+kuvvetle\s+muhtemel|ait\s+olma\s+ihtimal"
                                                           r"|aidiyet\S*\s+(?:\S+\s+){0,3}şüphe|sonradan\s+eklenmiş\s+olması\s+muhtemel"
                                                           r"|ait\s+olmayan|ait\s+olmadığı"),
         "editor's note: variant text": re.compile(r"(?i)bölüm\s+mevcuttur"),
         "editor's note: text changed by the editor": re.compile(r"(?i)de[ğġ]iştiril|aktarılmıştır|hiçbirinde\s+bu|neşirlerine\s+tabi"
                                                                 r"|(?:TD|aruz|vezin|anlam)\S*\s+(?:\S+\s+){0,12}?tercih"
                                                                 r"|tercih\S*\s+doğrultusunda"),
         "editor's note: struck out in the manuscript": re.compile(r"(?i)üzeri\s+çizil"
                                                                    r"|(?:beyit\S*|mısra\S*)(?:\s+\S+){0,2}\s+silinmi[şs]")}
REF = re.compile(r"(?:^|\s)(\d+)([ab]{1,2})?\s*[-.]?\s*(?:Bu\s+\S+\s+|Mısra\s+|[A-D]\.?(?:\s+ve\s+[A-D])?\s+mısra\S*\s+[^.]*?)?$")

def editor_notes(pdf: Path):
    notes = {}
    for n, page in enumerate(fitz.open(pdf), 1):
        rows = nfc(page.get_text()).split("\n")
        for j, row in enumerate(rows):
            for why, pattern in NOTES.items():
                for m in pattern.finditer(row):
                    ref = REF.search(row[:m.start()])
                    if ref:
                        kind, k = ("beyit" if ref.group(2) else "marker"), int(ref.group(1))
                    elif j and re.match(r"^\s*(\d+)\.?\s*$", rows[j - 1]):
                        kind, k = "marker", int(rows[j - 1].strip(" .\t"))
                    else:
                        kind, k = "page", None
                    notes.setdefault(n, []).append((kind, k, why))
    return notes

def md5(path: Path) -> str:
    if not path.exists():
        raise SystemExit(f"missing {path}")
    return hashlib.md5(path.read_bytes()).hexdigest()

def main() -> None:
    if "STORE" not in os.environ:
        raise SystemExit("STORE is not set (see ~/.bashrc)")
    store = Path(os.environ["STORE"])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=Path, default=store / "projects/aruz/latoc/YOUR_PATH/output_xml_cleaned")
    parser.add_argument("--pdfs", type=Path, default=store / "datasets/ottoman_poetry/latoc/pdf")
    parser.add_argument("--out", type=Path, default=store / "datasets/ottoman_poetry/processed/latoc.jsonl")
    parser.add_argument("--allow-other-pdfs", action="store_true",
                        help="go on when a PDF's MD5 differs from data/books.tsv (another edition or printing: the "
                             "corpus will differ from the paper's)")
    args = parser.parse_args()
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "src", "configs", "scripts"],
                                capture_output=True, text=True).stdout.strip())

    books = {b: (poet, int(century), split) for b, poet, century, split in read_tsv(ROOT / "configs/splits.tsv")}
    overlay = json.loads((ROOT / "configs/latoc_overlay.json").read_text(encoding="utf-8"))
    glyphs = {}
    for b, pattern, repl in read_tsv(ROOT / "configs/glyph_context.tsv"):
        glyphs.setdefault(b, []).append((pattern, repl))
    silent, context = {}, {}
    for row in read_tsv(ROOT / "configs/orthography.tsv"):
        silent[row[0]], context[row[0]] = (row[1] if len(row) > 1 else ""), (row[2] if len(row) > 2 else "")
    files = {nfc(p.stem): p.stem for p in args.xml.glob("*.xml")}
    found = sorted(files)
    table = ROOT / "data/books.tsv"
    if table.exists():
        expected = {nfc(row[0]): row[5] for row in read_tsv(table) if row[0] != "book"}
        other = [f"{b}: {md5(args.pdfs / f'{files[b]}.pdf')} (expected {expected[b]})" for b in found
                 if b in expected and md5(args.pdfs / f"{files[b]}.pdf") != expected[b]]
        if other:
            msg = "PDFs that differ from the paper's (data/books.tsv):\n  " + "\n  ".join(other)
            if not args.allow_other_pdfs:
                raise SystemExit(msg + "\n(--allow-other-pdfs to go on with them)")
            print("warning: " + msg)
    unlisted = [b for b in found if b not in books or b not in silent]
    if unlisted:
        raise SystemExit(f"books missing from configs/splits.tsv or configs/orthography.tsv: {unlisted}")

    attested = {}
    for b, glyph, letter in read_tsv(ROOT / "configs/attested_glyphs.tsv"):
        attested.setdefault(b, []).append((glyph, letter))
    notes = {b: editor_notes(args.pdfs / f"{files[b]}.pdf") for b in found}
    for b in found:
        print(f"{b}: editor's notes {Counter((why, k) for ns in notes[b].values() for k, _, why in ns)}")
    parsed = {b: parse(args.xml / f"{files[b]}.xml", glyphs.get(b, []), notes[b]) for b in found}
    if attested:
        def wkey(w):
            w = re.split(r"[-'’]", w)[0]
            return "".join(c for c in unicodedata.normalize("NFD", w.lower()) if c.isalpha() or c in "ʿʾ\u0327\u0306")
        vocab = {}
        for b, (poems, _) in parsed.items():
            for p in poems:
                for _, ms in p.units:
                    for m in ms:
                        for w in m.split():
                            vocab.setdefault(wkey(w.strip(".,;:!?“”\"'’")), set()).add(b)
        for b, rules in attested.items():
            changed = Counter()
            for p in parsed[b][0]:
                for _, ms in p.units:
                    for i, m in enumerate(ms):
                        words, uncertain = m.split(), set()
                        for j, w in enumerate(words):
                            for glyph, letter in rules:
                                if not w.startswith(glyph) or len(w) < 3:
                                    continue
                                cand = letter + w[1].upper() + w[2:]
                                elsewhere = any(o != b for o in vocab.get(wkey(cand.strip(".,;:!?'’")), ()))
                                if elsewhere and not any(o != b for o in vocab.get(wkey(w.strip(".,;:!?'’")), ())):
                                    changed[f"{w} -> {cand}"] += 1
                                    words[j] = cand
                                elif not elsewhere and not any(o != b for o in vocab.get(wkey(w.strip(".,;:!?'’")), ())):
                                    uncertain.add(i)
                        if " ".join(words) != m and m in p.notes:
                            p.notes[" ".join(words)] = p.notes.pop(m)
                        ms[i] = " ".join(words)
                        if i in uncertain:
                            p.notes[ms[i]] = f"letter uncertain ({glyph} may be {letter})"
            print(f"{b}: attested glyph changes {sum(changed.values())}: {changed.most_common(12)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    unknown: Counter = Counter()
    all_recs = []
    renumbered = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for b in found:
            poet, century, split = books[b]
            poems, stats = parsed[b]
            shapes = Counter(len(m) for p in poems for _, m in p.units)
            named = 0
            recs = []
            for k, p in enumerate(poems):
                mid = meter_id(p.pattern)
                named += mid is not None
                if mid is None:
                    unknown[p.pattern, p.meter_raw] += 1
                rec = {"id": f"{b}#{k}", "book": b, "poet": poet, "century": century, "split": split,
                       "meter": mid, "silent": silent[b], "context": context[b], **asdict(p)}
                rec["meter_raw"] = re.sub(r"(?<=[^\W\d_])\d+", "", rec["meter_raw"])
                rec["meter_raw"] = re.sub(r"^[\[(]|[\])]$", "", " ".join(rec["meter_raw"].split()))
                rec["meter_raw"] = re.sub(r"(l[üu]n|tün)[a-zı]\b", r"\1", rec["meter_raw"])
                rec["title"] = clean_title(rec["title"]) if rec["title"] != ["(verse that no poem took)"] else rec["title"]
                rec["units"] = [{"n": n, "misras": ms} for n, ms in p.units]
                for i, u in enumerate(rec["units"][:-1]):
                    prev = rec["units"][i - 1]["n"] if i else 0
                    if u["n"] == rec["units"][i + 1]["n"] and prev < u["n"] - 1:
                        u["n"] -= 1
                        renumbered += 1
                recs.append(rec)
            health = check(recs)
            all_recs += recs
            with open(args.out.with_name(f"unassigned_{b}.txt"), "w", encoding="utf-8") as un:
                un.writelines(f"p{pg}\t{line}\n" for pg, line in stats.unassigned)

            summary = {"poems": len(poems), "units": stats["units"], "with_inventory_meter": named,
                       "misras_per_unit": dict(sorted(shapes.items())),
                       "damaged_lines": sum(p.damaged for p in poems),
                       "without_number": sum(1 for p in poems if p.number is None),
                       "empty": sum(1 for p in poems if not p.units),
                       "meters": dict(Counter(p.pattern for p in poems).most_common()),
                       "health": dict(sorted(health.items()))}
            manifest = {"book": b, "poet": poet, "century": century, "split": split,
                        "pdf_md5": md5(args.pdfs / f"{files[b]}.pdf"), "xml_md5": md5(args.xml / f"{files[b]}.xml"),
                        "latoc_overlay": {k.split("|", 1)[1]: v for k, v in overlay.items() if nfc(k).startswith(b + ".xml|")}, "glyph_context": glyphs.get(b, []), "attested_glyphs": attested.get(b, []), "silent": silent[b], "context": context[b], "git_commit": commit, "git_dirty": dirty,
                        "stats": summary}
            (ROOT / "data/manifests").mkdir(parents=True, exist_ok=True)
            with open(ROOT / f"data/manifests/{b}.json", "w", encoding="utf-8") as m:
                json.dump(manifest, m, ensure_ascii=False, indent=2)
            print(f"{b} [{split}]: {len(poems)} poems, {stats['units']} units, "
                  f"{named} with an inventory meter, mısras per unit {dict(sorted(shapes.items()))}, "
                  f"{sum(p.damaged for p in poems)} damaged lines, "
                  f"{sum(1 for p in poems if p.number is None)} without a number, "
                  f"{sum(1 for p in poems if not p.units)} empty, {dict(stats)}")
            print(f"    usable units: {health['usable units']}/{health['units']} "
                  f"({health['usable units'] / max(health['units'], 1):.1%})")
            for k, v in sorted(health.items(), key=lambda x: -x[1]):
                if k not in ("units", "usable units"):
                    print(f"      {v:6d}  {k}")
        where = {}
        for rec in all_recs:
            for u in rec["units"]:
                for m in u["misras"]:
                    k = "".join(c for c in unicodedata.normalize("NFD", m.lower()) if c.isalpha() and not unicodedata.combining(c))
                    if len(k) >= 20:
                        where.setdefault(k, set()).add(rec["book"])
        shared = 0
        for rec in all_recs:
            for u in rec["units"]:
                ks = ["".join(c for c in unicodedata.normalize("NFD", m.lower()) if c.isalpha() and not unicodedata.combining(c)) for m in u["misras"]]
                if any(len(where.get(k, ())) > 1 for k in ks):
                    u["issues"] = sorted(set(u["issues"]) | {"shared with another book"})
                    shared += 1
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\nunits sharing a mısra with another book: {shared}")
    print(f"beyit numbers repaired (repeated, the one before missing): {renumbered}")
    print(f"wrote {args.out}")
    print("meters not in the inventory:")
    for (pattern, raw), n in unknown.most_common():
        print(f"  {n:4d}  {pattern:20s} {raw}")

if __name__ == "__main__":
    main()
