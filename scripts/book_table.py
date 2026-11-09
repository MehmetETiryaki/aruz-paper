import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def main():
    sources = json.loads((ROOT / "data/book_sources.json").read_text(encoding="utf-8"))
    rows = []
    for p in sorted((ROOT / "data/manifests").glob("*.json")):
        m = json.loads(p.read_text(encoding="utf-8"))
        s = sources.get(m["book"]) or {}
        if not s.get("landing_page"):
            raise SystemExit(f"no landing page recorded for {m['book']!r} in data/book_sources.json")
        rows.append([m["book"], m["poet"], str(m["century"]), m["split"], f"{m['book']}.pdf", m["pdf_md5"], s["landing_page"]])
    head = ["book", "poet", "century", "split", "pdf file name", "pdf md5", "download page"]
    out = ROOT / "data/books.tsv"
    out.write_text("\n".join("\t".join(r) for r in [head] + rows) + "\n", encoding="utf-8")
    print(f"wrote {out}: {len(rows)} books")

if __name__ == "__main__":
    main()
