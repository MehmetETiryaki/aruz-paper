import argparse
import json
import os
import shutil
import time
import unicodedata
from pathlib import Path

LATOC = Path(__file__).resolve().parent.parent / "third_party" / "latoc"
STEPS = {1: "1_pdf_extractor.py", 2: "2_preprocessing.py", 3: "3_xml_cleaner.py"}

def main() -> None:
    if "STORE" not in os.environ:
        raise SystemExit("STORE is not set (see ~/.bashrc)")
    store = Path(os.environ["STORE"])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdfs", type=Path, default=store / "datasets/ottoman_poetry/latoc/pdf")
    parser.add_argument("--work", type=Path, default=store / "projects/aruz/latoc")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"])
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 2, 3], choices=sorted(STEPS))
    args = parser.parse_args()
    missing = [f for f in list(STEPS.values()) + ["controversy_cache.json", "exclude_pages.txt"] if not (LATOC / f).exists()]
    if missing:
        raise SystemExit(f"LATOC is not in {LATOC} ({', '.join(missing)} missing): download the files of "
                         "Zenodo record 10.5281/zenodo.19445113 into that folder, unchanged (see README)")

    pdfs = sorted(args.pdfs.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"no PDFs in {args.pdfs}")
    excluded = {}
    for line in (LATOC / "exclude_pages.txt").read_text(encoding="utf-8").splitlines():
        if ".xml:" in line:
            name = line.split(".xml:")[0]
            excluded[unicodedata.normalize("NFC", name)] = name
    for pdf in pdfs:
        exact = excluded.get(unicodedata.normalize("NFC", pdf.stem))
        if exact is None:
            print(f"warning: {pdf.stem} has no entry in exclude_pages.txt; its front matter will be kept")
        elif exact != pdf.stem:
            raise SystemExit(f"{pdf.name}: name differs from exclude_pages.txt only in Unicode form; "
                             f"rename it to {exact!r}.pdf")
    root = args.work / "YOUR_PATH"
    root.mkdir(parents=True, exist_ok=True)
    if not (root / "input").exists():
        (root / "input").symlink_to(args.pdfs.resolve())
    shutil.copy2(LATOC / "exclude_pages.txt", root / "exclude_pages.txt")
    cache = json.loads((LATOC / "controversy_cache.json").read_text(encoding="utf-8"))
    overlay = json.loads((LATOC.parent.parent / "configs/latoc_overlay.json").read_text(encoding="utf-8"))
    overlay.pop("_comment", None)
    for key, value in overlay.items():
        print(f"overlay: {key} -> {value!r} (LATOC had {cache.get(key, 'nothing')!r})")
    cache.update(overlay)
    (root / "controversy_cache.json").write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(pdfs)} PDFs in {args.pdfs}; working in {args.work}")

    if args.device == "cuda" and 1 in args.steps:
        import torch
        if not torch.cuda.is_available():
            raise SystemExit("CUDA is not available on this node; refusing to run (LATOC would only log per-book errors)")

    os.chdir(args.work)
    started = time.time()
    for step in args.steps:
        path = LATOC / STEPS[step]
        source = path.read_text(encoding="utf-8")
        if step == 1:
            if source.count('DEVICE = "mps"') != 1:
                raise SystemExit(f"can't find the DEVICE line in {path}; has the script changed?")
            source = source.replace('DEVICE = "mps"', f'DEVICE = "{args.device}"')
        print(f"\n=== step {step}: {path.name}")
        exec(compile(source, str(path), "exec"), {"__name__": "__main__", "__file__": str(path)})
        if step == 1:
            out = root / "output_pdf_extractor"
            stale = [p.stem for p in pdfs
                     if not (out / f"{p.stem}_layout.json").exists() or (out / f"{p.stem}_layout.json").stat().st_mtime < started]
            if stale:
                raise SystemExit(f"step 1 produced no new layout for {len(stale)} PDFs: {stale}")

if __name__ == "__main__":
    main()
