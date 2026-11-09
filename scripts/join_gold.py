import argparse
import json
import os
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from task2_metrics import stream

def letter_units(text):
    nfd = unicodedata.normalize("NFD", text.replace("İ", "i").replace("I", "ı"))
    units, k = [], 0
    while k < len(nfd):
        c = nfd[k]
        if c.isalpha() or c in "ʿʾʼ":
            j = k + 1
            while j < len(nfd) and unicodedata.combining(nfd[j]):
                j += 1
            units.append(nfd[k:j])
            k = j
        else:
            k += 1
    return units

def gold_units(text, edits):
    units = letter_units(text)
    if not edits:
        return units
    out, at = [], 0
    for tag, i1, i2, repl in edits:
        out += units[at:i1]
        if tag in ("replace", "insert"):
            out += list(repl)
        at = i2
    return out + units[at:]

def rebuild(label, text):
    units = gold_units(text, label["stream_edits"])
    sylls, at = [], 0
    for n, slots, kusur in label["syllables"]:
        sylls.append([unicodedata.normalize("NFC", "".join(units[at:at + n])), slots, kusur])
        at += n
    if at != len(units):
        raise SystemExit(f"{label['item']}: {at} letters in the labels, {len(units)} in the line")
    return {"id": label["id"], "unit": label["unit"], "misra": label["misra"], "text": text, "meter": label["meter"],
            "variants": label["variants"], "annotated_text": text, "syllables": sylls, "item": label["item"]}

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, default=Path(os.environ["STORE"]) / "datasets/ottoman_poetry/final")
    ap.add_argument("--out", type=Path, default=ROOT / "data/annotation")
    args = ap.parse_args()
    for split in ("dev", "test"):
        lines = {}
        for r in map(json.loads, open(args.corpus / f"{split}.jsonl", encoding="utf-8")):
            for u in r["units"]:
                for k, m in enumerate(u["misras"], 1):
                    lines[r["id"], u["n"], k] = m
        gold = []
        for label in map(json.loads, open(ROOT / f"data/annotation/{split}_gold_labels.jsonl", encoding="utf-8")):
            key = (label["id"], label["unit"], label["misra"])
            if key not in lines:
                raise SystemExit(f"{label['item']}: line {key} is not in {args.corpus / f'{split}.jsonl'}")
            gold.append(rebuild(label, lines[key]))
        p = args.out / f"{split}_gold.jsonl"
        p.write_text("".join(json.dumps(g, ensure_ascii=False) + "\n" for g in gold), encoding="utf-8")
        print(f"{p}: {len(gold)} gold lines")

if __name__ == "__main__":
    main()
