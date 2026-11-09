import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from aruz.validate import POEM_RISKS, UNIT_RISKS

ROOT = Path(__file__).resolve().parent.parent
KEEP = ["id", "book", "poet", "century", "split", "number", "page", "meter", "meter_raw", "variants",
        "vezin_note", "inherited", "title", "silent", "context"]

def main() -> None:
    store = Path(os.environ["STORE"])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=store / "datasets/ottoman_poetry/processed/latoc.jsonl")
    parser.add_argument("--out", type=Path, default=store / "datasets/ottoman_poetry/final")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    label_check = ROOT / "data/label_check.json"
    if not label_check.exists():
        raise SystemExit("data/label_check.json is missing: run scripts/label_check.py first")
    suspect_labels = json.loads(label_check.read_text(encoding="utf-8"))
    source_check = ROOT / "data/source_check.json"
    if not source_check.exists():
        raise SystemExit("data/source_check.json is missing: run scripts/source_check.py first")
    unverified = json.loads(source_check.read_text(encoding="utf-8"))
    fit_check = ROOT / "data/fit_check.json"
    if not fit_check.exists():
        raise SystemExit("data/fit_check.json is missing: run scripts/fit_check.py first")
    misfit = json.loads(fit_check.read_text(encoding="utf-8"))
    macron_check = ROOT / "data/macron_check.json"
    if not macron_check.exists():
        raise SystemExit("data/macron_check.json is missing: run scripts/macron_check.py first")
    misprint = json.loads(macron_check.read_text(encoding="utf-8"))
    curated = {}
    for line in open(ROOT / "configs/excluded_units.tsv", encoding="utf-8"):
        if line.strip() and not line.startswith("#"):
            rid, n, first, why = line.rstrip("\n").split("\t")
            curated[rid, None if n == "*" else int(n)] = (first, why)
    fixes = defaultdict(list)
    for line in open(ROOT / "configs/metadata_fixes.tsv", encoding="utf-8"):
        if line.strip() and not line.startswith("#"):
            rid, field, old, new, _ = line.rstrip("\n").split("\t")
            fixes[rid].append((field, json.loads(old), json.loads(new)))
    seen = set()
    found = set()
    out = {s: open(args.out / f"{s}.jsonl", "w", encoding="utf-8") for s in ("train", "dev", "test")}
    stats = defaultdict(Counter)
    for line in open(args.corpus, encoding="utf-8"):
        r = json.loads(line)
        st = stats[r["book"]]
        st["poems in corpus"] += 1
        seen.add(r["id"])
        st["units in corpus"] += len(r["units"])
        if (r["id"], None) in curated:
            if not r["units"] or r["units"][0]["misras"][0] != curated[r["id"], None][0]:
                raise SystemExit(f"configs/excluded_units.tsv: {r['id']} does not start with the listed mısra (ids shifted?)")
            found.add((r["id"], None))
        for u in r["units"]:
            if (r["id"], u["n"]) in curated:
                if u["misras"][0] != curated[r["id"], u["n"]][0]:
                    raise SystemExit(f"configs/excluded_units.tsv: {r['id']} unit {u['n']} starts "
                                     f"{u['misras'][0]!r}, not the listed mısra (ids shifted?)")
                found.add((r["id"], u["n"]))
        risks = sorted(set(r["issues"]) & POEM_RISKS) + \
            ["vezin label suspect (one position misfits in most lines)"] * (r["id"] in suspect_labels) + \
            (["source: " + unverified[r["id"]]["poem"]] if "poem" in unverified.get(r["id"], {}) else []) + \
            ["fit: the label does not fit most mısras (wrong label)"] * ("poem" in misfit.get(r["id"], {})) + \
            (["curated: " + curated[r["id"], None][1].split(";")[0]] if (r["id"], None) in curated else [])
        if r["split"] not in out:
            st["poem left out: book not exported (configs/splits.tsv)"] += 1
            continue
        if risks:
            for x in risks:
                st[f"poem left out: {x}"] += 1
            continue
        if r["split"] != "train" and not r["variants"]:
            st["poem left out: no printed vezin (dev/test need a label)"] += 1
            continue
        units = []
        misshapen = {str(u["n"]) for u in r["units"] if any("mısras where the poem has" in i for i in u["issues"])}
        broken = min([int(n) for n, v in unverified.get(r["id"], {}).items() if n != "poem" and n not in misshapen
                      and any(why == "mısras not consecutive in the PDF" for _, why in v)], default=None)
        for u in r["units"]:
            bad = sorted(set(u["issues"]) & UNIT_RISKS) + \
                ["shape differs from the poem"] * any("mısras where the poem has" in i for i in u["issues"]) + \
                ["single mısra"] * (len(u["misras"]) < 2) + \
                ["not a beyit (odd mısra count)"] * (len(u["misras"]) != 2 and r.get("regrouped") != "stanzas") + \
                ["source: " + why for _, why in unverified.get(r["id"], {}).get(str(u["n"]), [])] + \
                ["fit: " + why for _, why in misfit.get(r["id"], {}).get("units", {}).get(str(u["n"]), [])] + \
                ["spelling: a vowel-length misprint the meter rejects"] * (str(u["n"]) in misprint.get(r["id"], {})) + \
                (["curated: " + curated[r["id"], u["n"]][1]] if (r["id"], u["n"]) in curated else []) + \
                ["source: after a non-consecutive unit (the pairing may be shifted)"] * (broken is not None and u["n"] > broken)
            if not u["usable"]:
                st["unit left out: not usable"] += 1
            elif bad:
                st[f"unit left out: {bad[0]}"] += 1
            else:
                units.append({"n": u["n"], "misras": u["misras"],
                              "notes": [i for i in u["issues"] if i.startswith("length")]})
        if not units:
            st["poem left out: no clean units"] += 1
            continue
        rec = {k: r[k] for k in KEEP}
        rec["units"] = units
        for field, old, new in fixes.get(r["id"], []):
            if field.startswith("n:"):
                u = next((u for u in units if u["n"] == old), None)
                if u is None and any(v["n"] == old for v in r["units"]):
                    continue
                if u is None or any(v["n"] == new for v in units):
                    raise SystemExit(f"configs/metadata_fixes.tsv: {r['id']} has no unit {old}, or already a unit {new}")
                u["n"] = new
            else:
                if rec[field] != old:
                    raise SystemExit(f"configs/metadata_fixes.tsv: {r['id']} {field} is {rec[field]!r}, not {old!r}")
                rec[field] = new
        out[r["split"]].write(json.dumps(rec, ensure_ascii=False) + "\n")
        st["poems kept"] += 1
        st["units kept"] += len(units)
        st["mısras kept"] += sum(len(u["misras"]) for u in units)
    for f in out.values():
        f.close()
    if set(fixes) - seen:
        raise SystemExit(f"configs/metadata_fixes.tsv names records not in the corpus: {sorted(set(fixes) - seen)}")
    if set(curated) - found:
        raise SystemExit(f"configs/excluded_units.tsv names units not in the corpus: {sorted(map(str, set(curated) - found))}")

    total = defaultdict(Counter)
    for book, st in stats.items():
        split = next(json.loads(l)["split"] for l in open(args.corpus, encoding="utf-8") if json.loads(l)["book"] == book)
        if split not in out:
            continue
        for k in ("poems kept", "units kept", "mısras kept"):
            total[split][k] += st[k]
    report = {"per_book": {b: dict(sorted(st.items())) for b, st in sorted(stats.items())},
              "per_split": {s: dict(c) for s, c in total.items()}}
    (ROOT / "data/final_stats.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for s, c in total.items():
        print(f"{s:5s}: {c['poems kept']:5d} poems, {c['units kept']:6d} units, {c['mısras kept']:6d} mısras")
    print(f"wrote {args.out}/{{train,dev,test}}.jsonl and data/final_stats.json")

if __name__ == "__main__":
    main()
