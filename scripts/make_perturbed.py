import argparse
import importlib.util
import json
import os
import random
import shutil
from collections import Counter
from pathlib import Path

from aruz.editions import apply_orthography
from aruz.perturb import Generator, letters, line_weights, word_weights

ROOT = Path(__file__).resolve().parent.parent
STORE = Path(os.environ["STORE"])

def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

meter_eval = module("eval_meter")
survey = module("web_survey")
popular = meter_eval.popular

FILES = {
    "web": ("web", 1., ("forms", "damage", "clutter", "structure")),
    "web_x2": ("web", 2., ("forms", "damage", "clutter", "structure")),
    "web_forms": ("web", 1., ("forms",)),
    "web_damage": ("web", 1., ("damage",)),
    "web_clutter": ("web", 1., ("clutter",)),
    "web_structure": ("web", 1., ("structure",)),
    "ocr": ("ocr", 1., ("forms", "damage", "clutter")),
    "ocr_x2": ("ocr", 2., ("forms", "damage", "clutter")),
    "ocr_forms": ("ocr", 1., ("forms",)),
    "ocr_damage": ("ocr", 1., ("damage",)),
    "ocr_clutter": ("ocr", 1., ("clutter",)),
}

GROUPS = ("forms", "damage", "clutter", "structure")

def generate(gen, recs, rng, gold, f):
    c = Counter()
    for r in recs:
        intensity = gen.intensity(rng)
        keep = {k for k, u in enumerate(r["units"]) if any((r["id"], m) in gold for m in u["misras"])}
        skipped = gen.skipped(r["units"], keep, rng, intensity)
        for k, u in enumerate(r["units"]):
            for i, m in enumerate(u["misras"]):
                raw = m.split()
                sch = apply_orthography(m, r["silent"], r["context"]).split()
                if len(raw) != len(sch):
                    raise SystemExit(f"{r['id']}: raw and scholarly tokens differ: {m!r}")
                clean = " ".join(popular(s) for s in sch)
                if k in skipped:
                    text, edits = None, [{"kind": "beyit left out", "group": "structure"}]
                    c["left out"] += 1
                else:
                    meta = {"poet": r["poet"], "number": r.get("number", ""), "beyit": u["n"]}
                    text, edits = gen.line(raw, sch, meta, rng, intensity)
                    cf, pf = survey.folded(clean, popular), survey.folded(text, popular)
                    c["lines"] += 1
                    c["words"] += len(cf)
                    for ei, wi in survey.align(cf, pf):
                        a, b = [cf[x] for x in ei], [pf[y] for y in wi]
                        if a != b:
                            c["deviations"] += 1
                            c["damage deviations"] += line_weights(a).replace(" ", "") != line_weights(b).replace(" ", "")
                    c["letters differ"] += cf != pf
                    c["syllables differ"] += sum(map(len, map(word_weights, cf))) != sum(map(len, map(word_weights, pf)))
                    c["weights differ"] += line_weights(cf) != line_weights(pf)
                    c["cer num"] += survey.levenshtein(" ".join(cf), " ".join(pf))
                    c["cer den"] += len(" ".join(cf))
                if f is not None:
                    f.write(json.dumps({"id": r["id"], "unit": u["n"], "misra": i, "text": m, "plain": clean,
                                        "perturbed": text, "edits": edits}, ensure_ascii=False) + "\n")
    return {"mısras": c["lines"] + c["left out"], "mısras left out": c["left out"],
            "word deviations per word": round(c["deviations"] / c["words"], 4),
            "damage deviations per word": round(c["damage deviations"] / c["words"], 4),
            "mısras whose letters differ": round(c["letters differ"] / c["lines"], 4),
            "mısras whose syllable count differs": round(c["syllables differ"] / c["lines"], 4),
            "mısras whose weights differ": round(c["weights differ"] / c["lines"], 4),
            "character error rate": round(c["cer num"] / c["cer den"], 4)}

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", default=["test"])
    parser.add_argument("--seed", type=int, default=2030)
    args = parser.parse_args()
    base = STORE / "datasets/ottoman_poetry/perturbed"
    old = STORE / "datasets/ottoman_poetry/perturbed_families_2026-10-29"
    if base.exists() and not old.exists():
        shutil.move(str(base), str(old))
    profiles = json.loads((ROOT / "data/robustness/profiles.json").read_text(encoding="utf-8"))["profiles"]
    vocab = Counter()
    for line in open(STORE / "datasets/ottoman_poetry/final/train.jsonl", encoding="utf-8"):
        r = json.loads(line)
        for u in r["units"]:
            for m in u["misras"]:
                vocab.update(letters(w) for w in popular(apply_orthography(m, r["silent"], r["context"])).split())
    vocab.pop("", None)
    manifest = {"seed": args.seed, "files": {}, "profiles measured": {k: v["measured"] for k, v in profiles.items()}}
    for split in args.splits:
        recs = [json.loads(l) for l in open(STORE / f"datasets/ottoman_poetry/final/{split}.jsonl", encoding="utf-8")]
        gold = set()
        gp = ROOT / f"data/annotation/{split}_gold.jsonl"
        if gp.exists():
            gold = {(g["id"], g["text"]) for g in map(json.loads, open(gp, encoding="utf-8"))}
        out = base / split
        out.mkdir(parents=True, exist_ok=True)
        factor = {}
        for prof in profiles:
            M = profiles[prof]["measured"]
            sample = random.Random(f"{args.seed}-amount").sample(recs, min(200, len(recs)))
            fac = {"forms": 1., "damage": 1., "clutter": 1., "structure": 1.}
            for _ in range(6):
                gen = Generator(profiles[prof], fac, GROUPS, vocab, popular)
                got = generate(gen, sample, random.Random(f"{args.seed}-amount-{prof}"), gold, None)
                fac["damage"] *= M["mısras whose weights differ"] / got["mısras whose weights differ"]
                fac["forms"] *= max(.05, M["word deviations per word"] - got["damage deviations per word"]) / \
                    max(.01, got["word deviations per word"] - got["damage deviations per word"])
            factor[prof] = {k: round(v, 4) for k, v in fac.items()}
            print(f"{prof}: amount factors {factor[prof]}; last sample {got}; measured {M}", flush=True)
        manifest["amount factor"] = factor
        for name, (prof, scale, groups) in FILES.items():
            gen = Generator(profiles[prof], {k: v * scale for k, v in factor[prof].items()}, groups, vocab, popular)
            with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
                stats = generate(gen, recs, random.Random(f"{args.seed}-{split}-{name}"), gold, f)
            stats = {"profile": prof, "scale": scale, "groups": list(groups), **stats}
            manifest["files"][f"{split}/{name}"] = stats
            print(f"{split}/{name}", stats, flush=True)
    (base / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("measured profiles:", json.dumps(manifest["profiles measured"], ensure_ascii=False))

if __name__ == "__main__":
    main()
