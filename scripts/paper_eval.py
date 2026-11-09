import argparse
import importlib.util
import json
import random
import os
import pickle
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from math import inf, log
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STORE = Path(os.environ["STORE"])
MODELS = STORE / "projects/aruz/models"
NO_FIT = 10.

SYSTEMS = {
    "frequency":                    dict(reader="aligner", cands="frequent", model=None, input="B", standard=True),
    "aligner":                      dict(reader="aligner", cands=None, model=None, input="B"),
    "aligner + lexicon":            dict(reader="aligner", cands="lexicon", model=None, input="B"),
    "aligner + restorer":           dict(reader="aligner", cands="restorer", model=None, input="B"),
    "Kurt & Kara + restorer":       dict(reader="kk", cands="restorer1", model=None, input="B"),
    "Kurt & Kara (scholarly text)": dict(reader="kk", cands=None, model=None, input="A"),
    "Kurt & Kara (plain letters)":  dict(reader="kk", cands=None, model=None, input="B"),
    "HMM + restorer":               dict(reader="hmm", cands="restorer", model="hmm_em.pkl", input="B"),
    "HMM + lexicon":                dict(reader="hmm", cands="lexicon", model="hmm_em.pkl", input="B"),
    "HMM (no EM) + restorer":       dict(reader="hmm", cands="restorer", model="untrained", input="B"),
    "HMM + restorer (flat)":        dict(reader="hmm", cands="restorer_flat", model="hmm_em.pkl", input="B"),
    "HMM (structure only) + restorer": dict(reader="hmm", cands="restorer_flat", model="untrained", input="B"),
    "HMM (no reading symbols) + restorer": dict(reader="hmm", cands="restorer", model="hmm_em_nosymbols.pkl", input="B"),
    "HMM (untied emissions) + restorer": dict(reader="hmm", cands="restorer", model="hmm_em_untied.pkl", input="B"),
    "HMM (no indels) + restorer":   dict(reader="hmm", cands="restorer", model="hmm_em_noindels.pkl", input="B"),
    "aligner (scholarly text)":     dict(reader="aligner", cands=None, model=None, input="A"),
    "HMM (scholarly text)":         dict(reader="hmm", cands=None, model="hmm_em.pkl", input="A"),
}

def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

W = {}

def init(system, gold_texts, perturbed=None, restorer_path=None, model_path=None):
    import torch
    torch.set_num_threads(1)
    from aruz.hmm.lattice import LatticeHMM
    from aruz.lexicon import Lexicon
    from aruz.meter import build
    es = module("eval_scansion")
    spec = SYSTEMS[system]
    W.update(es=es, spec=spec, gold_texts=gold_texts, cands=None, line_cands=None, perturbed=bool(perturbed))
    if perturbed:
        es.meter_eval.PERTURBED.update(perturbed)
    if spec["cands"] in ("lexicon", "frequent"):
        lex = Lexicon()
        for line in open(STORE / "datasets/ottoman_poetry/final/train.jsonl", encoding="utf-8"):
            r = json.loads(line)
            for u in r["units"]:
                for m in u["misras"]:
                    lex.update(es.tokens(m, r, "A"))
        if spec["cands"] == "lexicon":
            W["cands"] = lex.alternatives
        else:
            from aruz.lexicon import PUNCT, lower, normalize

            def frequent(w):
                counts = lex.forms.get(normalize(w))
                return [(counts.most_common(1)[0][0] if counts else lower(w.strip(PUNCT)), 0.)]
            W["cands"] = frequent
    elif spec["cands"] in ("restorer", "restorer1", "restorer_flat"):
        with open(restorer_path or MODELS / "restorer.pkl", "rb") as f:
            restorer = pickle.load(f)
        in_context = hasattr(restorer, "alternatives_line")
        if spec["cands"] == "restorer":
            W["cands"] = restorer.alternatives
            W["line_cands"] = restorer.alternatives_line if in_context else None
        elif spec["cands"] == "restorer_flat":
            W["cands"] = lambda w: [(f, 0.) for f, _ in restorer.alternatives(w)]
        else:
            W["cands"] = lambda w: restorer.alternatives(w)[:1]
            W["line_cands"] = (lambda ws: [[(f, 0.)] for f in restorer.best_line(ws)]) if in_context else None
    if spec["reader"] == "hmm":
        model, starts, slots = build(innate=True)
        if model_path:
            model = LatticeHMM.load(model_path)
        elif spec["model"] != "untrained":
            model = LatticeHMM.load(MODELS / spec["model"])
        W.update(model=model, starts=starts, slots=slots,
                 prior={m: log(model.A[model.bos_t, st].sum().item()) for m, st in starts.items()})

class Path_:
    def __init__(self, path):
        self.path = path

def poem(r):
    from aruz.align import align
    from aruz.scan import lattice
    es, spec = W["es"], W["spec"]
    lines = [m for u in r["units"] for m in u["misras"]]
    words = [es.tokens(m, r, "P" if W["perturbed"] and spec["input"] == "B" else spec["input"]) for m in lines]
    def cands_of(ws):
        if W["line_cands"]:
            return W["line_cands"](ws)
        return [W["cands"](w) for w in ws] if W["cands"] else [[(w, 0.)] for w in ws]
    cands = [cands_of(ws) for ws in words]
    names = [n for n, _ in es.INVENTORY]
    if spec["reader"] == "kk":
        from aruz.kurtkara import read_poem
        forms = [[c[0][0] for c in cs] for cs in cands]
        found_kk, per_line_kk, under = read_poem(forms, es.INVENTORY)
        scores = [{n: 0. for n in names} for _ in lines]
        prior = {n: 0. for n in names}
        found = found_kk[0] if found_kk else "none"
        per_line = [p[0] if p else "none" for p in per_line_kk]
        cache = {}
        out = []
        for i, m in enumerate(lines):
            row = {"text": m, "meters": {"found": found, "line": per_line[i], "gold": r["meter"]}, "decodes": {}}
            for mode, n in row["meters"].items():
                if (n, i) not in cache:
                    syl = under(n, i) if n in names else None
                    cache[n, i] = None if syl is None else (forms[i], syl)
                d = cache[n, i]
                keep = m in W["gold_texts"]
                row["decodes"][mode] = None if d is None else {"forms": d[0], "syllables": d[1] if keep else None}
            out.append(row)
        return {"id": r["id"], "gold": r["meter"], "found": found, "lines": out}
    if spec["reader"] == "aligner":
        standard = spec.get("standard", False)
        best = {n: [min((x for x in (align(c, p, standard=standard) for p in pats) if x), default=None, key=lambda x: x[0])
                    for c in cands] for n, pats in es.INVENTORY}
        scores = [{n: -(best[n][i][0] if best[n][i] else NO_FIT) for n in names} for i in range(len(lines))]
        prior = {n: 0. for n in names}

        def decode(n, i):
            x = best[n][i]
            return (list(x[1]), x[2]) if x else None
    else:
        model, starts, slots = W["model"], W["starts"], W["slots"]
        lats = [lattice(model, ws, alternatives=cs) for ws, cs in zip(words, cands)]
        prior = W["prior"]
        scores = []
        for lat in lats:
            row = {}
            for n in names:
                lat.starts = starts[n]
                row[n] = model.lattice_forward(lat).item() - prior[n]
            lat.starts = None
            scores.append(row)

        def decode(n, i):
            lat = lats[i]
            lat.starts = starts[n]
            try:
                _, path = model.lattice_viterbi(lat)
            except ValueError:
                return None
            finally:
                lat.starts = None
            forms = {}
            for a, _ in path:
                forms.setdefault(a.info.word, a.info.form)
            syl = es.from_hmm(Path_([(a, slots[t]) for a, t in path]))
            return [forms.get(k) for k in range(len(words[i]))], syl
    total = {n: prior[n] + sum(s[n] for s in scores if any(v > -inf for v in s.values())) for n in names}
    found = max(names, key=lambda n: total[n])
    per_line = [max(names, key=lambda n: s[n] + prior[n]) for s in scores]
    cache = {}
    out = []
    for i, m in enumerate(lines):
        row = {"text": m, "meters": {"found": found, "line": per_line[i], "gold": r["meter"]}, "decodes": {}}
        for mode, n in row["meters"].items():
            if (n, i) not in cache:
                cache[n, i] = decode(n, i)
            d = cache[n, i]
            keep = m in W["gold_texts"]
            row["decodes"][mode] = None if d is None else {"forms": d[0], "syllables": d[1] if keep else None}
        out.append(row)
    return {"id": r["id"], "gold": r["meter"], "found": found, "lines": out}

def f1_macro(gold, pred):
    f1s = []
    for m in set(gold):
        tp = sum(g == p == m for g, p in zip(gold, pred))
        fp = sum(p == m != g for g, p in zip(gold, pred))
        fn = sum(g == m != p for g, p in zip(gold, pred))
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.)
    return sum(f1s) / len(f1s)

def task1(gold, pred):
    return {"accuracy": sum(g == p for g, p in zip(gold, pred)) / len(gold), "macro-F1": f1_macro(gold, pred),
            "n": len(gold)}

def words_metric(es, recs, results, mode):
    from aruz.lexicon import reduce
    c = Counter()
    for res in results:
        r = recs[res["id"]]
        for row in res["lines"]:
            gws = [reduce(w) for w in es.tokens(row["text"], r, "A")]
            d = row["decodes"][mode]
            got = d["forms"] if d else [None] * len(gws)
            plain = es.tokens(row["text"], r, "B")
            if len(got) != len(gws):
                continue
            for f, g, p in zip(got, gws, plain):
                f = reduce(f if f is not None else p)
                c["words"] += 1
                c["spelling"] += f.replace("-", "") == g.replace("-", "")
                c["weights"] += es.weights(f) == es.weights(g)
    return {"words": c["words"], "spelling": c["spelling"] / max(c["words"], 1), "weights": c["weights"] / max(c["words"], 1)}

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--systems", nargs="+", default=list(SYSTEMS))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("WORKERS", os.cpu_count() or 4)))
    parser.add_argument("--limit", type=int, default=0, help="first N poems only (a smoke test)")
    parser.add_argument("--perturbed", type=Path, help="a robustness benchmark file (scripts/make_perturbed.py): each "
                        "line's plain input is replaced by its perturbed text, read through the same normalizer")
    parser.add_argument("--poem-sample", type=int, default=0,
                        help="Task 1 on the poems with gold lines plus seeded others up to N (the robustness runs)")
    parser.add_argument("--sample-seed", type=int, default=2029)
    parser.add_argument("--restorer", default="restorer.pkl", help="the restorer's pickle: a file name under the models "
                        "directory, or a path (an experiment's variant)")
    parser.add_argument("--results-subdir", default="", help="write under data/results/<split>/<this> (an experiment)")
    parser.add_argument("--model", type=Path, help="a trained HMM to use in place of the systems' own (a diagnostic; "
                                                   "with --results-subdir so the paper's results are not overwritten)")
    args = parser.parse_args()
    if args.split == "test" and not args.allow_test:
        raise SystemExit("test is sealed: pass --allow-test only for the final evaluation")
    es = module("eval_scansion")
    from aruz.lexicon import reduce
    from aruz.meter import METERS
    names = {m.name for m in METERS}
    final = STORE / "datasets/ottoman_poetry/final"
    recs = {r["id"]: r for r in map(json.loads, open(final / f"{args.split}.jsonl", encoding="utf-8")) if r["meter"] in names}
    gold = [json.loads(l) for l in open(ROOT / f"data/annotation/{args.split}_gold.jsonl", encoding="utf-8")]
    poems = list(recs.values())
    if args.limit:
        keep = {g["id"] for g in gold}
        poems = [r for r in poems if r["id"] in keep][:args.limit]
        gold = [g for g in gold if g["id"] in {r["id"] for r in poems}]
    out_dir = ROOT / f"data/results/{args.split}" / args.results_subdir
    restorer_path = Path(args.restorer) if "/" in args.restorer else MODELS / args.restorer
    if args.poem_sample:
        keep = {g["id"] for g in gold}
        rest = sorted(r["id"] for r in poems if r["id"] not in keep)
        random.Random(args.sample_seed).shuffle(rest)
        ids = sorted(keep) + rest[:max(0, args.poem_sample - len(keep))]
        sample_file = out_dir / "perturbed/poems.json"
        sample_file.parent.mkdir(parents=True, exist_ok=True)
        if sample_file.exists() and json.loads(sample_file.read_text()) != ids:
            raise SystemExit(f"{sample_file} holds a different poem sample: delete it or use the same settings")
        sample_file.write_text(json.dumps(ids, ensure_ascii=False))
        poems = [recs[i] for i in ids]
    perturbed = None
    if args.perturbed:
        perturbed = {}
        for line in open(args.perturbed, encoding="utf-8"):
            p = json.loads(line)
            perturbed[p["id"], p["text"]] = p["perturbed"]
        missing = [(r["id"], m) for r in poems for u in r["units"] for m in u["misras"] if (r["id"], m) not in perturbed]
        if missing:
            raise SystemExit(f"{args.perturbed}: {len(missing)} lines of the evaluated poems are missing, e.g. {missing[:2]}")
        poems = es.meter_eval.drop_skipped(poems, perturbed)
    if args.perturbed or args.poem_sample:
        out_dir = out_dir / "perturbed" / (args.perturbed.stem if args.perturbed else "clean")
    gold_texts = {g["text"] for g in gold}
    out_dir.mkdir(parents=True, exist_ok=True)
    for system in args.systems:
        spec = SYSTEMS[system]
        if spec["model"] not in (None, "untrained") and not (MODELS / spec["model"]).exists():
            raise SystemExit(f"{system}: model {MODELS / spec['model']} is missing")
        if args.model and not args.results_subdir:
            raise SystemExit("--model needs --results-subdir: it must not overwrite the paper's results")
        with ProcessPoolExecutor(args.workers, initializer=init, initargs=(system, gold_texts, perturbed, restorer_path, args.model)) as ex:
            results = list(ex.map(poem, poems, chunksize=2))
        by_id = {res["id"]: res for res in results}
        rep = {"system": system, "spec": dict(spec, restorer=restorer_path.name, **({"model": str(args.model)} if args.model else {})),
               "split": args.split, "poems": len(results),
               "input": str(args.perturbed) if args.perturbed else "clean"}
        rep["task1"] = {"poem": task1([res["gold"] for res in results], [res["found"] for res in results]),
                        "line": task1([res["gold"] for res in results for _ in res["lines"]],
                                      [row["meters"]["line"] for res in results for row in res["lines"]]),
                        "decisions": {"poem": [[res["gold"], res["found"]] for res in results],
                                      "line": [[res["gold"], row["meters"]["line"]] for res in results for row in res["lines"]]}}
        for mode in ("found", "line", "gold"):
            outs, oks = [], []
            for g in gold:
                row = next(x for x in by_id[g["id"]]["lines"] if x["text"] == g["text"])
                d = row["decodes"][mode]
                outs.append(d["syllables"] if d else None)
                oks.append(row["meters"][mode] == g["meter"])
            print(f"\n{system}, meter: {mode}")
            rep[mode] = {"task2": es.score_scansion(system, gold, outs, oks),
                         "words": words_metric(es, recs, results, mode)}
        print(f"{system}: task 1 {rep['task1']}; words {rep['found']['words']}")
        (out_dir / f"{system}.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    if args.perturbed or args.poem_sample:
        return
    train = [json.loads(l) for l in open(final / "train.jsonl", encoding="utf-8")]
    freq = Counter(r["meter"] for r in train if r["meter"] in names)
    from aruz.scan import syllabify
    lengths = {n: {len(p) for p in pats} for n, pats in es.INVENTORY}
    maj = freq.most_common(1)[0][0]
    pg, pl, gp, gl = [], [], [], []
    for r in poems:
        votes = Counter()
        for u in r["units"]:
            for m in u["misras"]:
                k = sum(len(syllabify(w)) for w in es.tokens(m, r, "B"))
                fits = [n for n in lengths if k in lengths[n]]
                best = max(fits, key=lambda n: freq.get(n, 0)) if fits else maj
                pl.append(best)
                gl.append(r["meter"])
                votes.update(fits)
        pg.append(max(names, key=lambda n: (votes[n], freq.get(n, 0))))
        gp.append(r["meter"])
    base = {"majority": {"task1": {"poem": task1(gp, [maj] * len(gp)), "line": task1(gl, [maj] * len(gl))}},
            "syllable count": {"task1": {"poem": task1(gp, pg), "line": task1(gl, pl)}}}
    (out_dir / "task1_baselines.json").write_text(json.dumps(base, ensure_ascii=False, indent=1), encoding="utf-8")
    print(base)

if __name__ == "__main__":
    main()
