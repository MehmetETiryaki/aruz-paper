import argparse
import difflib
import importlib.util
import json
import os
import pickle
import random
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STORE = Path(os.environ["STORE"])
MODELS = STORE / "projects/aruz/models"
PERT = STORE / "datasets/ottoman_poetry/perturbed"
CAL = ROOT / "data/detection/calibration.json"
W = {}

def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

def init(perturbed, calibration):
    import torch
    torch.set_num_threads(1)
    from aruz.detect import Detector
    from aruz.hmm.lattice import LatticeHMM
    from aruz.meter import build
    es = module("eval_scansion")
    _, starts, slots = build()
    model = LatticeHMM.load(MODELS / "hmm_em.pkl")
    with open(MODELS / "restorer.pkl", "rb") as f:
        restorer = pickle.load(f)
    names = [n for n, _ in es.INVENTORY]
    det = Detector(model, {n: starts[n] for n in names}, slots, restorer, es.from_hmm, names=names)
    if calibration:
        det.stats = {k: tuple(v) for k, v in calibration["stats"].items()}
        det.threshold = calibration["threshold"]
    es.meter_eval.PERTURBED.clear()
    es.meter_eval.PERTURBED.update(perturbed or {})
    W.update(es=es, det=det, perturbed=bool(perturbed))

def tokens(m, r, inp):
    es = W["es"]
    es.meter_eval.INPUT = inp
    return es.meter_eval.words(m, r)

def run_poem(r):
    es, det = W["es"], W["det"]
    lines = [(u["n"], k + 1, m) for u in r["units"] for k, m in enumerate(u["misras"])]
    clean = [tokens(m, r, "B") for _, _, m in lines]
    read = [tokens(m, r, "P") for _, _, m in lines] if W["perturbed"] else clean
    keep = [i for i, ws in enumerate(read) if ws]
    if not keep:
        return []
    if det.stats:
        res = det.poem([read[i] for i in keep])
        rows = res["lines"]
        meter = res["meter"]
    else:
        meter, lats = det.find_meter([read[i] for i in keep])
        rows = []
        for i, lat in zip(keep, lats):
            line, word, _ = det.raw_signals(read[i], lat, meter)
            rows.append({"signals": line, "word signals": word})
    out = []
    for i, row in zip(keep, rows):
        n, k, m = lines[i]
        rec = {"id": r["id"], "unit": n, "misra": k, "clean": clean[i], "read": read[i], "meter": meter,
               "signals": row["signals"], "word signals": row["word signals"]}
        for key in ("score", "flagged", "suspects", "reasons", "warning"):
            if key in row:
                rec[key] = row[key]
        out.append(rec)
    return out

def faults(clean, read):
    from aruz.perturb import letters, line_weights, word_weights
    survey = W.get("survey") or module("web_survey")
    W["survey"] = survey
    cf, rf = [letters(w) for w in clean], [letters(w) for w in read]
    bad, delta = [], {}
    for ei, wi in survey.align(cf, rf):
        a, b = [cf[i] for i in ei], [rf[j] for j in wi]
        if line_weights(a).replace(" ", "") != line_weights(b).replace(" ", ""):
            d = sum(len(word_weights(x)) for x in b) - sum(len(word_weights(x)) for x in a)
            for j in wi:
                bad.append(j)
                delta[j] = d
    return line_weights(cf) != line_weights(rf), bad, delta, cf != rf

def auroc(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return None
    ranked = sorted((s, l) for s, l in zip(scores, labels))
    rank_sum, i = 0., 0
    while i < len(ranked):
        j = i
        while j < len(ranked) and ranked[j][0] == ranked[i][0]:
            j += 1
        r = (i + j + 1) / 2
        rank_sum += r * sum(l for _, l in ranked[i:j])
        i = j
    return (rank_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

def evaluate_rows(rows):
    labels, scores, flagged, located, top1, top2, kind_ok, kind_n, offered, offer_n = [], [], [], 0, 0, 0, 0, 0, 0, 0
    forms_only, forms_alarms = 0, 0
    for row in rows:
        faulty, bad, delta, differs = faults(row["clean"], row["read"])
        row["changed"] = bad
        labels.append(faulty)
        scores.append(row["score"])
        flagged.append(row["flagged"])
        if differs and not faulty:
            forms_only += 1
            forms_alarms += row["flagged"]
        if faulty and row["flagged"] and bad:
            ranked = _rank(row)
            located += 1
            top1 += ranked[0] in bad
            top2 += any(j in bad for j in ranked[:2])
            wn = row.get("warning")
            if wn and wn["position"] in bad and wn["meter wants"] and wn["word gives"]:
                said = len(wn["meter wants"]) - len(wn["word gives"])
                d = delta[wn["position"]]
                kind_n += 1
                kind_ok += (said > 0 and d < 0) or (said < 0 and d > 0) or (said == 0 and d == 0)
                if wn["position"] < len(row["clean"]) and len(row["clean"]) == len(row["read"]):
                    from aruz.lexicon import normalize
                    offer_n += 1
                    offered += normalize(row["clean"][wn["position"]]) in wn["fitting"]
    tp = sum(f and l for f, l in zip(flagged, labels))
    fp = sum(f and not l for f, l in zip(flagged, labels))
    pos = sum(labels)
    return {"lines": len(rows), "faulty lines": pos, "auroc": auroc(scores, labels),
            "precision": tp / (tp + fp) if tp + fp else None, "recall": tp / pos if pos else None,
            "flagged": sum(flagged) / len(rows) if rows else None,
            "false alarms on lines that keep the scansion": fp / (len(rows) - pos) if len(rows) > pos else None,
            "lines changed but scansion kept": forms_only,
            "false alarms on them": forms_alarms / forms_only if forms_only else None,
            "located lines": located, "top-1 word": top1 / located if located else None,
            "recall@2 word": top2 / located if located else None,
            "kind right": kind_ok / kind_n if kind_n else None,
            "edition word offered": offered / offer_n if offer_n else None}

def _rank(row):
    seen, order = {}, []
    for w, _ in row["suspects"]:
        j = [k for k, x in enumerate(row["read"]) if x == w and k not in seen]
        if j:
            seen[j[0]] = True
            order.append(j[0])
    return order

def sample_poems(recs, n, seed, must):
    rng = random.Random(seed)
    rest = [r for r in recs if r["id"] not in must]
    rng.shuffle(rest)
    chosen = [r for r in recs if r["id"] in must] + rest[:max(0, n - len(must))]
    return chosen

def run(recs, perturbed, calibration, workers):
    with Pool(workers, initializer=init, initargs=(perturbed, calibration)) as pool:
        return [row for rows in pool.imap(run_poem, recs, chunksize=1) for row in rows]

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--poems", type=int, default=300)
    parser.add_argument("--seed", type=int, default=2029)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("WORKERS", os.cpu_count() or 4)))
    parser.add_argument("--sample", type=Path, help="a JSON list of test poem ids (the robustness job's sample)")
    parser.add_argument("--limit-files", type=int, default=0, help="smoke test: only this many perturbed files")
    args = parser.parse_args()
    final = STORE / "datasets/ottoman_poetry/final"
    names = {n for n, _ in module("eval_scansion").INVENTORY}

    if args.calibrate:
        dev = [r for r in map(json.loads, open(final / "dev.jsonl", encoding="utf-8")) if r["meter"] in names]
        poems = sample_poems(dev, args.poems, args.seed, set())
        rows = run(poems, None, None, args.workers)
        from statistics import mean, pstdev
        from aruz.detect import LINE_SIGNALS, WORD_SIGNALS
        stats = {s: (mean(r["signals"][s] for r in rows), pstdev(r["signals"][s] for r in rows)) for s in LINE_SIGNALS}
        ws = [w for r in rows for w in r["word signals"]]
        stats.update({s: (mean(w[s] for w in ws), pstdev(w[s] for w in ws)) for s in WORD_SIGNALS})
        line_scores = sorted(sum((r["signals"][s] - stats[s][0]) / stats[s][1] if stats[s][1] > 0 else 0.
                                 for s in LINE_SIGNALS) / len(LINE_SIGNALS) for r in rows)
        threshold = line_scores[int(.95 * len(line_scores))]
        CAL.parent.mkdir(parents=True, exist_ok=True)
        CAL.write_text(json.dumps({"poems": [r["id"] for r in poems], "lines": len(rows), "stats": stats,
                                   "threshold": threshold, "false-alarm target": .05}, ensure_ascii=False, indent=1),
                       encoding="utf-8")
        print(f"calibrated on {len(rows)} clean dev lines of {len(poems)} poems; threshold {threshold:.3f}")
        for s, (m, sd) in stats.items():
            print(f"  {s:14s} mean {m:8.3f}  sd {sd:7.3f}")

    if args.evaluate:
        if not CAL.exists():
            raise SystemExit("no calibration: run --calibrate first")
        calibration = json.loads(CAL.read_text(encoding="utf-8"))
        manifest = json.loads((PERT / "manifest.json").read_text(encoding="utf-8"))
        if "test/web" not in manifest.get("files", {}):
            raise SystemExit("the perturbed benchmark is not the profile-based one: regenerate it first")
        test = [r for r in map(json.loads, open(final / "test.jsonl", encoding="utf-8")) if r["meter"] in names]
        by_id = {r["id"]: r for r in test}
        if args.sample and args.sample.exists():
            ids = json.loads(args.sample.read_text(encoding="utf-8"))
            ids = ids["poems"] if isinstance(ids, dict) else ids
            poems = [by_id[i] for i in ids if i in by_id]
            if len(poems) != len(ids):
                raise SystemExit(f"{len(ids) - len(poems)} sample poems are not in the final test set")
        else:
            gold = {json.loads(l)["id"] for l in open(ROOT / "data/annotation/test_gold.jsonl", encoding="utf-8")}
            poems = sample_poems(test, args.poems, args.seed, gold)
        out = ROOT / "data/results/test/detection"
        out.mkdir(parents=True, exist_ok=True)
        summary = {"poems": [r["id"] for r in poems], "threshold": calibration["threshold"], "files": {}}
        files = [("clean", None)] + [(k.split("/", 1)[1], PERT / f"{k}.jsonl") for k in manifest["files"]
                                     if k.startswith("test/")]
        if args.limit_files:
            files = files[:args.limit_files + 1]
        ids = {r["id"] for r in poems}
        for name, path in files:
            perturbed = None
            if path:
                perturbed = {}
                for row in map(json.loads, open(path, encoding="utf-8")):
                    if row["id"] in ids:
                        perturbed[row["id"], row["text"]] = row["perturbed"]
                missing = [(r["id"], m) for r in poems for u in r["units"] for m in u["misras"] if (r["id"], m) not in perturbed]
                if missing:
                    raise SystemExit(f"{name}: {len(missing)} sample lines have no perturbed text, e.g. {missing[0]}")
            sub = module("eval_meter").drop_skipped(poems, perturbed) if perturbed else poems
            rows = run(sub, perturbed, calibration, args.workers)
            res = evaluate_rows(rows)
            summary["files"][name] = res
            (out / f"{name}.json").write_text(json.dumps({"summary": res, "lines": [
                {k: r[k] for k in ("id", "unit", "misra", "read", "meter", "score", "flagged", "reasons", "changed",
                                   "warning")}
                | {"suspects": [w for w, _ in r["suspects"][:3]]} for r in rows]}, ensure_ascii=False), encoding="utf-8")
            print(f"{name:14s} " + json.dumps(res), flush=True)
        (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")

if __name__ == "__main__":
    main()
