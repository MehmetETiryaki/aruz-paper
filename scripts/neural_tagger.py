import argparse
import importlib.util
import json
import os
import random
import unicodedata
from collections import Counter
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
STORE = Path(os.environ["STORE"])

def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

es = module("eval_scansion")
mc = module("meter_classifier")
pe = module("paper_eval")
from aruz.lexicon import lower, reduce
from aruz.meter import METERS
from aruz.restore import units
from aruz.scan import VOWELS, syllabify, weight

NFC = lambda s: unicodedata.normalize("NFC", s)
CHOICES = ["-", ".", "-."]
NEG = -1e4

def plain_syllables(words):
    spans, at = [], 0
    for w in words:
        ss = syllabify(w)
        if "".join(t for t, _, _ in ss) != w:
            return None
        for t, _, _ in ss:
            spans.append((at, at + len(t) - 1))
            at += len(t)
        at += 1
    return spans

def can_link(words):
    out = []
    for w, nxt in zip(words, words[1:] + [""]):
        ss = syllabify(w)
        out += [False] * (len(ss) - 1) + [ss[-1][2] > 0 and nxt[:1] in VOWELS and nxt != ""]
    return out

def line_example(r, m, meter, inp="B"):
    wa = [reduce(w) for w in es.tokens(m, r, "A")]
    wb = [lower(w) for w in es.tokens(m, r, inp)]
    ex = {"id": r["id"], "text": m, "plain": " ".join(wb), "words_b": wb, "forms_a": wa, "meter": meter,
          "spans": plain_syllables(wb)}
    if ex["spans"] is not None:
        ex["can link"] = can_link(wb)
    return ex

def units_of(ex, stats):
    unit_targets = []
    for w, f in zip(ex["words_b"], ex["forms_a"]):
        us = units(f)
        if NFC("".join(p for p, _ in us)) != NFC(w) or len(us) != len(w):
            unit_targets += [None] * len(w)
            stats["words without a letter alignment (restoration not trained on them)"] += 1
        else:
            unit_targets += [s for _, s in us]
        unit_targets.append(None)
    return unit_targets[:-1]

_MASKS = {}

def pattern_masks(p):
    if p not in _MASKS:
        P = len(p)
        m1 = torch.full((P, len(CHOICES)), NEG)
        m2 = torch.full((P, len(CHOICES)), NEG)
        for c, slots in enumerate(CHOICES):
            for k in range(P - len(slots) + 1):
                if all(q == "x" or q == s for q, s in zip(p[k:], slots)):
                    (m1 if len(slots) == 1 else m2)[k, c] = 0.
        _MASKS[p] = (m1, m2)
    return _MASKS[p]

def choice_logp(logits):
    return torch.log_softmax(logits, -1)

def log_marginal(lp, patterns):
    n = lp.shape[0]
    total = []
    for p in patterns:
        P = len(p)
        m1, m2 = pattern_masks(p)
        dp = torch.full((P + 1,), NEG)
        dp[0] = 0.
        for i in range(n - 1):
            one = torch.logsumexp(dp[:P].unsqueeze(1) + lp[i].unsqueeze(0) + m1, 1)
            two = torch.logsumexp(dp[:P].unsqueeze(1) + lp[i].unsqueeze(0) + m2, 1)
            dp = torch.logaddexp(torch.cat([torch.tensor([NEG]), one]), torch.cat([torch.tensor([NEG, NEG]), two[:-1]]))
        total.append(dp[P - 1])
    return torch.logsumexp(torch.stack(total), 0)

class Tagger(nn.Module):
    def __init__(self, chars, meters, units_, emb=64, hidden=256, layers=2, dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding(chars, emb, padding_idx=0)
        self.meter = nn.Embedding(meters, emb)
        self.rnn = nn.LSTM(emb, hidden, layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.choice = nn.Linear(4 * hidden, len(CHOICES))
        self.unit = nn.Linear(2 * hidden, units_)

    def forward(self, x, lengths, meter, spans):
        e = self.emb(x) + self.meter(meter).unsqueeze(1)
        h, _ = self.rnn(nn.utils.rnn.pack_padded_sequence(e, lengths.cpu(), batch_first=True, enforce_sorted=False))
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True)
        h = self.drop(h)
        b, s, k = spans[:, 0], spans[:, 1], spans[:, 2]
        syllable = torch.cat([h[b, s], h[b, k]], -1)
        return self.choice(syllable), self.unit(h)

def batch(examples, chars, meters):
    n = max(len(e["plain"]) for e in examples)
    x = torch.zeros(len(examples), n, dtype=torch.long)
    spans = []
    for i, e in enumerate(examples):
        x[i, :len(e["plain"])] = torch.tensor([chars.get(c, 1) for c in e["plain"]])
        spans += [(i, a, b) for a, b in e["spans"]]
    return x, torch.tensor([len(e["plain"]) for e in examples]), torch.tensor([meters[e["meter"]] for e in examples]), \
        torch.tensor(spans, dtype=torch.long)

def constrained(scores, patterns):
    best = None
    n = len(scores)
    for p in patterns:
        P = len(p)
        dp = [[None] * (P + 1) for _ in range(n + 1)]
        dp[0][0] = (0., None, None)
        for i in range(n):
            for k in range(P + 1):
                if dp[i][k] is None:
                    continue
                if i == n - 1:
                    if k + 1 == P and (dp[n][P] is None or dp[i][k][0] > dp[n][P][0]):
                        dp[n][P] = (dp[i][k][0], k, 0)
                    continue
                for t, slots in enumerate(CHOICES):
                    if scores[i][t] < NEG / 2 or k + len(slots) > P or any(q != "x" and q != c for q, c in zip(p[k:], slots)):
                        continue
                    kk = k + len(slots)
                    score = dp[i][k][0] + scores[i][t]
                    if dp[i + 1][kk] is None or score > dp[i + 1][kk][0]:
                        dp[i + 1][kk] = (score, k, t)
        if dp[n][P] and (best is None or dp[n][P][0] > best[0]):
            seq, k = [], P
            for i in range(n, 0, -1):
                _, pk, t = dp[i][k]
                seq.append(t)
                k = pk
            best = (dp[n][P][0], seq[::-1])
    return best[1] if best else None

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--limit", type=int, default=0, help="first N train and eval poems only (a smoke test)")
    parser.add_argument("--model", type=Path, help="save the trained classifier and tagger here, or load them if "
                        "the file exists (so the robustness runs reuse one training)")
    parser.add_argument("--perturbed", type=Path, nargs="*", default=[],
                        help="robustness benchmark files: evaluate on each (and on clean input) over --poem-sample")
    parser.add_argument("--poem-sample", type=int, default=0)
    parser.add_argument("--results-subdir", default="", help="write under data/results/<split>/<this> (another seed)")
    args = parser.parse_args()
    if args.split == "test" and not args.allow_test:
        raise SystemExit("test is sealed: pass --allow-test only for the final evaluation")
    torch.manual_seed(args.seed)
    torch.set_num_threads(int(os.environ.get("WORKERS", os.cpu_count() or 4)))
    rng = random.Random(args.seed)
    names = [m.name for m in METERS]
    meters = {n: i for i, n in enumerate(names)}
    patterns = {n: [p for p in pats] for n, pats in es.INVENTORY}
    final = STORE / "datasets/ottoman_poetry/final"
    train = [r for r in map(json.loads, open(final / "train.jsonl", encoding="utf-8")) if r["meter"] in meters]
    ev = [r for r in map(json.loads, open(final / f"{args.split}.jsonl", encoding="utf-8")) if r["meter"] in meters]
    gold = [json.loads(l) for l in open(ROOT / f"data/annotation/{args.split}_gold.jsonl", encoding="utf-8")]
    if args.limit:
        keep = {g["id"] for g in gold}
        train = train[:args.limit]
        ev = [r for r in ev if r["id"] in keep][:args.limit]
        gold = [g for g in gold if g["id"] in {r["id"] for r in ev}]

    saved = torch.load(args.model, weights_only=False) if args.model and args.model.exists() else None
    if saved and "choice.weight" not in saved["net"]:
        raise SystemExit(f"{args.model} is a tagger of an earlier design: retrain it")
    mc.em.INPUT = "B"
    rng_c = random.Random(args.seed)
    shuffled = list(train)
    rng_c.shuffle(shuffled)
    cut = max(1, len(shuffled) // 10)
    held_c, fit_c = mc.lines_of(shuffled[:cut]), mc.lines_of(shuffled[cut:])
    cchars = saved["cchars"] if saved else {c: i + 2 for i, c in enumerate(sorted({c for _, t, _ in fit_c for c in t}))}
    clf = mc.Net(len(cchars) + 2, len(names))
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
    best, state, bad = (saved["clf_best"], saved["clf"], 0) if saved else (-1, None, 0)
    for epoch in range(0 if saved else 1 if args.limit else args.epochs):
        clf.train()
        for x, lengths, y, _ in mc.batches(fit_c, cchars, meters, 64, True, rng_c):
            opt.zero_grad()
            nn.functional.cross_entropy(clf(x, lengths), y).backward()
            nn.utils.clip_grad_norm_(clf.parameters(), 5.)
            opt.step()
        lp = mc.predict(clf, held_c, cchars, meters)
        acc = sum(names[max(range(len(names)), key=p.__getitem__)] == m for p, (_, _, m) in zip(lp, held_c)) / len(held_c)
        print(f"classifier epoch {epoch + 1}: held-out train line accuracy {acc:.4f}", flush=True)
        if acc > best:
            best, state, bad = acc, {k: v.clone() for k, v in clf.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= args.patience:
                break
    clf.load_state_dict(state)
    clf.eval()
    clf_best = best

    def meters_of(ev):
        data_c = mc.lines_of(ev)
        lp = mc.predict(clf, data_c, cchars, meters)
        line_meter = [names[max(range(len(names)), key=p.__getitem__)] for p in lp]
        votes, logsum = {}, {}
        for (pid, _, _), pred, p in zip(data_c, line_meter, lp):
            votes.setdefault(pid, Counter())[pred] += 1
            acc_ = logsum.setdefault(pid, [0.] * len(names))
            for i, v in enumerate(p):
                acc_[i] += v
        return line_meter, {r["id"]: max(names, key=lambda n: (votes[r["id"]][n], logsum[r["id"]][meters[n]])) for r in ev}

    stats = Counter(saved.get("lines", {})) if saved else Counter()
    sets = {"held": [], "fit": []}
    if not saved:
        rng.shuffle(train)
        cut = max(1, len(train) // 10)
        for name, recs in (("held", train[:cut]), ("fit", train[cut:])):
            out = []
            for r in recs:
                for u in r["units"]:
                    for m in u["misras"]:
                        stats["train lines seen"] += 1
                        ex = line_example(r, m, r["meter"])
                        if ex["spans"] is None:
                            stats["skipped: plain syllables do not spell the words"] += 1
                            continue
                        if len(ex["forms_a"]) != len(ex["words_b"]):
                            stats["skipped: plain and scholarly word counts differ"] += 1
                            continue
                        uniform = torch.zeros(len(ex["spans"]), len(CHOICES))
                        if log_marginal(choice_logp(uniform), patterns[r["meter"]]).item() < NEG / 2:
                            stats["skipped: no choice sequence over the plain syllables fits the printed meter"] += 1
                            continue
                        ex["units"] = units_of(ex, stats)
                        stats["training lines"] += 1
                        out.append(ex)
            sets[name] = out
        if not sets["fit"] or not sets["held"]:
            raise SystemExit(f"no training lines: {dict(stats)}")
        print("training lines:", dict(stats), flush=True)
    if saved:
        chars, unit_list = saved["chars"], saved["unit_list"]
    else:
        chars = {c: i + 2 for i, c in enumerate(sorted({c for e in sets["fit"] for c in e["plain"]}))}
        unit_list = sorted({u for e in sets["fit"] for u in e["units"] if u is not None})
    unit_id = {u: i for i, u in enumerate(unit_list)}
    by_plain = {}
    for u in unit_list:
        p = NFC("".join(c for c in unicodedata.normalize("NFD", u) if c not in "ʿʾʼ’'ˇ-" and
                        (not unicodedata.combining(c) or c in "̧̆̈")))
        by_plain.setdefault(p, []).append(unit_id[u])
    net = Tagger(len(chars) + 2, len(names), len(unit_list))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    def losses(part):
        x, lengths, mt, spans = batch(part, chars, meters)
        choice_logits, unit_logits = net(x, lengths, mt, spans)
        nll, at = 0., 0
        for e in part:
            n = len(e["spans"])
            lp = choice_logp(choice_logits[at:at + n])
            nll = nll - log_marginal(lp, patterns[e["meter"]])
            at += n
        u = torch.full(unit_logits.shape[:2], -100, dtype=torch.long)
        for i, e in enumerate(part):
            for j, t in enumerate(e["units"]):
                if t is not None and t in unit_id:
                    u[i, j] = unit_id[t]
        return (nll / len(part) +
                nn.functional.cross_entropy(unit_logits.reshape(-1, len(unit_list)), u.reshape(-1), ignore_index=-100))

    def held_loss():
        net.eval()
        with torch.no_grad():
            total = sum(losses(sets["held"][k:k + 128]).item() * len(sets["held"][k:k + 128])
                        for k in range(0, len(sets["held"]), 128))
        return total / max(len(sets["held"]), 1)

    best, state, bad, epochs = (saved["best"], saved["net"], 0, saved["epochs"]) if saved else (float("inf"), None, 0, 0)
    for epoch in range(0 if saved else 1 if args.limit else args.epochs):
        net.train()
        order = list(range(len(sets["fit"])))
        rng.shuffle(order)
        for k in range(0, len(order), 64):
            opt.zero_grad()
            losses([sets["fit"][i] for i in order[k:k + 64]]).backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.)
            opt.step()
        held = held_loss()
        if held != held:
            raise SystemExit("the held-out loss is NaN: the training diverged")
        epochs = epoch + 1
        print(f"tagger epoch {epoch + 1}: held-out train loss {held:.4f} per line", flush=True)
        if held < best:
            best, state, bad = held, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= args.patience:
                break
    net.load_state_dict(state)
    net.eval()
    if args.model and not saved:
        args.model.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"cchars": cchars, "clf": clf.state_dict(), "clf_best": clf_best, "chars": chars,
                    "unit_list": unit_list, "net": net.state_dict(), "best": best, "epochs": epochs,
                    "lines": dict(stats)}, args.model)
        print(f"saved the classifier and tagger to {args.model}", flush=True)

    def decode(e, meter):
        if e["spans"] is None:
            return None
        with torch.no_grad():
            x, lengths, mt, spans = batch([dict(e, meter=meter)], chars, meters)
            cl, ul = net(x, lengths, mt, spans)
        scores = choice_logp(cl).tolist()
        seq = constrained(scores, patterns[meter])
        if seq is None:
            seq = [max(range(len(CHOICES)), key=row.__getitem__) for row in scores[:-1]] + [0]
        ul = ul[0]
        forms, at = [], 0
        for w in e["words_b"]:
            out = ""
            for c in w:
                cand = by_plain.get(c)
                out += unit_list[max(cand, key=lambda i: ul[at, i].item())] if cand else c
                at += 1
            at += 1
            forms.append(out)
        per_word = [syllabify(reduce(f)) for f in forms]
        texts = [t for ss in per_word for t, _, _ in ss]
        if len(texts) != len(seq):
            texts = [e["plain"][a:b + 1] for a, b in e["spans"]]
        syls = [[t, CHOICES[s], []] for t, s in zip(texts, seq)]
        for i in range(len(syls) - 1):
            if e["can link"][i] and syls[i][0][-1:].lower() not in VOWELS:
                syls[i + 1][0], syls[i][0] = syls[i][0][-1] + syls[i + 1][0], syls[i][0][:-1]
        for s in syls[:-1]:
            try:
                nat = weight(*syllabify(s[0])[-1])
            except ValueError:
                continue
            if nat == "S" and s[1] == "-":
                s[2] = ["imâle"]
            elif nat == "L" and s[1] == ".":
                s[2] = ["zihaf"]
            elif nat == "LH" and s[1] == "-":
                s[2] = ["no medd"]
        return forms, syls

    def evaluate(ev, gold, inp, out, inp_name="clean"):
        if inp == "P":
            mc.em.INPUT = "P"
        else:
            mc.em.INPUT = "B"
        line_meter, poem_meter = meters_of(ev)
        gold_texts = {g["text"] for g in gold}
        results = []
        k = 0
        for r in ev:
            rows = []
            for u in r["units"]:
                for m in u["misras"]:
                    e = line_example(r, m, None, inp)
                    ms = {"found": poem_meter[r["id"]], "line": line_meter[k], "gold": r["meter"]}
                    k += 1
                    cache, dec = {}, {}
                    for mode, n in ms.items():
                        if n not in cache:
                            cache[n] = decode(e, n)
                        d = cache[n]
                        dec[mode] = None if d is None else {"forms": d[0], "syllables": d[1] if m in gold_texts else None}
                    rows.append({"text": m, "meters": ms, "decodes": dec})
            results.append({"id": r["id"], "gold": r["meter"], "found": poem_meter[r["id"]], "lines": rows})
        if k != len(line_meter):
            raise SystemExit(f"classifier lines ({len(line_meter)}) and tagger lines ({k}) differ")

        recs = {r["id"]: r for r in ev}
        by_id = {res["id"]: res for res in results}
        system = "BiLSTM tagger"
        rep = {"system": system, "split": args.split, "poems": len(results), "input": inp_name,
               "training": {"lines": dict(stats), "units": len(unit_list), "epochs": epochs,
                            "held-out train loss per line": best, "classifier held-out line accuracy": clf_best,
                            "supervision": "the printed meter (constrained marginal over the readings that fit it) and "
                                           "the edition's spelling; no system's output"},
               "task1 source": "the character BiLSTM meter classifier (scripts/meter_classifier.py settings), retrained here"}
        rep["task1"] = {"poem": pe.task1([res["gold"] for res in results], [res["found"] for res in results]),
                        "line": pe.task1([res["gold"] for res in results for _ in res["lines"]],
                                         [row["meters"]["line"] for res in results for row in res["lines"]])}
        for mode in ("found", "line", "gold"):
            outs, oks = [], []
            for g in gold:
                row = next(x for x in by_id[g["id"]]["lines"] if x["text"] == g["text"])
                d = row["decodes"][mode]
                outs.append(d["syllables"] if d else None)
                oks.append(row["meters"][mode] == g["meter"])
            print(f"\n{system}, meter: {mode}")
            rep[mode] = {"task2": es.score_scansion(system, gold, outs, oks),
                         "words": pe.words_metric(es, recs, results, mode)}
        if args.limit:
            out = out.with_name(f"{system} (smoke).json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
        print(json.dumps({k: rep[k] for k in ("training", "task1")}, ensure_ascii=False, indent=1))
        print(f"wrote {out}")

    results_dir = ROOT / f"data/results/{args.split}"
    out_dir = results_dir / args.results_subdir
    if not args.perturbed and not args.poem_sample:
        evaluate(ev, gold, "B", out_dir / "BiLSTM tagger.json")
        return
    sample = results_dir / "perturbed/poems.json"
    if not sample.exists():
        raise SystemExit(f"{sample} is missing: run paper_eval.py --poem-sample first")
    ids = json.loads(sample.read_text())
    by = {r["id"]: r for r in ev}
    sub = [by[i] for i in ids]
    evaluate(sub, gold, "B", out_dir / "perturbed/clean/BiLSTM tagger.json")
    for f in args.perturbed:
        pert = {}
        for line in open(f, encoding="utf-8"):
            p = json.loads(line)
            pert[p["id"], p["text"]] = p["perturbed"]
        missing = [(r["id"], m) for r in sub for u in r["units"] for m in u["misras"] if (r["id"], m) not in pert]
        if missing:
            raise SystemExit(f"{f}: {len(missing)} lines of the evaluated poems are missing, e.g. {missing[:2]}")
        for mod in (es.meter_eval, mc.em):
            mod.PERTURBED.clear()
            mod.PERTURBED.update(pert)
        evaluate(es.meter_eval.drop_skipped(sub, pert), gold, "P", out_dir / f"perturbed/{f.stem}/BiLSTM tagger.json", str(f))

if __name__ == "__main__":
    main()
