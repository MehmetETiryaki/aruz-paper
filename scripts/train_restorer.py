import argparse
import json
import os
import pickle
from collections import Counter
from pathlib import Path

from aruz.lexicon import Lexicon, reduce
from aruz.restore import JointNgram, Restorer
from aruz.scan import syllabify, weight

ROOT = Path(__file__).resolve().parent.parent

def module(name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

meter_eval = module("eval_meter")

def weights(form):
    return [weight(*s) for s in syllabify(form)]

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(os.environ["STORE"]) / "projects/aruz/models/restorer.pkl")
    parser.add_argument("--orders", type=int, nargs="+", default=[3, 4, 5, 6])
    parser.add_argument("--weights", type=float, nargs="+", default=[.5, .8, .95])
    parser.add_argument("--cased", action="store_true", help="read capitals: the letter model keeps the plain letter's "
                        "case, and a word typed with a capital seen in training uses its own lexicon counts, backed off "
                        "to the lowercase ones (Witten-Bell)")
    args = parser.parse_args()
    final = Path(os.environ["STORE"]) / "datasets/ottoman_poetry/final"
    meter_eval.INPUT = "A"
    train, lexicon = Counter(), Lexicon()
    for l in open(final / "train.jsonl", encoding="utf-8"):
        r = json.loads(l)
        for u in r["units"]:
            for m in u["misras"]:
                ws = meter_eval.words(m, r)
                lexicon.update(ws)
                train.update(reduce(w, keep_case=args.cased) for w in ws)
    dev = Counter()
    for l in open(final / "dev.jsonl", encoding="utf-8"):
        r = json.loads(l)
        for u in r["units"]:
            for m in u["misras"]:
                gold = [reduce(w) for w in meter_eval.words(m, r)]
                plain = meter_eval.popular(" ".join(meter_eval.words(m, r))).split()
                if len(plain) == len(gold):
                    dev.update(zip(plain, gold))
    n = sum(dev.values())
    print(f"train: {len(train)} word types; dev: {n} words, {len(dev)} (plain, edition) types")

    best = None
    for order in args.orders:
        model = JointNgram(order, cased=args.cased)
        model.train(train.items())
        for w in args.weights:
            restorer = Restorer(lexicon, model, weight=w, cased=args.cased)
            right = oracle = 0
            for (p, g), c in dev.items():
                cands = [f for f, _ in restorer.alternatives(p)]
                right += c * (weights(cands[0]) == weights(g))
                oracle += c * any(weights(f) == weights(g) for f in cands)
            print(f"order {order}, lexicon weight {w}: top candidate has the edition's weights {right / n:.1%}; "
                  f"some candidate has them {oracle / n:.1%}")
            if best is None or oracle + right > best[0]:
                best = (oracle + right, order, w, restorer)
    _, order, w, restorer = best
    print(f"chosen: order {order}, lexicon weight {w}, cased {args.cased}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(restorer, f)
    print(f"saved {args.out}")

if __name__ == "__main__":
    main()
