import argparse
import importlib.util
import json
import logging
import os
import random
from pathlib import Path

import torch

from aruz.editions import apply_orthography
from aruz.hmm import Word
from aruz.lexicon import Lexicon
from aruz.meter import METERS, build, symbol
from aruz.scan import VOWELS, build_lattice

ROOT = Path(__file__).resolve().parent.parent

def words(m, r):
    t = apply_orthography(m, r["silent"], r["context"])
    return [w for w in (x.strip(".,;:!?“”\"()«»") for x in t.split()) if any(c in VOWELS for c in w)]

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(os.environ["STORE"]) / "projects/aruz/models/hmm_em.pkl")
    parser.add_argument("--lam", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--dev", type=int, default=500, help="dev mısras for the stopping criterion")
    parser.add_argument("--limit", type=int, default=0, help="first N train poems only (a smoke test)")
    parser.add_argument("--untied", action="store_true", help="ablation: every slot its own emission row instead of one "
                                                              "row per slot kind")
    parser.add_argument("--no-indels", action="store_true", help="ablation (and stage 1): no insert and no skip states; a "
                                                                 "line whose syllable count fits no pattern is left out")
    parser.add_argument("--no-symbols", action="store_true", help="ablation: no O and K symbols; no medd and no ulama "
                                                                  "emit the plain long weight")
    parser.add_argument("--converge", type=float, help="run EM until the training log-likelihood per mısra improves by "
                                                       "less than this fraction, the last epoch kept; else early stopping "
                                                       "on the dev log-likelihood")
    parser.add_argument("--emissions-from", type=Path, help="two-stage training: take the emission rows of the long, short "
                                                            "and free slots from this model and keep them fixed; EM then "
                                                            "learns only the transitions, the meter prior and the insert row")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.emissions_from and args.no_indels:
        raise SystemExit("--emissions-from needs the insert and skip states")
    final = Path(os.environ["STORE"]) / "datasets/ottoman_poetry/final"
    names = {m.name for m in METERS}
    train = [r for r in map(json.loads, open(final / "train.jsonl", encoding="utf-8")) if r["meter"] in names]
    dev = [r for r in map(json.loads, open(final / "dev.jsonl", encoding="utf-8")) if r["meter"] in names]
    if args.limit:
        train = train[:args.limit]
    innate = not args.no_symbols

    model, starts, slots = build(tie=not args.untied, innate=innate,
                                 **(dict(p_delete=0., p_insert=0.) if args.no_indels else {}))
    frozen = None
    if args.emissions_from:
        from aruz.hmm.lattice import LatticeHMM
        src = LatticeHMM.load(args.emissions_from)
        if [str(w) for w in src.vocab] != [str(w) for w in model.vocab] or src.B.shape != model.B.shape:
            raise SystemExit(f"{args.emissions_from}: a different vocabulary or state set")
        frozen = torch.tensor([s.kind != "I" for s in slots] + [False] * (model.B.size(0) - len(slots)))
        model.B[frozen] = src.B[frozen]
        print(f"emissions of the long, short and free slots fixed from {args.emissions_from}")
    lattices, skipped = [], 0
    for r in train:
        for u in r["units"]:
            for m in u["misras"]:
                lat = build_lattice(words(m, r), model.vocab, innate=innate)
                lat.starts = starts[r["meter"]]
                if model.lattice_forward(lat).item() == float("-inf"):
                    skipped += 1
                    continue
                lattices.append(lat)
    print(f"train: {len(lattices)} mısras ({skipped} impossible under the model, left out)")

    spec = importlib.util.spec_from_file_location("eval_meter", ROOT / "scripts/eval_meter.py")
    meter_eval = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(meter_eval)
    lexicon = Lexicon()
    for r in train:
        for u in r["units"]:
            for m in u["misras"]:
                lexicon.update(words(m, r))
    held = [(m, r) for r in dev for u in r["units"] for m in u["misras"]]
    held = random.Random(0).sample(held, args.dev)
    dev_lats = []
    for m, r in held:
        lat = build_lattice([w for w in meter_eval.popular(" ".join(words(m, r))).split()], model.vocab,
                            alternatives=lexicon.alternatives, innate=innate)
        lat.starts = starts[r["meter"]]
        dev_lats.append(lat)

    def loss(hmm):
        scores = [hmm.lattice_forward(lat).item() for lat in dev_lats]
        finite = [s for s in scores if s > float("-inf")]
        if len(finite) < len(scores):
            print(f"dev: {len(scores) - len(finite)} of {len(scores)} mısras impossible under the model, left out")
        total = sum(finite)
        print(f"dev: mean log-likelihood per mısra {total / len(finite):.4f}")
        return -total / len(finite)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    syms = [symbol(w, c) for c in ("tr", "ar") for w in (("L", "S", "H", "O", "K") if innate else ("L", "S", "H"))]

    def emissions(epoch):
        print(f"emissions after epoch {epoch}")
        print("  kind " + " ".join(f"{x:>6s}" for x in syms))
        for kind in ("L", "S", "X", "I"):
            t = next(i for i, sl in enumerate(slots) if sl.kind == kind)
            print(f"  {kind}    " + " ".join(f"{model.B[t, model.vocab.index(Word(x))].item():6.3f}" for x in syms))

    best = (loss(model), model.A.clone(), model.B.clone())
    prev_ll = None
    for epoch in range(args.epochs):
        model._zero_counts()
        train_ll = 0.
        for lat in lattices:
            if args.converge:
                train_ll += model.lattice_forward(lat).item()
            model.E_step(lat)
        model.M_step(args.lam)
        if frozen is not None:
            model.B[frozen] = src.B[frozen]
        if args.converge:
            print(f"train: mean log-likelihood per mısra {train_ll / len(lattices):.4f} (under the model before this M step)")
        dev_loss = loss(model)
        if args.converge:
            ll = train_ll / len(lattices)
            if prev_ll is not None and ll - prev_ll < args.converge * abs(prev_ll):
                print(f"converged: training log-likelihood per mısra {prev_ll:.5f} -> {ll:.5f} before epoch {epoch + 1}'s "
                      f"M step; the last epoch is kept")
                best = (dev_loss, model.A.clone(), model.B.clone())
                break
            prev_ll = ll
            best = (dev_loss, model.A.clone(), model.B.clone())
            if epoch == args.epochs - 1:
                raise SystemExit(f"EM did not converge in {args.epochs} epochs (tolerance {args.converge})")
            continue
        if dev_loss < best[0] * (1 - 1e-4):
            best = (dev_loss, model.A.clone(), model.B.clone())
        else:
            break
    model.A, model.B = best[1:]
    model.epochs = epoch + 1
    emissions("the kept one")
    print(f"variant: symbols {innate}, untied {args.untied}, no indels {args.no_indels}, converge {args.converge}, "
          f"emissions from {args.emissions_from}")
    print(f"kept epoch: dev mean log-likelihood {-best[0]:.4f}")
    model.save(args.out)
    print(f"saved {args.out}")

if __name__ == "__main__":
    main()
