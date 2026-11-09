# aruz

Repo for identification and scansion of Ottoman Turkish verse from plain modern Turkish letters. Code and models for the
paper. Unfortunately,  editions are copyrighted, so no text is included. The corpus is rebuilt from the publishers' PDFs with links provided, and the
gold annotation and the LLM answers are shipped without text and re-joined with the rebuilt corpus. You can search for LATOC and the links for pdfs are given here as well.

Web tool: https://mehmetemretiryaki.com/aruz

## Setup

    pip install -e .
    export STORE=/path/to/data      # all data, models and logs go under here

`WORKERS` sets the number of processes (default: all cores); `PYTHON` picks the interpreter (default `python`).

## PDFs and LATOC

`data/books.tsv` lists the 14 editions: download page, the file name to save the PDF under, and the MD5 of the
file we used. Put the PDFs in `$STORE/datasets/ottoman_poetry/latoc/pdf/` with those names. The corpus build
checks the MD5s and stops on a different file (`--allow-other-pdfs` to continue).

The text is extracted with LATOC: Yılandiloğlu, Enes (2026), *Latin-transliterated Ottoman Turkish Corpus (LATOC)*,
version 5, Zenodo, https://doi.org/10.5281/zenodo.19445113, CC BY 4.0. It is a separate project and not included here:
download the record's files into `third_party/latoc/` and leave them unchanged. `scripts/run_latoc.py` runs them on
our PDFs with our per-book glyph fixes (`configs/latoc_overlay.json`) merged over its `controversy_cache.json` in the
work copy. Its layout step needs its own packages and a GPU:

    pip install torch==2.5.1 torchvision==0.20.1 doclayout-yolo==0.0.4 pymupdf==1.24.14 opencv-python-headless huggingface_hub

Set `LATOC_PYTHON` to that interpreter if it differs from `PYTHON`, then:

    bash scripts/run_latoc.sh              # or: $LATOC_PYTHON scripts/run_latoc.py --device cpu
    bash scripts/rebuild_corpus.sh         # parser, checks, splits -> $STORE/datasets/ottoman_poetry/final/
    python scripts/join_gold.py            # gold files from data/annotation/*_gold_labels.jsonl + your splits

## Train and evaluate

    bash scripts/train.sh                  # restorer, HMM (two stages) and its ablations, tagger (5 seeds)
    bash scripts/benchmark.sh              # robustness benchmark from data/robustness/profiles.json
    bash scripts/evaluate.sh dev
    bash scripts/evaluate.sh test          # all systems, robustness, detector

Results land in `data/results/<split>/` as JSON, one file per system. The trained models are in `models/`; copy them
to `$STORE/projects/aruz/models/` to skip training.

LLM rows (API keys in the environment):

    bash scripts/llm_baseline.sh claude-opus-5.5 anthropic claude-opus-5-5 test --all-poems
    python scripts/llm_baseline.py score --label claude-opus-5.5 --split test --allow-test --all-poems

The paper's two runs are in `data/llm/cache/` as labels only; the `score` command reproduces their numbers without
calling the API.

## Check

    bash scripts/smoke.sh                  # the pipeline after the corpus build, on a few poems
