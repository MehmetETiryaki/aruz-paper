import argparse
import datetime as dt
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STORE = Path(os.environ["STORE"])
CACHE = Path(os.environ.get("ARUZ_LLM_CACHE", ROOT / "data/llm/cache"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from task2_metrics import stream as letter_stream

KUSUR = ("imâle", "zihaf", "no medd", "no ulama")

def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

def fewshot(es, n=6):
    from aruz.reading import from_annotation
    final = STORE / "datasets/ottoman_poetry/final"
    recs = {r["id"]: r for r in map(json.loads, open(final / "dev.jsonl", encoding="utf-8"))}
    gold = [json.loads(l) for l in open(ROOT / "data/annotation/dev_gold.jsonl", encoding="utf-8")]
    labels = lambda g: {s[2] for s in g["syllables"]} - {"none", "free"}
    want = set().union(*(labels(g) for g in gold))
    chosen, covered = [], set()
    while want - covered:
        best = max(gold, key=lambda g: (len(labels(g) & (want - covered)), -len(g["syllables"])))
        chosen.append(best); covered |= labels(best)
    for g in sorted(gold, key=lambda g: (len(g["syllables"]), g["id"])):
        if len(chosen) >= n:
            break
        if g["meter"] not in {c["meter"] for c in chosen} and g["id"] not in {c["id"] for c in chosen}:
            chosen.append(g)
    out = []
    for g in chosen:
        r = recs[g["id"]]
        plain = " ".join(es.tokens(g["text"], r, "B"))
        fold = es.meter_eval.popular
        syl = json.dumps([[fold(t), sl, k] for t, sl, k in from_annotation(g["syllables"])], ensure_ascii=False)
        out.append({"id": g["id"], "meter": g["meter"], "input": plain, "syllables": syl, "words": transcribed(g["text"])})
    return out

def transcribed(text):
    table = {"ā": "â", "ī": "î", "ū": "û", "Ā": "Â", "Ī": "Î", "Ū": "Û", "ʿ": "'", "ʾ": "'", "ʼ": "'", "’": "'",
             "ḫ": "h", "ḥ": "h", "ṣ": "s", "ṭ": "t", "ẓ": "z", "ḳ": "k", "ż": "z", "ẕ": "z", "ŝ": "s", "ḍ": "d", "ġ": "g",
             "ñ": "n", "Ḫ": "H", "Ḥ": "H", "Ṣ": "S", "Ṭ": "T", "Ẓ": "Z", "Ḳ": "K", "Ġ": "G"}
    return "".join(table.get(c, c) for c in text)

def instructions(examples=(), variant="plain"):
    from aruz.meter import METERS
    w = max(len(m.name) for m in METERS)
    meters = "\n".join(f"{m.name:<{w}}  {m.display:<52}  {' or '.join(p.replace(' ', '') for p in m.patterns)}" for m in METERS)
    if variant == "restore":
        shots = "\n\n".join(f"Meter: {e['meter']}\nLine: {e['input']}\nWords: {e['words']}\nScansion: {e['syllables']}"
                            for e in examples) or "(none)"
        task_tail = (" For every line you first write its words in full transcription, long vowels marked, then its "
                     "scansion: the syllables as read, the slot each fills, and its kusur.")
        words_par = """- words: the line's words in transcription, in Turkish letters with â î û for long vowels, ' for ayn and hemze, and hyphens for the izafet and compounds. Each syllable's own weight follows from this form; a kusur is noted only where the reading departs from it.
"""
        line_fmt = '{{"words": "<transcription>", "syllables": [[text, slots, kusur], ...]}}'
        each = "Each line is " + line_fmt + "; each syllable is [text, slots, kusur]."
    else:
        shots = "\n\n".join(f"Meter: {e['meter']}\nLine: {e['input']}\nScansion: {e['syllables']}" for e in examples) or "(none)"
        task_tail = " Your answer gives the poem's meter, one of the fourteen below, and the scansion of every line: the syllables as read, the slot each fills, and its kusur."
        words_par = ""
        each = "Each syllable is [text, slots, kusur]."
    return f"""You are an expert in Ottoman Turkish prosody (aruz).

# Task
You get one Ottoman Divan poem in plain modern Turkish letters. This spelling does not mark vowel length, ayn or hemze and writes no hyphens, so the words carry none of the information a transcription would; recognizing the words, their long vowels and their weights is part of the task.{task_tail}

# Meters
Patterns: - long slot, . short slot, x either. The last slot of a line takes any syllable.
{"id":<{w}}  {"feet":<52}  pattern
{meters}

# Scansion
Weights: a syllable is short when it is open with a short vowel, long when it is closed or has a long vowel, overlong when a long vowel or two consonants close it.
{each}
{words_par}- text: the syllable as read, in the input's letters. Linking (ulama): a word-final consonant before a vowel-initial word is normally read with the next word; write it on the next syllable (mihr ü -> mih, rü). Where it is not read over, keep it on its own word and mark "no ulama".
- slots: "-" long, "." short, "-." an overlong syllable filling a long and a short slot (medd). Joined over the line, the slots form one of the meter's patterns.
- kusur: how the syllable is read differently from its own weight so that it fits its slot. A syllable read at its own weight has none, whatever that weight is. One of
  "imâle"     a short syllable read long: a short open syllable in a long slot, the Persian izafet -i and the conjunction u/ü included when they are read long
  "zihaf"     a long syllable read short: a long vowel in a short slot
  "no medd"   an overlong syllable read as one long slot instead of two, not at the end of the line
  "no ulama"  a linkable consonant kept on its own word
  The last syllable of a line is free and has no kusur.

# Output
One JSON object, nothing else: {{"meter": "<id>", "lines": [<line 1>, <line 2>, ...]}}, one entry per input line, in order.

# Examples (single lines from other poems)
{shots}"""

_SYLLABLES = {"type": "array", "items": {
    "type": "array", "items": {"anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}]}}}
JSON_SCHEMA = {
    "plain": {"type": "object", "properties": {"meter": {"type": "string"}, "lines": {"type": "array", "items": _SYLLABLES}},
              "required": ["meter", "lines"], "additionalProperties": False},
    "restore": {"type": "object", "properties": {"meter": {"type": "string"}, "lines": {"type": "array", "items": {
                    "type": "object", "properties": {"words": {"type": "string"}, "syllables": _SYLLABLES},
                    "required": ["words", "syllables"], "additionalProperties": False}}},
                "required": ["meter", "lines"], "additionalProperties": False}}

def poem_prompt(r, es):
    lines = [" ".join(es.tokens(m, r, "B")) for u in r["units"] for m in u["misras"]]
    body = "\n".join(f"{i + 1}. {l}" for i, l in enumerate(lines))
    return lines, f"Poem ({len(lines)} lines), plain modern Turkish letters:\n{body}\n\nAnswer with the JSON object only."

def load(args):
    es = module("eval_scansion")
    shots = fewshot(es)
    args.shots = shots
    from aruz.meter import METERS
    names = {m.name for m in METERS}
    final = STORE / "datasets/ottoman_poetry/final"
    recs = {r["id"]: r for r in map(json.loads, open(final / f"{args.split}.jsonl", encoding="utf-8")) if r["meter"] in names}
    gold = [json.loads(l) for l in open(ROOT / f"data/annotation/{args.split}_gold.jsonl", encoding="utf-8")]
    with_gold = {g["id"] for g in gold}
    poems = list(recs.values()) if args.all_poems else [recs[i] for i in sorted(with_gold)]
    if args.split == "dev":
        shown = {e["id"] for e in shots}
        poems = [r for r in poems if r["id"] not in shown]
        gold = [g for g in gold if g["id"] not in shown]
        print(f"dev: {len(shown)} poems serve as the prompt's examples and are left out of the evaluation", file=sys.stderr)
    if args.limit:
        poems = [r for r in poems if r["id"] in with_gold][:args.limit]
        keep = {r["id"] for r in poems}
        gold = [g for g in gold if g["id"] in keep]
    return es, recs, gold, poems

def cache_dir(label):
    d = CACHE / label
    d.mkdir(parents=True, exist_ok=True)
    return d

def cache_file(label, poem_id):
    return cache_dir(label) / (re.sub(r"[^\w#-]+", "_", poem_id) + ".json")

def cached_cost(label, pid):
    f = cache_file(label, pid)
    if not f.exists():
        return 0.0
    try:
        return float((((json.loads(f.read_text(encoding="utf-8")).get("meta") or {}).get("usage") or {}).get("cost")) or 0)
    except (ValueError, TypeError, json.JSONDecodeError):
        return 0.0

def call_anthropic(model, system, user, effort, schema):
    import anthropic
    client = anthropic.Anthropic(max_retries=5)
    kw = dict(model=model, max_tokens=128000, messages=[{"role": "user", "content": user}],
              system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
              output_config={"format": {"type": "json_schema", "schema": schema}})
    if effort == "none":
        kw["thinking"] = {"type": "disabled"}
    elif effort:
        kw["output_config"]["effort"] = effort
    with client.messages.stream(**kw) as stream:
        r = stream.get_final_message()
    if r.stop_reason == "refusal":
        return None, {"stop_reason": "refusal", "usage": r.usage.to_dict()}
    text = "".join(b.text for b in r.content if b.type == "text")
    return text, {"stop_reason": r.stop_reason, "usage": r.usage.to_dict(), "model": r.model}

def call_openai(model, system, user, effort, schema):
    from openai import OpenAI
    client = OpenAI(max_retries=5)
    kw = dict(model=model, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
              response_format={"type": "json_object"})
    if effort:
        kw["reasoning_effort"] = effort
    r = client.chat.completions.create(**kw)
    c = r.choices[0]
    return c.message.content, {"finish_reason": c.finish_reason, "usage": r.usage.model_dump() if r.usage else None,
                               "model": r.model}

def call_gemini(model, system, user, effort, schema):
    from google import genai
    from google.genai import types
    client = genai.Client()
    cfg = types.GenerateContentConfig(system_instruction=system, response_mime_type="application/json")
    r = client.models.generate_content(model=model, contents=user, config=cfg)
    meta = r.usage_metadata
    return r.text, {"usage": meta.model_dump() if meta is not None else None, "model": model}

def call_openrouter(model, system, user, effort, schema):
    import os
    from openai import OpenAI
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"], max_retries=2,
                    timeout=3600)
    kw = dict(model=model, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
              response_format={"type": "json_object"}, max_tokens=128000,
              extra_body={"provider": {"allow_fallbacks": True, "data_collection": "deny"}})
    if effort:
        kw["extra_body"]["reasoning"] = {"effort": effort}
    r = client.chat.completions.create(**kw)
    c = r.choices[0]
    served = getattr(r, "provider", None) or (r.model_extra or {}).get("provider")
    if c.finish_reason == "error" or not c.message.content:
        raise RuntimeError(f"{served}: finish_reason {c.finish_reason!r}, empty answer")
    return c.message.content, {"finish_reason": c.finish_reason, "usage": r.usage.model_dump() if r.usage else None,
                               "model": r.model, "provider": served}

PROVIDERS = {"anthropic": call_anthropic, "openai": call_openai, "gemini": call_gemini, "openrouter": call_openrouter}

def parse_answer(text):
    if not text:
        return None
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    cand = m.group(1) if m else text[text.find("{"): text.rfind("}") + 1]
    try:
        return json.loads(cand)
    except json.JSONDecodeError:
        return None

def valid_line(syls, patterns):
    try:
        out = []
        for s in syls:
            text, slots, kusur = s[0], s[1], s[2]
            if isinstance(kusur, str):
                kusur = [k for k in re.split(r"\s*\+\s*|,\s*", kusur) if k and k != "none"]
            kusur = [k.replace("imale", "imâle") for k in kusur]
            if slots not in (".", "-", "-.", "x") or any(k not in KUSUR for k in kusur) or not isinstance(text, str):
                return None
            out.append([text, slots, kusur])
    except (TypeError, IndexError, KeyError):
        return None
    joined = "".join(s[1] for s in out)
    ok = any(len(joined) == len(p) and all(a == b or b == "x" or i == len(p) - 1 for i, (a, b) in enumerate(zip(joined, p)))
             and all(a != "x" or b == "x" or i == len(p) - 1 for i, (a, b) in enumerate(zip(joined, p))) for p in patterns)
    return out if ok else None

def cmd_estimate(args):
    es, recs, gold, poems = load(args)
    sys_tokens = len(instructions(args.shots, args.prompt)) / 3.5
    chars = sum(len(l) for r in poems for l in poem_prompt(r, es)[0])
    nlines = sum(len(u["misras"]) for r in poems for u in r["units"])
    in_tok = len(poems) * sys_tokens + chars / 3
    out_tok = nlines * 90
    print(f"{args.split}: {len(poems)} poems, {nlines} lines, {chars:,} characters of poem text")
    print(f"estimated tokens: input {in_tok / 1e6:.2f}M (instructions {len(poems) * sys_tokens / 1e6:.2f}M + poems "
          f"{chars / 3e6:.2f}M), output {out_tok / 1e6:.2f}M (more for a model that bills its thinking as output)")
    if args.price_in is not None and args.price_out is not None:
        cost = in_tok / 1e6 * args.price_in + out_tok / 1e6 * args.price_out
        print(f"at ${args.price_in}/M in and ${args.price_out}/M out: about ${cost:.2f} per run "
              f"(instructions cached: about ${(chars / 3e6 * args.price_in + out_tok / 1e6 * args.price_out):.2f})")

def cmd_run(args):
    es, recs, gold, poems = load(args)
    if args.provider not in PROVIDERS:
        raise SystemExit(f"provider {args.provider!r}: one of {sorted(PROVIDERS)}")
    call = PROVIDERS[args.provider]
    system = instructions(args.shots, args.prompt)
    todo = [r for r in poems if not cache_file(args.label, r["id"]).exists()]
    with_gold = {g["id"] for g in gold}
    todo.sort(key=lambda r: r["id"] not in with_gold)
    print(f"{args.label}: {len(poems)} poems, {len(poems) - len(todo)} cached, {len(todo)} to ask ({args.provider} {args.model})")
    if args.max_calls:
        todo = todo[:args.max_calls]
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading
    spent = [sum(cached_cost(args.label, r["id"]) for r in poems)]
    lock = threading.Lock()
    if args.budget_usd:
        print(f"budget ${args.budget_usd:.2f}; spent so far ${spent[0]:.2f} (from the cached answers' billed cost)")

    def ask(r):
        if args.budget_usd and spent[0] >= args.budget_usd:
            return None
        lines, user = poem_prompt(r, es)
        t0 = time.time()
        text, meta = call(args.model, system, user, args.effort, JSON_SCHEMA[args.prompt])
        with lock:
            spent[0] += ((meta or {}).get("usage") or {}).get("cost") or 0
        rec = {"id": r["id"], "provider": args.provider, "model": args.model, "effort": args.effort, "prompt": args.prompt,
               "lines": lines,
               "asked": dt.datetime.now().isoformat(timespec="seconds"), "seconds": round(time.time() - t0, 1),
               "answer": text, "meta": meta}
        cache_file(args.label, r["id"]).write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        return rec

    done = skipped = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(ask, r): r for r in todo}
        for fut in as_completed(futures):
            r = futures[fut]; done += 1
            try:
                rec = fut.result()
            except Exception as e:
                print(f"  {done}/{len(todo)} {r['id']}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
                if args.stop_on_error:
                    raise
                continue
            if rec is None:
                skipped += 1
                continue
            u = (rec["meta"] or {}).get("usage") or {}
            short = {k: u.get(k) for k in ("prompt_tokens", "completion_tokens", "input_tokens", "output_tokens",
                                           "prompt_token_count", "candidates_token_count", "thoughts_token_count") if u.get(k) is not None}
            print(f"  {done}/{len(todo)} {r['id']}: {'ok' if parse_answer(rec['answer']) else 'NO JSON'} {rec['seconds']}s {short}", flush=True)
    if args.budget_usd:
        print(f"billed so far ${spent[0]:.2f} of the ${args.budget_usd:.2f} budget"
              + (f"; BUDGET REACHED: {skipped} poems not asked" if skipped else ""))
    print(f"cached answers: {sum(1 for r in poems if cache_file(args.label, r['id']).exists())} of {len(poems)}; "
          f"now: python scripts/llm_baseline.py score --label {args.label} --split {args.split}"
          + (" --allow-test" if args.split == "test" else ""))

def cmd_prepare_web(args):
    es, recs, gold, poems = load(args)
    d = CACHE / "web" / args.label
    d.mkdir(parents=True, exist_ok=True)
    (d / "INSTRUCTIONS.md").write_text(instructions(args.shots, args.prompt) + "\n", encoding="utf-8")
    manifest = []
    for b, start in enumerate(range(0, len(poems), args.batch_size), 1):
        batch = poems[start:start + args.batch_size]
        parts = [instructions(args.shots, args.prompt), "", f"Below are {len(batch)} poems, each with an id. Answer with one JSON object per poem, "
                 "each inside its own ```json block, in order, with the poem's id added as \"id\". Nothing else."]
        for r in batch:
            lines, user = poem_prompt(r, es)
            parts += ["", f"### id: {r['id']}", user.replace("\n\nAnswer with the JSON object only.", "")]
        (d / f"batch_{b:03d}.md").write_text("\n".join(parts) + "\n", encoding="utf-8")
        manifest.append({"batch": b, "ids": [r["id"] for r in batch]})
    (d / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(manifest)} batch files in {d}: paste each batch_NNN.md into a fresh chat (memory, personalization, "
          f"browsing and tools off), save the whole answer as batch_NNN.response.txt next to it, note the model and "
          f"date in notes.txt, then: ingest-web --label {args.label}")

def cmd_ingest_web(args):
    es, recs, gold, poems = load(args)
    d = CACHE / "web" / args.label
    manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    n = 0
    for entry in manifest:
        f = d / f"batch_{entry['batch']:03d}.response.txt"
        if not f.exists():
            print(f"missing: {f.name}", file=sys.stderr)
            continue
        text = f.read_text(encoding="utf-8")
        blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S) or re.findall(r"(\{.*?\})(?=\s*\{|\s*$)", text, re.S)
        answers = {}
        for i, blk in enumerate(blocks):
            try:
                obj = json.loads(blk)
            except json.JSONDecodeError:
                continue
            pid = obj.get("id") or (entry["ids"][i] if i < len(entry["ids"]) else None)
            if pid in entry["ids"]:
                answers[pid] = blk
        for pid in entry["ids"]:
            r = recs[pid]
            lines, _ = poem_prompt(r, es)
            rec = {"id": pid, "provider": "web", "model": args.model or args.label, "lines": lines,
                   "asked": dt.datetime.fromtimestamp(f.stat().st_mtime).isoformat(timespec="seconds"),
                   "answer": answers.get(pid), "meta": {"batch": entry["batch"], "parsed": pid in answers}}
            cache_file(args.label, pid).write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
            n += pid in answers
    print(f"{n} answers ingested into {cache_dir(args.label)}; poems without an answer count as wrong at scoring")

def cmd_selftest(args):
    from aruz.reading import from_annotation
    es, recs, gold, poems = load(args)
    by_text = {(g["id"], g["text"]): g for g in gold}
    for r in poems:
        lines, _ = poem_prompt(r, es)
        out = []
        for u in r["units"]:
            for m in u["misras"]:
                g = by_text.get((r["id"], m))
                syl = from_annotation(g["syllables"]) if g else [[w, "-", []] for w in m.split()]
                out.append({"words": transcribed(g["text"] if g else m), "syllables": syl} if args.prompt == "restore" else syl)
        rec = {"id": r["id"], "provider": "selftest", "model": "gold", "lines": lines,
               "answer": json.dumps({"meter": r["meter"], "lines": out}, ensure_ascii=False), "meta": {}}
        cache_file(args.label, r["id"]).write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
    print(f"selftest cache written for {len(poems)} poems under label {args.label!r}")
    cmd_score(args)

def rebuild_answer(labels, plain_lines):
    if not isinstance(labels, dict):
        return None
    rows = labels.get("lines")
    if not isinstance(rows, list):
        return {"meter": labels.get("meter"), "lines": None}
    out = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            out.append(None)
            continue
        chars = list(letter_stream(plain_lines[i])) if i < len(plain_lines) else []
        got, at = [], 0
        for tag, i1, i2, repl in row.get("edits") or []:
            got += chars[at:i1]
            if tag in ("replace", "insert"):
                got += list(repl)
            at = i2
        got += chars[at:]
        syls, at = [], 0
        for n, slots, kusur in row["syllables"]:
            syls.append(["".join(got[at:at + n]), slots, kusur])
            at += n
        out.append(syls)
    return {"meter": labels.get("meter"), "lines": out}

def cmd_score(args):
    es, recs, gold, poems = load(args)
    pe = module("paper_eval")
    patterns = dict(es.INVENTORY)
    system = f"LLM {args.label}"
    gp, pp, gl, pl = [], [], [], []
    outs, oks = [], []
    answered = parsed_n = 0
    meta_models = set()
    per_poem = {}
    for r in poems:
        f = cache_file(args.label, r["id"])
        rec = json.loads(f.read_text(encoding="utf-8")) if f.exists() else None
        lines = [m for u in r["units"] for m in u["misras"]]
        if rec is not None and "labels" in rec and "answer" not in rec:
            obj = rebuild_answer(rec["labels"], poem_prompt(r, es)[0])
            answered += bool(rec.get("answered"))
        else:
            obj = parse_answer(rec["answer"]) if rec else None
            answered += bool(rec and rec.get("answer"))
        if rec and (rec.get("meta") or {}).get("model"):
            meta_models.add(rec["meta"]["model"])
        found = obj.get("meter") if isinstance(obj, dict) else None
        if found not in patterns:
            found = None
        rows = obj.get("lines") if isinstance(obj, dict) and isinstance(obj.get("lines"), list) else None
        if rows is not None and len(rows) != len(lines):
            rows = None
        parsed_n += rows is not None and found is not None
        per_poem[r["id"]] = {"found": found, "rows": rows}
        gp.append(r["meter"]); pp.append(found or "none")
        for i, m in enumerate(lines):
            gl.append(r["meter"]); pl.append(found or "none")
            pass
    for g in gold:
        pp_ = per_poem[g["id"]]
        lines = [m for u in recs[g["id"]]["units"] for m in u["misras"]]
        i = lines.index(g["text"])
        syl = None
        if pp_["found"] and pp_["rows"] is not None:
            row = pp_["rows"][i]
            row = row.get("syllables") if isinstance(row, dict) else row
            syl = valid_line(row, patterns[pp_["found"]]) if isinstance(row, list) else None
        outs.append(syl)
        oks.append(pp_["found"] == g["meter"])
    print(f"\n{system}: {len(poems)} poems, {answered} answered, {parsed_n} parsed and well-formed")
    t2 = es.score_scansion(system, gold, outs, oks)
    rep = {"system": system, "spec": {"reader": "llm", "label": args.label, "models": sorted(meta_models),
                                      "cache": str(cache_dir(args.label)), "poems asked": len(poems),
                                      "answered": answered, "well-formed": parsed_n},
           "split": args.split, "poems": len(poems), "input": "clean",
           "task1": {"poem": pe.task1(gp, pp), "line": pe.task1(gl, pl)},
           "found": {"task2": t2, "words": {"words": 0, "spelling": None, "weights": None}}}
    rep["line"] = rep["found"]
    out = ROOT / f"data/results/{args.split}" / getattr(args, "results_subdir", "") / f"{system}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"task 1 {rep['task1']}")
    print(f"wrote {out}")
    if not args.all_poems:
        print("note: Task 1 here is over the poems with gold lines only; --all-poems asks every test poem "
              "(about six times the cost) for a Task 1 figure comparable with Table 1")

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    def common(p, label=True):
        p.add_argument("--split", default="dev", choices=["dev", "test"])
        p.add_argument("--allow-test", action="store_true")
        p.add_argument("--all-poems", action="store_true", help="every poem of the split, not only those with gold lines")
        p.add_argument("--limit", type=int, default=0, help="first N poems with gold lines (a pilot)")
        p.add_argument("--prompt", default="plain", choices=["plain", "restore"],
                       help="plain: scansion only; restore: each line's words in transcription first, then the scansion")
        if label:
            p.add_argument("--label", required=True, help="the system's label: data/results/<split>/LLM <label>.json")
    p = sub.add_parser("estimate"); common(p, label=False)
    p.add_argument("--price-in", type=float); p.add_argument("--price-out", type=float)
    p = sub.add_parser("run"); common(p)
    p.add_argument("--provider", required=True, choices=sorted(PROVIDERS)); p.add_argument("--model", required=True)
    p.add_argument("--effort", default=None, help="anthropic: output_config.effort (low|medium|high|xhigh|max), or none "
                   "= thinking disabled (pre-5 models); openai/openrouter: reasoning_effort, none = no reasoning; gemini: ignored")
    p.add_argument("--max-calls", type=int, default=0, help="stop after N calls (a pilot)")
    p.add_argument("--workers", type=int, default=4, help="parallel requests")
    p.add_argument("--stop-on-error", action="store_true")
    p.add_argument("--budget-usd", type=float, default=0, help="stop asking once this label's billed cost (summed from the "
                   "cached answers' usage.cost, which OpenRouter reports per call) reaches this many dollars")
    p = sub.add_parser("prepare-web"); common(p); p.add_argument("--batch-size", type=int, default=5)
    p = sub.add_parser("ingest-web"); common(p); p.add_argument("--model", default=None, help="the model name shown by the interface")
    p = sub.add_parser("score"); common(p); p.add_argument("--results-subdir", default="")
    p = sub.add_parser("selftest"); common(p)
    args = ap.parse_args()
    if args.split == "test" and not args.allow_test:
        raise SystemExit("test is sealed: pass --allow-test only for the final evaluation")
    {"estimate": cmd_estimate, "run": cmd_run, "prepare-web": cmd_prepare_web, "ingest-web": cmd_ingest_web,
     "score": cmd_score, "selftest": cmd_selftest}[args.cmd](args)

if __name__ == "__main__":
    main()
