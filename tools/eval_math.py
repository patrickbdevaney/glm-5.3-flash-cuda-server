#!/usr/bin/env python3
"""Numeric-answer benchmarks (GSM8K today) against a running CUDA server.

WHY THIS EXISTS. The REAP calibration corpus is weighted toward coding, agentic, math, finance
and science, but the only things measured were HumanEval (coding) and BFCL live_multiple
(tool-use). Two of the five axes. A criterion tuned on math with no math eval is an untested
claim, so comparing one REAP plan against another on coding alone cannot settle whether a
selection criterion is better on the axes it was calibrated for.

GSM8K is grade-school arithmetic word problems: 1319 test items, one exact integer answer each,
so scoring is unambiguous and needs no model judge. It is a FLOOR, not a ceiling -- frontier
models saturate it. It is here to catch pruning damage to multi-step arithmetic, which is what
expert pruning plausibly breaks, not to rank strong models against each other. MATH-500,
AIME_2024, GPQA-Diamond and the finqa family are already in the HF cache; TASKS below is the
extension point.

CONFIG IS PINNED, NOT INHERITED. See the long note in eval_humaneval.py: the two servers default
to different reasoning_effort and glm5's default (Max) never terminates. Every request carries an
explicit effort and budget, and every row records whether it was truncated.
"""
import argparse, glob, json, re, sys, time, urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8080"

TASKS = {
    "gsm8k": {
        "glob": "/home/patrickd/.cache/huggingface/hub/datasets--openai--gsm8k/**/main/test-*.parquet",
        "question": "question",
        # GSM8K ships the worked solution; the graded answer is whatever follows '####'.
        "gold": lambda r: r["answer"].split("####")[-1].strip(),
        "instruct": ("Solve the problem. Reason briefly, then give the final numeric answer on "
                     "its own last line as:\nANSWER: <number>\n\n"),
    },
}

NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def norm(s):
    """'$1,234.00' and '1234' are the same answer. Integral floats compare as ints."""
    if s is None: return None
    s = s.replace(",", "").replace("$", "").replace("%", "").strip().rstrip(".")
    try:
        f = float(s)
    except ValueError:
        return None
    return str(int(f)) if f == int(f) else str(f)


def extract(text):
    """Prefer the declared ANSWER: line; fall back to the last number in the reply.

    The fallback matters: a model that reasons correctly but ignores the output format should be
    scored on the arithmetic, not on its obedience. `has_answer_line` records which path was used
    so format-following can be separated from correctness after the fact.
    """
    m = re.findall(r"ANSWER:\s*(.+)", text)
    if m:
        n = NUM.search(m[-1])
        if n: return norm(n.group(0)), True
    n = NUM.findall(text)
    return (norm(n[-1]) if n else None), False


def post(obj, timeout=1800):
    r = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(obj).encode(),
                               {"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as fh:
        return json.loads(fh.read())


def main():
    global BASE
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="gsm8k", choices=sorted(TASKS))
    ap.add_argument("--limit", type=int, default=200, help="0 = the whole split")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--reasoning-effort", default="high", choices=["low", "high"])
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--label", default="glm5")
    ap.add_argument("--out", default="artifacts/gsm8k.json")
    a = ap.parse_args()
    BASE = a.base
    t = TASKS[a.task]

    import pyarrow.parquet as pq
    f = sorted(glob.glob(t["glob"], recursive=True))
    if not f: raise SystemExit(f"{a.task} parquet not found in the HF cache")
    rows_in = pq.read_table(f[0]).to_pylist()
    if a.limit: rows_in = rows_in[:a.limit]

    print(f"{a.task}: {len(rows_in)} items against {BASE} [{a.label}] "
          f"effort={a.reasoning_effort} max_tokens={a.max_tokens}", flush=True)
    npass = ntrunc = nfmt = 0
    rows = []
    t0 = time.time()
    for i, r in enumerate(rows_in):
        gold = norm(t["gold"](r))
        try:
            resp = post({"model": a.label, "temperature": 0.0, "max_tokens": a.max_tokens,
                         "reasoning_effort": a.reasoning_effort,
                         "messages": [{"role": "user",
                                       "content": t["instruct"] + r[t["question"]]}]})
            txt = resp["choices"][0]["message"]["content"] or ""
            ct = int(resp.get("usage", {}).get("completion_tokens", 0))
            err = None
        except Exception as e:
            txt, ct, err = "", 0, repr(e)[:160]
        got, fmt = extract(txt)
        ok = got is not None and got == gold
        trunc = ct >= a.max_tokens
        npass += ok; ntrunc += trunc; nfmt += fmt
        rows.append({"i": i, "pass": bool(ok), "gold": gold, "got": got,
                     "completion_tokens": ct, "truncated": bool(trunc),
                     "has_answer_line": bool(fmt), "error": err,
                     "reply_tail": txt[-300:]})
        print(f"[{i+1}/{len(rows_in)}] {'PASS' if ok else 'fail'} "
              f"got={got} gold={gold} {ct} tok{' TRUNC' if trunc else ''}  "
              f"running {npass}/{i+1} = {npass/(i+1):.1%}", flush=True)
        Path(a.out).write_text(json.dumps(
            {"task": a.task, "n": len(rows), "pass": npass, "acc": npass / len(rows),
             "max_tokens": a.max_tokens, "reasoning_effort": a.reasoning_effort,
             "truncated": ntrunc, "answer_line": nfmt, "label": a.label, "rows": rows}, indent=1))
    ct = sorted(r["completion_tokens"] for r in rows)
    print(f"\n{a.task} accuracy = {npass}/{len(rows)} = {npass/len(rows):.1%} "
          f"({(time.time()-t0)/60:.0f} min)")
    print(f"completion_tokens  median {ct[len(ct)//2]}  max {ct[-1]}   budget {a.max_tokens}")
    print(f"followed ANSWER: format {nfmt}/{len(rows)}")
    print(f"TRUNCATED {ntrunc}/{len(rows)}"
          + ("  <-- budget-bound, accuracy is NOT a capability number" if ntrunc else "  (clean)"))


main()
