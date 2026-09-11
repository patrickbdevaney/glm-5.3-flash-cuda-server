#!/usr/bin/env python3
"""HumanEval pass@1 against the running NVFP4 CUDA server. Executable, not a proxy.

WHY THIS EXISTS. Every quality number this project has produced -- dNLL, top-1 agreement vs
the FP8 teacher, reconstruction error, retained routing mass -- is a PROXY. None of them is
"does it write working code". Three independent papers in the 2026 compression literature
measured perplexity MIS-RANKING compressed models, one noting it "can rate a broken model
above an intact one", so the entire quality picture rests on metrics that literature warns
against using alone.

HumanEval is chosen over a larger suite deliberately: 164 problems with executable tests, a
couple of hours, and it is what the MoE-compression papers report, so the number is directly
comparable to published pruning/narrowing results rather than only to ourselves.

Generated code is executed in a subprocess with a timeout and a hard memory cap. It is model
output; it is not trusted.
"""
import argparse, glob, json, re, subprocess, sys, tempfile, time, urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8080"   # overridden by --base
HE = "/home/patrickd/.cache/huggingface/hub/datasets--openai--openai_humaneval/**/*.parquet"

RUNNER = r'''
import resource, sys
resource.setrlimit(resource.RLIMIT_AS, (2*1024**3, 2*1024**3))
sys.setrecursionlimit(10000)
exec(open(sys.argv[1]).read(), {"__name__": "__main__"})
print("__PASS__")
'''


def load_problems(limit):
    import pyarrow.parquet as pq
    f = sorted(glob.glob(HE, recursive=True))
    if not f:
        raise SystemExit("HumanEval parquet not found in the HF cache")
    rows = pq.read_table(f[0]).to_pylist()
    return rows[:limit] if limit else rows


def post(path, obj, timeout=1800):
    r = urllib.request.Request(BASE + path, json.dumps(obj).encode(),
                               {"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as fh:
        return json.loads(fh.read())


def extract(text, entry_point):
    """Pull the function body out of a chat reply.

    The model emits reasoning before the answer, so a naive 'first code block' grab picks up
    scratch work. Prefer the LAST fenced block that actually defines the entry point.
    """
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    cand = [b for b in blocks if f"def {entry_point}" in b]
    if cand:
        return cand[-1]
    if blocks:
        return blocks[-1]
    # unfenced: take from the first def onward
    i = text.find(f"def {entry_point}")
    return text[i:] if i >= 0 else text


def run_one(code, test, entry_point, timeout=15):
    prog = f"{code}\n\n{test}\n\ncheck({entry_point})\n"
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "cand.py"; p.write_text(prog)
        r = Path(d) / "run.py"; r.write_text(RUNNER)
        try:
            out = subprocess.run([sys.executable, str(r), str(p)], capture_output=True,
                                 text=True, timeout=timeout)
            return "__PASS__" in out.stdout, (out.stderr or "")[-200:]
        except subprocess.TimeoutExpired:
            return False, "timeout"
        except Exception as e:
            return False, repr(e)[:200]


def main():
    global BASE
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 = all 164")
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--out", default="artifacts/humaneval.json")
    ap.add_argument("--base", default=BASE,
                    help="server root; lets the SAME harness score both stacks")
    ap.add_argument("--label", default="glm5")
    # MUST be set explicitly, and identically for both stacks. The two servers disagree on the
    # default: dsv4 defaults to "low" (its openai_api.h:34) while glm5 renders an absent value as
    # Max (its openai_api.h:63) -- and glm5 at Max DEGENERATES, looping until it hits the token
    # cap with an empty `content` and everything stranded in `reasoning_content`. Measured
    # 2026-09-11: HumanEval/0 at Max burned 8192 tok, repetition fraction 0.61, no answer; the
    # same problem at "high" finished in 172 tok. The 41.5%-vs-77.4% result of that morning was
    # a looping GLM scored against a low-effort DeepSeek, not a capability gap.
    ap.add_argument("--reasoning-effort", default="high", choices=["low", "high"],
                    help="sent in the request body; 'max'/unset is the degenerate path on glm5")
    a = ap.parse_args()
    BASE = a.base
    probs = load_problems(a.limit)
    print(f"HumanEval: {len(probs)} problems against {BASE} [{a.label}] "
          f"effort={a.reasoning_effort} max_tokens={a.max_tokens}", flush=True)
    rows, npass, ntrunc = [], 0, 0
    t0 = time.time()
    for i, p in enumerate(probs):
        prompt = ("Complete this Python function. Reply with the full function in a single "
                  "```python code block, no explanation.\n\n" + p["prompt"])
        try:
            r = post("/v1/chat/completions",
                     {"model": a.label, "temperature": 0.0, "max_tokens": a.max_tokens,
                      "reasoning_effort": a.reasoning_effort,
                      "messages": [{"role": "user", "content": prompt}]})
            txt = r["choices"][0]["message"]["content"]
            ct = int(r.get("usage", {}).get("completion_tokens", 0))
            fr = r["choices"][0].get("finish_reason")
            err = None
        except Exception as e:
            txt, ct, fr, err = "", 0, None, repr(e)[:200]
        code = extract(txt, p["entry_point"])
        ok, detail = (False, err) if err else run_one(code, p["test"], p["entry_point"])
        npass += ok
        # WHY THESE FIELDS. The 2026-09-11 run scored GLM at 41.5% and DeepSeek at 77.4%, and
        # the gap was ENTIRELY budget: 92 of GLM's 96 failures emitted no `def` at all because
        # 1536 tokens of reasoning ran out before any code was written, and ZERO were wrong
        # answers. A pass@1 with no truncation column cannot tell a weak model from a starved
        # one. dsv4's server hardcodes finish_reason="stop", so `truncated` is derived from the
        # token count, which is true for both stacks.
        trunc = ct >= a.max_tokens
        rows.append({"task_id": p["task_id"], "pass": bool(ok), "detail": detail,
                     "completion_tokens": ct, "finish_reason": fr, "truncated": bool(trunc),
                     "has_def": f"def {p['entry_point']}" in code,
                     "reply_head": txt[:400], "reply_tail": txt[-400:]})
        ntrunc += trunc
        print(f"[{i+1}/{len(probs)}] {p['task_id']} {'PASS' if ok else 'fail'}  "
              f"{ct} tok{' TRUNC' if trunc else ''}  "
              f"running {npass}/{i+1} = {npass/(i+1):.1%}", flush=True)
        Path(a.out).write_text(json.dumps(
            {"n": len(rows), "pass": npass, "pass@1": npass/len(rows),
             "max_tokens": a.max_tokens, "truncated": ntrunc, "label": a.label,
             "reasoning_effort": a.reasoning_effort,
             "rows": rows}, indent=1))
    dt = time.time() - t0
    ct = sorted(r["completion_tokens"] for r in rows)
    def pct(q): return ct[min(len(ct) - 1, int(q * len(ct)))] if ct else 0
    print(f"\nHumanEval pass@1 = {npass}/{len(rows)} = {npass/len(rows):.1%}  ({dt/60:.0f} min)")
    print(f"completion_tokens  median {pct(.5)}  p90 {pct(.9)}  p99 {pct(.99)}  max {ct[-1] if ct else 0}"
          f"   budget {a.max_tokens}")
    print(f"TRUNCATED {ntrunc}/{len(rows)}"
          + ("  <-- budget-bound, pass@1 is NOT a capability number" if ntrunc else "  (clean)"))


main()
