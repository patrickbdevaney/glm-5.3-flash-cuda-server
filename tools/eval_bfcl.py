#!/usr/bin/env python3
"""BFCL live_multiple: tool selection + argument extraction. The agentic proxy HumanEval isn't.

WHY THIS AND NOT SWE-BENCH. SWE-bench eval images publish linux/amd64 only; this box is
aarch64, so running them means QEMU at 10-50x slowdown -- hours per instance. Building arm64
environments per repo is a multi-day project with per-commit dependency resolution. BFCL needs
no repo environment at all and tests the capabilities that actually decide agentic daily
driving: does the model pick the RIGHT function from several, and fill its arguments correctly.

SCORING is BFCL's AST match, not string equality. ground_truth gives each argument a LIST of
acceptable values ("boiling hot", "served boiling hot", ...), because many phrasings are
correct. A call counts only if the function name matches AND every ground-truth argument is
present with an accepted value. Extra arguments the schema allows are tolerated; a wrong
function name is an immediate miss.
"""
import argparse, json, glob, os, re, sys, time, urllib.request

SNAP = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--gorilla-llm--Berkeley-Function-Calling-Leaderboard"
    "/snapshots/*/BFCL_v3_live_multiple.json"))
BASE = "http://127.0.0.1:8080"

def to_openai_tools(fns):
    """BFCL schemas say type 'dict'/'float'; JSON Schema wants 'object'/'number'."""
    fix = {"dict": "object", "float": "number", "tuple": "array", "any": "string"}
    def walk(o):
        if isinstance(o, dict):
            o = {k: walk(v) for k, v in o.items()}
            if isinstance(o.get("type"), str):
                o["type"] = fix.get(o["type"], o["type"])
            return o
        return [walk(x) for x in o] if isinstance(o, list) else o
    out = []
    for f in fns:
        p = walk(f.get("parameters") or {"type": "object", "properties": {}})
        p.setdefault("type", "object"); p.setdefault("properties", {})
        out.append({"type": "function", "function": {
            "name": f["name"], "description": f.get("description", "")[:1024], "parameters": p}})
    return out

def matches(call, gts):
    """call = (name, args dict); gts = ground_truth list of {name: {arg: [accepted...]}}"""
    name, args = call
    for gt in gts:
        for gname, gargs in gt.items():
            if gname != name:
                continue
            ok = True
            for a, accepted in gargs.items():
                if not isinstance(accepted, list):
                    accepted = [accepted]
                if a not in args:
                    # an empty-string / empty-list option means "may be omitted"
                    if any(x in ("", [], {}, None) for x in accepted):
                        continue
                    ok = False; break
                v = args[a]
                if not any(v == x or str(v).strip().lower() == str(x).strip().lower()
                           for x in accepted):
                    ok = False; break
            if ok:
                return True
    return False

def post(obj, timeout=900):
    r = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(obj).encode(),
                               {"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as f:
        return json.loads(f.read())

def extract(msg):
    """Prefer real tool_calls; fall back to a JSON object in the text (models vary)."""
    tc = msg.get("tool_calls") or []
    if tc:
        fn = tc[0].get("function", {})
        try: args = json.loads(fn.get("arguments") or "{}")
        except Exception: args = {}
        return (fn.get("name"), args)
    txt = msg.get("content") or ""
    m = re.search(r'\{.*\}', txt, re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            if "name" in d: return (d["name"], d.get("arguments") or d.get("parameters") or {})
        except Exception: pass
    return (None, {})

def main():
    global BASE
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--base", default=BASE); ap.add_argument("--label", default="glm5")
    ap.add_argument("--out", default="artifacts/bfcl.json")
    ap.add_argument("--max-tokens", type=int, default=2048)
    # See the long note in eval_humaneval.py: the two servers disagree on the default effort and
    # glm5's default (Max) loops until the cap with an empty `content` and no tool_calls. Both
    # arms MUST be given the same explicit value or the comparison is meaningless.
    ap.add_argument("--reasoning-effort", default="high", choices=["low", "high"])
    a = ap.parse_args()
    BASE = a.base
    if not SNAP: raise SystemExit("BFCL not in the HF cache")
    qp = SNAP[0]; ap_ = os.path.join(os.path.dirname(qp), "possible_answer",
                                     "BFCL_v3_live_multiple.json")
    qs = [json.loads(l) for l in open(qp)][:a.limit]
    gt = {json.loads(l)["id"]: json.loads(l)["ground_truth"] for l in open(ap_)}
    print(f"BFCL live_multiple: {len(qs)} cases against {BASE} [{a.label}]", flush=True)
    npass = 0; ntrunc = 0; rows = []; t0 = time.time()
    for i, q in enumerate(qs):
        msgs = [m for turn in q["question"] for m in turn]
        try:
            r = post({"model": a.label, "temperature": 0.0, "max_tokens": a.max_tokens,
                      "reasoning_effort": a.reasoning_effort,
                      "messages": msgs, "tools": to_openai_tools(q["function"])})
            msg = r["choices"][0]["message"]
            call = extract(msg); err = None
            ct = int(r.get("usage", {}).get("completion_tokens", 0))
            trunc = ct >= a.max_tokens
        except Exception as e:
            call, err, ct, trunc = (None, {}), repr(e)[:160], 0, False
        ok = bool(call[0]) and matches(call, gt.get(q["id"], []))
        npass += ok
        rows.append({"id": q["id"], "pass": bool(ok), "called": call[0], "error": err,
                     "completion_tokens": ct, "truncated": bool(trunc)})
        ntrunc += trunc
        print(f"[{i+1}/{len(qs)}] {q['id']} {'PASS' if ok else 'fail'}  "
              f"running {npass}/{i+1} = {npass/(i+1):.1%}", flush=True)
        json.dump({"n": len(rows), "pass": npass, "acc": npass/len(rows),
                   "max_tokens": a.max_tokens, "reasoning_effort": a.reasoning_effort,
                   "truncated": ntrunc, "label": a.label, "rows": rows},
                  open(a.out, "w"), indent=1)
    print(f"\nBFCL live_multiple accuracy = {npass}/{len(rows)} = {npass/len(rows):.1%} "
          f"({(time.time()-t0)/60:.0f} min)")
    print(f"TRUNCATED {ntrunc}/{len(rows)}"
          + ("  <-- budget-bound, accuracy is NOT a capability number" if ntrunc else "  (clean)"))

main()
