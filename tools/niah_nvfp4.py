#!/usr/bin/env python3
"""NIAH against the NVFP4 CUDA server -- tests the REAL DSA sparse path.

WHY THIS IS NOT A REPEAT of artifacts/niah_GLM-5.3-Flash-REAP50-Q4_K_M.json:
llama.cpp does not implement DSA, so that run (9/9 to 32K) exercised DENSE attention.
This server runs the actual indexer + top-2048 sparse selection above DENSE_CTX_LIMIT=2051.
32K is therefore a CONTROL: same length, known-dense answer (3/3 found), DSA the only
variable. 64K/128K are new ground for both the kernel and the 2048-token saliency question.

Lengths are hit exactly via /tokenize rather than by counting characters.
"""
import json, sys, time, urllib.request, argparse

BASE = "http://127.0.0.1:8080"
FILLER = ("The city council met on Tuesday to discuss the new drainage plan. Rainfall in the "
          "northern district has exceeded the seasonal average for three consecutive years. "
          "Local farmers reported that the soil retains water longer than it once did. "
          "A survey of the riverbank was commissioned and the results are pending review. ")
NEEDLE = "The secret passphrase for the Halvorsen archive is {code}."
QUESTION = "\n\nQuestion: What is the secret passphrase for the Halvorsen archive?\nAnswer:"
CODE = "crimson-lantern-47"

def post(path, obj, timeout=14400):
    r = urllib.request.Request(BASE+path, json.dumps(obj).encode(),
                               {"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as f:
        return json.loads(f.read())

def ntok(s):
    return len(post("/tokenize", {"content": s})["tokens"])

def build(target, depth):
    """Filler padded to `target` tokens with the needle at fractional `depth`."""
    per = ntok(FILLER)
    reps = max(1, target // per)
    body = FILLER * reps
    while ntok(body) < target:                       # top up; tokenizer is not linear in chars
        body += FILLER
    words = body.split()
    # binary-search a word count that lands on `target` tokens
    lo, hi = 0, len(words)
    while lo < hi:
        mid = (lo+hi)//2
        if ntok(" ".join(words[:mid])) < target: lo = mid+1
        else: hi = mid
    words = words[:lo]
    cut = int(len(words)*depth)
    needle = NEEDLE.format(code=CODE)
    return " ".join(words[:cut] + [needle] + words[cut:])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", default="32000,64000,128000")
    ap.add_argument("--depths", default="0.1,0.5,0.9")
    ap.add_argument("--out", default="artifacts/niah_nvfp4_dsa.json")
    a = ap.parse_args()
    lengths = [int(x) for x in a.lengths.split(",")]
    depths  = [float(x) for x in a.depths.split(",")]
    rows = []
    for L in lengths:
        for d in depths:
            prompt = build(L, d) + QUESTION
            n = ntok(prompt)
            t0 = time.time()
            try:
                r = post("/v1/completions", {"prompt": prompt, "max_tokens": 24,
                                             "temperature": 0.0, "model": "glm5"})
                txt = r["choices"][0]["text"]
                err = None
            except Exception as e:
                txt, err = "", repr(e)
            dt = time.time()-t0
            found = CODE in txt
            rows.append({"target": L, "tokens": n, "depth": d, "found": found,
                         "secs": round(dt,1), "reply": txt.strip()[:120], "error": err})
            print(json.dumps(rows[-1]), flush=True)
            json.dump(rows, open(a.out, "w"), indent=1)
    ok = sum(r["found"] for r in rows)
    print(f"NIAH DSA: {ok}/{len(rows)} found", flush=True)

main()
