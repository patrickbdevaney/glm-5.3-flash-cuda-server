#!/usr/bin/env python3
"""End-to-end server smoke test. Checks PROPERTIES, not text quality.

Run against a server started with --n-layer 3: the generated text is meaningless, but every
property below is about plumbing, and plumbing is exactly what a 3-layer load exercises for real.

    python3 tools/smoke_server.py [port]
"""
import json, sys, urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8177
BASE = f"http://127.0.0.1:{PORT}"
ok = fail = 0

def ck(cond, what, detail=""):
    global ok, fail
    if cond: ok += 1; print(f"  ok   {what}")
    else:    fail += 1; print(f"  FAIL {what}  {detail}")

def post(path, body, raw=False):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            data = r.read().decode("utf-8", "replace")
            return r.status, (data if raw else json.loads(data))
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")

def get(path):
    with urllib.request.urlopen(BASE + path, timeout=30) as r:
        return r.read().decode()

def chat(msgs, **kw):
    b = {"messages": msgs, "max_tokens": 12, "temperature": 0.8, "top_p": 0.95, "seed": 1234}
    b.update(kw)
    return post("/v1/chat/completions", b)

print("=== smoke_server ===")

h = json.loads(get("/health"))
ck(h["status"] == "ok", "health")
ck(get("/").startswith("<!DOCTYPE html>"), "web UI is served")
ck("glm5_requests_total" in get("/metrics"), "metrics are Prometheus text")

# A seeded run must be exactly reproducible. Without this, nothing else here is a test.
s1, a = chat([{"role": "user", "content": "Say something."}])
s2, b = chat([{"role": "user", "content": "Say something."}])
ck(s1 == 200 and s2 == 200, "chat returns 200", f"{s1} {s2}")
t1 = a["choices"][0]["message"]
t2 = b["choices"][0]["message"]
ck(t1 == t2, "the same seed reproduces the same message", f"{t1} != {t2}")

# The prompt ends with <think>, so a run cut off by max_tokens is all reasoning, no content.
ck(t1.get("content", "") == "" and t1.get("reasoning_content"),
   "truncated generation is reasoning, not content", json.dumps(t1)[:160])
ck(a["choices"][0]["finish_reason"] == "length", "finish_reason is length when truncated")

# STREAMING MUST AGREE WITH NON-STREAMING for the same seed. These are two different code paths
# over the same tokens, and the smoke test exists mostly to keep them from drifting apart.
body = {"messages": [{"role": "user", "content": "Say something."}], "max_tokens": 12,
        "temperature": 0.8, "top_p": 0.95, "seed": 1234, "stream": True}
st, raw = post("/v1/chat/completions", body, raw=True)
chunks = [json.loads(l[6:]) for l in raw.splitlines()
          if l.startswith("data: ") and l[6:].strip() != "[DONE]"]
sr = "".join(c["choices"][0]["delta"].get("reasoning_content", "")
             for c in chunks if c.get("choices"))
sc = "".join(c["choices"][0]["delta"].get("content", "")
             for c in chunks if c.get("choices"))
ck(raw.rstrip().endswith("data: [DONE]"), "stream terminates with [DONE]")
ck(sr == t1.get("reasoning_content", "") and sc == t1.get("content", ""),
   "streaming text equals non-streaming text", f"{sr!r} vs {t1.get('reasoning_content','')!r}")
usage = [c for c in chunks if "usage" in c]
ck(len(usage) == 1 and usage[0]["usage"]["completion_tokens"] == 12, "final chunk carries usage")

# PREFIX REUSE. Turn two of a conversation extends turn one's rendered prompt, so the resident
# recurrent state should be reused and cached_tokens should be most of the prompt.
first = [{"role": "user", "content": "First question."}]
s, r1 = chat(first, max_tokens=4, temperature=0)
reply = r1["choices"][0]["message"]
second = first + [{"role": "assistant", "content": reply.get("content", ""),
                   "reasoning_content": reply.get("reasoning_content", "")},
                  {"role": "user", "content": "Second question."}]
s, r2 = chat(second, max_tokens=4, temperature=0)
cached = r2["usage"]["prompt_tokens_details"]["cached_tokens"]
ck(cached > 0, "a follow-up turn reuses the resident prefix", f"cached={cached}")
ck(cached < r2["usage"]["prompt_tokens"], "reuse is a prefix, not the whole prompt",
   f"cached={cached} prompt={r2['usage']['prompt_tokens']}")

# A fresh, unrelated conversation must NOT claim a cache hit — the state cannot be rewound, so a
# false hit here would mean generating from someone else's context.
s, r3 = chat([{"role": "user", "content": "Completely different opening."}], max_tokens=4, temperature=0)
ck(r3["usage"]["prompt_tokens_details"]["cached_tokens"] == 0,
   "a divergent prompt resets rather than reusing")

# Stop strings.
s, r = chat([{"role": "user", "content": "x"}], max_tokens=24, temperature=0, stop=["a"])
ck(s == 200 and "a" not in r["choices"][0]["message"].get("reasoning_content", ""),
   "a stop string truncates the text")

# /v1/completions
s, r = post("/v1/completions", {"prompt": "The capital of France is", "max_tokens": 6, "temperature": 0})
ck(s == 200 and "text" in r["choices"][0] and r["usage"]["completion_tokens"] > 0, "/v1/completions", str(r)[:200])
ck("timings" in r, "/v1/completions reports timings")

# Errors must be errors, with a body. A bare 500 with an empty body is the failure this whole
# server is written to avoid.
s, r = post("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}],
                                     "max_tokens": 100000})
ck(s == 400 and "context" in str(r), "over-context is a 400 with a message", f"{s} {str(r)[:120]}")
req = urllib.request.Request(BASE + "/v1/chat/completions", data=b"{not json",
                             headers={"Content-Type": "application/json"})
try:
    urllib.request.urlopen(req, timeout=30); code, msg = 200, ""
except urllib.error.HTTPError as e:
    code, msg = e.code, e.read().decode()
ck(code == 400 and "invalid JSON" in msg, "malformed JSON is a 400 with a message")

# Tools round-trip through the prompt.
s, r = chat([{"role": "user", "content": "weather?"}], max_tokens=4, temperature=0,
            tools=[{"type": "function", "function": {"name": "get_weather", "parameters": {}}}])
ck(s == 200, "a request with tools is accepted")

print(f"--- {ok} passed, {fail} failed ---")
print("ALL SMOKE CHECKS PASS" if not fail else "SMOKE FAILED")
sys.exit(1 if fail else 0)
