#!/usr/bin/env python3
"""Golden vectors for tests/gate_tokenizer.cpp — HF's own answer for a set of adversarial strings.

Run from ~/glm-5.3-reap with its venv:
    ./.venv/bin/python ~/glm-5.3-flash-cuda-server/tools/gen_tokenizer_vectors.py

Tokenizer-only: it never loads the model, so it costs no GPU and ~100 MB of RSS. The cases are
chosen to hit the places where a hand-written pre-tokenizer diverges from the regex, not to be a
representative corpus -- a corpus agrees almost everywhere and hides exactly these.
"""
import json, os, sys

SRC = os.path.expanduser("~/glm-5.3-reap/source/GLM-5.3-Flash")
OUT = os.path.expanduser("~/glm-5.3-flash-cuda-server/ref/tokenizer_vectors.json")

CASES = [
    # --- ordinary text, the baseline ---
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    # --- alternative 0: contractions, and the case-insensitivity ---
    "don't  can't  it's  we've  they'll  I'd  he's",
    "DON'T CAN'T IT'S WE'VE THEY'LL I'D",
    "'s at the start, and a bare ' apostrophe, and 'x which is not a contraction",
    # --- alternative 1 vs 3: both can open with a space ---
    " word", " .", "  .", " ?word", " word", "a-b", "-a", "--a",
    # --- alternative 2: digits group in threes, so 4+ digits split ---
    "1 12 123 1234 12345 1234567890",
    "3.14159 2,718 0x1F 1e10 -42",
    # --- alternatives 4/5/6: the whitespace cases, which is where backtracking shows ---
    "trailing spaces   ",
    "a  b   c    d",
    "line1\nline2\r\nline3\n\n\nline4",
    "tab\there\tand\t\tthere",
    "   \n   leading blank line",
    "ends with a single space ",
    "\n\n\n",
    "   ",
    # --- CJK, kana, and combining marks: \p{L}+ swallows a whole run, then BPE splits it ---
    "你好，世界！这是一个测试。",
    "こんにちは世界テスト",
    "한국어 테스트",
    "café naïve résumé",           # NFD: letter + combining mark
    "السلام عليكم",
    "Русский текст",
    # --- emoji and symbols: astral plane, and runs of them ---
    "\U0001f600\U0001f601\U0001f602 emoji \U0001f680 test",
    "→←↑↓ ≤≥≠ ±×÷",
    "\U0001f1fa\U0001f1f8 \U0001f469‍\U0001f4bb",      # flags and ZWJ sequences
    # --- code, which is mostly alternative 3 ---
    "def f(x):\n    return x ** 2  # square\n",
    "int main() { printf(\"%d\\n\", 42); return 0; }",
    "a[i] = b->c.d(e, f);",
    "{\"key\": [1, 2, 3], \"n\": null}",
    "https://example.com/a/b?c=d&e=f#g",
    # --- added tokens: matched literally on the raw text, before everything else ---
    "<|user|>hello<|assistant|>",
    "[gMASK]<sop><|system|>be brief<|user|>hi",
    "<think>reasoning here</think>the answer",
    "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>",
    "text<|endoftext|>more",
    "/nothink what is 2+2",
    "an <|image|> inline",
    # --- added tokens with adjacent text and no separator, plus a near-miss ---
    "x<|user|>y", "<|user|", "<|use", "<<|user|>>",
    # --- degenerate ---
    "", " ", "\n", "a",
    # --- raw bytes that are not valid on their own, exercising the ByteLevel alphabet ---
    "ÿþý",
    "null\x00byte" if False else "control\x01\x02chars",
]

def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(SRC)
    out = []
    for s in CASES:
        ids = tok.encode(s, add_special_tokens=False)
        out.append({
            "text": s,
            "ids": ids,
            "decoded_keep": tok.decode(ids, skip_special_tokens=False),
            "decoded_skip": tok.decode(ids, skip_special_tokens=True),
        })
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({"cases": out}, f, ensure_ascii=False, indent=1)
    n = sum(len(c["ids"]) for c in out)
    print(f"wrote {len(out)} cases / {n} tokens -> {OUT}")
    # A round-trip that is not identity is a property of the vocab, not a bug, but it is worth
    # seeing which cases those are: they are where decode gating has to be exact.
    for c in out:
        if c["decoded_keep"] != c["text"]:
            print(f"  non-identity round-trip: {c['text']!r} -> {c['decoded_keep']!r}")

if __name__ == "__main__":
    main()
