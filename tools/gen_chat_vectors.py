#!/usr/bin/env python3
"""Golden prompt strings for tests/gate_encoding.cpp, rendered by the checkpoint's own Jinja.

    ./.venv/bin/python ~/glm-5.3-flash-cuda-server/tools/gen_chat_vectors.py

The C++ encoder is gated BYTE-EXACT against these. It is not gated against my reading of
chat_template.jinja, because the file is 257 lines of Jinja with whitespace-control markers that
differ line by line ({%- vs {%), and reading those correctly by eye is exactly the kind of thing
that looks right and is off by one newline.
"""
import json, os

SRC = os.path.expanduser("~/glm-5.3-reap/source/GLM-5.3-Flash")
OUT = os.path.expanduser("~/glm-5.3-flash-cuda-server/ref/chat_vectors.json")

WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather in a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["c", "f"]},
            },
            "required": ["city"],
        },
    },
}
SEARCH = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search the web.",
        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]},
    },
}

CASES = [
    ("user_only", [{"role": "user", "content": "Hello!"}], None, {}),
    ("system_user",
     [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Hi"}], None, {}),
    ("effort_low", [{"role": "user", "content": "Hi"}], None, {"reasoning_effort": "low"}),
    ("effort_high", [{"role": "user", "content": "Hi"}], None, {"reasoning_effort": "high"}),
    ("effort_max", [{"role": "user", "content": "Hi"}], None, {"reasoning_effort": "max"}),
    ("effort_bogus", [{"role": "user", "content": "Hi"}], None, {"reasoning_effort": "medium"}),
    ("no_gen_prompt", [{"role": "user", "content": "Hi"}], None, {"add_generation_prompt": False}),
    ("multiturn",
     [{"role": "user", "content": "What is 2+2?"},
      {"role": "assistant", "content": "4"},
      {"role": "user", "content": "And 3+3?"}], None, {}),
    ("assistant_with_think",
     [{"role": "user", "content": "q"},
      {"role": "assistant", "content": "<think>let me add</think>4"},
      {"role": "user", "content": "again"}], None, {}),
    ("assistant_reasoning_field",
     [{"role": "user", "content": "q"},
      {"role": "assistant", "content": "4", "reasoning_content": "adding two and two"},
      {"role": "user", "content": "again"}], None, {}),
    ("clear_thinking",
     [{"role": "user", "content": "q"},
      {"role": "assistant", "content": "<think>hidden</think>4"},
      {"role": "user", "content": "again"}], None, {"clear_thinking": True}),
    ("tools_one", [{"role": "user", "content": "weather in Paris?"}], [WEATHER], {}),
    ("tools_two", [{"role": "user", "content": "hi"}], [WEATHER, SEARCH], {}),
    ("tool_call_roundtrip",
     [{"role": "user", "content": "weather in Paris?"},
      {"role": "assistant", "content": "",
       "tool_calls": [{"id": "c1", "type": "function",
                       "function": {"name": "get_weather", "arguments": {"city": "Paris", "unit": "c"}}}]},
      {"role": "tool", "tool_call_id": "c1", "content": "18C, clear"}], [WEATHER], {}),
    ("two_tool_results",
     [{"role": "user", "content": "both"},
      {"role": "assistant", "content": "",
       "tool_calls": [{"id": "a", "type": "function", "function": {"name": "search", "arguments": {"q": "x"}}},
                      {"id": "b", "type": "function", "function": {"name": "search", "arguments": {"q": "y"}}}]},
      {"role": "tool", "tool_call_id": "a", "content": "res x"},
      {"role": "tool", "tool_call_id": "b", "content": "res y"}], [SEARCH], {}),
    ("multimodal_text_parts",
     [{"role": "user", "content": [{"type": "text", "text": "part one "}, {"type": "text", "text": "part two"}]}],
     None, {}),
    ("image_part",
     [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}},
                                   {"type": "text", "text": "what is this?"}]}], None, {}),
    ("unicode", [{"role": "user", "content": "你好 \U0001f600 café"}], None, {}),
    ("empty_content", [{"role": "user", "content": ""}], None, {}),
]

def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(SRC)
    out = []
    for name, msgs, tools, kw in CASES:
        kwargs = dict(tokenize=False, add_generation_prompt=True)
        kwargs.update(kw)
        if tools:
            kwargs["tools"] = tools
        try:
            text = tok.apply_chat_template(msgs, **kwargs)
        except Exception as e:
            print(f"  {name}: template raised {type(e).__name__}: {e}")
            continue
        ids = tok.encode(text, add_special_tokens=False)
        out.append({"name": name, "messages": msgs, "tools": tools, "kwargs": kw,
                    "prompt": text, "ids": ids})
        print(f"  {name:24s} {len(text):5d} chars {len(ids):5d} tok")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({"cases": out}, f, ensure_ascii=False, indent=1)
    print(f"wrote {len(out)} cases -> {OUT}")

if __name__ == "__main__":
    main()
