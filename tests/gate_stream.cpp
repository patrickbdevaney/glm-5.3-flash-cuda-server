// gate_stream.cpp — the streaming splitter, without a GPU.
//
// The property that matters is BYTE-BOUNDARY INDEPENDENCE: the deltas a client receives must not
// depend on how the generated text happened to be chopped into tokens. A splitter that passes on
// whole-string input and fails one byte at a time is the normal failure, and it shows up in
// production as a stray "</thi" in the answer — never as an error.
#include "../include/stream_parse.h"
#include <cstdio>
#include <string>
#include <vector>

using glm5srv::StreamSplitter;
static int pass = 0, fail = 0;
static void ck(bool ok, const char* what) {
    if (ok) { ++pass; printf("  ok   %s\n", what); }
    else    { ++fail; printf("  FAIL %s\n", what); }
}

struct Split { std::string r, c, tools; };

// Feed `text` in chunks of `n` bytes (n <= 0 means all at once).
static Split run(const std::string& text, int n) {
    StreamSplitter sp(true);
    Split out;
    if (n <= 0) { sp.feed(text, out.r, out.c); }
    else for (size_t i = 0; i < text.size(); i += n)
        sp.feed(text.substr(i, n), out.r, out.c);
    sp.finish(out.r, out.c);
    out.tools = sp.tools_buf;
    return out;
}

int main() {
    printf("=== gate_stream ===\n");

    const std::string simple = "I should add them.</think>The answer is 4.";
    { const Split a = run(simple, 0);
      ck(a.r == "I should add them." && a.c == "The answer is 4.", "reasoning and content split"); }

    // One byte at a time is the adversarial case: every marker straddles a boundary.
    { const Split a = run(simple, 0), b = run(simple, 1), c = run(simple, 3), d = run(simple, 7);
      ck(a.r == b.r && a.c == b.c, "1-byte chunks agree with one shot");
      ck(a.r == c.r && a.c == c.c, "3-byte chunks agree");
      ck(a.r == d.r && a.c == d.c, "7-byte chunks agree"); }

    // Tool calls: everything from the first <tool_call> is buffered, none of it leaks into content.
    { const std::string t = "thinking</think>Let me check.<tool_call>get_weather"
                            "<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>";
      const Split a = run(t, 0), b = run(t, 1);
      ck(a.c == "Let me check." && a.tools.find("get_weather") != std::string::npos,
         "tool block is buffered, not streamed as content");
      ck(a.r == b.r && a.c == b.c && a.tools == b.tools, "tool split is chunk-independent"); }

    // Several calls in a row, which is GLM's shape: no outer wrapper, so the buffer just runs on.
    { const std::string t = "</think><tool_call>a</tool_call><tool_call>b</tool_call>";
      const Split x = run(t, 1);
      ck(x.c.empty() && x.tools.find("<tool_call>a") != std::string::npos &&
         x.tools.find("<tool_call>b") != std::string::npos, "consecutive tool calls all buffer"); }

    // Multi-byte UTF-8 must never be cut mid-character by the holdback logic. Splitting at every
    // possible offset is the only way to be sure, since the bug depends on where the boundary lands.
    { const std::string t = "推理过程</think>答案是四。\xF0\x9F\x98\x80";
      const Split ref = run(t, 0);
      bool ok = true;
      for (int n = 1; n <= 9; ++n) {
          const Split a = run(t, n);
          if (a.r != ref.r || a.c != ref.c) ok = false;
      }
      ck(ok, "UTF-8 survives every chunk size");
      ck(ref.r + ref.c == "推理过程" + std::string("答案是四。\xF0\x9F\x98\x80"),
         "no bytes are lost or duplicated"); }

    // Degenerate inputs a real generation actually produces.
    { const Split a = run("</think>", 1);
      ck(a.r.empty() && a.c.empty(), "an immediate </think> yields nothing"); }
    { const Split a = run("still thinking when the budget ran out", 1);
      ck(a.c.empty() && a.r == "still thinking when the budget ran out",
         "an unterminated think block stays reasoning"); }
    { const Split a = run("", 1);
      ck(a.r.empty() && a.c.empty(), "empty generation"); }
    // A marker that is a prefix of the held-back tail at the very end must still be flushed.
    { const Split a = run("done</thi", 1);
      ck(a.r == "done</thi" && a.c.empty(), "a partial marker at EOS is flushed as text"); }

    printf("--- %d passed, %d failed ---\n", pass, fail);
    printf(fail ? "GATE FAILED\n" : "ALL STREAM GATES PASS\n");
    return fail ? 1 : 0;
}
