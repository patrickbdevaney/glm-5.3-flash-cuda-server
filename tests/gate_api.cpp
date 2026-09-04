// gate_api.cpp — request parsing and response shaping, without a GPU.
//
// These are the defaults and field names a client actually sees. They are gated because every one
// of them fails silently: a wrong temperature default just makes the model a bit worse, an ignored
// `stop` array just makes it ramble, a missing `finish_reason` just confuses one client library.
#include "../include/openai_api.h"
#include <cstdio>

using glm5api::json;
static int pass = 0, fail = 0;
static void ck(bool ok, const char* what) {
    if (ok) { ++pass; printf("  ok   %s\n", what); }
    else    { ++fail; printf("  FAIL %s\n", what); }
}

int main() {
    printf("=== gate_api ===\n");

    // Defaults come from generation_config.json, NOT from habit. 0.7/0.9 is gemma's and would be
    // wrong here in a way nothing reports.
    { const auto r = glm5api::parse_chat_request(json::parse(R"({"messages":[]})"));
      ck(r.sampling.temperature == 1.0, "default temperature is 1.0");
      ck(r.sampling.top_p == 0.95, "default top_p is 0.95");
      ck(r.sampling.max_tokens == 512, "default max_tokens");
      ck(r.reasoning_effort.empty(), "reasoning_effort defaults to unset (renders as Max)"); }

    { const auto r = glm5api::parse_chat_request(json::parse(
          R"({"messages":[{"role":"user","content":"hi"}],"temperature":0.6,"top_p":0.8,
              "max_completion_tokens":99,"seed":7,"stop":["\n\n","END"],"stream":true})"));
      ck(r.sampling.temperature == 0.6 && r.sampling.top_p == 0.8, "explicit sampling wins");
      ck(r.sampling.max_tokens == 99, "max_completion_tokens is accepted as max_tokens");
      ck(r.sampling.has_seed && r.sampling.seed == 7, "seed");
      ck(r.sampling.stop.size() == 2 && r.sampling.stop[1] == "END", "stop array");
      ck(r.stream, "stream flag"); }

    { const auto r = glm5api::parse_chat_request(json::parse(
          R"({"messages":[],"chat_template_kwargs":{"reasoning_effort":"low","clear_thinking":true}})"));
      ck(r.reasoning_effort == "low" && r.clear_thinking, "chat_template_kwargs is honoured"); }

    // 'medium' is not a valid effort for this template — it renders as Max. The server passes it
    // through rather than rejecting it, so the prompt matches what HF's Jinja would have produced.
    { auto r = glm5api::parse_chat_request(json::parse(
          R"({"messages":[{"role":"user","content":"hi"}],"reasoning_effort":"medium"})"));
      const std::string p = glm5api::build_prompt(r);
      ck(p.find("Reasoning Effort: Max") != std::string::npos, "an invalid effort renders as Max"); }

    { auto r = glm5api::parse_chat_request(json::parse(
          R"({"messages":[{"role":"user","content":"hi"}],
              "tools":[{"type":"function","function":{"name":"f","parameters":{}}}]})"));
      const std::string p = glm5api::build_prompt(r);
      ck(p.find("# Tools") != std::string::npos && p.find("\"name\": \"f\"") != std::string::npos,
         "tools reach the prompt from the request, not a system message"); }

    { auto r = glm5api::parse_chat_request(json::parse(R"({"messages":[{"role":"user","content":"hi"}]})"));
      const std::string p = glm5api::build_prompt(r);
      ck(p.rfind("<|assistant|><think>") == p.size() - 20, "the prompt ends with the think opener"); }

    // Response shaping.
    { json parsed = json::object();
      parsed["content"] = "hello"; parsed["reasoning_content"] = "thought";
      const json o = glm5api::chat_completion_response("abcdef0123", "m", parsed, 10, 3, 1700000000);
      ck(o["choices"][0]["finish_reason"] == "stop", "finish_reason stop");
      ck(o["choices"][0]["message"]["reasoning_content"] == "thought", "reasoning surfaced separately");
      ck(o["usage"]["total_tokens"] == 13, "usage totals"); }

    { json parsed = glm5enc::parse_message_from_completion_text(
          "t</think>ok<tool_call>f<arg_key>a</arg_key><arg_value>1</arg_value></tool_call>");
      const json o = glm5api::chat_completion_response("abcdef0123", "m", parsed, 1, 1, 0);
      ck(o["choices"][0]["finish_reason"] == "tool_calls", "finish_reason tool_calls");
      ck(o["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "f", "tool call shaped"); }

    { const json o = glm5api::chat_completion_response("abcdef0123", "m", json::object(), 1, 1, 0, "length");
      ck(o["choices"][0]["finish_reason"] == "length", "truncation reports length"); }

    { const std::string s = glm5api::sse_chunk("id", "m", 0, "hi", "", nullptr);
      ck(s.rfind("data: ", 0) == 0 && s.size() > 12 && s.substr(s.size() - 2) == "\n\n",
         "sse framing"); }
    // Invalid UTF-8 in a delta must not throw — byte-level BPE can emit it mid-character.
    { const std::string bad = "\xC3";                    // a lone lead byte
      bool threw = false;
      try { (void)glm5api::sse_chunk("id", "m", 0, bad, "", nullptr); } catch (...) { threw = true; }
      ck(!threw, "invalid UTF-8 in a delta does not throw"); }

    printf("--- %d passed, %d failed ---\n", pass, fail);
    printf(fail ? "GATE FAILED\n" : "ALL API GATES PASS\n");
    return fail ? 1 : 0;
}
