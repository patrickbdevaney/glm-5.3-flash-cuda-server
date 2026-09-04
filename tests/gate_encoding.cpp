// gate_encoding.cpp — the C++ chat encoder against HF's own Jinja, byte for byte.
//
// Vectors: tools/gen_chat_vectors.py -> ref/chat_vectors.json.
#include "../include/encoding_glm5.h"
#include "../include/tokenizer_glm5.h"
#include <cstdio>
#include <fstream>

using json = nlohmann::ordered_json;
static int pass = 0, fail = 0;

// Where two prompts diverge is the whole diagnosis; the byte offset plus 40 bytes of context
// localises a missing newline or a sorted key immediately.
static void diff(const std::string& want, const std::string& got) {
    size_t d = 0;
    while (d < want.size() && d < got.size() && want[d] == got[d]) ++d;
    const size_t lo = d > 40 ? d - 40 : 0;
    printf("    diverges at byte %zu (want %zu bytes, got %zu)\n", d, want.size(), got.size());
    printf("    want ...%s\n", want.substr(lo, 100).c_str());
    printf("    got  ...%s\n", got.substr(lo, 100).c_str());
}

int main(int argc, char** argv) {
    const std::string vecpath = argc > 1 ? argv[1]
        : std::string(getenv("HOME")) + "/glm-5.3-flash-cuda-server/ref/chat_vectors.json";
    const std::string tokpath = argc > 2 ? argv[2]
        : std::string(getenv("HOME")) + "/glm-5.3-reap/source/GLM-5.3-Flash/tokenizer.json";

    printf("=== gate_encoding ===\n");
    std::ifstream f(vecpath);
    if (!f) { printf("  FAIL cannot open %s\n", vecpath.c_str()); return 1; }
    json v; f >> v;

    for (auto& c : v["cases"]) {
        const std::string name = c["name"].get<std::string>();
        glm5enc::Options opt;
        const auto& kw = c["kwargs"];
        if (kw.contains("reasoning_effort")) opt.reasoning_effort = kw["reasoning_effort"].get<std::string>();
        if (kw.contains("add_generation_prompt")) opt.add_generation_prompt = kw["add_generation_prompt"].get<bool>();
        if (kw.contains("clear_thinking")) opt.clear_thinking = kw["clear_thinking"].get<bool>();

        const std::string got = glm5enc::encode_messages(c["messages"], c["tools"], opt);
        const std::string want = c["prompt"].get<std::string>();
        if (got == want) { ++pass; printf("  ok   %-26s %5zu bytes\n", name.c_str(), want.size()); }
        else { ++fail; printf("  FAIL %-26s\n", name.c_str()); diff(want, got); }
    }

    // The prompt is only correct if it also TOKENISES to what HF got. The two gates are
    // independent: a byte-exact prompt fed through a wrong tokenizer is still wrong ids, and this
    // is the only place the two halves are checked together.
    {
        glm5tok::Tokenizer tok;
        tok.load(tokpath);
        int tp = 0, tf = 0;
        for (auto& c : v["cases"]) {
            glm5enc::Options opt;
            const auto& kw = c["kwargs"];
            if (kw.contains("reasoning_effort")) opt.reasoning_effort = kw["reasoning_effort"].get<std::string>();
            if (kw.contains("add_generation_prompt")) opt.add_generation_prompt = kw["add_generation_prompt"].get<bool>();
            if (kw.contains("clear_thinking")) opt.clear_thinking = kw["clear_thinking"].get<bool>();
            const auto ids = tok.encode(glm5enc::encode_messages(c["messages"], c["tools"], opt));
            if (ids == c["ids"].get<std::vector<int>>()) ++tp;
            else { ++tf; printf("  FAIL ids %-22s got %zu want %zu\n", c["name"].get<std::string>().c_str(),
                               ids.size(), c["ids"].size()); }
        }
        printf("  prompt->ids: %d ok, %d failed\n", tp, tf);
        pass += tp; fail += tf;
    }

    // Round-trip: the parser must recover what the encoder emitted for an assistant turn.
    {
        const std::string completion =
            "reasoning here</think>Sure.<tool_call>get_weather"
            "<arg_key>city</arg_key><arg_value>Paris</arg_value>"
            "<arg_key>unit</arg_key><arg_value>c</arg_value></tool_call>";
        const json m = glm5enc::parse_message_from_completion_text(completion);
        const bool ok = m["content"] == "Sure." &&
                        m["reasoning_content"] == "reasoning here" &&
                        m["tool_calls"].size() == 1 &&
                        m["tool_calls"][0]["function"]["name"] == "get_weather" &&
                        m["tool_calls"][0]["function"]["arguments"] ==
                            "{\"city\": \"Paris\", \"unit\": \"c\"}";
        if (ok) { ++pass; printf("  ok   parse_completion\n"); }
        else { ++fail; printf("  FAIL parse_completion: %s\n", m.dump().c_str()); }
    }
    // A call truncated by the token budget must still parse, not throw or hang.
    {
        const json m = glm5enc::parse_message_from_completion_text(
            "</think><tool_call>search<arg_key>q</arg_key><arg_value>partial");
        const bool ok = m.contains("tool_calls") && m["tool_calls"].size() == 1 &&
                        m["tool_calls"][0]["function"]["arguments"] == "{\"q\": \"partial\"}";
        if (ok) { ++pass; printf("  ok   parse_truncated_call\n"); }
        else { ++fail; printf("  FAIL parse_truncated_call: %s\n", m.dump().c_str()); }
    }

    printf("--- %d passed, %d failed ---\n", pass, fail);
    printf(fail ? "GATE FAILED\n" : "ALL ENCODING GATES PASS\n");
    return fail ? 1 : 0;
}
