// stream_parse.h — split a streamed completion into reasoning_content / content / tool-call deltas.
//
// The prompt always ends with `<think>`, so generation STARTS inside the reasoning block and the
// raw output for a turn looks like
//     {reasoning}</think>{content}<tool_call>…</tool_call><tool_call>…</tool_call>
// A client streaming this needs the parts separated as they arrive, not after the fact.
//
// Two things make that harder than a find():
//   1. A marker can straddle a chunk boundary. "</thi" + "nk>" must not be emitted as content, so
//      the splitter holds back the longest possible marker prefix and re-examines it next feed.
//   2. A held-back boundary can fall INSIDE a UTF-8 character, and every delta ends up in a JSON
//      string — nlohmann throws on invalid UTF-8, which would turn a Chinese answer into a 500.
//      So emit boundaries are rounded down to a character boundary.
//
// Tool calls are buffered rather than streamed as deltas: GLM emits a SEQUENCE of <tool_call>
// blocks with no outer wrapper, so "the calls are finished" is only known at end of generation.
// The server emits them in one final chunk, which is what vLLM and SGLang do.
//
// Gated by tests/gate_stream.cpp. The property that gate actually checks is byte-boundary
// independence: feeding the same text one byte at a time must produce the same deltas.
#pragma once
#include <string>

namespace glm5srv {

inline const char* THINK_END_MARK = "</think>";
inline const char* TOOL_OPEN_MARK = "<tool_call>";

// Largest number of trailing bytes that could still turn out to be the start of a marker.
inline size_t stream_holdback() {
    const size_t a = std::char_traits<char>::length(THINK_END_MARK);
    const size_t b = std::char_traits<char>::length(TOOL_OPEN_MARK);
    return (a > b ? a : b) - 1;
}

// Round `end` down so it never splits a UTF-8 character.
inline size_t utf8_floor(const std::string& s, size_t end) {
    while (end > 0 && ((unsigned char)s[end] & 0xC0) == 0x80) --end;
    return end;
}

struct StreamSplitter {
    bool in_reasoning = true;      // the prompt ends with <think>, so generation starts inside it
    bool in_tools     = false;
    std::string pending;           // bytes held back pending a marker decision
    std::string raw;               // everything ever fed, for the final authoritative parse
    std::string tools_buf;         // from the first <tool_call> onward

    StreamSplitter() = default;
    explicit StreamSplitter(bool thinking) : in_reasoning(thinking) {}

    // Consume `chunk`; append newly-decided output to the two deltas.
    void feed(const std::string& chunk, std::string& r_delta, std::string& c_delta) {
        raw += chunk;
        pending += chunk;
        for (;;) {
            if (in_tools) { tools_buf += pending; pending.clear(); return; }
            const char* mark = in_reasoning ? THINK_END_MARK : TOOL_OPEN_MARK;
            const size_t mlen = std::char_traits<char>::length(mark);
            const size_t p = pending.find(mark);
            std::string& out = in_reasoning ? r_delta : c_delta;
            if (p != std::string::npos) {
                out += pending.substr(0, p);
                if (in_reasoning) { in_reasoning = false; pending.erase(0, p + mlen); }
                else              { in_tools = true; tools_buf = pending.substr(p); pending.clear(); }
                continue;                       // the same chunk may contain the next marker too
            }
            // No marker yet: emit everything that cannot be the start of one.
            const size_t hb = stream_holdback();
            size_t keep = pending.size() > hb ? pending.size() - hb : 0;
            keep = utf8_floor(pending, keep);
            if (keep) { out += pending.substr(0, keep); pending.erase(0, keep); }
            return;
        }
    }

    // Generation ended: nothing more can arrive, so the held-back tail is real output.
    void finish(std::string& r_delta, std::string& c_delta) {
        if (pending.empty()) return;
        if (in_tools) tools_buf += pending;
        else if (in_reasoning) r_delta += pending;
        else c_delta += pending;
        pending.clear();
    }
};

}  // namespace glm5srv
