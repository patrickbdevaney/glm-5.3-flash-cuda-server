// encoding_glm5.h — C++ port of the checkpoint's chat_template.jinja. No Python on the hot path.
//
// The template is 257 lines of Jinja whose whitespace-control markers differ line by line, so this
// is NOT written from a reading of it. It is written to reproduce what the template actually
// emits, and gated BYTE-EXACT against 19 fixtures rendered by HF's own Jinja engine
// (tools/gen_chat_vectors.py -> ref/chat_vectors.json, tests/gate_encoding.cpp).
//
// WHAT THE TEMPLATE PRODUCES, once whitespace control is applied:
//
//   [gMASK]<sop>
//   <|system|>Reasoning Effort: {Low|High|Max}          -- always present; see below
//   [<|system|>\n# Tools\n\n...<tools>\n{schemas}\n</tools>\n\n...]   -- only when tools are given
//   per message:
//     user       <|user|>{text}
//     system     <|system|>{text}
//     assistant  <|assistant|><think>{reasoning}</think>{content}{tool_calls}
//     tool       <|observation|> once per contiguous run, then <tool_response>{r}</tool_response>*
//   <|assistant|><think>                                -- when add_generation_prompt
//
// THREE THINGS THAT ARE NOT OBVIOUS FROM THE TEMPLATE AND ARE LOAD-BEARING:
//
//   1. `reasoning_effort` accepts ONLY 'low' and 'high'. Anything else -- including 'medium', and
//      including omitting it -- renders as "Max". So the default is the most expensive setting,
//      and a client passing 'medium' silently gets 'Max' rather than an error.
//   2. An assistant turn ALWAYS carries a think block. With no reasoning it is the empty
//      `<think></think>`, never nothing. Omitting it puts the prompt off-distribution in the one
//      place the model is most sensitive to.
//   3. Tool results are re-ordered to match the preceding assistant's tool_calls when every result
//      carries a tool_call_id, no id repeats, and every id matches a call. Otherwise message order
//      is kept. A client that returns results out of order is silently corrected -- but only if it
//      supplied ids, so dropping ids changes the prompt.
#pragma once
#include "third_party/json.hpp"
#include <string>
#include <vector>
#include <stdexcept>

namespace glm5enc {

// ordered_json, NOT json: nlohmann's default `json` sorts object keys, while Python's json.dumps
// preserves insertion order. Tool schemas are embedded in the prompt verbatim, so key order is
// load-bearing and sorting them fails the byte-exact gate.
using json = nlohmann::ordered_json;

// ---- special tokens (added_tokens in tokenizer.json) ----
inline constexpr const char* GMASK        = "[gMASK]";
inline constexpr const char* SOP          = "<sop>";
inline constexpr const char* SYSTEM_SP    = "<|system|>";
inline constexpr const char* USER_SP      = "<|user|>";
inline constexpr const char* ASSISTANT_SP = "<|assistant|>";
inline constexpr const char* OBSERVATION  = "<|observation|>";
inline constexpr const char* THINK_START  = "<think>";
inline constexpr const char* THINK_END    = "</think>";
inline constexpr const char* TOOL_CALL_S  = "<tool_call>";
inline constexpr const char* TOOL_CALL_E  = "</tool_call>";
inline constexpr const char* ARG_KEY_S    = "<arg_key>";
inline constexpr const char* ARG_KEY_E    = "</arg_key>";
inline constexpr const char* ARG_VAL_S    = "<arg_value>";
inline constexpr const char* ARG_VAL_E    = "</arg_value>";
inline constexpr const char* TOOL_RESP_S  = "<tool_response>";
inline constexpr const char* TOOL_RESP_E  = "</tool_response>";
inline constexpr const char* IMAGE_TRIPLE = "<|begin_of_image|><|image|><|end_of_image|>";
inline constexpr const char* VIDEO_TRIPLE = "<|begin_of_video|><|video|><|end_of_video|>";
inline constexpr const char* AUDIO_PAIR   = "<|begin_of_audio|><|end_of_audio|>";

// ---- helpers -----------------------------------------------------------------------------------

// Mimic Python's json.dumps(value, ensure_ascii=False) EXACTLY. Two differences from nlohmann's
// dump() both break the byte-exact gate: Python's default separators are ", " and ": " (with
// spaces), and key order (handled by ordered_json above). Scalars delegate to dump(), which
// already escapes correctly and emits raw UTF-8.
inline std::string to_json(const json& v) {
    if (v.is_object()) {
        std::string s = "{";
        bool first = true;
        for (auto it = v.begin(); it != v.end(); ++it) {
            if (!first) s += ", ";
            first = false;
            s += json(it.key()).dump() + ": " + to_json(it.value());
        }
        return s + "}";
    }
    if (v.is_array()) {
        std::string s = "[";
        for (size_t i = 0; i < v.size(); ++i) { if (i) s += ", "; s += to_json(v[i]); }
        return s + "]";
    }
    return v.dump();
}

// Jinja's .strip() is Python's: ASCII whitespace on both ends.
inline std::string strip(const std::string& s) {
    const char* ws = " \t\n\r\f\v";
    const size_t a = s.find_first_not_of(ws);
    if (a == std::string::npos) return "";
    return s.substr(a, s.find_last_not_of(ws) - a + 1);
}

inline bool contains(const std::string& h, const std::string& n) {
    return h.find(n) != std::string::npos;
}

// Python's str.split(sep)[0] / [-1], which is not the same as find(): [-1] is the tail after the
// LAST separator, and [0] the head before the FIRST.
inline std::string split_first(const std::string& s, const std::string& sep) {
    const size_t p = s.find(sep);
    return p == std::string::npos ? s : s.substr(0, p);
}
inline std::string split_last(const std::string& s, const std::string& sep) {
    const size_t p = s.rfind(sep);
    return p == std::string::npos ? s : s.substr(p + sep.size());
}

// The template's visible_text() macro: a string passes through; a list of parts is concatenated,
// with image/video/audio parts replaced by their placeholder triples.
inline std::string visible_text(const json& content) {
    if (content.is_null()) return "";
    if (content.is_string()) return content.get<std::string>();
    if (content.is_array()) {
        std::string o;
        for (const auto& item : content) {
            if (item.is_string()) { o += item.get<std::string>(); continue; }
            if (!item.is_object()) continue;
            const std::string t = item.value("type", "");
            if (t == "text") o += item.value("text", "");
            else if (t == "image" || t == "image_url") o += IMAGE_TRIPLE;
            else if (t == "video" || t == "video_url") o += VIDEO_TRIPLE;
            else if (t == "audio" || t == "audio_url" || t == "input_audio") o += AUDIO_PAIR;
        }
        return o;
    }
    return to_json(content);
}

// ---- reasoning effort ---------------------------------------------------------------------------
// effective = reasoning_effort if it is exactly 'low' or 'high', else 'max'; then capitalize().
inline std::string effort_word(const std::string& req) {
    if (req == "low")  return "Low";
    if (req == "high") return "High";
    return "Max";
}

// ---- tools --------------------------------------------------------------------------------------

// tool_to_json: drop `defer_loading` and `strict`, keep every other key in order.
inline std::string tool_to_json(const json& tool_in) {
    const json& tool = tool_in.contains("function") ? tool_in["function"] : tool_in;
    std::string s = "{";
    bool first = true;
    for (auto it = tool.begin(); it != tool.end(); ++it) {
        if (it.key() == "defer_loading" || it.key() == "strict") continue;
        if (!first) s += ", ";
        first = false;
        s += json(it.key()).dump() + ": " + to_json(it.value());
    }
    return s + "}";
}

// The tools system block, verbatim. Do NOT paraphrase: it sits near position 0 where it also keys
// the prefix cache, and a reworded instruction is a different prompt.
inline std::string render_tools(const json& tools) {
    std::string schemas;
    bool first = true;
    for (const auto& t : tools) {
        const json& fn = t.contains("function") ? t["function"] : t;
        if (fn.value("defer_loading", false)) continue;
        if (!first) schemas += "\n";
        first = false;
        schemas += tool_to_json(t);
    }
    return std::string(SYSTEM_SP) +
        "\n# Tools\n\n"
        "You may call one or more functions to assist with the user query.\n\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n"
        "<tools>\n" + schemas + "\n</tools>\n\n"
        "For each function call, output the function name and arguments within the following XML format:\n"
        "<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value>"
        "<arg_key>{arg-key-2}</arg_key><arg_value>{arg-value-2}</arg_value>...</tool_call>";
}

// One assistant tool call. Argument VALUES: a string goes in verbatim, anything else is JSON.
// OpenAI clients send `arguments` as a JSON *string*, but the template iterates it as a mapping,
// so it must be parsed first — passing the raw string through emits one arg_key of garbage.
inline std::string render_tool_call(const json& tc_in) {
    const json& tc = tc_in.contains("function") ? tc_in["function"] : tc_in;
    std::string s = std::string(TOOL_CALL_S) + tc.value("name", "");
    json args = tc.contains("arguments") ? tc["arguments"] : json::object();
    if (args.is_string()) {
        const std::string raw = args.get<std::string>();
        args = json::parse(raw, nullptr, false);           // no-throw: bad JSON must not 500
        if (args.is_discarded() || !args.is_object()) args = json::object();
    }
    if (args.is_object())
        for (auto it = args.begin(); it != args.end(); ++it) {
            const std::string v = it.value().is_string() ? it.value().get<std::string>()
                                                         : to_json(it.value());
            s += std::string(ARG_KEY_S) + it.key() + ARG_KEY_E + ARG_VAL_S + v + ARG_VAL_E;
        }
    return s + TOOL_CALL_E;
}

// ---- tool-result blocks --------------------------------------------------------------------------
// A contiguous run of role=="tool" messages. `is_list_of_outputs` is the responses-API shape where
// one message carries several results; the ordinary shape is one message per result.
inline bool is_list_of_outputs(const json& m) {
    return m.contains("content") && m["content"].is_array() && !m["content"].empty() &&
           m["content"][0].is_object() && m["content"][0].contains("output");
}
inline std::string id_of(const json& o) {
    if (o.contains("tool_call_id") && o["tool_call_id"].is_string()) return o["tool_call_id"];
    if (o.contains("id") && o["id"].is_string()) return o["id"];
    return "";
}
inline std::string tool_response(const std::string& text) {
    return std::string(TOOL_RESP_S) + text + TOOL_RESP_E;
}
// Everything one tool message contributes, in message order.
inline std::string render_tool_message(const json& m) {
    if (m.contains("content") && m["content"].is_string())
        return tool_response(m["content"].get<std::string>());
    if (is_list_of_outputs(m)) {
        std::string o;
        for (const auto& tr : m["content"]) o += tool_response(visible_text(tr["output"]));
        return o;
    }
    return tool_response(visible_text(m.value("content", json())));
}

// ---- the encoder ---------------------------------------------------------------------------------
struct Options {
    std::string reasoning_effort;          // "" -> Max, same as the template's default
    bool add_generation_prompt = true;
    bool clear_thinking = false;
};

inline std::string encode_messages(const json& messages, const json& tools, const Options& opt) {
    std::string s = std::string(GMASK) + SOP;
    s += std::string(SYSTEM_SP) + "Reasoning Effort: " + effort_word(opt.reasoning_effort);
    if (!tools.is_null() && tools.is_array() && !tools.empty()) s += render_tools(tools);

    const int n = (int)messages.size();
    int last_user = -1;
    for (int i = 0; i < n; ++i)
        if (messages[i].value("role", "") == "user") last_user = i;

    for (int i = 0; i < n; ++i) {
        const json& m = messages[i];
        const std::string role = m.value("role", "");

        if (role == "user") {
            s += std::string(USER_SP) + visible_text(m.value("content", json()));
            continue;
        }
        if (role == "system") {
            s += std::string(SYSTEM_SP) + visible_text(m.value("content", json()));
            continue;
        }
        if (role == "assistant") {
            s += ASSISTANT_SP;
            std::string content = visible_text(m.value("content", json()));
            std::string reasoning;
            bool have_reasoning = false;
            if (m.contains("reasoning_content") && m["reasoning_content"].is_string()) {
                reasoning = m["reasoning_content"].get<std::string>();
                have_reasoning = true;
            } else if (contains(content, THINK_END)) {
                reasoning = split_last(split_first(content, THINK_END), THINK_START);
                content   = split_last(content, THINK_END);
                have_reasoning = true;
            }
            // clear_thinking drops reasoning from every turn at or before the last user message —
            // the history is replayed without its scratchpad, but the turn still in progress keeps
            // one. The empty think block is emitted either way.
            if (have_reasoning && (!opt.clear_thinking || i > last_user))
                s += std::string(THINK_START) + reasoning + THINK_END;
            else
                s += std::string(THINK_START) + THINK_END;
            const std::string body = strip(content);
            if (!body.empty()) s += body;
            if (m.contains("tool_calls") && m["tool_calls"].is_array())
                for (const auto& tc : m["tool_calls"]) s += render_tool_call(tc);
            continue;
        }
        if (role == "tool") {
            // Only the first message of a contiguous run opens the observation block; the rest are
            // consumed here, so skip them when the loop reaches them.
            if (i > 0 && messages[i-1].value("role", "") == "tool") continue;
            s += OBSERVATION;
            int end = i;
            while (end + 1 < n && messages[end+1].value("role", "") == "tool") ++end;

            // Sort results into tool_calls order when — and only when — every result has an id, no
            // id repeats, and every id matches a call in the preceding assistant turn.
            const json* calls = nullptr;
            if (i > 0 && messages[i-1].value("role", "") == "assistant" &&
                messages[i-1].contains("tool_calls") && messages[i-1]["tool_calls"].is_array() &&
                !messages[i-1]["tool_calls"].empty())
                calls = &messages[i-1]["tool_calls"];

            bool can_sort = calls != nullptr;
            std::vector<std::string> ids;                    // one per result, in message order
            std::vector<const json*> results;
            if (can_sort) {
                for (int k = i; k <= end; ++k) {
                    if (is_list_of_outputs(messages[k]))
                        for (const auto& e : messages[k]["content"]) { ids.push_back(id_of(e)); results.push_back(&e); }
                    else { ids.push_back(id_of(messages[k])); results.push_back(&messages[k]); }
                }
                for (size_t a = 0; a < ids.size() && can_sort; ++a) {
                    if (ids[a].empty()) { can_sort = false; break; }
                    for (size_t b = a + 1; b < ids.size(); ++b)
                        if (ids[b] == ids[a]) { can_sort = false; break; }
                    bool found = false;
                    for (const auto& tc : *calls) if (id_of(tc) == ids[a]) { found = true; break; }
                    if (!found) can_sort = false;
                }
                for (const auto& tc : *calls) if (id_of(tc).empty()) { can_sort = false; break; }
            }

            if (can_sort) {
                for (const auto& tc : *calls) {
                    const std::string want = id_of(tc);
                    for (size_t a = 0; a < ids.size(); ++a) {
                        if (ids[a] != want) continue;
                        const json& r = *results[a];
                        s += r.contains("output") ? tool_response(visible_text(r["output"]))
                                                  : render_tool_message(r);
                    }
                }
            } else {
                for (int k = i; k <= end; ++k) s += render_tool_message(messages[k]);
            }
            i = end;
            continue;
        }
        // Any other role is dropped, exactly as the template's if/elif chain drops it.
    }

    if (opt.add_generation_prompt) s += std::string(ASSISTANT_SP) + THINK_START;
    return s;
}

// ---- the inverse: completion text -> an OpenAI message -------------------------------------------
// The engine is prompted with a trailing `<think>`, so the model's output STARTS inside the
// reasoning block and the opening tag is never generated.
//
// `starts_in_reasoning` is not a detail. When generation is cut off by max_tokens before the model
// ever closes the block, there is no `</think>` to find — and treating that as content puts raw
// scratchpad in the `content` field. The STREAMING path has this right for free (its splitter is
// told where it starts), so getting it wrong here makes the two paths disagree about the same
// generation: caught by exactly that comparison in the 3-layer smoke test.
inline json parse_message_from_completion_text(const std::string& text_in,
                                               bool starts_in_reasoning = true) {
    std::string text = text_in;
    json msg = json::object();
    msg["role"] = "assistant";

    std::string reasoning;
    const size_t te = text.find(THINK_END);
    if (te != std::string::npos) {
        reasoning = text.substr(0, te);
        // A stray opening tag, if the model emitted one anyway.
        const size_t ts = reasoning.rfind(THINK_START);
        if (ts != std::string::npos) reasoning = reasoning.substr(ts + std::string(THINK_START).size());
        text = text.substr(te + std::string(THINK_END).size());
    } else if (starts_in_reasoning) {
        reasoning = text;                     // the whole generation is an unterminated think block
        text.clear();
    }

    json tool_calls = json::array();
    std::string content;
    size_t p = 0;
    while (true) {
        const size_t a = text.find(TOOL_CALL_S, p);
        if (a == std::string::npos) { content += text.substr(p); break; }
        content += text.substr(p, a - p);
        const size_t body = a + std::string(TOOL_CALL_S).size();
        size_t b = text.find(TOOL_CALL_E, body);
        const bool closed = b != std::string::npos;
        if (!closed) b = text.size();                     // truncated call: parse what there is
        const std::string blk = text.substr(body, b - body);

        // <tool_call>NAME<arg_key>k</arg_key><arg_value>v</arg_value>...
        const size_t name_end = blk.find(ARG_KEY_S);
        const std::string name = strip(name_end == std::string::npos ? blk : blk.substr(0, name_end));
        json args = json::object();
        size_t q = 0;
        while (true) {
            const size_t ks = blk.find(ARG_KEY_S, q);
            if (ks == std::string::npos) break;
            const size_t ke = blk.find(ARG_KEY_E, ks);
            if (ke == std::string::npos) break;
            const std::string key = blk.substr(ks + std::string(ARG_KEY_S).size(),
                                               ke - ks - std::string(ARG_KEY_S).size());
            const size_t vs = blk.find(ARG_VAL_S, ke);
            if (vs == std::string::npos) break;
            size_t ve = blk.find(ARG_VAL_E, vs);
            if (ve == std::string::npos) ve = blk.size();
            const std::string val = blk.substr(vs + std::string(ARG_VAL_S).size(),
                                               ve - vs - std::string(ARG_VAL_S).size());
            args[key] = val;                              // values are emitted as text, so they stay text
            q = ve + 1;
        }
        json tc = json::object();
        tc["id"] = "call_" + std::to_string(tool_calls.size());
        tc["type"] = "function";
        tc["function"] = json::object();
        tc["function"]["name"] = name;
        tc["function"]["arguments"] = to_json(args);       // OpenAI wants a JSON string here
        tool_calls.push_back(tc);
        p = closed ? b + std::string(TOOL_CALL_E).size() : text.size();
    }

    msg["content"] = strip(content);
    if (!reasoning.empty()) msg["reasoning_content"] = reasoning;
    if (!tool_calls.empty()) msg["tool_calls"] = tool_calls;
    return msg;
}

} // namespace glm5enc
