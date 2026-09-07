// openai_api.h — OpenAI-compatible request/response shaping for GLM-5.3-Flash-REAP50.
//
// Kept separate from the HTTP transport and from the engine so it can be gated without a GPU
// (tests/gate_api.cpp). The engine-facing surface is deliberately tiny: a prompt string in,
// generated text out.
//
// Model-specific defaults that differ from the other servers in this family — do NOT carry
// gemma's 0.7/0.9 or DeepSeek's 1.0/1.0 over by habit. generation_config.json ships
// temperature 1.0 and top_p 0.95, and those are the numbers the checkpoint was tuned at.
//
// There is NO thinking_mode. Every assistant turn carries a think block; a client that wants a
// short answer says so in the prompt or uses the `/nothink` token, it does not flip a server flag.
// Accepting `enable_thinking` and silently ignoring it would be worse than rejecting it, so it is
// not accepted at all.
#pragma once
#include "encoding_glm5.h"
#include <string>
#include <vector>
#include <cstdint>

namespace glm5api {

using glm5enc::json;

struct SamplingParams {
    double temperature = 1.0;      // generation_config.json
    double top_p       = 0.95;     // generation_config.json
    int    top_k       = 0;
    double min_p       = 0.0;
    int    max_tokens  = 512;
    std::vector<std::string> stop;
    uint64_t seed = 0;
    bool has_seed = false;
};

struct ChatRequest {
    std::string model = "glm-5.3-flash-reap50-nvfp4";
    json messages = json::array();
    json tools = json();
    std::string reasoning_effort;      // "" -> Max, matching the template's own default
    bool clear_thinking = false;
    bool stream = false;
    bool include_usage = true;
    SamplingParams sampling;
};

inline bool truthy(const json& v, bool dflt) {
    if (v.is_boolean()) return v.get<bool>();
    if (v.is_number())  return v.get<double>() != 0.0;
    return dflt;
}

// Parse an OpenAI /v1/chat/completions body. Tolerant of the usual client variations.
inline ChatRequest parse_chat_request(const json& b) {
    ChatRequest r;
    if (b.contains("model") && b["model"].is_string()) r.model = b["model"].get<std::string>();
    if (b.contains("messages")) r.messages = b["messages"];
    if (b.contains("stream"))   r.stream   = truthy(b["stream"], false);
    // Tools stay on the REQUEST, where OpenAI puts them; the template reads them from there too,
    // so unlike the DeepSeek encoder there is nothing to fold into a system message.
    if (b.contains("tools") && b["tools"].is_array() && !b["tools"].empty()) r.tools = b["tools"];

    // reasoning_effort: only 'low' and 'high' are real. Anything else renders as Max, and that is
    // the template's behaviour, not a fallback invented here — so 'medium' is passed through
    // rather than rejected, and comes out as Max exactly as HF's Jinja would render it.
    if (b.contains("reasoning_effort") && b["reasoning_effort"].is_string())
        r.reasoning_effort = b["reasoning_effort"].get<std::string>();
    if (b.contains("chat_template_kwargs") && b["chat_template_kwargs"].is_object()) {
        const auto& k = b["chat_template_kwargs"];
        if (k.contains("reasoning_effort") && k["reasoning_effort"].is_string())
            r.reasoning_effort = k["reasoning_effort"].get<std::string>();
        if (k.contains("clear_thinking")) r.clear_thinking = truthy(k["clear_thinking"], false);
    }

    auto& s = r.sampling;
    if (b.contains("temperature") && b["temperature"].is_number()) s.temperature = b["temperature"].get<double>();
    if (b.contains("top_p") && b["top_p"].is_number()) s.top_p = b["top_p"].get<double>();
    if (b.contains("top_k") && b["top_k"].is_number()) s.top_k = b["top_k"].get<int>();
    if (b.contains("min_p") && b["min_p"].is_number()) s.min_p = b["min_p"].get<double>();
    if (b.contains("max_tokens") && b["max_tokens"].is_number())            s.max_tokens = b["max_tokens"].get<int>();
    else if (b.contains("max_completion_tokens") && b["max_completion_tokens"].is_number())
                                                                            s.max_tokens = b["max_completion_tokens"].get<int>();
    if (b.contains("seed") && b["seed"].is_number()) { s.seed = b["seed"].get<uint64_t>(); s.has_seed = true; }
    if (b.contains("stop")) {
        if (b["stop"].is_string()) s.stop.push_back(b["stop"].get<std::string>());
        else if (b["stop"].is_array()) for (auto& x : b["stop"]) if (x.is_string()) s.stop.push_back(x.get<std::string>());
    }
    if (b.contains("stream_options") && b["stream_options"].is_object() &&
        b["stream_options"].contains("include_usage"))
        r.include_usage = truthy(b["stream_options"]["include_usage"], true);
    return r;
}

// Image URLs in message order — the SAME order build_prompt emits <|image|> in, which is what
// lets the k-th token be matched to the k-th image. If you change one, change both.
inline std::vector<std::string> collect_image_urls(const nlohmann::json& body) {
    std::vector<std::string> out;
    if (!body.contains("messages") || !body["messages"].is_array()) return out;
    for (const auto& m : body["messages"]) {
        if (!m.is_object() || !m.contains("content")) continue;
        const auto& c = m["content"];
        if (!c.is_array()) continue;
        for (const auto& it : c) {
            if (!it.is_object()) continue;
            const std::string t = it.value("type", "");
            if (t != "image" && t != "image_url") continue;
            if (it.contains("image_url")) {
                const auto& iu = it["image_url"];
                if (iu.is_string()) out.push_back(iu.get<std::string>());
                else if (iu.is_object()) out.push_back(iu.value("url", std::string()));
            } else if (it.contains("image") && it["image"].is_string()) {
                out.push_back(it["image"].get<std::string>());
            } else out.push_back(std::string());
        }
    }
    return out;
}

inline std::string build_prompt(const ChatRequest& r) {
    glm5enc::Options o;
    o.reasoning_effort = r.reasoning_effort;
    o.add_generation_prompt = true;
    o.clear_thinking = r.clear_thinking;
    return glm5enc::encode_messages(r.messages, r.tools, o);
}

// Non-streaming response. `parsed` comes from glm5enc::parse_message_from_completion_text.
inline json chat_completion_response(const std::string& id, const std::string& model,
                                     const json& parsed, int prompt_tokens, int completion_tokens,
                                     long created, const char* finish_override = nullptr) {
    json msg = json::object();
    msg["role"] = "assistant";
    msg["content"] = parsed.value("content", "");
    const std::string rc = parsed.value("reasoning_content", "");
    if (!rc.empty()) msg["reasoning_content"] = rc;      // surfaced separately, as vLLM/SGLang do
    const bool has_tc = parsed.contains("tool_calls") && !parsed["tool_calls"].empty();
    if (has_tc) {
        json tcs = json::array();
        int i = 0;
        for (const auto& tc : parsed["tool_calls"]) {
            json o = tc;
            o["id"] = "call_" + id.substr(0, 8) + "_" + std::to_string(i++);
            tcs.push_back(o);
        }
        msg["tool_calls"] = tcs;
    }
    json choice = json::object();
    choice["index"] = 0;
    choice["message"] = msg;
    choice["finish_reason"] = finish_override ? finish_override : (has_tc ? "tool_calls" : "stop");

    json out = json::object();
    out["id"] = "chatcmpl-" + id;
    out["object"] = "chat.completion";
    out["created"] = created;
    out["model"] = model;
    out["choices"] = json::array({choice});
    json usage = json::object();
    usage["prompt_tokens"] = prompt_tokens;
    usage["completion_tokens"] = completion_tokens;
    usage["total_tokens"] = prompt_tokens + completion_tokens;
    out["usage"] = usage;
    return out;
}

// One SSE chunk of a streamed completion.
inline std::string sse_chunk(const std::string& id, const std::string& model, long created,
                             const std::string& delta_content, const std::string& delta_reasoning,
                             const char* finish_reason) {
    json d = json::object();
    if (!delta_content.empty())   d["content"] = delta_content;
    if (!delta_reasoning.empty()) d["reasoning_content"] = delta_reasoning;
    json choice = json::object();
    choice["index"] = 0;
    choice["delta"] = d;
    if (finish_reason) choice["finish_reason"] = finish_reason; else choice["finish_reason"] = nullptr;
    json o = json::object();
    o["id"] = "chatcmpl-" + id;
    o["object"] = "chat.completion.chunk";
    o["created"] = created;
    o["model"] = model;
    o["choices"] = json::array({choice});
    // dump with the replace handler, not the throwing default: deltas carry model-generated text
    // and byte-level BPE can emit sequences that are not valid UTF-8 on their own. See
    // dump_lossy() in server.cpp for the incident on the sibling server this comes from.
    return "data: " + o.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
}

inline json models_response(const std::string& model, long created) {
    json m = json::object();
    m["id"] = model; m["object"] = "model"; m["created"] = created; m["owned_by"] = "local";
    json o = json::object();
    o["object"] = "list";
    o["data"] = json::array({m});
    return o;
}

}  // namespace glm5api
