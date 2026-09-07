// vision_http.h — turning an OpenAI `image_url` content part into engine input.
//
// Two jobs the server needs and nothing else does: pull the image bytes out of a chat request, and
// expand the single `<|image|>` token the chat encoder emits into the N the tower actually
// produces, recording where they landed so the embeddings can be spliced there.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace glm5 {

// GLM-5.3 multimodal token ids, from config.json.
inline constexpr int IMAGE_TOKEN_ID = 154854;   // <|image|>

// Decode a base64 payload. Tolerates whitespace and missing padding; returns false on a stray
// character rather than guessing, because this is reachable from the network.
inline bool b64_decode(const std::string& in, std::vector<uint8_t>& out) {
    auto val = [](char c) -> int {
        if (c >= 'A' && c <= 'Z') return c - 'A';
        if (c >= 'a' && c <= 'z') return c - 'a' + 26;
        if (c >= '0' && c <= '9') return c - '0' + 52;
        if (c == '+') return 62;
        if (c == '/') return 63;
        return -1;
    };
    out.clear(); out.reserve(in.size() * 3 / 4);
    int acc = 0, bits = 0;
    for (char c : in) {
        if (c == '=' ) break;
        if (c == '\n' || c == '\r' || c == ' ' || c == '\t') continue;
        const int v = val(c);
        if (v < 0) return false;
        acc = (acc << 6) | v; bits += 6;
        if (bits >= 8) { bits -= 8; out.push_back((uint8_t)((acc >> bits) & 0xFF)); }
    }
    return true;
}

// Accepts `data:image/...;base64,<payload>`. A plain http(s) URL is REFUSED, deliberately: making
// the inference server fetch arbitrary URLs on a caller's behalf is server-side request forgery,
// and a box holding a 100 GiB checkpoint on someone's LAN is not the place to add one.
inline bool image_bytes_from_url(const std::string& url, std::vector<uint8_t>& bytes,
                                 std::string& err) {
    const std::string pfx = "data:";
    if (url.rfind(pfx, 0) != 0) {
        err = "only data: URIs are accepted; remote URL fetching is disabled (SSRF)";
        return false;
    }
    const size_t comma = url.find(',');
    if (comma == std::string::npos) { err = "malformed data: URI"; return false; }
    const std::string meta = url.substr(0, comma);
    if (meta.find(";base64") == std::string::npos) { err = "data: URI must be base64"; return false; }
    if (!b64_decode(url.substr(comma + 1), bytes)) { err = "invalid base64"; return false; }
    if (bytes.size() < 16) { err = "image too small to be real"; return false; }
    return true;
}

// Replace each IMAGE_TOKEN_ID in `ids` with `counts[k]` copies, and report where each run starts.
// The chat encoder emits exactly one <|image|> per image, in message order, so the k-th token
// corresponds to the k-th image. Returns false if the counts and the tokens disagree, which means
// the encoder and the request parser have drifted apart and is not something to paper over.
inline bool expand_image_tokens(std::vector<int>& ids, const std::vector<int>& counts,
                                std::vector<int>& starts) {
    std::vector<int> out; out.reserve(ids.size() + 256);
    starts.clear();
    size_t k = 0;
    for (int id : ids) {
        if (id != IMAGE_TOKEN_ID) { out.push_back(id); continue; }
        if (k >= counts.size()) return false;
        starts.push_back((int)out.size());
        out.insert(out.end(), (size_t)counts[k], IMAGE_TOKEN_ID);
        ++k;
    }
    if (k != counts.size()) return false;
    ids.swap(out);
    return true;
}

}  // namespace glm5
