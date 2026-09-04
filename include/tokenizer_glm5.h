// tokenizer_glm5.h — pure-C++ GLM-5.3-Flash tokenizer. Loads the checkpoint's own tokenizer.json.
//
// WHY A THIRD TOKENIZER ON THIS BOX. gemma's is SentencePiece-style. DeepSeek-V4's
// (`tokenizer_dsv4.h`) is ByteLevel BPE but with a FOUR-stage pre-tokenizer: digits, CJK, a big
// alternation, then ByteLevel. GLM-5.3 is ByteLevel BPE with a ONE-stage pre-tokenizer holding the
// standard GPT-4/cl100k alternation, and it sets `ignore_merges`. Reusing dsv4's by habit produces
// ids that are wrong but plausible — the worst failure mode a decode server has, because nothing
// errors and the model simply talks slightly-wrong. Written against tokenizer.json and gated
// against HF in tests/gate_tokenizer.cpp.
//
// PIPELINE (tokenizer.json, verbatim):
//   normalizer    null                                        -- genuinely absent, not empty
//   pre_tokenizer Sequence[
//      Split(Regex <the GPT-4 alternation>, Isolated)
//      ByteLevel(add_prefix_space=false, use_regex=false)      -- byte -> unicode, no splitting
//   ]
//   model         BPE(dropout=null, unk=null, byte_fallback=false, ignore_merges=TRUE)
//   post_processor ByteLevel                                   -- adds NO bos/eos
//   decoder       ByteLevel
// added_tokens (36) are matched literally on the RAW text, before any of the above.
//
// THREE THINGS THAT DIFFER FROM dsv4 AND WILL BITE IF ASSUMED:
//   1. `ignore_merges: true`. If a whole pre-token is already a vocab entry, it is emitted
//      directly and BPE never runs on it. Without this the common words still tokenise, but a
//      minority take a different merge path and land on different ids.
//   2. The post-processor adds nothing. There is no BOS. The prompt's `[gMASK]<sop>` prefix comes
//      from the chat template as ordinary added tokens (encoding_glm5.h), not from here.
//   3. Only 18 of the 36 added tokens are `special`. `<think>`, `</think>`, `<tool_call>` and the
//      `<arg_key>`/`<arg_value>` markers are added-but-NOT-special, so `skip_special_tokens`
//      must not drop them — they are exactly what the stream splitter and tool parser look for.
#pragma once
#include "third_party/json.hpp"
#include "unicode_cat.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <cstdint>
#include <climits>

namespace glm5tok {

// ---- UTF-8 <-> codepoints ---------------------------------------------------------------------
inline int u8len(unsigned char c) {
    return c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : (c >> 3) == 0x1E ? 4 : 1;
}
inline void u8_decode(const std::string& s, std::vector<uint32_t>& cp, std::vector<int>& off) {
    cp.clear(); off.clear();
    for (size_t i = 0; i < s.size();) {
        const int L = u8len((unsigned char)s[i]);
        uint32_t c = 0;
        if (L == 1) c = (unsigned char)s[i];
        else if (L == 2) c = ((s[i] & 0x1F) << 6) | (s[i+1] & 0x3F);
        else if (L == 3) c = ((s[i] & 0x0F) << 12) | ((s[i+1] & 0x3F) << 6) | (s[i+2] & 0x3F);
        else             c = ((s[i] & 0x07) << 18) | ((s[i+1] & 0x3F) << 12) | ((s[i+2] & 0x3F) << 6) | (s[i+3] & 0x3F);
        cp.push_back(c); off.push_back((int)i);
        i += L;
    }
    off.push_back((int)s.size());
}
inline void u8_append(std::string& s, uint32_t c) {
    if (c < 0x80) s.push_back((char)c);
    else if (c < 0x800) { s.push_back((char)(0xC0 | (c >> 6))); s.push_back((char)(0x80 | (c & 0x3F))); }
    else if (c < 0x10000) { s.push_back((char)(0xE0 | (c >> 12))); s.push_back((char)(0x80 | ((c >> 6) & 0x3F))); s.push_back((char)(0x80 | (c & 0x3F))); }
    else { s.push_back((char)(0xF0 | (c >> 18))); s.push_back((char)(0x80 | ((c >> 12) & 0x3F))); s.push_back((char)(0x80 | ((c >> 6) & 0x3F))); s.push_back((char)(0x80 | (c & 0x3F))); }
}

// ---- character classes used by the pre-tokenizer -----------------------------------------------
// \s is Unicode White_Space=Yes. The set is small and fixed; spelling it out is more honest than
// generating a sixth table for six ranges.
inline bool is_space(uint32_t c) {
    return (c >= 0x09 && c <= 0x0D) || c == 0x20 || c == 0x85 || c == 0xA0 || c == 0x1680 ||
           (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F ||
           c == 0x205F || c == 0x3000;
}
inline bool is_nl(uint32_t c) { return c == '\r' || c == '\n'; }
inline uint32_t lower_ascii(uint32_t c) { return (c >= 'A' && c <= 'Z') ? c + 32 : c; }

// ---- the GPT-2 byte <-> unicode alphabet -------------------------------------------------------
struct ByteLevel {
    uint32_t b2u[256];
    std::unordered_map<uint32_t, int> u2b;
    ByteLevel() {
        std::vector<int> bs;
        for (int b = 33; b < 127; ++b) bs.push_back(b);
        for (int b = 161; b < 173; ++b) bs.push_back(b);
        for (int b = 174; b < 256; ++b) bs.push_back(b);
        std::vector<uint32_t> cs(bs.begin(), bs.end());
        int n = 0;
        for (int b = 0; b < 256; ++b)
            if (std::find(bs.begin(), bs.end(), b) == bs.end()) { bs.push_back(b); cs.push_back(256 + n++); }
        for (size_t i = 0; i < bs.size(); ++i) { b2u[bs[i]] = cs[i]; u2b[cs[i]] = bs[i]; }
    }
    std::string encode(const std::string& raw) const {
        std::string o;
        for (unsigned char ch : raw) u8_append(o, b2u[ch]);
        return o;
    }
    // Inverse. Unmapped codepoints cannot occur in vocab strings, so they are dropped rather than
    // guessed at.
    std::string decode(const std::string& mapped) const {
        std::vector<uint32_t> cp; std::vector<int> off;
        u8_decode(mapped, cp, off);
        std::string o;
        for (uint32_t c : cp) { auto it = u2b.find(c); if (it != u2b.end()) o.push_back((char)it->second); }
        return o;
    }
};

// ---- the pre-tokenizer -------------------------------------------------------------------------
// The GPT-4 alternation, verbatim from tokenizer.json:
//
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)      0  contractions, ASCII-case-insensitive
//   [^\r\n\p{L}\p{N}]?\p{L}+          1  optional lead char, then letters
//   \p{N}{1,3}                        2  digits, in groups of at most three
//    ?[^\s\p{L}\p{N}]+[\r\n]*         3  optional space, then symbols/punct, then newlines
//   \s*[\r\n]+                        4  a blank-line run
//   \s+(?!\S)                         5  trailing whitespace, minus its last character
//   \s+                               6  whatever whitespace is left
//
// The pattern carries a lookahead, so HF drives it through fancy-regex: ordered alternation with
// ordinary backtracking, first alternative that matches at the position wins. That is what this
// reproduces. Returns the match length in CODEPOINTS, or -1.
//
// ORDER IS LOAD-BEARING and not obvious: alternatives 1 and 3 can both begin with a space, so
// " ." must be tried against 1 (fails, '.' is not a letter) before 3 (matches). Hoisting the
// whitespace cases up front — which reads as a tidy fast path — silently changes " word" into
// " " + "word" and shifts a large fraction of all ids.
inline int gpt4_match(const std::vector<uint32_t>& c, int i) {
    const int n = (int)c.size();

    // 0 — (?i:'s|'t|'re|'ve|'m|'ll|'d)
    if (c[i] == '\'' && i + 1 < n) {
        const uint32_t a = lower_ascii(c[i+1]);
        if (a == 's' || a == 't' || a == 'm' || a == 'd') return 2;
        if ((a == 'r' || a == 'v' || a == 'l') && i + 2 < n) {
            const uint32_t b = lower_ascii(c[i+2]);
            if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) return 3;
        }
    }
    // 1 — [^\r\n\p{L}\p{N}]?\p{L}+ ; the optional char is greedy, so try consumed then zero-width.
    for (int opt = 1; opt >= 0; --opt) {
        int j = i;
        if (opt) {
            if (is_nl(c[j]) || uc_is_L(c[j]) || uc_is_N(c[j])) continue;
            ++j;
        }
        int k = j;
        while (k < n && uc_is_L(c[k])) ++k;
        if (k > j) return k - i;
    }
    // 2 — \p{N}{1,3}
    if (uc_is_N(c[i])) {
        int j = i;
        while (j < n && j - i < 3 && uc_is_N(c[j])) ++j;
        return j - i;
    }
    // 3 —  ?[^\s\p{L}\p{N}]+[\r\n]*
    // The class excludes \s, so [\r\n]* can never steal from the greedy +; there is no backtrack
    // between them. Only the leading space is optional.
    for (int opt = 1; opt >= 0; --opt) {
        int j = i;
        if (opt) { if (c[j] != ' ') continue; ++j; }
        int k = j;
        while (k < n && !is_space(c[k]) && !uc_is_L(c[k]) && !uc_is_N(c[k])) ++k;
        if (k > j) {
            while (k < n && is_nl(c[k])) ++k;
            return k - i;
        }
    }
    // 4/5/6 — all three require whitespace at i, so one test gates them.
    if (is_space(c[i])) {
        int run = i;
        while (run < n && is_space(c[run])) ++run;       // one past the whitespace run
        // 4 — \s* is greedy, then backtracks to the last position where [\r\n]+ can start.
        for (int s = run; s >= i; --s) {
            if (s >= n || !is_nl(c[s])) continue;
            int k = s;
            while (k < n && is_nl(c[k])) ++k;
            return k - i;
        }
        // 5 — \s+ greedy, backtracked until the next character is not \S. At EOF the lookahead is
        // satisfied outright; otherwise c[run] is \S by construction, so give the last space back.
        const int k = run - i;                            // maximal \s+ length, >= 1
        if (run >= n) return k;
        if (k >= 2) return k - 1;
        // 6 — \s+
        return k;
    }
    return -1;
}

// Split `s` on every match, keeping gaps and matches (HF's SplitDelimiterBehavior::Isolated).
inline void split_isolated(const std::string& s, std::vector<std::string>& out) {
    std::vector<uint32_t> cp; std::vector<int> off;
    u8_decode(s, cp, off);
    const int n = (int)cp.size();
    int gap = 0;                                    // codepoint index where the current gap started
    for (int i = 0; i < n;) {
        const int L = gpt4_match(cp, i);
        if (L <= 0) { ++i; continue; }
        if (i > gap) out.push_back(s.substr(off[gap], off[i] - off[gap]));
        out.push_back(s.substr(off[i], off[i + L] - off[i]));
        i += L; gap = i;
    }
    if (gap < n) out.push_back(s.substr(off[gap]));
}

// ---- the tokenizer ------------------------------------------------------------------------------
struct Tokenizer {
    std::unordered_map<std::string, int> vocab;                 // ByteLevel-space token -> id
    std::vector<std::string> id2tok;
    std::vector<char> id_is_added;                              // added tokens decode verbatim
    std::vector<char> id_is_special;                            // subset of the above that skips
    std::unordered_map<uint64_t, std::pair<int,int>> merges;    // (a,b) -> (rank, merged)
    std::vector<std::pair<std::string,int>> added;              // (content,id), longest-first
    // Added tokens bucketed by first byte, so the literal scan at each offset checks only the
    // patterns that could start there rather than all 36.
    std::vector<std::vector<int>> added_by_first{256};
    ByteLevel bl;
    bool ignore_merges = false;

    // generation_config.json: three of them, and all three must stop a decode loop. A GGUF header
    // holds only one, which is precisely the trap recorded in `gguf-multi-eos-token-trap`.
    int eos_ids[3] = { 154820, 154827, 154829 };
    int pad_id = 154820;
    bool is_eos(int id) const { return id == eos_ids[0] || id == eos_ids[1] || id == eos_ids[2]; }

    static uint64_t pk(int a, int b) { return ((uint64_t)(uint32_t)a << 32) | (uint32_t)b; }

    void load(const std::string& path) {
        std::ifstream f(path);
        if (!f) throw std::runtime_error("tokenizer: cannot open " + path);
        nlohmann::json j; f >> j;

        auto& model = j["model"];
        ignore_merges = model.value("ignore_merges", false);

        auto& v = model["vocab"];
        int maxid = 0;
        for (auto it = v.begin(); it != v.end(); ++it) maxid = std::max(maxid, it.value().get<int>());
        if (j.contains("added_tokens"))
            for (auto& a : j["added_tokens"]) maxid = std::max(maxid, a["id"].get<int>());
        id2tok.assign(maxid + 1, std::string());
        id_is_added.assign(maxid + 1, 0);
        id_is_special.assign(maxid + 1, 0);
        for (auto it = v.begin(); it != v.end(); ++it) {
            const int id = it.value().get<int>();
            vocab[it.key()] = id;
            id2tok[id] = it.key();
        }
        int rank = 0;
        for (auto& m : model["merges"]) {
            std::string a, b;
            if (m.is_array()) { a = m[0].get<std::string>(); b = m[1].get<std::string>(); }
            else { const std::string s = m.get<std::string>(); const size_t sp = s.find(' '); a = s.substr(0, sp); b = s.substr(sp + 1); }
            auto ia = vocab.find(a), ib = vocab.find(b), ic = vocab.find(a + b);
            if (ia != vocab.end() && ib != vocab.end() && ic != vocab.end())
                merges[pk(ia->second, ib->second)] = { rank, ic->second };
            ++rank;
        }
        if (j.contains("added_tokens")) for (auto& a : j["added_tokens"]) {
            const std::string c = a["content"].get<std::string>();
            const int id = a["id"].get<int>();
            added.push_back({ c, id });
            id2tok[id] = c;
            id_is_added[id] = 1;
            id_is_special[id] = a.value("special", false) ? 1 : 0;
        }
        // Longest-first, so a literal match at a position takes the longest added token there.
        std::sort(added.begin(), added.end(),
                  [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });
        for (int i = 0; i < (int)added.size(); ++i)
            if (!added[i].first.empty())
                added_by_first[(unsigned char)added[i].first[0]].push_back(i);
    }

    int id_for(const std::string& content) const {          // added-token lookup by literal text
        for (auto& a : added) if (a.first == content) return a.second;
        auto it = vocab.find(content);
        return it == vocab.end() ? -1 : it->second;
    }

    // BPE over one pre-token, already in ByteLevel space.
    void bpe(const std::string& piece, std::vector<int>& out) const {
        // `ignore_merges`: a pre-token that is itself a vocab entry is emitted whole and the merge
        // loop never runs. Dropping this does not fail loudly — most words come out the same way
        // regardless — it just quietly re-routes a minority onto a different merge path.
        if (ignore_merges) {
            auto whole = vocab.find(piece);
            if (whole != vocab.end()) { out.push_back(whole->second); return; }
        }
        std::vector<uint32_t> cp; std::vector<int> off;
        u8_decode(piece, cp, off);
        std::vector<int> s; s.reserve(cp.size());
        for (size_t i = 0; i < cp.size(); ++i) {
            const std::string ch = piece.substr(off[i], off[i+1] - off[i]);
            auto it = vocab.find(ch);
            if (it != vocab.end()) s.push_back(it->second);   // ByteLevel guarantees this hits
        }
        while (s.size() >= 2) {
            int bestK = -1, bestRank = INT_MAX, bestMerged = -1;
            for (size_t k = 0; k + 1 < s.size(); ++k) {
                auto m = merges.find(pk(s[k], s[k+1]));
                if (m != merges.end() && m->second.first < bestRank) {
                    bestRank = m->second.first; bestK = (int)k; bestMerged = m->second.second;
                }
            }
            if (bestK < 0) break;
            s[bestK] = bestMerged;
            s.erase(s.begin() + bestK + 1);
        }
        out.insert(out.end(), s.begin(), s.end());
    }

    // One added-token-free text run: pre-tokenize, ByteLevel-map, BPE.
    void encode_run(const std::string& text, std::vector<int>& out) const {
        if (text.empty()) return;
        std::vector<std::string> pieces;
        split_isolated(text, pieces);
        for (auto& p : pieces) bpe(bl.encode(p), out);
    }

    // The post-processor adds no BOS and no EOS, so neither does this.
    std::vector<int> encode(const std::string& text) const {
        std::vector<int> out;
        size_t run = 0;                                      // start of the pending plain-text run
        for (size_t i = 0; i < text.size(); ++i) {
            const auto& cand = added_by_first[(unsigned char)text[i]];
            if (cand.empty()) continue;
            // `added` is sorted longest-first, so the first hit here is the longest.
            for (int ai : cand) {
                const std::string& s = added[ai].first;
                if (i + s.size() > text.size()) continue;
                if (text.compare(i, s.size(), s) != 0) continue;
                if (i > run) encode_run(text.substr(run, i - run), out);
                out.push_back(added[ai].second);
                i += s.size() - 1;                           // -1: the loop's ++i lands after it
                run = i + 1;
                break;
            }
        }
        if (run < text.size()) encode_run(text.substr(run), out);
        return out;
    }

    std::string decode(const std::vector<int>& ids, bool skip_special = true) const {
        std::string mapped;                                  // pending ByteLevel-space run
        std::string out;
        for (int id : ids) {
            if (id < 0 || id >= (int)id2tok.size()) continue;
            if (id_is_added[id]) {
                out += bl.decode(mapped); mapped.clear();    // flush before a verbatim token
                if (!(skip_special && id_is_special[id])) out += id2tok[id];
                continue;
            }
            mapped += id2tok[id];
        }
        out += bl.decode(mapped);
        return out;
    }

    // Streaming-safe: decode ids[from..] but return only the prefix that is complete UTF-8, so a
    // multi-byte character split across two tokens is never emitted as broken bytes.
    //
    // A SERVER must pass skip_special=false. `<think>`/`</think>` and the `<tool_call>` markers
    // are added tokens; dropping them deletes precisely what the stream splitter and the tool-call
    // parser exist to find, so reasoning arrives glued into content and tool calls arrive as
    // unparseable text. Nothing errors; it just stops working. The engine never delivers an EOS to
    // its callback, so keeping specials cannot leak a stop token into the output.
    std::string decode_stream(const std::vector<int>& ids, size_t from, size_t& consumed_bytes,
                              bool skip_special = true) const {
        std::vector<int> tail(ids.begin() + from, ids.end());
        const std::string s = decode(tail, skip_special);
        size_t end = s.size();
        while (end > 0) {                                    // back off any incomplete final char
            size_t st = end - 1;
            while (st > 0 && ((unsigned char)s[st] & 0xC0) == 0x80) --st;
            const int need = u8len((unsigned char)s[st]);
            if (st + need <= end) break;
            end = st;
        }
        consumed_bytes = end;
        return s.substr(0, end);
    }
};

} // namespace glm5tok
