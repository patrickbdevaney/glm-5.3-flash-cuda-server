// gate_tokenizer.cpp — the C++ tokenizer against HF's own answer, id-for-id.
//
// Gate policy (CLAUDE.md §2): a tokenizer that is 99% right is worse than one that is obviously
// broken, because the model keeps producing fluent text from subtly wrong ids and nothing in the
// stack reports an error. So this is exact-match on every id of every case, not a similarity.
//
// Vectors: tools/gen_tokenizer_vectors.py -> ref/tokenizer_vectors.json.
#include "../include/tokenizer_glm5.h"
#include <cstdio>
#include <string>

using json = nlohmann::json;

static int pass = 0, fail = 0;

static std::string show(const std::string& s, size_t cap = 48) {
    std::string o;
    for (size_t i = 0; i < s.size() && o.size() < cap; ++i) {
        const unsigned char c = s[i];
        if (c == '\n') o += "\\n";
        else if (c == '\r') o += "\\r";
        else if (c == '\t') o += "\\t";
        else if (c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\x%02x", c); o += b; }
        else o.push_back((char)c);
    }
    if (o.size() >= cap) o += "...";
    return o;
}

static void check_ids(const std::string& text, const std::vector<int>& got,
                      const std::vector<int>& want) {
    if (got == want) { ++pass; return; }
    ++fail;
    printf("  FAIL encode %-50s\n", show(text).c_str());
    printf("    want[%zu]:", want.size());
    for (size_t i = 0; i < want.size() && i < 24; ++i) printf(" %d", want[i]);
    printf("\n    got [%zu]:", got.size());
    for (size_t i = 0; i < got.size() && i < 24; ++i) printf(" %d", got[i]);
    // The first divergence is the whole diagnosis: everything after it is downstream noise.
    size_t d = 0;
    while (d < want.size() && d < got.size() && want[d] == got[d]) ++d;
    printf("\n    first divergence at index %zu\n", d);
}

static void check_str(const char* what, const std::string& text,
                      const std::string& got, const std::string& want) {
    if (got == want) { ++pass; return; }
    ++fail;
    printf("  FAIL %s %-40s\n    want %s\n    got  %s\n",
           what, show(text).c_str(), show(want, 80).c_str(), show(got, 80).c_str());
}

int main(int argc, char** argv) {
    const std::string tokpath = argc > 1 ? argv[1]
        : std::string(getenv("HOME")) + "/glm-5.3-reap/source/GLM-5.3-Flash/tokenizer.json";
    const std::string vecpath = argc > 2 ? argv[2]
        : std::string(getenv("HOME")) + "/glm-5.3-flash-cuda-server/ref/tokenizer_vectors.json";

    printf("=== gate_tokenizer ===\n");
    glm5tok::Tokenizer tok;
    tok.load(tokpath);
    printf("vocab %zu  merges %zu  added %zu  ignore_merges %d\n",
           tok.vocab.size(), tok.merges.size(), tok.added.size(), (int)tok.ignore_merges);

    // Structural facts the rest of the server depends on. If any of these move, the checkpoint is
    // not the one this was written against and the id gate below would be meaningless.
    if (tok.vocab.size() != 154820) { printf("  FAIL vocab size %zu != 154820\n", tok.vocab.size()); ++fail; } else ++pass;
    if (tok.added.size() != 36)     { printf("  FAIL added %zu != 36\n", tok.added.size()); ++fail; } else ++pass;
    if (!tok.ignore_merges)         { printf("  FAIL ignore_merges is not set\n"); ++fail; } else ++pass;
    for (int e : { 154820, 154827, 154829 })
        if (!tok.is_eos(e)) { printf("  FAIL %d is not eos\n", e); ++fail; } else ++pass;
    // `</think>` is added but NOT special, so a server decoding with skip_special must keep it.
    const int think_end = tok.id_for("</think>");
    if (think_end != 154842 || tok.id_is_special[think_end]) {
        printf("  FAIL </think> id %d special %d (want 154842, not special)\n",
               think_end, think_end >= 0 ? (int)tok.id_is_special[think_end] : -1); ++fail;
    } else ++pass;

    std::ifstream f(vecpath);
    if (!f) { printf("  FAIL cannot open %s\n", vecpath.c_str()); return 1; }
    json v; f >> v;

    for (auto& c : v["cases"]) {
        const std::string text = c["text"].get<std::string>();
        check_ids(text, tok.encode(text), c["ids"].get<std::vector<int>>());
        const auto ids = c["ids"].get<std::vector<int>>();
        check_str("decode(keep)", text, tok.decode(ids, false), c["decoded_keep"].get<std::string>());
        check_str("decode(skip)", text, tok.decode(ids, true),  c["decoded_skip"].get<std::string>());
    }

    // decode_stream must never emit a partial UTF-8 character. Feed the ids of a multi-byte string
    // one at a time and assert every prefix returned is valid UTF-8 and that the concatenation is
    // the whole string. A server that gets this wrong shows mojibake only on non-ASCII output.
    {
        const std::string text = "你好，世界 \U0001f600 café";
        const auto ids = tok.encode(text);
        std::string acc;
        size_t from = 0;
        std::vector<int> fed;
        for (int id : ids) {
            fed.push_back(id);
            size_t used = 0;
            const std::string piece = tok.decode_stream(fed, from, used, false);
            // Every byte handed out must be part of a complete character.
            for (size_t i = 0; i < piece.size();) {
                const int L = glm5tok::u8len((unsigned char)piece[i]);
                if (i + L > piece.size()) { printf("  FAIL decode_stream emitted a partial char\n"); ++fail; break; }
                i += L;
            }
            acc += piece;
            if (used == tok.decode(std::vector<int>(fed.begin() + from, fed.end()), false).size()) {
                from = fed.size();                 // fully consumed: advance the window
            }
        }
        // Whatever is still pending at EOS is flushed by the caller the same way.
        acc += tok.decode(std::vector<int>(fed.begin() + from, fed.end()), false);
        check_str("decode_stream", text, acc, text);
    }

    printf("--- %d passed, %d failed ---\n", pass, fail);
    printf(fail ? "GATE FAILED\n" : "ALL TOKENIZER GATES PASS\n");
    return fail ? 1 : 0;
}
