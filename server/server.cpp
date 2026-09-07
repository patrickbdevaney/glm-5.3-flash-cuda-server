// server.cpp — OpenAI-compatible HTTP server for GLM-5.3-Flash-REAP50-NVFP4 on Jetson AGX Thor.
//
// One process, one binary, no Python anywhere on the request path: httplib and json.hpp are
// header-only and vendored, the chat encoder is include/encoding_glm5.h (byte-exact against the
// checkpoint's own Jinja over 19 fixtures), the tokenizer is include/tokenizer_glm5.h (id-exact
// against HF over 54 adversarial cases), and the model is src/engine.cu.
//
//   GET  /                      the web UI (self-contained, no CDN)
//   GET  /health                readiness + resident context length
//   GET  /metrics               cumulative counters, Prometheus text format
//   GET  /v1/models
//   POST /v1/chat/completions   streaming and non-streaming; tools; thinking blocks
//   POST /v1/completions        raw prompt in, text out
//
// CONCURRENCY IS DELIBERATELY ONE-AT-A-TIME. The engine owns a single recurrent KDA state and one
// MLA latent cache, and on this box that is the right shape: a 122 GiB unified-memory part running
// a 19.76 GB/token weight read has no headroom for a second context, and batching a decode that is
// bandwidth-bound at M=1 buys nothing. Requests serialise on `g_lock`; the queue depth is reported
// in /metrics rather than hidden.
#include "third_party/httplib.h"
#include "openai_api.h"
#include "encoding_glm5.h"
#include "vision_http.h"
#include "vision_preproc.h"
#include <memory>
#include <cuda_runtime.h>
#define CUDA_OK(x) do { cudaError_t e_=(x); if(e_){ \
    fprintf(stderr, "cuda %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
    res.status = 500; res.set_content("{\"error\":{\"message\":\"cuda failure\"}}", \
        "application/json"); return; } } while(0)
#include "tokenizer_glm5.h"
#include "stream_parse.h"
#include "webui.h"
#include "engine.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <functional>
#include <mutex>
#include <random>
#include <string>

using glm5enc::json;

static glm5tok::Tokenizer g_tok;
static glm5::Engine*      g_eng = nullptr;
static std::mutex         g_lock;
static std::string        g_model_name = "glm-5.3-flash-reap50-nvfp4";

static std::atomic<long> m_requests{0}, m_prompt_tok{0}, m_cached_tok{0}, m_completion_tok{0},
                         m_queued{0}, m_errors{0};
static std::atomic<long long> m_decode_us{0}, m_prefill_us{0};

static long now_s() { return (long)std::time(nullptr); }

static std::string rand_id() {
    static std::mt19937_64 rng{ (uint64_t)std::chrono::steady_clock::now().time_since_epoch().count() };
    static const char* hexd = "0123456789abcdef";
    std::string s(24, '0');
    for (auto& c : s) c = hexd[rng() & 15];
    return s;
}

// Stop strings are a text-level concept; the engine works in tokens. Rather than pretend otherwise,
// the check runs on the decoded text as it accumulates and truncates there.
static bool hit_stop(const std::string& text, const std::vector<std::string>& stops, size_t& cut) {
    for (const auto& s : stops) {
        if (s.empty()) continue;
        const size_t p = text.find(s);
        if (p != std::string::npos) { cut = p; return true; }
    }
    return false;
}

struct RunResult {
    std::string raw;                 // the full completion text
    glm5::GenStats stats;
    bool truncated = false;          // stopped on max_tokens rather than EOS
};

// OpenAI's logprobs shape. `bytes` is the raw UTF-8 of the piece, which clients use to re-align
// token boundaries with the text they were streamed.
static json logprobs_json(const std::vector<glm5::TokenLogprob>& lps) {
    json content = json::array();
    for (const auto& l : lps) {
        const std::string piece = g_tok.decode({l.id}, /*skip_special=*/false);
        json bytes = json::array();
        for (unsigned char c : piece) bytes.push_back((int)c);
        json e{{"token", piece}, {"logprob", l.logprob}, {"bytes", bytes}};
        json top = json::array();
        for (const auto& t : l.top) {
            const std::string tp = g_tok.decode({t.first}, false);
            json tb = json::array();
            for (unsigned char c : tp) tb.push_back((int)c);
            top.push_back(json{{"token", tp}, {"logprob", t.second}, {"bytes", tb}});
        }
        e["top_logprobs"] = top;
        content.push_back(e);
    }
    return json{{"content", content}};
}

// One generation, with the token->text->stop-string plumbing shared by both endpoints.
// `on_delta(reasoning, content)` is called as text becomes final; return false to stop.
static RunResult run_generation(const std::vector<int>& ids, const glm5::GenParams& gp,
                                bool thinking, const std::vector<std::string>& stops,
                                const std::function<bool(const std::string&, const std::string&)>& on_delta) {
    RunResult out;
    glm5srv::StreamSplitter sp(thinking);
    std::vector<int> gen;
    size_t emitted = 0;
    bool stopped = false;

    out.stats = g_eng->generate(ids, gp, [&](int tok) -> bool {
        gen.push_back(tok);
        size_t nb = 0;
        // skip_special=FALSE on purpose: `</think>` and `<tool_call>` are added tokens, and the
        // splitter downstream is looking for exactly those.
        const std::string all = g_tok.decode_stream(gen, 0, nb, /*skip_special=*/false);
        if (nb <= emitted) return true;                 // incomplete UTF-8 so far; wait for more
        const std::string chunk = all.substr(emitted);
        emitted = nb;

        std::string r, c;
        sp.feed(chunk, r, c);

        size_t cut = 0;
        if (!stops.empty() && hit_stop(sp.raw, stops, cut)) stopped = true;
        if ((!r.empty() || !c.empty()) && on_delta && !on_delta(r, c)) stopped = true;
        return !stopped;
    });

    { std::string r, c;
      sp.finish(r, c);
      if ((!r.empty() || !c.empty()) && on_delta) on_delta(r, c); }

    out.raw = sp.raw;
    out.truncated = !out.stats.hit_eos && !stopped;
    size_t cut = 0;
    if (!stops.empty() && hit_stop(out.raw, stops, cut)) out.raw = out.raw.substr(0, cut);
    return out;
}

static void account(const glm5::GenStats& s) {
    m_prompt_tok += s.prompt_tokens;
    m_cached_tok += s.cached_tokens;
    m_completion_tok += s.completion_tokens;
    m_decode_us += (long long)(s.decode_ms * 1000);
    m_prefill_us += (long long)(s.prefill_ms * 1000);
}

// Dump a response that carries MODEL-GENERATED TEXT.
//
// nlohmann's default dump() THROWS type_error.316 on invalid UTF-8, and this model can produce it:
// the tokenizer is byte-level BPE, so a generation can contain byte sequences that are not valid
// UTF-8 on their own. On the sibling 0731 server that surfaced as a bare 500 with an EMPTY BODY
// and destroyed two whole benchmarks before the handler was made to catch and report it.
//
// `error_handler_t::replace` substitutes U+FFFD instead of throwing. A response is not the place to
// be strict: the alternative to a replacement character is no answer at all. Valid output is
// byte-for-byte unaffected, since the handler only fires on bytes that could not have been
// serialised anyway.
static inline std::string dump_lossy(const json& j) {
    return j.dump(-1, ' ', false, json::error_handler_t::replace);
}

static json timings_json(const glm5::GenStats& s) {
    return json{{"prefill_ms", s.prefill_ms},
                {"decode_ms", s.decode_ms},
                {"tokens_per_second", s.tok_per_s},
                {"cached_prompt_tokens", s.cached_tokens}};
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::string host = "0.0.0.0";
    int port = 8080;
    glm5::EngineConfig ec;
    ec.model_dir = std::string(getenv("HOME")) + "/glm-5.3-reap/output/glm-5.3-flash-reap50-nvfp4-pass2";
    std::string tokdir;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--ckpt")         ec.model_dir = next();
        else if (a == "--tokenizer") tokdir = next();
        else if (a == "--host")    host = next();
        else if (a == "--port")    port = atoi(next().c_str());
        else if (a == "--seqmax" || a == "--ctx") ec.max_ctx = atoi(next().c_str());
        // Loading fewer layers is NOT a quality knob — it is a plumbing smoke test. A 3-layer load
        // fits in ~6 GiB and exercises tokenizer -> prompt -> engine -> sampler -> SSE end to end
        // on a box that cannot currently hold the 98 GiB checkpoint. The text is meaningless.
        else if (a == "--n-layer") ec.n_layer = atoi(next().c_str());
        else if (a == "--model")   g_model_name = next();
        else if (a == "--help") {
            printf("usage: %s [--ckpt DIR] [--tokenizer DIR] [--host H] [--port P]\n"
                   "       [--ctx N | --seqmax N] [--n-layer N] [--model NAME]\n"
                   "  --ctx    KV/context length; sizes the MLA latent cache (11 layers x N x 512 x 4 B)\n"
                   "           and the indexer state. The 34 KDA layers cost a FIXED 145.56 MiB\n"
                   "           regardless of N, so context is cheaper here than in a dense model.\n", argv[0]);
            return 0;
        }
    }
    if (tokdir.empty()) tokdir = ec.model_dir;

    printf("[server] tokenizer %s/tokenizer.json\n", tokdir.c_str());
    g_tok.load(tokdir + "/tokenizer.json");
    { // The same structural gate tests/gate_tokenizer.cpp runs. If this ever fails, every prompt is
      // silently wrong — which is the one failure mode a decode server cannot detect on its own.
        if (g_tok.vocab.size() != 154820 || !g_tok.ignore_merges || g_tok.added.size() != 36) {
            fprintf(stderr, "[server] FATAL: tokenizer gate failed (vocab %zu, added %zu, ignore_merges %d)\n",
                    g_tok.vocab.size(), g_tok.added.size(), (int)g_tok.ignore_merges);
            return 1;
        }
    }
    printf("[server] tokenizer ok (vocab %zu, %zu added, eos %d/%d/%d)\n",
           g_tok.vocab.size(), g_tok.added.size(),
           g_tok.eos_ids[0], g_tok.eos_ids[1], g_tok.eos_ids[2]);

    glm5::Engine eng(ec);
    g_eng = &eng;

    httplib::Server srv;
    srv.new_task_queue = [] { return new httplib::ThreadPool(4); };

    auto cors = [](httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    };
    srv.Options(".*", [&](const httplib::Request&, httplib::Response& res) { cors(res); });

    srv.Get("/", [&](const httplib::Request&, httplib::Response& res) {
        res.set_content(WEBUI_HTML, "text/html; charset=utf-8");
    });

    srv.Get("/health", [&](const httplib::Request&, httplib::Response& res) {
        cors(res);
        json o;
        o["status"] = "ok";
        o["model"] = g_model_name;
        o["n_layer"] = g_eng->nLayer();
        o["max_ctx"] = g_eng->maxCtx();
        o["resident_gib"] = g_eng->residentGiB();
        o["resident_seq_len"] = g_eng->seqLen();
        o["busy"] = m_queued.load() > 0;
        res.set_content(o.dump(), "application/json");
    });

    srv.Get("/metrics", [&](const httplib::Request&, httplib::Response& res) {
        cors(res);
        char buf[2048];
        const double dsec = m_decode_us.load() / 1e6;
        snprintf(buf, sizeof buf,
            "# HELP glm5_requests_total Completed requests.\n# TYPE glm5_requests_total counter\n"
            "glm5_requests_total %ld\n"
            "glm5_errors_total %ld\n"
            "glm5_prompt_tokens_total %ld\n"
            "glm5_cached_prompt_tokens_total %ld\n"
            "glm5_completion_tokens_total %ld\n"
            "glm5_decode_seconds_total %.3f\n"
            "glm5_prefill_seconds_total %.3f\n"
            "glm5_decode_tokens_per_second %.3f\n"
            "glm5_prefix_cache_hit_ratio %.4f\n"
            "glm5_queue_depth %ld\n",
            m_requests.load(), m_errors.load(), m_prompt_tok.load(), m_cached_tok.load(),
            m_completion_tok.load(), dsec, m_prefill_us.load() / 1e6,
            dsec > 0 ? m_completion_tok.load() / dsec : 0.0,
            m_prompt_tok.load() > 0 ? (double)m_cached_tok.load() / m_prompt_tok.load() : 0.0,
            m_queued.load());
        res.set_content(buf, "text/plain; version=0.0.4");
    });

    // ---- tokenizer introspection -------------------------------------------------------------
    // Clients that budget context need to count tokens without guessing, and a wrong count is
    // invisible until a prompt silently overruns. The same encoder the chat path uses.
    srv.Post("/tokenize", [&](const httplib::Request& req, httplib::Response& res) {
        cors(res);
        try {
            const json body = json::parse(req.body);
            const std::string content = body.value("content", std::string());
            const std::vector<int> ids = g_tok.encode(content);
            json out{{"tokens", ids}, {"count", (int)ids.size()}};
            if (body.value("with_pieces", false)) {
                json pieces = json::array();
                for (int t : ids) pieces.push_back(json{{"id", t}, {"piece", g_tok.decode({t})}});
                out["pieces"] = pieces;
            }
            res.set_content(out.dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(json{{"error", {{"message", e.what()}}}}.dump(), "application/json");
        }
    });

    srv.Post("/detokenize", [&](const httplib::Request& req, httplib::Response& res) {
        cors(res);
        try {
            const json body = json::parse(req.body);
            std::vector<int> ids;
            if (body.contains("tokens") && body["tokens"].is_array())
                for (const auto& t : body["tokens"]) ids.push_back(t.get<int>());
            res.set_content(json{{"content", g_tok.decode(ids)}}.dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(json{{"error", {{"message", e.what()}}}}.dump(), "application/json");
        }
    });

    // ---- server properties ---------------------------------------------------------------------
    srv.Get("/props", [&](const httplib::Request&, httplib::Response& res) {
        cors(res);
        res.set_content(json{
            {"model", g_model_name},
            {"n_ctx", eng.maxCtx()},
            {"n_layer", ec.n_layer},
            {"vocab_size", (int)glm5::VOCAB},
            {"hidden_size", (int)glm5::HIDDEN},
            {"resident_gib", eng.residentGiB()},
            {"vision", eng.hasVision()},
            {"eos_ids", json::array({g_tok.eos_ids[0], g_tok.eos_ids[1], g_tok.eos_ids[2]})},
            {"chat_template", "glm5-next (built in, include/encoding_glm5.h)"},
            {"capabilities", json::array({"chat", "completions", "embeddings", "tools",
                                          "reasoning", "streaming",
                                          eng.hasVision() ? "vision" : "text-only"})}
        }.dump(), "application/json");
    });

    // ---- embeddings ------------------------------------------------------------------------------
    // The LAST token's pooled hidden state, L2-normalized. For a causal decoder that is the
    // standard choice: every earlier position has only seen a prefix, so mean-pooling them mixes
    // representations of different amounts of the input. Said plainly because "embeddings" without
    // a pooling rule is ambiguous and the rule changes the numbers.
    srv.Post("/v1/embeddings", [&](const httplib::Request& req, httplib::Response& res) {
        cors(res);
        std::vector<std::string> inputs;
        try {
            const json body = json::parse(req.body);
            const auto& in = body.at("input");
            if (in.is_string()) inputs.push_back(in.get<std::string>());
            else if (in.is_array()) for (const auto& x : in)
                if (x.is_string()) inputs.push_back(x.get<std::string>());
            if (inputs.empty()) throw std::runtime_error("input must be a string or array of strings");
        } catch (const std::exception& e) {
            ++m_errors; res.status = 400;
            res.set_content(json{{"error", {{"message", e.what()},
                {"type", "invalid_request_error"}}}}.dump(), "application/json");
            return;
        }
        std::lock_guard<std::mutex> lk(g_lock);
        json data = json::array();
        long total = 0;
        std::vector<float> host(glm5::HIDDEN);
        for (size_t i = 0; i < inputs.size(); ++i) {
            std::vector<int> ids = g_tok.encode(inputs[i]);
            if (ids.empty()) ids.push_back(g_tok.eos_ids[0]);
            if ((int)ids.size() > eng.maxCtx()) ids.resize(eng.maxCtx());
            total += (long)ids.size();
            eng.reset();
            eng.prefill(ids, nullptr);
            if (cudaMemcpy(host.data(), eng.pooledDev(), host.size() * 4,
                           cudaMemcpyDeviceToHost) != cudaSuccess) {
                ++m_errors; res.status = 500;
                res.set_content(json{{"error", {{"message", "embedding readback failed"}}}}.dump(),
                                "application/json");
                return;
            }
            double n2 = 0; for (float v : host) n2 += (double)v * v;
            const double inv = n2 > 0 ? 1.0 / sqrt(n2) : 1.0;
            json vec = json::array();
            for (float v : host) vec.push_back((double)v * inv);
            data.push_back(json{{"object", "embedding"}, {"index", (int)i}, {"embedding", vec}});
        }
        eng.reset();   // the embedding pass trampled the resident prefix; do not let a later
                       // chat request believe it can reuse it
        res.set_content(json{{"object", "list"}, {"data", data}, {"model", g_model_name},
                             {"usage", {{"prompt_tokens", total}, {"total_tokens", total}}}}.dump(),
                        "application/json");
    });

    srv.Get("/v1/models", [&](const httplib::Request&, httplib::Response& res) {
        cors(res);
        res.set_content(glm5api::models_response(g_model_name, now_s()).dump(), "application/json");
    });

    // ---- POST /v1/chat/completions ----------------------------------------------------------
    srv.Post("/v1/chat/completions", [&](const httplib::Request& req, httplib::Response& res) {
        cors(res);
        json body;
        try { body = json::parse(req.body); }
        catch (const std::exception& e) {
            ++m_errors;
            res.status = 400;
            res.set_content(json{{"error", {{"message", std::string("invalid JSON: ") + e.what()},
                                            {"type", "invalid_request_error"}}}}.dump(), "application/json");
            return;
        }

        glm5api::ChatRequest cr;
        std::string prompt;
        try {
            cr = glm5api::parse_chat_request(body);
            prompt = glm5api::build_prompt(cr);
        } catch (const std::exception& e) {
            ++m_errors;
            res.status = 400;
            res.set_content(json{{"error", {{"message", e.what()}, {"type", "invalid_request_error"}}}}.dump(),
                            "application/json");
            return;
        }

        std::vector<int> ids = g_tok.encode(prompt);

        // ---- images ----------------------------------------------------------------------
        // The chat encoder emits one <|image|> per image part, in message order. Each becomes the
        // N embeddings the tower produces for that image, and the engine splices them at those
        // absolute positions. Vision work happens under the same lock as generation because it
        // uses the engine's single vision workspace.
        std::vector<std::string> img_urls;
        try { img_urls = glm5api::collect_image_urls(body); }
        catch (...) { img_urls.clear(); }
        std::vector<int> img_starts, img_counts;
        // SHARED OWNERSHIP, not a scope guard. The SSE path hands generation to a chunked content
        // provider that captures by value and runs AFTER this handler returns, so a guard scoped
        // here would free the embeddings out from under the stream. Whichever of the two outlives
        // the other drops the last reference and the deleter runs exactly once.
        auto img_devs_p = std::shared_ptr<std::vector<const float*>>(
            new std::vector<const float*>(),
            [](std::vector<const float*>* v) {
                g_eng->clear_image_embeds();
                for (auto* d : *v) cudaFree((void*)d);
                delete v;
            });
        std::vector<const float*>& img_devs = *img_devs_p;
        std::unique_lock<std::mutex> vis_lk(g_lock, std::defer_lock);
        if (!img_urls.empty()) {
            if (!g_eng->hasVision()) {
                ++m_errors; res.status = 400;
                res.set_content(json{{"error", {{"message",
                    "this engine has no vision tower resident (reduced --n-layer?)"},
                    {"type", "invalid_request_error"}}}}.dump(), "application/json");
                return;
            }
            vis_lk.lock();
            for (const auto& u : img_urls) {
                std::vector<uint8_t> bytes; std::string err;
                if (!glm5::image_bytes_from_url(u, bytes, err)) {
                    ++m_errors; res.status = 400;
                    res.set_content(json{{"error", {{"message", err},
                        {"type", "invalid_request_error"}}}}.dump(), "application/json");
                    return;
                }
                std::vector<uint8_t> rgb; int ih = 0, iw = 0;
                if (!glm5::vision_decode_image(bytes.data(), bytes.size(), rgb, &ih, &iw)) {
                    ++m_errors; res.status = 400;
                    res.set_content(json{{"error", {{"message", "could not decode image"},
                        {"type", "invalid_request_error"}}}}.dump(), "application/json");
                    return;
                }
                const float* dev = nullptr;
                const int n = g_eng->encodeImage(rgb.data(), ih, iw, &dev);
                if (n <= 0) {
                    ++m_errors; res.status = 400;
                    res.set_content(json{{"error", {{"message", "vision encode failed"},
                        {"type", "invalid_request_error"}}}}.dump(), "application/json");
                    return;
                }
                // encodeImage reuses one output buffer, so each image's rows must be taken away
                // before the next call overwrites them.
                float* keep = nullptr;
                CUDA_OK(cudaMalloc(&keep, (size_t)n * glm5::HIDDEN * 4));
                CUDA_OK(cudaMemcpy(keep, dev, (size_t)n * glm5::HIDDEN * 4, cudaMemcpyDeviceToDevice));
                img_devs.push_back(keep);
                img_counts.push_back(n);
            }
            // RELEASE BEFORE GENERATION. g_lock is not recursive and both the blocking and the
            // streaming paths take it again to generate; holding it here deadlocked the request
            // while /metrics kept answering, which made it look alive. The encoded rows were
            // already copied out of the engine's reusable buffer above, so nothing needs the lock
            // past this point.
            vis_lk.unlock();
            if (!glm5::expand_image_tokens(ids, img_counts, img_starts)) {
                ++m_errors; res.status = 400;
                res.set_content(json{{"error", {{"message",
                    "image parts and <|image|> tokens disagree"},
                    {"type", "invalid_request_error"}}}}.dump(), "application/json");
                return;
            }
        }
        const std::string id = rand_id();
        const long created = now_s();

        glm5::GenParams gp;
        // OpenAI chat uses a boolean `logprobs` plus an optional `top_logprobs`; /v1/completions
        // uses an integer. Both land on the same engine fields.
        std::vector<glm5::TokenLogprob> lps;
        bool want_lp = false;
        try {
            const json b2 = json::parse(req.body);
            want_lp = b2.value("logprobs", false);
            if (want_lp) { gp.out_logprobs = &lps; gp.n_logprobs = b2.value("top_logprobs", 0); }
        } catch (...) {}
        gp.sampling.temperature = (float)cr.sampling.temperature;
        gp.sampling.top_p = (float)cr.sampling.top_p;
        gp.sampling.top_k = cr.sampling.top_k;
        gp.sampling.min_p = (float)cr.sampling.min_p;
        gp.max_tokens = cr.sampling.max_tokens;
        gp.seed = cr.sampling.seed;
        gp.has_seed = cr.sampling.has_seed;
        gp.eos_ids.assign(g_tok.eos_ids, g_tok.eos_ids + 3);

        if ((int)ids.size() + gp.max_tokens + 8 > g_eng->maxCtx()) {
            ++m_errors;
            res.status = 400;
            char m[256];
            snprintf(m, sizeof m, "prompt (%zu tokens) + max_tokens (%d) exceeds context %d",
                     ids.size(), gp.max_tokens, g_eng->maxCtx());
            res.set_content(json{{"error", {{"message", m}, {"type", "context_length_exceeded"}}}}.dump(),
                            "application/json");
            return;
        }

        if (!cr.stream) {
            std::lock_guard<std::mutex> lk(g_lock);
            ++m_requests;
            // EVERYTHING inside the lock is wrapped. On the sibling 0731 server this path had no
            // try/catch while the streaming one did, so any throw escaped the handler and httplib
            // turned it into a bare 500 with an EMPTY BODY — no message, no log line, nothing to
            // debug from. It took three sessions and a lost 198-item benchmark to find. Catch it,
            // say what it was, and log it.
            try {
                for (size_t k = 0; k < img_starts.size(); ++k)
                    g_eng->set_image_embeds(img_starts[k], img_counts[k], img_devs[k]);
                const RunResult r = run_generation(ids, gp, /*thinking=*/true, cr.sampling.stop, nullptr);
                account(r.stats);
                // The prompt ends with <think>, so the model's output starts INSIDE the reasoning
                // block and the opening tag was never generated — the parser has to be told, or a
                // generation truncated before `</think>` lands in `content` as raw scratchpad.
                const json parsed = glm5enc::parse_message_from_completion_text(r.raw, true);
                json out = glm5api::chat_completion_response(id, cr.model, parsed,
                        r.stats.prompt_tokens, r.stats.completion_tokens, created,
                        r.truncated ? "length" : nullptr);
                out["usage"]["prompt_tokens_details"] = json{{"cached_tokens", r.stats.cached_tokens}};
                if (want_lp) out["choices"][0]["logprobs"] = logprobs_json(lps);
                out["timings"] = timings_json(r.stats);
                res.set_content(dump_lossy(out), "application/json");
            } catch (const std::exception& e) {
                ++m_errors;
                fprintf(stderr, "[server] generation failed (%zu prompt tokens, max_tokens %d): %s\n",
                        ids.size(), gp.max_tokens, e.what());
                fflush(stderr);
                res.status = 500;
                res.set_content(json{{"error", {{"message", e.what()},
                                                {"type", "generation_error"}}}}.dump(), "application/json");
            }
            return;
        }

        // ---- streaming (SSE) ----
        res.set_header("Cache-Control", "no-cache");
        res.set_header("Connection", "keep-alive");
        res.set_header("X-Accel-Buffering", "no");
        res.set_chunked_content_provider("text/event-stream",
            [id, created, ids, gp, cr, img_starts, img_counts, img_devs_p](size_t, httplib::DataSink& sink) -> bool {
                std::lock_guard<std::mutex> lk(g_lock);
                ++m_requests;
                bool alive = true;
                auto send = [&](const std::string& s) {
                    if (!alive) return;
                    if (!sink.write(s.data(), s.size())) alive = false;   // client hung up
                };
                // A first chunk carrying only the role is what OpenAI clients expect, and it also
                // flushes headers so a slow first token does not look like a dead connection.
                { json d{{"role", "assistant"}};
                  json ch{{"index", 0}, {"delta", d}, {"finish_reason", nullptr}};
                  json o{{"id", "chatcmpl-" + id}, {"object", "chat.completion.chunk"},
                         {"created", created}, {"model", cr.model}, {"choices", json::array({ch})}};
                  send("data: " + dump_lossy(o) + "\n\n"); }

                RunResult r;
                try {
                    for (size_t k = 0; k < img_starts.size(); ++k)
                        g_eng->set_image_embeds(img_starts[k], img_counts[k], (*img_devs_p)[k]);
                    r = run_generation(ids, gp, /*thinking=*/true, cr.sampling.stop,
                        [&](const std::string& rr, const std::string& cc) -> bool {
                            if (!rr.empty() || !cc.empty())
                                send(glm5api::sse_chunk(id, cr.model, created, cc, rr, nullptr));
                            return alive;
                        });
                } catch (const std::exception& e) {
                    ++m_errors;
                    fprintf(stderr, "[server] stream failed: %s\n", e.what()); fflush(stderr);
                    send(std::string("data: ") + json{{"error", {{"message", e.what()}}}}.dump() + "\n\n");
                    sink.done();
                    return true;
                }
                account(r.stats);

                // Tool calls arrive as one final delta. GLM emits a SEQUENCE of <tool_call> blocks
                // with no outer wrapper, so "the calls are complete" is only known at end of
                // generation — there is no closing marker to stream against.
                const json parsed = glm5enc::parse_message_from_completion_text(r.raw, true);
                const bool has_tc = parsed.contains("tool_calls") && !parsed["tool_calls"].empty();
                if (has_tc) {
                    json tcs = json::array();
                    int i = 0;
                    for (const auto& tc : parsed["tool_calls"]) {
                        json o = tc;
                        o["index"] = i;
                        o["id"] = "call_" + id.substr(0, 8) + "_" + std::to_string(i);
                        ++i;
                        tcs.push_back(o);
                    }
                    json ch{{"index", 0}, {"delta", json{{"tool_calls", tcs}}}, {"finish_reason", nullptr}};
                    json o{{"id", "chatcmpl-" + id}, {"object", "chat.completion.chunk"},
                           {"created", created}, {"model", cr.model}, {"choices", json::array({ch})}};
                    send("data: " + dump_lossy(o) + "\n\n");
                }
                send(glm5api::sse_chunk(id, cr.model, created, "", "",
                                        has_tc ? "tool_calls" : (r.truncated ? "length" : "stop")));

                if (cr.include_usage) {
                    json o{{"id", "chatcmpl-" + id}, {"object", "chat.completion.chunk"},
                           {"created", created}, {"model", cr.model}, {"choices", json::array()},
                           {"usage", json{{"prompt_tokens", r.stats.prompt_tokens},
                                          {"completion_tokens", r.stats.completion_tokens},
                                          {"total_tokens", r.stats.prompt_tokens + r.stats.completion_tokens},
                                          {"prompt_tokens_details", json{{"cached_tokens", r.stats.cached_tokens}}}}},
                           {"timings", timings_json(r.stats)}};
                    send("data: " + dump_lossy(o) + "\n\n");
                }
                send("data: [DONE]\n\n");
                sink.done();
                return true;
            });
    });

    // ---- POST /v1/completions ---------------------------------------------------------------
    srv.Post("/v1/completions", [&](const httplib::Request& req, httplib::Response& res) {
        cors(res);
        json body;
        try { body = json::parse(req.body); }
        catch (const std::exception& e) {
            ++m_errors; res.status = 400;
            res.set_content(json{{"error", {{"message", std::string("invalid JSON: ") + e.what()}}}}.dump(),
                            "application/json");
            return;
        }
        // Wrapped for the same reason the chat path is — see above. This is also the endpoint any
        // long unattended eval drives, so an unexplained 500 here costs a whole run, not a turn.
        try {
            const std::string prompt = body.value("prompt", "");
            glm5::GenParams gp;
            gp.sampling.temperature = body.contains("temperature") && body["temperature"].is_number()
                                    ? (float)body["temperature"].get<double>() : 1.0f;
            gp.sampling.top_p = body.contains("top_p") && body["top_p"].is_number()
                              ? (float)body["top_p"].get<double>() : 0.95f;
            gp.max_tokens = body.contains("max_tokens") && body["max_tokens"].is_number()
                          ? body["max_tokens"].get<int>() : 128;
            if (body.contains("seed") && body["seed"].is_number()) {
                gp.seed = body["seed"].get<uint64_t>(); gp.has_seed = true; }
            gp.eos_ids.assign(g_tok.eos_ids, g_tok.eos_ids + 3);
            std::vector<std::string> stops;
            if (body.contains("stop")) {
                if (body["stop"].is_string()) stops.push_back(body["stop"].get<std::string>());
                else if (body["stop"].is_array()) for (auto& x : body["stop"]) if (x.is_string()) stops.push_back(x.get<std::string>());
            }
            // A raw completion is a raw completion: the prompt as given, no chat template, and no
            // BOS — this tokenizer's post-processor adds none, and inventing one here would put
            // every /v1/completions request off-distribution relative to /v1/chat/completions.
            std::vector<glm5::TokenLogprob> lps;
            {   // /v1/completions takes an INTEGER logprobs, unlike chat's boolean pair.
                const int nlp = body.value("logprobs", 0);
                if (nlp > 0) { gp.out_logprobs = &lps; gp.n_logprobs = nlp; }
            }
            std::vector<int> ids = g_tok.encode(prompt);
            if (ids.empty()) { res.status = 400;
                res.set_content(json{{"error", {{"message", "empty prompt"}}}}.dump(), "application/json"); return; }

            std::lock_guard<std::mutex> lk(g_lock);
            ++m_requests;
            const RunResult r = run_generation(ids, gp, /*thinking=*/false, stops, nullptr);
            account(r.stats);
            json choice{{"index", 0}, {"text", r.raw},
                        {"finish_reason", r.truncated ? "length" : "stop"}, {"logprobs", nullptr}};
            if (gp.out_logprobs) choice["logprobs"] = logprobs_json(lps);
            json out{{"id", "cmpl-" + rand_id()}, {"object", "text_completion"}, {"created", now_s()},
                     {"model", g_model_name}, {"choices", json::array({choice})},
                     {"usage", json{{"prompt_tokens", r.stats.prompt_tokens},
                                    {"completion_tokens", r.stats.completion_tokens},
                                    {"total_tokens", r.stats.prompt_tokens + r.stats.completion_tokens}}}};
            // Timings on this path are not cosmetic: it is the endpoint an unattended eval drives,
            // and on the sibling server the omission meant every long-context continuation ever run
            // recorded nothing — the exact regime the decode model most needed to be fitted against.
            out["timings"] = timings_json(r.stats);
            res.set_content(dump_lossy(out), "application/json");
        } catch (const std::exception& e) {
            ++m_errors;
            res.status = 500;
            res.set_content(json{{"error", {{"message", std::string("completion failed: ") + e.what()},
                                            {"type", "server_error"}}}}.dump(), "application/json");
        } catch (...) {
            ++m_errors;
            res.status = 500;
            res.set_content(json{{"error", {{"message", "completion failed: unknown exception"},
                                            {"type", "server_error"}}}}.dump(), "application/json");
        }
    });

    srv.set_pre_routing_handler([](const httplib::Request&, httplib::Response&) {
        ++m_queued;
        return httplib::Server::HandlerResponse::Unhandled;
    });
    srv.set_post_routing_handler([](const httplib::Request&, httplib::Response&) { --m_queued; });

    printf("\n[server] listening on http://%s:%d   (web UI at /)\n", host.c_str(), port);
    printf("[server] model %s, %d layers, %.2f GiB resident, max_ctx %d\n",
           g_model_name.c_str(), g_eng->nLayer(), g_eng->residentGiB(), g_eng->maxCtx());
    if (g_eng->nLayer() < glm5::N_LAYER)
        printf("[server] *** %d of %d layers loaded: this is a PLUMBING SMOKE TEST, the text is meaningless ***\n",
               g_eng->nLayer(), glm5::N_LAYER);
    if (!srv.listen(host.c_str(), port)) {
        fprintf(stderr, "[server] FATAL: cannot bind %s:%d\n", host.c_str(), port);
        return 1;
    }
    return 0;
}
