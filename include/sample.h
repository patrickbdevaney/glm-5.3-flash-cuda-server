// sample.h — logits -> one token id. Host-side, header-only, no CUDA.
//
// WHY THE HOST. A device sampler avoids a 620 KB D2H copy per token (154,880 fp32). At the AR wall
// of 12-25 tok/s that copy is ~0.03 ms against a ~50 ms step: 0.06% of the budget, for which a
// device top-p would buy a partial sort kernel that has to be gated and a source of
// non-determinism from atomics. It is the wrong trade at M=1. Revisit only if batching lands, when
// the copy scales with the batch and the step does not.
//
// Determinism is a requirement, not a nicety: without it, an accept/reject speculative decoder
// cannot be checked against its own AR path, which is the only test that proves speculation is
// lossless. So the RNG is an explicit seeded counter, never a global, and the candidate ordering
// breaks ties by id.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace glm5 {

struct SampleParams {
    float temperature = 1.0f;      // generation_config.json; 0 means greedy
    float top_p       = 0.95f;     // generation_config.json
    int   top_k       = 0;         // 0 = disabled
    float min_p       = 0.0f;      // 0 = disabled
    uint64_t seed     = 0;
};

// splitmix64: one multiply-xor chain, no state beyond the counter, identical on every platform.
inline uint64_t splitmix64(uint64_t& x) {
    uint64_t z = (x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
inline float next_unit(uint64_t& s) {
    return (float)((splitmix64(s) >> 40) * (1.0 / 16777216.0));   // 24 bits, [0,1)
}

inline int argmax(const float* logits, int n) {
    int best = 0;
    float bv = logits[0];
    for (int i = 1; i < n; ++i) if (logits[i] > bv) { bv = logits[i]; best = i; }
    return best;
}

// Full sampling pipeline: temperature -> top_k -> top_p -> min_p -> multinomial.
//
// The order matters and is the one every other server uses: top_k first (a cheap cardinality cut),
// then top_p over what survives, then min_p as an absolute floor relative to the peak. Applying
// top_p before top_k would make top_k a no-op whenever the nucleus is already smaller.
//
// `state` is advanced, so a caller stepping a sequence gets a different draw per token from one
// seed. Greedy (temperature <= 0) does not touch it — a greedy run must not consume randomness, or
// switching temperature would shift every later draw.
inline int sample(const float* logits, int n, const SampleParams& p, uint64_t& state,
                  std::vector<std::pair<float,int>>& scratch) {
    if (p.temperature <= 0.0f) return argmax(logits, n);

    // Softmax in fp32 with the standard max subtraction. exp() of a 154k-wide logit vector at
    // temperature 0.6 underflows to zero for all but a few hundred entries, which is fine — those
    // entries were never going to be drawn — but the max subtraction is what keeps the top of the
    // distribution from overflowing instead.
    float mx = logits[0];
    for (int i = 1; i < n; ++i) mx = std::max(mx, logits[i]);
    const float inv_t = 1.0f / p.temperature;

    scratch.clear();
    scratch.reserve(n);
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        const float e = std::exp((logits[i] - mx) * inv_t);
        if (e > 0.0f) { scratch.push_back({e, i}); sum += e; }
    }
    if (scratch.empty()) return argmax(logits, n);

    // Descending by probability, ties broken by ascending id so the order is total and stable.
    std::sort(scratch.begin(), scratch.end(),
              [](const std::pair<float,int>& a, const std::pair<float,int>& b) {
                  return a.first != b.first ? a.first > b.first : a.second < b.second;
              });

    size_t keep = scratch.size();
    if (p.top_k > 0) keep = std::min(keep, (size_t)p.top_k);

    if (p.top_p > 0.0f && p.top_p < 1.0f) {
        // The nucleus INCLUDES the token that crosses the threshold. Cutting before it makes
        // top_p=0.0 select nothing at all, and top_p slightly above the top token's probability
        // behave like greedy — both are off-by-one bugs that only show as a subtly flattened
        // distribution, never as an error.
        double acc = 0.0;
        size_t k = 0;
        const double target = (double)p.top_p * sum;
        while (k < keep) { acc += scratch[k].first; ++k; if (acc >= target) break; }
        keep = k;
    }

    if (p.min_p > 0.0f) {
        const float floor_p = p.min_p * scratch[0].first;
        size_t k = 0;
        while (k < keep && scratch[k].first >= floor_p) ++k;
        keep = std::max<size_t>(k, 1);
    }

    double kept = 0.0;
    for (size_t i = 0; i < keep; ++i) kept += scratch[i].first;
    const double r = (double)next_unit(state) * kept;
    double acc = 0.0;
    for (size_t i = 0; i < keep; ++i) {
        acc += scratch[i].first;
        if (r < acc) return scratch[i].second;
    }
    return scratch[keep - 1].second;              // fp rounding at the very top of the range
}

}  // namespace glm5
