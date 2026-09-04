// gate_sample.cpp — the sampler's boundary behaviour, without a GPU.
//
// Everything here is a property that fails silently in a running server: a nucleus off by one
// token, a greedy path that consumes randomness, a draw that is not reproducible from a seed.
#include "../include/sample.h"
#include <cstdio>
#include <map>

using namespace glm5;
static int pass = 0, fail = 0;
static void ck(bool ok, const char* what) {
    if (ok) { ++pass; printf("  ok   %s\n", what); }
    else    { ++fail; printf("  FAIL %s\n", what); }
}

int main() {
    printf("=== gate_sample ===\n");
    std::vector<std::pair<float,int>> scratch;

    // A distribution with known probabilities: exp(logit) proportional to 8:4:2:1:1 over 5 ids.
    std::vector<float> lg = { std::log(8.f), std::log(4.f), std::log(2.f), std::log(1.f), std::log(1.f) };
    const int N = (int)lg.size();

    { SampleParams p; p.temperature = 0.f;
      uint64_t s = 42;
      ck(sample(lg.data(), N, p, s, scratch) == 0, "temperature 0 is greedy");
      ck(s == 42, "greedy consumes no randomness"); }

    // top_p must INCLUDE the token that crosses the threshold. 8/16 = 0.5, so top_p = 0.5 keeps
    // exactly one token, and top_p = 0.51 keeps two.
    { SampleParams p; p.temperature = 1.f; p.top_p = 0.5f;
      std::map<int,int> seen;
      for (uint64_t i = 0; i < 400; ++i) { uint64_t s = i; seen[sample(lg.data(), N, p, s, scratch)]++; }
      ck(seen.size() == 1 && seen.count(0), "top_p=0.5 keeps exactly the 0.5 token"); }
    { SampleParams p; p.temperature = 1.f; p.top_p = 0.51f;
      std::map<int,int> seen;
      for (uint64_t i = 0; i < 400; ++i) { uint64_t s = i; seen[sample(lg.data(), N, p, s, scratch)]++; }
      ck(seen.size() == 2 && seen.count(0) && seen.count(1), "top_p just above keeps two"); }

    { SampleParams p; p.temperature = 1.f; p.top_p = 1.f; p.top_k = 2;
      std::map<int,int> seen;
      for (uint64_t i = 0; i < 400; ++i) { uint64_t s = i; seen[sample(lg.data(), N, p, s, scratch)]++; }
      ck(seen.size() == 2, "top_k=2 keeps two"); }

    // min_p is relative to the PEAK: 0.3 * 8 = 2.4, so only ids 0 and 1 (8 and 4) survive.
    { SampleParams p; p.temperature = 1.f; p.top_p = 1.f; p.min_p = 0.3f;
      std::map<int,int> seen;
      for (uint64_t i = 0; i < 400; ++i) { uint64_t s = i; seen[sample(lg.data(), N, p, s, scratch)]++; }
      ck(seen.size() == 2 && seen.count(0) && seen.count(1), "min_p cuts relative to the peak"); }

    // Reproducibility: the same seed must give the same SEQUENCE, not just the same first draw.
    { SampleParams p; p.temperature = 1.f; p.top_p = 1.f;
      uint64_t a = 7, b = 7;
      bool same = true;
      for (int i = 0; i < 64; ++i)
          if (sample(lg.data(), N, p, a, scratch) != sample(lg.data(), N, p, b, scratch)) same = false;
      ck(same, "a seed reproduces the whole sequence"); }

    // The empirical distribution must track the analytic one. 8:4:2:1:1 over 16 -> 0.5 for id 0.
    // A 20k-draw estimate has a standard error of ~0.0035, so 0.02 is a ~6-sigma band: wide enough
    // never to flake, tight enough to catch an off-by-one in the CDF walk.
    { SampleParams p; p.temperature = 1.f; p.top_p = 1.f;
      std::map<int,int> seen;
      uint64_t s = 12345;
      const int T = 20000;
      for (int i = 0; i < T; ++i) seen[sample(lg.data(), N, p, s, scratch)]++;
      const double want[5] = { 8/16., 4/16., 2/16., 1/16., 1/16. };
      bool ok = true;
      for (int i = 0; i < 5; ++i) {
          const double got = (double)seen[i] / T;
          if (std::fabs(got - want[i]) > 0.02) { ok = false;
              printf("    id %d: want %.4f got %.4f\n", i, want[i], got); }
      }
      ck(ok, "empirical distribution matches the analytic one"); }

    // Temperature must actually flatten. At T=100 the 8:1 ratio compresses to near-uniform.
    { SampleParams p; p.temperature = 100.f; p.top_p = 1.f;
      std::map<int,int> seen;
      uint64_t s = 999;
      for (int i = 0; i < 20000; ++i) seen[sample(lg.data(), N, p, s, scratch)]++;
      ck(std::fabs(seen[0] / 20000.0 - 0.2) < 0.02, "high temperature flattens toward uniform"); }

    // A degenerate distribution (one finite logit, the rest -inf) must not divide by zero.
    { std::vector<float> d(64, -INFINITY); d[17] = 1.0f;
      SampleParams p; p.temperature = 1.f; p.top_p = 0.95f;
      uint64_t s = 1;
      ck(sample(d.data(), 64, p, s, scratch) == 17, "a single finite logit is always drawn"); }

    printf("--- %d passed, %d failed ---\n", pass, fail);
    printf(fail ? "GATE FAILED\n" : "ALL SAMPLE GATES PASS\n");
    return fail ? 1 : 0;
}
