// gate_vision_preproc.cpp — image bytes to patch rows, against transformers' image processor.
//
// Two checks, deliberately separate. `patchify` is exact integer arithmetic on a given pixel
// buffer and must match the oracle to the bit -- it is fed the ORACLE's own resized/normalized
// image so nothing else can contaminate it. The full path additionally does the canvas fit, the
// zero pad and (when the budget bites) an antialiased bicubic, which is implementation-defined;
// it is checked closely rather than exactly, and the two are reported apart so a failure says
// which half moved.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include "vision_preproc.h"
#include "vision.h"

using namespace glm5;
static int g_pass = 0, g_fail = 0;

static std::vector<float> readf(const std::string& p, size_t n) {
    FILE* f = fopen(p.c_str(), "rb");
    if (!f) { fprintf(stderr, "MISSING %s — run ref/gen_vision_preproc.py first\n", p.c_str()); exit(2); }
    std::vector<float> v(n);
    if (fread(v.data(), 4, n, f) != n) { fprintf(stderr, "SHORT %s\n", p.c_str()); exit(2); }
    fclose(f); return v;
}
static std::vector<int32_t> readi(const std::string& p, size_t n) {
    FILE* f = fopen(p.c_str(), "rb");
    if (!f) { fprintf(stderr, "MISSING %s\n", p.c_str()); exit(2); }
    std::vector<int32_t> v(n);
    if (fread(v.data(), 4, n, f) != n) { fprintf(stderr, "SHORT %s\n", p.c_str()); exit(2); }
    fclose(f); return v;
}
static void cmp(const char* name, const std::vector<float>& a, const std::vector<float>& b,
                double rel_max) {
    if (a.size() != b.size()) {
        printf("  %-22s FAIL  size %zu vs %zu\n", name, a.size(), b.size()); ++g_fail; return; }
    double d2 = 0, n2 = 0, mx = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double e = (double)a[i] - b[i]; d2 += e * e; n2 += (double)b[i] * b[i];
        if (fabs(e) > mx) mx = fabs(e);
    }
    const double rel = sqrt(d2) / (sqrt(n2) + 1e-30);
    const bool ok = rel <= rel_max;
    printf("  %-22s %s  relL2 %.3e  max_abs %.3e  (n=%zu)\n", name, ok ? "PASS" : "FAIL", rel, mx, a.size());
    ok ? ++g_pass : ++g_fail;
}
static void ck(bool c, const char* m) {
    printf("  %-22s %s\n", m, c ? "PASS" : "FAIL"); c ? ++g_pass : ++g_fail;
}

static void run_case(const std::string& R, const char* suf, const char* label) {
    printf("--- %s ---\n", label);
    auto hw   = readi(R + "test_image_hw" + suf + ".bin", 2);
    auto grid = readi(R + "preproc_grid" + suf + ".bin", 3);
    auto chw_hw = readi(R + "preproc_chw_hw" + suf + ".bin", 2);
    const int H = hw[0], W = hw[1], GH = grid[1], GW = grid[2];
    const int CH = chw_hw[0], CW = chw_hw[1];

    // 1. the canvas the processor chose
    int th = 0, tw = 0;
    vision_smart_resize(H, W, VIS_PATCH * VIS_MERGE, VIS_TPATCH, 16, 8000, &th, &tw);
    printf("  image %dx%d -> canvas %dx%d (oracle %dx%d), grid %dx%d\n", H, W, th, tw, CH, CW, GH, GW);
    ck(th == CH && tw == CW, "smart_resize canvas");

    // 2. patchify, on the ORACLE's own normalized pixels: exact arithmetic, so exact agreement
    auto chw = readf(R + "preproc_chw" + suf + ".bin", (size_t)VIS_CHAN * CH * CW);
    int gh = 0, gw = 0; std::vector<float> mine;
    vision_patchify(chw.data(), CH, CW, &gh, &gw, mine);
    ck(gh == GH && gw == GW, "patchify grid");
    auto want = readf(R + "preproc_patches" + suf + ".bin", (size_t)GH * GW * VIS_IN_DIM);
    // 1e-6, not 0: patchify itself is pure rearrangement, but the oracle's patches come from the
    // processor's own normalize while preproc_chw.bin comes from this script's, and those two
    // float32 op orders differ by one ulp (max_abs 4.77e-07 = 2^-21). That is rounding, not a
    // rearrangement error, and demanding bit-equality across two different float32 paths would be
    // a gate that can only fail.
    cmp("patchify", mine, want, 1e-6);

    // 3. the whole path from the decoded image
    std::vector<uint8_t> rgb;
    int dh = 0, dw = 0;
    if (!vision_load_image(R + "test_image" + suf + ".png", rgb, &dh, &dw)) {
        printf("  %-22s FAIL  could not decode png\n", "stb decode"); ++g_fail; return; }
    ck(dh == H && dw == W, "stb decode size");
    PreprocResult P = vision_preprocess_rgb(rgb.data(), dh, dw, 8000);
    ck(P.grid_h == GH && P.grid_w == GW, "full path grid");
    cmp("full path patches", P.patches, want, 1e-6);
}

int main() {
    const std::string R = std::string(getenv("HOME")) + "/glm-5.3-flash-cuda-server/ref/vision/";
    printf("=== gate_vision_preproc (vs Glm5NextImageProcessor) ===\n");
    run_case(R, "", "small image (pad path: content fits, zero-padded)");
    run_case(R, "_big", "large image (same path at scale)");
    printf("\n--- %d passed, %d failed ---\n", g_pass, g_fail);
    if (g_fail) { printf("GATE FAILED\n"); return 1; }
    printf("ALL VISION PREPROC GATES PASS\n");
    return 0;
}
