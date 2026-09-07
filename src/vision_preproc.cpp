#include "vision_preproc.h"
#include "vision.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#define STBI_ONLY_BMP
#include "third_party/stb_image.h"

namespace glm5 {

// OPENAI_CLIP_MEAN / OPENAI_CLIP_STD, which is what the processor defaults to.
static const float kMean[3] = {0.48145466f, 0.4578275f, 0.40821073f};
static const float kStd[3]  = {0.26862954f, 0.26130258f, 0.27577711f};

static inline int align_up(int v, int f) { return (int)(((v + f - 1) / f) * f); }

void vision_smart_resize(int h, int w, int factor, int temporal_factor,
                         int min_tokens, int max_tokens, int* out_h, int* out_w) {
    const long long ppt = (long long)temporal_factor * factor * factor;
    const long long min_pixels = (long long)min_tokens * ppt;
    const long long max_pixels = (long long)max_tokens * ppt;
    const int aligned_frames = std::max(temporal_factor,
                                        (int)llround((double)temporal_factor / temporal_factor) * temporal_factor);
    int ah = align_up(h, factor), aw = align_up(w, factor);
    long long budget = (long long)aligned_frames * ah * aw;
    if (budget < min_pixels) {
        const double scale = sqrt((double)min_pixels / ((double)temporal_factor * h * w));
        ah = align_up(std::max(1, (int)ceil(h * scale)), factor);
        aw = align_up(std::max(1, (int)ceil(w * scale)), factor);
        budget = (long long)aligned_frames * ah * aw;
    }
    if (budget > max_pixels) {
        // Binary search the largest content height whose aligned canvas fits the budget. Mirrors
        // fit_within_budget: the search is over CONTENT height, the test is on the ALIGNED canvas.
        int low = 1, high = h, bh = factor, bw = factor;
        while (low <= high) {
            const int ch = (low + high) / 2;
            const int cw = std::max(1, (int)floor((double)w * ch / h));
            const int candh = align_up(ch, factor), candw = align_up(cw, factor);
            if ((long long)aligned_frames * candh * candw <= max_pixels) {
                bh = candh; bw = candw; low = ch + 1;
            } else high = ch - 1;
        }
        ah = bh; aw = bw;
    }
    *out_h = ah; *out_w = aw;
}

// Separable antialiased bicubic, the convention torchvision uses: filter support scales with the
// downsampling ratio, and source coordinates are pixel-centre aligned.
static float cubic(float x) {
    const float a = -0.5f;                     // torchvision/PIL use a = -0.5
    x = fabsf(x);
    if (x < 1.f) return ((a + 2.f) * x - (a + 3.f)) * x * x + 1.f;
    if (x < 2.f) return (((x - 5.f) * x + 8.f) * x - 4.f) * a;
    return 0.f;
}
static void resize_axis(const std::vector<float>& in, int ih, int iw, int C,
                        int oh, int ow, std::vector<float>& out, bool vertical) {
    const int isz = vertical ? ih : iw, osz = vertical ? oh : ow;
    const float ratio = (float)isz / (float)osz;
    const float sup = ratio > 1.f ? 2.f * ratio : 2.f;      // antialias: widen when shrinking
    const float inv = ratio > 1.f ? 1.f / ratio : 1.f;
    out.assign((size_t)C * oh * ow, 0.f);
    for (int o = 0; o < osz; ++o) {
        const float centre = (o + 0.5f) * ratio;
        const int lo = std::max(0, (int)floorf(centre - sup + 0.5f));
        const int hi = std::min(isz - 1, (int)ceilf(centre + sup - 0.5f));
        float wsum = 0.f;
        static thread_local std::vector<float> wt; wt.assign(hi - lo + 1, 0.f);
        for (int i = lo; i <= hi; ++i) {
            const float t = cubic(((i + 0.5f) - centre) * inv);
            wt[i - lo] = t; wsum += t;
        }
        if (wsum == 0.f) wsum = 1.f;
        for (int c = 0; c < C; ++c)
            for (int q = 0; q < (vertical ? ow : oh); ++q) {
                float acc = 0.f;
                for (int i = lo; i <= hi; ++i) {
                    const size_t si = vertical ? ((size_t)c * ih + i) * iw + q
                                               : ((size_t)c * ih + q) * iw + i;
                    acc += wt[i - lo] * in[si];
                }
                const size_t di = vertical ? ((size_t)c * oh + o) * ow + q
                                           : ((size_t)c * oh + q) * ow + o;
                out[di] = acc / wsum;
            }
    }
}

void vision_patchify(const float* chw, int H, int W, int* grid_h, int* grid_w,
                     std::vector<float>& out) {
    const int ps = VIS_PATCH, m = VIS_MERGE, T = VIS_TPATCH, C = VIS_CHAN;
    const int gh = H / ps, gw = W / ps;
    *grid_h = gh; *grid_w = gw;
    out.assign((size_t)gh * gw * VIS_IN_DIM, 0.f);
    // Row order is (hb, wb, i, j): patches walk in merge x merge blocks, NOT raster order, which
    // is the same order vision_position_ids uses. Within a row the layout is (c, t, py, px), and
    // the temporal slots are duplicates of the same frame for a still image (the oracle's
    // unsqueeze/expand).
    size_t r = 0;
    for (int hb = 0; hb < gh / m; ++hb)
      for (int wb = 0; wb < gw / m; ++wb)
        for (int i = 0; i < m; ++i)
          for (int j = 0; j < m; ++j, ++r) {
            float* dst = out.data() + r * VIS_IN_DIM;
            const int py0 = (hb * m + i) * ps, px0 = (wb * m + j) * ps;
            for (int c = 0; c < C; ++c)
              for (int t = 0; t < T; ++t)
                for (int py = 0; py < ps; ++py)
                  for (int px = 0; px < ps; ++px)
                    dst[((c * T + t) * ps + py) * ps + px] =
                        chw[((size_t)c * H + (py0 + py)) * W + (px0 + px)];
          }
}

PreprocResult vision_preprocess_rgb(const uint8_t* rgb, int h, int w) {
    const int factor = VIS_PATCH * VIS_MERGE;         // 28
    int th = 0, tw = 0;
    vision_smart_resize(h, w, factor, VIS_TPATCH, 16, 8000, &th, &tw);

    // Content fit. An image already inside the budget is NOT upscaled -- scale is clamped to 1 --
    // it is placed at the top-left and the canvas is zero-padded.
    double scale = std::min((double)th / h, (double)tw / w);
    const long long ppt = (long long)VIS_TPATCH * factor * factor;
    if ((long long)VIS_TPATCH * h * w >= ppt * 16) scale = std::min(1.0, scale);
    const int ch = std::max(1, std::min(th, (int)floor(h * scale)));
    const int cw = std::max(1, std::min(tw, (int)floor(w * scale)));

    std::vector<float> src((size_t)VIS_CHAN * h * w);
    for (int c = 0; c < VIS_CHAN; ++c)
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
                src[((size_t)c * h + y) * w + x] = (float)rgb[((size_t)y * w + x) * 3 + c];

    std::vector<float> content;
    if (ch != h || cw != w) {
        std::vector<float> tmp;
        resize_axis(src, h, w, VIS_CHAN, ch, w, tmp, true);
        resize_axis(tmp, ch, w, VIS_CHAN, ch, cw, content, false);
    } else content.swap(src);

    // Pad to the canvas with ZERO, then rescale and normalize -- in that order, so the padded
    // region ends up at -mean/std rather than at 0.
    std::vector<float> canvas((size_t)VIS_CHAN * th * tw, 0.f);
    for (int c = 0; c < VIS_CHAN; ++c)
        for (int y = 0; y < ch; ++y)
            memcpy(&canvas[((size_t)c * th + y) * tw], &content[((size_t)c * ch + y) * cw],
                   (size_t)cw * sizeof(float));
    for (int c = 0; c < VIS_CHAN; ++c)
        for (size_t i = 0; i < (size_t)th * tw; ++i) {
            float& v = canvas[(size_t)c * th * tw + i];
            v = (v * (1.f / 255.f) - kMean[c]) / kStd[c];
        }

    PreprocResult R;
    R.canvas_h = th; R.canvas_w = tw;
    vision_patchify(canvas.data(), th, tw, &R.grid_h, &R.grid_w, R.patches);
    return R;
}

bool vision_decode_image(const uint8_t* bytes, size_t n, std::vector<uint8_t>& rgb, int* h, int* w) {
    int comp = 0;
    unsigned char* p = stbi_load_from_memory(bytes, (int)n, w, h, &comp, 3);
    if (!p) return false;
    rgb.assign(p, p + (size_t)(*h) * (*w) * 3);
    stbi_image_free(p);
    return true;
}
bool vision_load_image(const std::string& path, std::vector<uint8_t>& rgb, int* h, int* w) {
    int comp = 0;
    unsigned char* p = stbi_load(path.c_str(), w, h, &comp, 3);
    if (!p) return false;
    rgb.assign(p, p + (size_t)(*h) * (*w) * 3);
    stbi_image_free(p);
    return true;
}

}  // namespace glm5
