// vision_preproc.h — image bytes to the [n_patch, 1176] rows the vision tower eats.
//
// Host-side and Python-free, which is the point: the rest of this server has no Python on the
// request path and an image endpoint that shelled out to torchvision would be the only exception.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace glm5 {

struct PreprocResult {
    std::vector<float> patches;   // [grid_h*grid_w, 1176], row-major, 2x2-block order
    int grid_t = 1, grid_h = 0, grid_w = 0;
    int canvas_h = 0, canvas_w = 0;
};

// Aligned canvas for an image, matching transformers' smart_resize exactly: align both sides up to
// `factor`, and only if that exceeds the token budget binary-search a smaller content height.
void vision_smart_resize(int h, int w, int factor, int temporal_factor,
                         int min_tokens, int max_tokens, int* out_h, int* out_w);

// Full path: RGB8 HWC -> canvas (fit + zero pad, or antialiased bicubic downscale when the budget
// bites) -> rescale 1/255 -> CLIP normalize -> patchify.
PreprocResult vision_preprocess_rgb(const uint8_t* rgb, int h, int w);

// Patchify a CHW float image that is ALREADY resized and normalized. Split out because this half
// is exact integer arithmetic and must match the oracle to the bit, while the resize half is
// implementation-defined -- keeping them separate means a gate failure says which one moved.
void vision_patchify(const float* chw, int H, int W, int* grid_h, int* grid_w,
                     std::vector<float>& out);

// Decode a PNG/JPEG from disk or memory into RGB8. Returns false and leaves the vectors empty on
// a malformed file rather than aborting: this one is reachable from a network request.
bool vision_load_image(const std::string& path, std::vector<uint8_t>& rgb, int* h, int* w);
bool vision_decode_image(const uint8_t* bytes, size_t n, std::vector<uint8_t>& rgb, int* h, int* w);

}  // namespace glm5
