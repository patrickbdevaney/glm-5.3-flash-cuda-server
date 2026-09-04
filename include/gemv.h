#pragma once
#include <cuda_runtime.h>
namespace glm5 {
enum { GEMV_F32 = 0, GEMV_BF16 = 1 };
// y[N] = W[N,K] @ x[K],  W row-major, x and y fp32.
void gemv(float* y, const void* W, const float* x, int N, int K, int dtype, cudaStream_t s);
}

// fp32 -> bf16 in place on device, for gates/benches that hold fp32 oracle weights but must
// exercise the production dtype. Production loads bf16 straight from the checkpoint.
namespace glm5 { void f32_to_bf16(void* dst, const float* src, size_t n, cudaStream_t s); }
namespace glm5 { void f32_from_bf16_dev(float* dst, const void* src, size_t n, cudaStream_t s); }
