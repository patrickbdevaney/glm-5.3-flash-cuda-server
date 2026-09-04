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

namespace glm5 {
// y[M, N] = x[M, K] @ W[N, K]^T,  W row-major [N, K], x and y fp32 and row-major.
//
// THE WHOLE POINT is that W is read ONCE for all M rows of x. At M=1 this is gemv and produces
// BIT-IDENTICAL results (same loop order, same reduction tree) — tests/gate_batch.cu checks that,
// because a batched path that merely agrees to 1e-6 with the sequential one cannot be used to
// verify speculative drafts losslessly.
//
// M is arbitrary; the dispatcher covers 1..8, 16 and 32 exactly and splits anything else into
// those chunks, re-reading W once per chunk (which is what a caller would have paid anyway).
void gemm(float* y, const void* W, const float* x, int M, int N, int K, int dtype, cudaStream_t s);
}
