#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>
namespace glm5 {
enum { GEMV_F32 = 0, GEMV_BF16 = 1 };

// A weight matrix as the engine hands it to gemv/gemm: either a plain row-major buffer of
// `dtype`, or an NVFP4 triple. NVFP4 takes precedence whenever `packed` is set, and `dtype` is
// then ignored — which is what lets ROOFLINE §3 be gated FAMILY BY FAMILY at load time with no
// runtime branch and no second code path in the callers. A family that is not in the overlay
// simply keeps its bf16 pointer.
//
// The implicit constructor from `const void*` is deliberate: every existing assignment and every
// gate that hands these structs a raw pointer keeps compiling and keeps meaning bf16.
struct WRef {
    const void*    p      = nullptr;   // bf16/f32 weights, row-major [N, K]
    const uint8_t* packed = nullptr;   // NVFP4 [N, K/2], low nibble first
    const uint8_t* scale  = nullptr;   // NVFP4 [N, K/16], fp8-e4m3
    const float*   gscale = nullptr;   // NVFP4 [1]
    WRef() = default;
    WRef(const void* q) : p(q) {}                                   // NOLINT: implicit on purpose
    WRef(std::nullptr_t) {}
    WRef(const uint8_t* pk, const uint8_t* sc, const float* gs) : packed(pk), scale(sc), gscale(gs) {}
    bool nvfp4() const { return packed != nullptr; }
};

// y[N] = W[N,K] @ x[K],  W row-major, x and y fp32.
void gemv(float* y, const WRef& W, const float* x, int N, int K, int dtype, cudaStream_t s);

// ---- gemv_multi ------------------------------------------------------------------------------
//
// Several INDEPENDENT gemvs that share K and dtype, issued as ONE launch.
//
// The KDA decode path is not weight-bandwidth-bound, it is LAUNCH-bound. kda_gates issues seven
// kernels per layer x 34 layers = 238 launches per token to move 2.7 MB, and dprof measures it at
// 16.3 GB/s -- 7% of achievable, while the MoE gemvs on either side reach 66-70%. gemv_bs() was
// already tuned for these shapes (see its comment) and did not fix it, because the residue is the
// launch itself, not occupancy.
//
// Each job keeps its own y, W, x and N. Nothing about a row's reduction changes -- a block still
// owns the same rows and sums them in the same order -- so the result is BIT-IDENTICAL to calling
// gemv() on each job in turn. tests/gate_gemv_multi.cu asserts exactly that on real checkpoint
// weights, which is the gate that lets speculative verification keep using this path.
//
// Jobs whose W is not NVFP4, or a count outside 2..GEMV_MULTI_MAX, fall back to a gemv() loop:
// correct, just not fused. In production every KDA/MLA projection is in the dense NVFP4 overlay,
// so the fused path is the one that runs; the fallback covers the fp32 oracle gates.
enum { GEMV_MULTI_MAX = 4 };
struct GemvJob {
    float*       y = nullptr;
    WRef         W;
    const float* x = nullptr;
    int          N = 0;
};
void gemv_multi(const GemvJob* jobs, int n, int K, int dtype, cudaStream_t s);
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
// verify speculative drafts losslessly. For NVFP4 that identity is structural rather than
// argued: gemv IS k_gemm_nvfp4<BS, 1>.
//
// M is arbitrary; the dispatcher covers 1..8, 16 and 32 exactly and splits anything else into
// those chunks, re-reading W once per chunk (which is what a caller would have paid anyway).
void gemm(float* y, const WRef& W, const float* x, int M, int N, int K, int dtype, cudaStream_t s);
}
