// kda.cu — Kimi Delta Attention, decode step (batch 1).
//
// 34 of 45 layers are KDA and they are 47.4% of per-token bandwidth (ROOFLINE.md §1), so this
// file is the single largest cost centre in the engine. Two very different problems live here:
//
//   * the projections (q/k/v/o_proj, 4 x [8192, 4096]) — pure streaming, handled by gemv.cu;
//   * the recurrence — 64 independent heads, each a [128 x 128] fp32 state that must be read,
//     updated and written exactly once. 4 MiB per layer, 136 MiB total, CONTEXT-INDEPENDENT.
//
// The recurrence is where a naive implementation loses. The state has to be touched twice (once
// to reduce over k, once to apply the rank-1 update) but must only cross HBM once, so it is
// staged in 64 KiB of dynamic shared memory per head. That needs the >48 KB opt-in, done once in
// kda_recurrence().
//
// Reference: transformers Glm5NextTextLinearAttention / recurrent_kimi_delta_attention.
// Gated against it on real layer-0 weights by tests/gate_kda.cu.
#include "kda.h"
#include "gemv.h"
#include "glm5_config.h"
#include <cstdio>
#include <cmath>

namespace glm5 {

#define CU(x) do { cudaError_t e_ = (x); if (e_) { \
    fprintf(stderr, "cuda %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); abort(); } } while (0)

static constexpr int H  = KDA_HEADS;      // 64
static constexpr int D  = KDA_HEAD_DIM;   // 128
static constexpr int Q  = KDA_QKV_DIM;    // 8192
static constexpr int CK = KDA_CONV_K;     // 4
static constexpr int CS = KDA_CONV_STATE; // 3

// ---------------------------------------------------------------- workspace layout
// [0]        q_raw   Q      (q_proj output, pre-conv)
// [Q]        k_raw   Q
// [2Q]       v_raw   Q
// [3Q]       f_lo    GATE_RANK
// [3Q+128]   g_lo    GATE_RANK
// [3Q+256]   f_hi    Q       (f_b output)
// [4Q+256]   gate    Q       (g_b output)
// [5Q+256]   g       Q       (forget gate, log-space)
// [6Q+256]   beta    H
// [6Q+256+H] q_n     Q
// [7Q+256+H] k_n     Q
// [8Q+256+H] core    Q
// [9Q+256+H] normed  Q
size_t kda_workspace_floats() { return 10 * (size_t)Q + 256 + H; }

// ---------------------------------------------------------------- conv + silu
// Depthwise causal conv, kernel 4, one channel per thread. The rolling window lives in
// conv_state[c][0..2]; the new sample is appended and the window shifts by one.
//
// The checkpoint splits this into q_conv1d / k_conv1d / v_conv1d, each [8192, 1, 4]; the loader
// concatenates them in q,k,v order to match the reference's single [24576, 1, 4].
__global__ void k_conv_silu(float* __restrict__ out, float* __restrict__ conv_state,
                            const float* __restrict__ qkv_raw, const float* __restrict__ wgt,
                            int C) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float* st = conv_state + (size_t)c * CS;
    const float* w = wgt + (size_t)c * CK;
    const float xn = qkv_raw[c];

    float acc = st[0] * w[0] + st[1] * w[1] + st[2] * w[2] + xn * w[3];
    st[0] = st[1]; st[1] = st[2]; st[2] = xn;          // shift the window

    out[c] = acc / (1.f + __expf(-acc));                // silu
}

// ---------------------------------------------------------------- forget / input / output gates
// g[h][d] = lower_bound * sigmoid( exp(A_log[h]) * (f_b(f_a(x))[h][d] + dt_bias[h][d]) )
// with lower_bound = -5.0, so g is in (-5, 0) and exp(g) in (0.0067, 1).
__global__ void k_forget_gate(float* __restrict__ g, const float* __restrict__ f_hi,
                              const float* __restrict__ dt_bias, const float* __restrict__ A_log) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= Q) return;
    const int h = i / D;
    const float decay = __expf(A_log[h]);
    const float v = decay * (f_hi[i] + dt_bias[i]);
    g[i] = KDA_LOWER_BOUND / (1.f + __expf(-v));
}

__global__ void k_sigmoid(float* __restrict__ y, const float* __restrict__ x, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = 1.f / (1.f + __expf(-x[i]));
}

// ---------------------------------------------------------------- l2 norm over head_dim
// FLA convention: inv = sqrt(sum(x^2) + eps), x / inv.  NOT max(norm, eps) — the difference is
// small but systematic, and matching it is why this gates instead of merely correlating.
// One warp per head; q additionally scaled by 1/sqrt(D).
__global__ void k_l2norm(float* __restrict__ q_n, float* __restrict__ k_n,
                         const float* __restrict__ qkv) {
    const int h = blockIdx.x;
    const int t = threadIdx.x;                    // 128 threads, one per dim
    const float qv = qkv[(size_t)h * D + t];
    const float kv = qkv[(size_t)Q + (size_t)h * D + t];

    __shared__ float sq[4], sk[4];
    float aq = qv * qv, ak = kv * kv;
    for (int o = 16; o; o >>= 1) {
        aq += __shfl_down_sync(0xffffffff, aq, o);
        ak += __shfl_down_sync(0xffffffff, ak, o);
    }
    const int lane = t & 31, warp = t >> 5;
    if (lane == 0) { sq[warp] = aq; sk[warp] = ak; }
    __syncthreads();
    float tq = sq[0] + sq[1] + sq[2] + sq[3];
    float tk = sk[0] + sk[1] + sk[2] + sk[3];
    const float iq = sqrtf(tq + KDA_L2_EPS), ik = sqrtf(tk + KDA_L2_EPS);

    q_n[(size_t)h * D + t] = qv / iq * rsqrtf((float)D);
    k_n[(size_t)h * D + t] = kv / ik;
}

// ---------------------------------------------------------------- the recurrence
// One block per head. 128 threads, thread v owns state COLUMN v.
//
//   S     = S * exp(g)[k]                      decay, per k-row
//   kv_mem[v] = sum_k S[k][v] * k[k]
//   delta[v]  = (v_in[v] - kv_mem[v]) * beta
//   S[k][v]  += k[k] * delta[v]
//   out[v]    = sum_k S[k][v] * q[k]
//
// S is row-major [k][v], so for a fixed k all 128 threads read 128 consecutive floats: coalesced
// in HBM and bank-conflict-free in shared memory. The state crosses HBM exactly once each way.
__global__ void k_recurrence(float* __restrict__ core, float* __restrict__ S,
                             const float* __restrict__ q_n, const float* __restrict__ k_n,
                             const float* __restrict__ v_in, const float* __restrict__ g,
                             const float* __restrict__ beta) {
    extern __shared__ float Ss[];                    // [D][D] = 64 KiB
    const int h = blockIdx.x, v = threadIdx.x;
    float* Sh = S + (size_t)h * D * D;

    __shared__ float eg[D], kk[D], qq[D];
    eg[v] = __expf(g[(size_t)h * D + v]);
    kk[v] = k_n[(size_t)h * D + v];
    qq[v] = q_n[(size_t)h * D + v];
    const float vv = v_in[(size_t)h * D + v];
    const float b  = beta[h];
    __syncthreads();

    // pass 1: decay into shared memory, reduce over k for kv_mem[v]
    float kv_mem = 0.f;
    #pragma unroll 8
    for (int k = 0; k < D; ++k) {
        const float s = Sh[(size_t)k * D + v] * eg[k];
        Ss[k * D + v] = s;
        kv_mem += s * kk[k];
    }

    const float delta = (vv - kv_mem) * b;

    // pass 2: rank-1 update, write S back, reduce over k for the output
    float out = 0.f;
    #pragma unroll 8
    for (int k = 0; k < D; ++k) {
        const float s = Ss[k * D + v] + kk[k] * delta;
        Sh[(size_t)k * D + v] = s;
        out += s * qq[k];
    }
    core[(size_t)h * D + v] = out;
}

// ---------------------------------------------------------------- gated output norm
// RMSNorm over head_dim in strict fp32 (upstream does not downcast the weights here), then
// multiply by sigmoid(gate). One block per head.
__global__ void k_out_norm_f32(float* __restrict__ normed, const float* __restrict__ core,
                               const float* __restrict__ gate, const float* __restrict__ w) {
    const int h = blockIdx.x, d = threadIdx.x;
    const float c = core[(size_t)h * D + d];
    __shared__ float red[4];
    float a = c * c;
    for (int o = 16; o; o >>= 1) a += __shfl_down_sync(0xffffffff, a, o);
    const int lane = d & 31, warp = d >> 5;
    if (lane == 0) red[warp] = a;
    __syncthreads();
    const float var = (red[0] + red[1] + red[2] + red[3]) / (float)D;
    const float y = c * rsqrtf(var + RMS_EPS) * w[d];
    const float gt = gate[(size_t)h * D + d];
    normed[(size_t)h * D + d] = y * (1.f / (1.f + __expf(-gt)));
}

__global__ void k_out_norm_bf16(float* __restrict__ normed, const float* __restrict__ core,
                                const float* __restrict__ gate, const __nv_bfloat16* __restrict__ w) {
    const int h = blockIdx.x, d = threadIdx.x;
    const float c = core[(size_t)h * D + d];
    __shared__ float red[4];
    float a = c * c;
    for (int o = 16; o; o >>= 1) a += __shfl_down_sync(0xffffffff, a, o);
    const int lane = d & 31, warp = d >> 5;
    if (lane == 0) red[warp] = a;
    __syncthreads();
    const float var = (red[0] + red[1] + red[2] + red[3]) / (float)D;
    const float y = c * rsqrtf(var + RMS_EPS) * __bfloat162float(w[d]);
    const float gt = gate[(size_t)h * D + d];
    normed[(size_t)h * D + d] = y * (1.f / (1.f + __expf(-gt)));
}

// ---------------------------------------------------------------- stage entry points

void kda_qkv_conv(const float* x, const KdaWeights& W, float* conv_state, float* qkv, float* ws,
                  cudaStream_t s) {
    float* q_raw = ws;
    gemv(q_raw,         W.q_proj, x, Q, HIDDEN, W.dtype, s);
    gemv(q_raw + Q,     W.k_proj, x, Q, HIDDEN, W.dtype, s);
    gemv(q_raw + 2 * Q, W.v_proj, x, Q, HIDDEN, W.dtype, s);
    const int C = 3 * Q;
    // conv weights are fp32 in the gate and bf16 in production; upconvert once at load, not here.
    k_conv_silu<<<(C + 255) / 256, 256, 0, s>>>(qkv, conv_state, q_raw, (const float*)W.conv1d, C);
}

void kda_gates(const float* x, const KdaWeights& W, float* g, float* beta, float* gate, float* ws,
               cudaStream_t s) {
    float* f_lo = ws;               // [128]
    float* g_lo = ws + 128;         // [128]
    float* f_hi = ws + 256;         // [Q]
    gemv(f_lo, W.f_a, x, KDA_GATE_RANK, HIDDEN, W.dtype, s);
    gemv(g_lo, W.g_a, x, KDA_GATE_RANK, HIDDEN, W.dtype, s);
    gemv(f_hi, W.f_b, f_lo, Q, KDA_GATE_RANK, W.dtype, s);
    gemv(gate, W.g_b, g_lo, Q, KDA_GATE_RANK, W.dtype, s);
    k_forget_gate<<<(Q + 255) / 256, 256, 0, s>>>(g, f_hi, W.dt_bias, W.A_log);
    gemv(beta, W.b_proj, x, H, HIDDEN, W.dtype, s);
    k_sigmoid<<<1, H, 0, s>>>(beta, beta, H);
}

void kda_norm_qk(const float* qkv, float* q_n, float* k_n, cudaStream_t s) {
    k_l2norm<<<H, D, 0, s>>>(q_n, k_n, qkv);
}

void kda_recurrence(const float* q_n, const float* k_n, const float* v_in, const float* g,
                    const float* beta, float* S, float* core, cudaStream_t s) {
    static bool optin = false;
    const size_t smem = (size_t)D * D * sizeof(float);      // 64 KiB
    if (!optin) {
        CU(cudaFuncSetAttribute(k_recurrence, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
        optin = true;
    }
    k_recurrence<<<H, D, smem, s>>>(core, S, q_n, k_n, v_in, g, beta);
}

void kda_out_norm(const float* core, const float* gate, const void* w, int dtype, float* normed,
                  cudaStream_t s) {
    if (dtype == GEMV_F32) k_out_norm_f32<<<H, D, 0, s>>>(normed, core, gate, (const float*)w);
    else                   k_out_norm_bf16<<<H, D, 0, s>>>(normed, core, gate, (const __nv_bfloat16*)w);
}

void kda_decode_step(const float* x, const KdaWeights& W, float* conv_state, float* S, float* y,
                     float* ws, cudaStream_t s) {
    float* qkv    = ws;                          // reuse q_raw..v_raw in place after the conv
    float* f_area = ws + 3 * Q;
    float* gate   = ws + 4 * Q + 256;
    float* g      = ws + 5 * Q + 256;
    float* beta   = ws + 6 * Q + 256;
    float* q_n    = ws + 6 * Q + 256 + H;
    float* k_n    = ws + 7 * Q + 256 + H;
    float* core   = ws + 8 * Q + 256 + H;
    float* normed = ws + 9 * Q + 256 + H;

    kda_qkv_conv(x, W, conv_state, qkv, ws, s);
    kda_gates(x, W, g, beta, gate, f_area, s);
    kda_norm_qk(qkv, q_n, k_n, s);
    kda_recurrence(q_n, k_n, qkv + 2 * Q, g, beta, S, core, s);
    kda_out_norm(core, gate, W.o_norm, W.dtype, normed, s);
    gemv(y, W.o_proj, normed, HIDDEN, Q, W.dtype, s);
}

}  // namespace glm5
