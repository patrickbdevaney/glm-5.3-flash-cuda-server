// indexer.cu — DSA indexer. See include/indexer.h for the semantics and the four traps.
#include "indexer.h"
#include "gemv.h"
#include "topk_radix.h"
#include <cuda_bf16.h>
#include <cfloat>
#include <cstdio>

namespace glm5 {

#define CU(x) do { cudaError_t e_=(x); if(e_){ fprintf(stderr,"cuda %s:%d %s\n",__FILE__,__LINE__, \
    cudaGetErrorString(e_)); abort(); } } while(0)

static constexpr int HD = IDX_HEAD_DIM;   // 128
static constexpr int NH = IDX_HEADS;      // 32
static constexpr int KP = IDX_KPOOL;      // 4

size_t indexer_state_floats(int max_ctx) {
    return (size_t)idx_max_pools(max_ctx) * HD + 2 * (size_t)KP * HD;
}
// k_raw + gate + q + wts + scores(one per pool)
size_t indexer_workspace_floats(int max_ctx) {
    return (size_t)HD + HD + (size_t)NH * HD + NH + (size_t)idx_max_pools(max_ctx) + 64;
}

// ---------------------------------------------------------------- LayerNorm, WITH BIAS
// Not the RMSNorm used everywhere else in this model: this one subtracts the mean. One block,
// HD threads. Getting it wrong yields scores that are plausible and wrong.
template <typename W>
__global__ void k_layernorm_store(float* __restrict__ roll_k, const float* __restrict__ x,
                                  const W* __restrict__ w, const W* __restrict__ b,
                                  int slot, float eps) {
    const int d = threadIdx.x;
    __shared__ float red[HD / 32];
    float v = x[d];

    float sum = v;
    for (int o = 16; o; o >>= 1) sum += __shfl_down_sync(0xffffffff, sum, o);
    if ((d & 31) == 0) red[d >> 5] = sum;
    __syncthreads();
    float mean = 0.f;
    #pragma unroll
    for (int i = 0; i < HD / 32; ++i) mean += red[i];
    mean /= (float)HD;
    __syncthreads();

    const float c = v - mean;
    float var = c * c;
    for (int o = 16; o; o >>= 1) var += __shfl_down_sync(0xffffffff, var, o);
    if ((d & 31) == 0) red[d >> 5] = var;
    __syncthreads();
    float tot = 0.f;
    #pragma unroll
    for (int i = 0; i < HD / 32; ++i) tot += red[i];
    tot /= (float)HD;

    const float wv = (float)w[d], bv = (float)b[d];
    roll_k[(size_t)slot * HD + d] = c * rsqrtf(tot + eps) * wv + bv;
}

__global__ void k_store_gate(float* __restrict__ roll_gate, const float* __restrict__ g, int slot) {
    const int d = threadIdx.x;
    roll_gate[(size_t)slot * HD + d] = g[d];
}

// ---------------------------------------------------------------- pool key, on completion
// pk[d] = sum_j softmax_j(gate[j][d] + ape[j][d]) * k[j][d]
//
// The softmax is over the FOUR TOKENS, per channel d — not over the 128 channels. One thread per
// channel, so the whole reduction is thread-local and there is nothing to get wrong about ordering.
template <typename W>
__global__ void k_pool_key(float* __restrict__ pool_keys, const float* __restrict__ roll_k,
                           const float* __restrict__ roll_gate, const W* __restrict__ ape,
                           int pool) {
    const int d = threadIdx.x;
    float l[KP], mx = -FLT_MAX;
    #pragma unroll
    for (int j = 0; j < KP; ++j) {
        l[j] = roll_gate[(size_t)j * HD + d] + (float)ape[(size_t)j * HD + d];
        mx = fmaxf(mx, l[j]);
    }
    float sum = 0.f;
    #pragma unroll
    for (int j = 0; j < KP; ++j) { l[j] = __expf(l[j] - mx); sum += l[j]; }
    float acc = 0.f;
    #pragma unroll
    for (int j = 0; j < KP; ++j) acc += (l[j] / sum) * roll_k[(size_t)j * HD + d];
    pool_keys[(size_t)pool * HD + d] = acc;
}

// ---------------------------------------------------------------- pool scores
// score[p] = sum_h wts[h] * relu( (q[h] . pk[p]) * softmax_scale )
//
// One block per pool, HD threads. The block loads its own pool key once and reuses it across all
// 32 heads; q is 16 KB and hits L2 on every block after the first. The whole stage is ~0.8% of
// B_tok, so this is deliberately the simple layout — measure before making it clever.
__global__ void k_pool_scores(float* __restrict__ scores, const float* __restrict__ q,
                              const float* __restrict__ pool_keys, const float* __restrict__ wts,
                              int n_pools, float scale) {
    const int p = blockIdx.x;
    if (p >= n_pools) return;
    const int d = threadIdx.x;
    const float pk = pool_keys[(size_t)p * HD + d];

    __shared__ float red[HD / 32];
    float total = 0.f;
    for (int h = 0; h < NH; ++h) {
        float dot = q[(size_t)h * HD + d] * pk;
        for (int o = 16; o; o >>= 1) dot += __shfl_down_sync(0xffffffff, dot, o);
        if ((d & 31) == 0) red[d >> 5] = dot;
        __syncthreads();
        float s = 0.f;
        #pragma unroll
        for (int i = 0; i < HD / 32; ++i) s += red[i];
        // relu AFTER scaling, and the head weight applied OUTSIDE the relu — wts can be negative,
        // which is what makes score signed and makes -0.0 reachable.
        total += wts[h] * fmaxf(s * scale, 0.f);
        __syncthreads();
    }
    if (d == 0) scores[p] = total;
}

__global__ void k_scale_wts(float* __restrict__ w, int n, float s) {
    const int i = threadIdx.x;
    if (i < n) w[i] *= s;
}

// ---------------------------------------------------------------- select + emit
// One block. Radix-selects the top `select_k` pools, expands each to its IDX_KPOOL raw token
// indices, appends the incomplete tail raw, and pads to IDX_OUT_WIDTH with -1.
__global__ void k_select_emit(int32_t* __restrict__ out, int32_t* __restrict__ out_n,
                              const float* __restrict__ scores, int* __restrict__ sel,
                              int n_pools, int select_k, int t) {
    __shared__ TopkRadixSmem S;
    // Every pool here is COMPLETE (n_pools = floor((t+1)/kpool)) and therefore ends at 4p+3 <= t,
    // so every one of them is visible to this query and no visibility mask is needed. That is only
    // true for the decode case — a prefill row in the middle of a batch would need the mask.
    //
    // early_out=FALSE deliberately. Its fast path returns the winners in INDEX order when
    // select_k == n_pools (nothing to exclude), while torch.topk always returns them in SCORE
    // order; matching the reference costs nothing at this width.
    topk_radix_select<TOPK_RADIX_NT>(sel, scores, n_pools, select_k, -FLT_MAX, S, false);

    const int tid = threadIdx.x;
    for (int i = tid; i < IDX_OUT_WIDTH; i += blockDim.x) out[i] = -1;

    // SORT THE SELECTED POOLS ASCENDING BY INDEX, discarding the score order.
    //
    // Two reasons, and the second is the one that matters. (1) Ascending pool ids expand to
    // ascending token ids, so the attention reads the latent cache SEQUENTIALLY instead of
    // jumping around it — at 8k context that is the difference between a streaming read and a
    // gather. (2) It makes the sparse path BIT-IDENTICAL to the dense path whenever the indexer
    // selects everything, because the fp32 context sum then accumulates in the same order. That
    // turns "sparse agrees with dense below 2051" into an exact test with no new oracle, which is
    // the only whole-path check available above the reference's reach.
    __shared__ int ps[TOPK_RADIX_CAP];
    ps[tid] = (tid < select_k && sel[tid] >= 0) ? sel[tid] : 0x7fffffff;   // pad sorts to the end
    __syncthreads();
    for (int k = 2; k <= TOPK_RADIX_CAP; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            const int ixj = tid ^ j;
            if (ixj > tid) {
                const int a = ps[tid], b = ps[ixj];
                const bool up = ((tid & k) == 0);
                if (up ? (a > b) : (a < b)) { ps[tid] = b; ps[ixj] = a; }
            }
            __syncthreads();
        }
    }

    // A hole (sel[j] < 0) can only arise if a score was NaN, since every finite score is admitted
    // by floorv = -FLT_MAX. It cannot be silently tolerated: the expansion below would leave -1 in
    // the middle of the list and attention would index the cache at -1. Detect it and report a
    // negative count so the caller aborts loudly instead.
    __shared__ int holes;
    if (tid == 0) holes = 0;
    __syncthreads();
    if (tid < select_k && sel[tid] < 0) atomicAdd(&holes, 1);
    __syncthreads();

    for (int j = tid; j < select_k; j += blockDim.x) {
        const int p = ps[j];
        if (p == 0x7fffffff) continue;
        #pragma unroll
        for (int c = 0; c < KP; ++c) out[j * KP + c] = p * KP + c;
    }
    // The trailing incomplete pool is never selectable, but its tokens ARE visible and are
    // appended raw. This is the branch that makes the dense limit 2051 rather than 2048. It sits
    // after every selected pool, and its token ids are the highest, so the list stays ascending.
    const int tail_start = n_pools * KP;
    const int tail = t + 1 - tail_start;
    for (int c = tid; c < tail; c += blockDim.x) out[select_k * KP + c] = tail_start + c;
    if (tid == 0) *out_n = holes ? -1 : select_k * KP + tail;
}

// ---------------------------------------------------------------- host entry points
void indexer_keys(const float* h, const IndexerWeights& W, IndexerState& S, int t,
                  float* ws, cudaStream_t s) {
    float* k_raw = ws;
    float* gate  = ws + HD;
    const int slot = t % KP;

    gemv(k_raw, W.wk, h, HD, HIDDEN, W.dtype, s);
    if (W.dtype == GEMV_F32)
        k_layernorm_store<float><<<1, HD, 0, s>>>(S.roll_k, k_raw, (const float*)W.k_norm_w,
                                                  (const float*)W.k_norm_b, slot, IDX_LN_EPS);
    else
        k_layernorm_store<__nv_bfloat16><<<1, HD, 0, s>>>(S.roll_k, k_raw,
                                                  (const __nv_bfloat16*)W.k_norm_w,
                                                  (const __nv_bfloat16*)W.k_norm_b, slot, IDX_LN_EPS);

    gemv(gate, W.compress_gate, h, HD, HIDDEN, W.dtype, s);
    k_store_gate<<<1, HD, 0, s>>>(S.roll_gate, gate, slot);

    if (slot == KP - 1) {                                  // this token completes pool t/KP
        const int pool = t / KP;
        if (W.dtype == GEMV_F32)
            k_pool_key<float><<<1, HD, 0, s>>>(S.pool_keys, S.roll_k, S.roll_gate,
                                               (const float*)W.compress_ape, pool);
        else
            k_pool_key<__nv_bfloat16><<<1, HD, 0, s>>>(S.pool_keys, S.roll_k, S.roll_gate,
                                               (const __nv_bfloat16*)W.compress_ape, pool);
    }
}

void indexer_scores(const float* h, const float* q_resid, const IndexerWeights& W,
                    const IndexerState& S, int n_pools, float* scores, float* ws, cudaStream_t s) {
    float* q   = ws + 2 * HD;
    float* wts = q + (size_t)NH * HD;
    gemv(q, W.wq_b, q_resid, NH * HD, MLA_Q_LORA, W.dtype, s);
    gemv(wts, W.weights_proj, h, NH, HIDDEN, W.dtype, s);
    k_scale_wts<<<1, 32, 0, s>>>(wts, NH, rsqrtf((float)NH));
    if (n_pools > 0)
        k_pool_scores<<<n_pools, HD, 0, s>>>(scores, q, S.pool_keys, wts, n_pools,
                                             rsqrtf((float)HD));
}

void indexer_decode_step(const float* h, const float* q_resid, const IndexerWeights& W,
                         IndexerState& S, int t, int max_ctx,
                         int32_t* out_idx, int32_t* out_n, float* ws, cudaStream_t s) {
    indexer_keys(h, W, S, t, ws, s);

    // Only COMPLETE pools are candidates; the count is floor((t+1)/kpool), which is exactly the
    // `keep = pool_valid.any(0)` trim the reference applies.
    const int n_pools  = (t + 1) / KP;
    const int select_k = n_pools < IDX_SELECT_MAX ? n_pools : IDX_SELECT_MAX;

    float* scores = ws + 2 * HD + (size_t)NH * HD + NH + 64;
    indexer_scores(h, q_resid, W, S, n_pools, scores, ws, s);

    static int32_t* sel = nullptr;                  // TOPK_RADIX_CAP ints, reused across calls
    if (!sel) CU(cudaMalloc(&sel, TOPK_RADIX_CAP * sizeof(int32_t)));
    k_select_emit<<<1, TOPK_RADIX_NT, 0, s>>>(out_idx, out_n, scores, sel, n_pools, select_k, t);
}

}  // namespace glm5
