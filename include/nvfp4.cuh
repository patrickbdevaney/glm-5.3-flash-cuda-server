// nvfp4.cuh — the NVFP4 primitives, shared by the MoE experts and the dense gemv/gemm.
//
// Layout, exactly as the checkpoint (and tools/requant_dense_nvfp4.py) writes it:
//   packed [out, in/2]  uint8, LOW nibble first
//   scale  [out, in/16] fp8-e4m3
//   gscale [1]          fp32
//   w = kE2M1[nib & 7] * (-1)^(nib >> 3) * fp8(scale) * (1 / gscale)
//
// The unpack is a HARDWARE instruction. __nv_cvt_fp4x2_to_halfraw2 lowers to
// `cvt.rn.f16x2.e2m1x2`: two e2m1 codes -> one half2, one instruction. It is available on sm_110a
// even though the FP4 *mma* path (mma.sync.kind::f8f6f4, tcgen05, block-scaled mxf4) is rejected
// by ptxas for this arch and cuBLASLt returns zero FP4 algos. Unpack yes, tensor-core FP4 no.
//
// The nibble LUT that this replaced lived in __constant__ memory, which broadcasts only when every
// lane reads the SAME address. Here every lane reads a different one, so all 16 lookups per group
// serialised up to 8 ways -- that, not the loads, was the MoE bottleneck (OPTIMIZATION_LOG #10).
#pragma once
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace glm5 {

// fp8-e4m3 -> float. Built from the exponent field rather than exp2f so it is exact.
// Callers stage this into a 256-entry SHARED table: a byte-indexed table is effectively random
// across a warp, so it bank-conflicts -- and replacing it with this arithmetic inline measured
// 19% SLOWER, because the kernel is instruction-bound, not shared-bandwidth-bound. Do not
// "optimise" the table away again.
__device__ __forceinline__ float fp8e4m3(uint8_t v) {
    const int s = v >> 7, e = (v >> 3) & 0xF, m = v & 0x7;
    const float sign = s ? -1.f : 1.f;
    if (e == 0) return sign * (float)m * (1.f / 8.f) * (1.f / 64.f);       // subnormal, 2^-6
    return sign * (1.f + (float)m * (1.f / 8.f)) * __int_as_float((e - 7 + 127) << 23);
}

// Two packed e2m1 codes -> two floats.
__device__ __forceinline__ float2 e2m1x2(unsigned char b) {
    __half2_raw r = __nv_cvt_fp4x2_to_halfraw2((__nv_fp4x2_storage_t)b, __NV_E2M1);
    __half2 h = *reinterpret_cast<__half2*>(&r);
    return make_float2(__low2float(h), __high2float(h));
}

// Two fp8-e4m3 group scales -> two floats, in ONE hardware instruction (cvt.rn.f16x2.e4m3x2).
//
// Not to be confused with the 256-entry shared LUT the MoE uses. That LUT beat *hand-rolled
// arithmetic* by 19%, which is why it is still there; it does not beat a hardware converter. The
// LUT is also indexed by a byte VALUE, so it is random across a warp and bank-conflicts — and in
// a dense gemv, where the whole matrix streams through, that conflict is on the critical path.
// Here consecutive lanes want consecutive scale bytes, so this reads them coalesced instead.
__device__ __forceinline__ float2 e4m3x2(unsigned short b) {
    __half2_raw r = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)b, __NV_E4M3);
    __half2 h = *reinterpret_cast<__half2*>(&r);
    return make_float2(__low2float(h), __high2float(h));
}

// One uint32 of packed weights -> 8 floats. uint32 is the WIDEST legal load on checkpoint
// tensors: safetensors aligns to 4 bytes and 777 weight_packed tensors per shard sit at offset
// 4 mod 8, so a uint2 faults with "misaligned address" (OPTIMIZATION_LOG #2).
__device__ __forceinline__ void e2m1x8(float* __restrict__ w, unsigned pv) {
    #pragma unroll
    for (int b = 0; b < 4; ++b) {
        const float2 f = e2m1x2((unsigned char)((pv >> (b * 8)) & 0xff));
        w[b * 2] = f.x; w[b * 2 + 1] = f.y;
    }
}

}  // namespace glm5
