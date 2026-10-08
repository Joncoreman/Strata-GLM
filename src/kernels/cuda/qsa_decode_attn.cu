// src/kernels/cuda/qsa_decode_attn.cu - see include/strata/kernels/qsa_decode_attn.hpp.
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/kv_q4.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int HD = 256;          // head_dim
constexpr int G = 12;            // query heads per KV head (24 / 2)
constexpr int CHUNK = 64;        // cells per block
constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// 8 consecutive values of one cell's key or value row for KV head `kvh`, dimensions [d0, d0+8).
// Per-format bodies; `load8` below is the KV_MODE dispatcher. value=false is the K side, true the V side.
__device__ __forceinline__ void load8_f16(const QsaAttnPools& p, bool value, long long row, int d0, float* out) {
    const uint16_t* base = (value ? p.v_pool : p.k_pool) + row * HD + d0;
    const uint4 raw = *reinterpret_cast<const uint4*>(base);
    const __half2* h2 = reinterpret_cast<const __half2*>(&raw);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float2 f = __half22float2(h2[j]);
        out[2 * j] = f.x;
        out[2 * j + 1] = f.y;
    }
}
__device__ __forceinline__ void load8_q8(const QsaAttnPools& p, bool value, long long row, int d0, float* out) {
    const int8_t* codes = (value ? p.v_q : p.k_q) + row * HD + d0;
    const uint16_t sbits = (value ? p.v_scale : p.k_scale)[row * (HD / KV_Q8_GROUP) + d0 / KV_Q8_GROUP];
    const float sc = __half2float(__ushort_as_half(sbits));
    const uint2 raw = *reinterpret_cast<const uint2*>(codes);
    const int8_t* c = reinterpret_cast<const int8_t*>(&raw);
#pragma unroll
    for (int j = 0; j < 8; ++j) out[j] = (float) c[j] * sc;
}
__device__ __forceinline__ void load8_q4(const QsaAttnPools& p, bool value, long long row, int d0, float* out) {
    constexpr int bytes_per_head = (HD / QK4_0) * sizeof(block_q4_0);
    const int b = d0 / QK4_0;
    const int rem = d0 % QK4_0;
    const block_q4_0* blk = reinterpret_cast<const block_q4_0*>((value ? p.v_q4 : p.k_q4) + row * bytes_per_head) + b;
    const float d = __half2float(__ushort_as_half(blk->d));
    const int j = (rem == 0 || rem == 16) ? 0 : 8;
    const uint8_t* bytes = blk->qs + j;
    if (rem < 16) {
#pragma unroll
        for (int k = 0; k < 8; ++k) out[k] = (float) ((int)(bytes[k] & 0x0F) - 8) * d;
    } else {
#pragma unroll
        for (int k = 0; k < 8; ++k) out[k] = (float) ((int)(bytes[k] >> 4) - 8) * d;
    }
}
template <int KV_MODE>
__device__ __forceinline__ void load8(const QsaAttnPools& p, bool value, long long row, int d0, float* out) {
    if constexpr (KV_MODE == 0) load8_f16(p, value, row, d0, out);
    else if constexpr (KV_MODE == 1) load8_q8(p, value, row, d0, out);
    else if constexpr (KV_MODE == 3) {
        // hybrid K8V4: both sides are defined - K unrotated INT8, V rotated Q4_0 - so a value=true call
        // reads the Q4_0 pool instead of dereferencing the null v_q (no call site does today; PR review)
        if (value) load8_q4(p, value, row, d0, out);
        else load8_q8(p, value, row, d0, out);
    } else load8_q4(p, value, row, d0, out);
}

template <int KV_MODE>
__global__ void __launch_bounds__(THREADS) attn_chunk_kernel(const float* __restrict__ q, QsaAttnPools p,
                                                             const int32_t* __restrict__ ids,
                                                             const int32_t* __restrict__ step, int n_kv_heads,
                                                             int page_size, float scale, float* __restrict__ part_acc,
                                                             float* __restrict__ part_m, float* __restrict__ part_l,
                                                             int n_chunks, int cap = 0, long long scratch_stride = 0) {
    // batched form: query blockIdx.z, with its own q row, selection, step and scratch
    q += (size_t) blockIdx.z * (size_t) (n_kv_heads * G) * HD;
    ids += (size_t) blockIdx.z * (size_t) cap;
    step += (size_t) blockIdx.z * kStepCount;
    part_acc += (size_t) blockIdx.z * (size_t) scratch_stride;
    part_m += (size_t) blockIdx.z * (size_t) scratch_stride;
    part_l += (size_t) blockIdx.z * (size_t) scratch_stride;
    __shared__ __align__(16) float sq[G][HD];     // 12 KB: this KV head's query heads
    __shared__ float sp[G][CHUNK];                // scores, then probabilities
    __shared__ long long srow[CHUNK];             // pool row of each cell (page, kv head, slot)
    const int n_ids = __ldg(step + kStepWidth);
    const int chunk = blockIdx.x, kvh = blockIdx.y;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int c0 = chunk * CHUNK;
    const int n_here = min(CHUNK, n_ids - c0);
    const int slot = kvh * n_chunks + chunk;
    if (n_here <= 0) {
        if (t < G) { part_m[slot * G + t] = -FLT_MAX; part_l[slot * G + t] = 0.0f; }
        return;
    }
    for (int i = t; i < G * HD; i += THREADS) sq[i / HD][i % HD] = q[(size_t) (kvh * G) * HD + i];
    if (t < CHUNK) {
        long long r = -1;
        if (t < n_here) {
            const int cell = ids[c0 + t];
            const long long page = (long long) p.page_table[cell / page_size];
            // a block the KV streaming could not make resident keeps page -1 (ctl[3]); its cells are masked
            // (score -FLT_MAX, weight 0) instead of being read from before the pool.
            if (page >= 0) r = (page * n_kv_heads + kvh) * page_size + (cell % page_size);
        }
        srow[t] = r;
    }
    __syncthreads();
    // scores: each warp takes cells warp, warp+8, ...; each lane holds 8 of the 256 dimensions.
    for (int c = warp; c < CHUNK; c += WARPS) {
        if (c >= n_here || srow[c] < 0) {
            if (lane < G) sp[lane][c] = -FLT_MAX;
            continue;
        }
        float k8[8];
        load8<KV_MODE>(p, false, srow[c], lane * 8, k8);
#pragma unroll
        for (int h = 0; h < G; ++h) {
            const float4 qa = *reinterpret_cast<const float4*>(&sq[h][lane * 8]);
            const float4 qb = *reinterpret_cast<const float4*>(&sq[h][lane * 8 + 4]);
            float s = k8[0] * qa.x + k8[1] * qa.y + k8[2] * qa.z + k8[3] * qa.w +
                      k8[4] * qb.x + k8[5] * qb.y + k8[6] * qb.z + k8[7] * qb.w;
            s = warp_sum(s);
            if (lane == 0) sp[h][c] = s * scale;
        }
    }
    __syncthreads();
    // per-head chunk max and exp-sum: warp w handles heads w and w+8.
    for (int h = warp; h < G; h += WARPS) {
        const float a = sp[h][lane], b = sp[h][lane + 32];
        const float m = warp_max(fmaxf(a, b));
        const float ea = (lane < n_here && srow[lane] >= 0) ? __expf(a - m) : 0.0f;
        const float eb = (lane + 32 < n_here && srow[lane + 32] >= 0) ? __expf(b - m) : 0.0f;
        sp[h][lane] = ea;
        sp[h][lane + 32] = eb;
        const float l = warp_sum(ea + eb);
        if (lane == 0) { part_m[slot * G + h] = m; part_l[slot * G + h] = l; }
    }
    __syncthreads();
    // values: thread t owns dimension t for all 12 heads.
    float acc[G];
#pragma unroll
    for (int h = 0; h < G; ++h) acc[h] = 0.0f;
    for (int c = 0; c < n_here; ++c) {
        if (srow[c] < 0) continue;   // masked above, weight 0
        float v;
        if constexpr (KV_MODE == 0) {
            v = __half2float(__ushort_as_half(p.v_pool[srow[c] * HD + t]));
        } else if constexpr (KV_MODE == 1) {
            const float sc = __half2float(__ushort_as_half(p.v_scale[srow[c] * (HD / KV_Q8_GROUP) + t / KV_Q8_GROUP]));
            v = (float) p.v_q[srow[c] * HD + t] * sc;
        } else {   // modes 2 and 3: V is rotated Q4_0 (kv_q4.hpp); the caller rotates the output back
            constexpr int bytes_per_head = (HD / QK4_0) * sizeof(block_q4_0);
            const int b = t / QK4_0;
            const int rem = t % QK4_0;
            const block_q4_0* blk = reinterpret_cast<const block_q4_0*>(p.v_q4 + srow[c] * bytes_per_head) + b;
            const float d = __half2float(__ushort_as_half(blk->d));
            const int j = rem < 16 ? rem : (rem - 16);
            const uint8_t byte = blk->qs[j];
            const int nibble = (rem < 16) ? ((byte & 0x0F) - 8) : ((byte >> 4) - 8);
            v = (float) nibble * d;
        }
#pragma unroll
        for (int h = 0; h < G; ++h) acc[h] = fmaf(sp[h][c], v, acc[h]);
    }
#pragma unroll
    for (int h = 0; h < G; ++h) part_acc[((size_t) slot * G + h) * HD + t] = acc[h];
}

__global__ void __launch_bounds__(HD) attn_merge_kernel(const float* __restrict__ part_acc,
                                                        const float* __restrict__ part_m,
                                                        const float* __restrict__ part_l, int n_chunks,
                                                        float* __restrict__ attn, long long scratch_stride = 0) {
    part_acc += (size_t) blockIdx.y * (size_t) scratch_stride;
    part_m += (size_t) blockIdx.y * (size_t) scratch_stride;
    part_l += (size_t) blockIdx.y * (size_t) scratch_stride;
    attn += (size_t) blockIdx.y * (size_t) gridDim.x * HD;
    const int h = blockIdx.x;                 // global query head
    const int kvh = h / G, hl = h % G;
    const int d = threadIdx.x;
    float M = -FLT_MAX;
    for (int c = 0; c < n_chunks; ++c) M = fmaxf(M, part_m[(kvh * n_chunks + c) * G + hl]);
    float L = 0.0f, acc = 0.0f;
    for (int c = 0; c < n_chunks; ++c) {
        const int slot = kvh * n_chunks + c;
        const float m = part_m[slot * G + hl];
        if (m == -FLT_MAX) continue;
        const float w = __expf(m - M);
        L = fmaf(part_l[slot * G + hl], w, L);
        acc = fmaf(part_acc[((size_t) slot * G + hl) * HD + d], w, acc);
    }
    attn[(size_t) h * HD + d] = L > 0.0f ? acc / L : 0.0f;
}

// ================================ glm5-next: MLA (absorbed) sparse attention, decode ==============================
// The same split-K flash-decode as attn_chunk_kernel (chunks of 64 selected cells -> per-chunk max / exp-sum / acc ->
// merge), for GLM-5.3-Flash's DSA: ONE 512-value latent per cell serves as K and V for all 64 query heads (MQA 64:1,
// llama.cpp b11461 glm5-next.cpp:917-950).  64 heads x 512 floats of q do not fit in shared memory, so a block takes
// a group of MGH heads (grid.y = head group); the latent rows are re-read per group from L2.  Pool: fp16 rows of 512
// at (page * 1 + 0) * page_size + slot (n_kv_heads = 1).  `scale` is the caller's (GLM: 1/sqrt(256), not 1/sqrt(512)).
constexpr int MHD = 512;            // latent dim
constexpr int MGH = 16;             // query heads per block
constexpr int MPL = MHD / 32;       // latent values per lane in the score pass (16)

// The latent row's formats: fp16 (512 x 2 bytes) or q4_0 (16 ggml blocks of 32: an fp16 scale + 16 bytes of nibbles,
// value i < 16 in the low nibble of qs[i], i >= 16 in the high nibble of qs[i - 16]; 288 bytes), 2026-10-08.
constexpr int MQ4_ROW = (MHD / QK4_0) * (int) sizeof(block_q4_0);
template <bool Q4>
__device__ __forceinline__ void mla_load8(const void* pool, long long row, int d0, float* out) {
    if constexpr (!Q4) {
        const uint4 raw = *reinterpret_cast<const uint4*>((const uint16_t*) pool + row * MHD + d0);
        const __half2* h2 = reinterpret_cast<const __half2*>(&raw);
#pragma unroll
        for (int j = 0; j < 4; ++j) { const float2 f = __half22float2(h2[j]); out[2 * j] = f.x; out[2 * j + 1] = f.y; }
    } else {
        const block_q4_0* blk = reinterpret_cast<const block_q4_0*>((const uint8_t*) pool + row * MQ4_ROW) + d0 / QK4_0;
        const float d = __half2float(__ushort_as_half(blk->d));
        const int rem = d0 % QK4_0;                       // 0, 8, 16 or 24
        const uint8_t* bytes = blk->qs + (rem & 15);
        if (rem < 16) {
#pragma unroll
            for (int k = 0; k < 8; ++k) out[k] = (float) ((int) (bytes[k] & 0x0F) - 8) * d;
        } else {
#pragma unroll
            for (int k = 0; k < 8; ++k) out[k] = (float) ((int) (bytes[k] >> 4) - 8) * d;
        }
    }
}
template <bool Q4>
__device__ __forceinline__ float mla_load1(const void* pool, long long row, int i) {
    if constexpr (!Q4) {
        return __half2float(__ushort_as_half(((const uint16_t*) pool)[row * MHD + i]));
    } else {
        const block_q4_0* blk = reinterpret_cast<const block_q4_0*>((const uint8_t*) pool + row * MQ4_ROW) + i / QK4_0;
        const int r = i % QK4_0;
        const int nib = r < 16 ? (blk->qs[r] & 0x0F) : (blk->qs[r - 16] >> 4);
        return (float) (nib - 8) * __half2float(__ushort_as_half(blk->d));
    }
}

template <bool Q4>
__global__ void __launch_bounds__(THREADS) mla_chunk_kernel(const float* __restrict__ q, const void* __restrict__ pool,
                                                            const int32_t* __restrict__ page_table,
                                                            const int32_t* __restrict__ ids,
                                                            const int32_t* __restrict__ step, int n_head, int page_size,
                                                            float scale, float* __restrict__ part_acc,
                                                            float* __restrict__ part_m, float* __restrict__ part_l,
                                                            int n_chunks, int cap, long long scratch_stride) {
    q += (size_t) blockIdx.z * (size_t) n_head * MHD;
    ids += (size_t) blockIdx.z * (size_t) cap;
    step += (size_t) blockIdx.z * kStepCount;
    part_acc += (size_t) blockIdx.z * (size_t) scratch_stride;
    part_m += (size_t) blockIdx.z * (size_t) scratch_stride;
    part_l += (size_t) blockIdx.z * (size_t) scratch_stride;
    __shared__ __align__(16) float sq[MGH][MHD];  // 32 KB: this group's query heads
    __shared__ float sp[MGH][CHUNK];              // scores, then probabilities
    __shared__ long long srow[CHUNK];
    const int n_ids = __ldg(step + kStepWidth);
    const int chunk = blockIdx.x, hg = blockIdx.y;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int c0 = chunk * CHUNK;
    const int n_here = min(CHUNK, n_ids - c0);
    const int h0 = hg * MGH;
    if (n_here <= 0) {
        if (t < MGH) { part_m[(size_t) chunk * n_head + h0 + t] = -FLT_MAX; part_l[(size_t) chunk * n_head + h0 + t] = 0.0f; }
        return;
    }
    for (int i = t; i < MGH * MHD; i += THREADS) sq[i / MHD][i % MHD] = q[(size_t) h0 * MHD + i];
    if (t < CHUNK) {
        long long r = -1;
        if (t < n_here) {
            const int cell = ids[c0 + t];
            const long long page = (long long) page_table[cell / page_size];
            if (page >= 0) r = page * page_size + (cell % page_size);
        }
        srow[t] = r;
    }
    __syncthreads();
    // scores: a warp per cell, each lane 16 of the 512 latent values (two 8-wide fp16 loads), used for MGH heads
    for (int c = warp; c < CHUNK; c += WARPS) {
        if (c >= n_here || srow[c] < 0) {
            if (lane < MGH) sp[lane][c] = -FLT_MAX;
            continue;
        }
        float k16[MPL];
#pragma unroll
        for (int half = 0; half < 2; ++half)   // dims lane*8 + [0,8) and 256 + lane*8 + [0,8)
            mla_load8<Q4>(pool, srow[c], half * 256 + lane * 8, k16 + half * 8);
#pragma unroll
        for (int h = 0; h < MGH; ++h) {
            float s = 0.f;
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const float4 qa = *reinterpret_cast<const float4*>(&sq[h][half * 256 + lane * 8]);
                const float4 qb = *reinterpret_cast<const float4*>(&sq[h][half * 256 + lane * 8 + 4]);
                const float* k8 = k16 + half * 8;
                s += k8[0] * qa.x + k8[1] * qa.y + k8[2] * qa.z + k8[3] * qa.w +
                     k8[4] * qb.x + k8[5] * qb.y + k8[6] * qb.z + k8[7] * qb.w;
            }
            s = warp_sum(s);
            if (lane == 0) sp[h][c] = s * scale;
        }
    }
    __syncthreads();
    for (int h = warp; h < MGH; h += WARPS) {
        const float a = sp[h][lane], b = sp[h][lane + 32];
        const float m = warp_max(fmaxf(a, b));
        const float ea = (lane < n_here && srow[lane] >= 0) ? __expf(a - m) : 0.0f;
        const float eb = (lane + 32 < n_here && srow[lane + 32] >= 0) ? __expf(b - m) : 0.0f;
        sp[h][lane] = ea;
        sp[h][lane + 32] = eb;
        const float l = warp_sum(ea + eb);
        if (lane == 0) { part_m[(size_t) chunk * n_head + h0 + h] = m; part_l[(size_t) chunk * n_head + h0 + h] = l; }
    }
    __syncthreads();
    // values (= the same latent): thread t owns dims t and t + 256 for the MGH heads
    float acc0[MGH], acc1[MGH];
#pragma unroll
    for (int h = 0; h < MGH; ++h) { acc0[h] = 0.0f; acc1[h] = 0.0f; }
    for (int c = 0; c < n_here; ++c) {
        if (srow[c] < 0) continue;
        const float v0 = mla_load1<Q4>(pool, srow[c], t);
        const float v1 = mla_load1<Q4>(pool, srow[c], 256 + t);
#pragma unroll
        for (int h = 0; h < MGH; ++h) { acc0[h] = fmaf(sp[h][c], v0, acc0[h]); acc1[h] = fmaf(sp[h][c], v1, acc1[h]); }
    }
#pragma unroll
    for (int h = 0; h < MGH; ++h) {
        float* a = part_acc + ((size_t) chunk * n_head + h0 + h) * MHD;
        a[t] = acc0[h];
        a[256 + t] = acc1[h];
    }
}

__global__ void __launch_bounds__(MHD) mla_merge_kernel(const float* __restrict__ part_acc, const float* __restrict__ part_m,
                                                        const float* __restrict__ part_l, int n_chunks, int n_head,
                                                        float* __restrict__ out, long long scratch_stride) {
    part_acc += (size_t) blockIdx.y * (size_t) scratch_stride;
    part_m += (size_t) blockIdx.y * (size_t) scratch_stride;
    part_l += (size_t) blockIdx.y * (size_t) scratch_stride;
    out += (size_t) blockIdx.y * (size_t) n_head * MHD;
    const int h = blockIdx.x, d = threadIdx.x;
    float M = -FLT_MAX;
    for (int c = 0; c < n_chunks; ++c) M = fmaxf(M, part_m[(size_t) c * n_head + h]);
    float L = 0.0f, acc = 0.0f;
    for (int c = 0; c < n_chunks; ++c) {
        const float m = part_m[(size_t) c * n_head + h];
        if (m == -FLT_MAX) continue;
        const float w = __expf(m - M);
        L = fmaf(part_l[(size_t) c * n_head + h], w, L);
        acc = fmaf(part_acc[((size_t) c * n_head + h) * MHD + d], w, acc);
    }
    out[(size_t) h * MHD + d] = L > 0.0f ? acc / L : 0.0f;
}

}  // namespace

uint64_t mla_decode_attn_scratch_floats(int64_t cap, int64_t n_head) {
    const int64_t chunks = (cap + CHUNK - 1) / CHUNK;
    return (uint64_t) chunks * (uint64_t) n_head * (MHD + 2) + 64;
}

namespace {
template <bool Q4>
void mla_decode_attn_any(const float* q_abs, const void* latent_pool, const int32_t* page_table, int page_size,
                         const int32_t* ids, const int32_t* steps, int64_t cap, int n_head, float scale,
                         float* scratch, float* out, int64_t n_q, void* stream) {
    if (n_q <= 0) return;
    if (n_head % MGH != 0 || cap <= 0 || !scratch || !ids || !steps || !page_table || !latent_pool || n_q > 65535) {
        std::fprintf(stderr, "mla_decode_attn_batch: unsupported geometry or missing buffers\n");
        std::exit(1);
    }
    const int n_chunks = (int) ((cap + CHUNK - 1) / CHUNK);
    const long long stride = (long long) mla_decode_attn_scratch_floats(cap, n_head);
    float* part_acc = scratch;
    float* part_m = scratch + (size_t) n_chunks * n_head * MHD;
    float* part_l = part_m + (size_t) n_chunks * n_head;
    cudaStream_t st = (cudaStream_t) stream;
    mla_chunk_kernel<Q4><<<dim3((unsigned) n_chunks, (unsigned) (n_head / MGH), (unsigned) n_q), THREADS, 0, st>>>(
        q_abs, latent_pool, page_table, ids, steps, n_head, page_size, scale, part_acc, part_m, part_l, n_chunks,
        (int) cap, stride);
    mla_merge_kernel<<<dim3((unsigned) n_head, (unsigned) n_q), MHD, 0, st>>>(part_acc, part_m, part_l, n_chunks, n_head,
                                                                            out, stride);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "mla_decode_attn_batch: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}
}  // namespace

void mla_decode_attn_batch(const float* q_abs, const uint16_t* latent_pool, const int32_t* page_table, int page_size,
                           const int32_t* ids, const int32_t* steps, int64_t cap, int n_head, float scale,
                           float* scratch, float* out, int64_t n_q, void* stream) {
    mla_decode_attn_any<false>(q_abs, latent_pool, page_table, page_size, ids, steps, cap, n_head, scale, scratch, out,
                               n_q, stream);
}

void mla_decode_attn_batch_q4(const float* q_abs, const uint8_t* latent_q4, const int32_t* page_table, int page_size,
                              const int32_t* ids, const int32_t* steps, int64_t cap, int n_head, float scale,
                              float* scratch, float* out, int64_t n_q, void* stream) {
    mla_decode_attn_any<true>(q_abs, latent_q4, page_table, page_size, ids, steps, cap, n_head, scale, scratch, out,
                              n_q, stream);
}

void qsa_decode_attn_batch(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
                           int64_t cap, const QsaShapes& s, float* scratch, float* attn, int64_t n_q, void* stream) {
    if (n_q <= 0) return;
    if (s.head_dim != HD || s.n_head != (int64_t) G * s.n_head_kv || cap <= 0 || !scratch || !ids || !steps ||
        !pools.page_table || n_q > 65535) {
        std::fprintf(stderr, "qsa_decode_attn_batch: unsupported geometry or missing buffers\n");
        std::exit(1);
    }
    const int kv_mode = pools.k_q4 != nullptr ? 2 : (pools.k_q != nullptr && pools.v_q4 != nullptr ? 3
                        : (pools.k_q != nullptr ? 1 : 0));
    const int n_chunks = (int) ((cap + CHUNK - 1) / CHUNK);
    // per query: [acc: n_chunks*n_head*HD][m: n_chunks*n_head][l: n_chunks*n_head], all offsets from one stride
    const long long stride = (long long) qsa_decode_attn_scratch_floats(cap, s);
    float* part_acc = scratch;
    float* part_m = scratch + (size_t) n_chunks * s.n_head * HD;
    float* part_l = part_m + (size_t) n_chunks * s.n_head;
    const float scale = 1.0f / sqrtf((float) HD);
    const dim3 grid((unsigned) n_chunks, (unsigned) s.n_head_kv, (unsigned) n_q);
    cudaStream_t st = (cudaStream_t) stream;
    if (kv_mode == 3)
        attn_chunk_kernel<3><<<grid, THREADS, 0, st>>>(q, pools, ids, steps, (int) s.n_head_kv, (int) s.page_size,
                                                        scale, part_acc, part_m, part_l, n_chunks, (int) cap, stride);
    else if (kv_mode == 2)
        attn_chunk_kernel<2><<<grid, THREADS, 0, st>>>(q, pools, ids, steps, (int) s.n_head_kv, (int) s.page_size,
                                                        scale, part_acc, part_m, part_l, n_chunks, (int) cap, stride);
    else if (kv_mode == 1)
        attn_chunk_kernel<1><<<grid, THREADS, 0, st>>>(q, pools, ids, steps, (int) s.n_head_kv, (int) s.page_size,
                                                        scale, part_acc, part_m, part_l, n_chunks, (int) cap, stride);
    else
        attn_chunk_kernel<0><<<grid, THREADS, 0, st>>>(q, pools, ids, steps, (int) s.n_head_kv, (int) s.page_size,
                                                        scale, part_acc, part_m, part_l, n_chunks, (int) cap, stride);
    attn_merge_kernel<<<dim3((unsigned) s.n_head, (unsigned) n_q), HD, 0, st>>>(part_acc, part_m, part_l, n_chunks,
                                                                                  attn, stride);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "qsa_decode_attn_batch: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

uint64_t qsa_decode_attn_scratch_floats(int64_t cap, const QsaShapes& s) {
    const int64_t chunks = (cap + CHUNK - 1) / CHUNK;
    return (uint64_t) chunks * (uint64_t) s.n_head * (HD + 2) + 64;
}

void qsa_decode_attn_step(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* step,
                          int64_t cap, const QsaShapes& s, float* scratch, float* attn, void* stream) {
    if (s.head_dim != HD || s.n_head != (int64_t) G * s.n_head_kv || cap <= 0 || !scratch || !ids || !step ||
        !pools.page_table) {
        std::fprintf(stderr, "qsa_decode_attn: unsupported geometry or missing buffers\n");
        std::exit(1);
    }
    const int kv_mode = pools.k_q4 != nullptr ? 2 : (pools.k_q != nullptr && pools.v_q4 != nullptr ? 3
                        : (pools.k_q != nullptr ? 1 : 0));
    if (kv_mode == 3 ? (!pools.k_scale || !pools.v_q4)
                     : (kv_mode == 2 ? (!pools.v_q4) : (kv_mode == 1 ? (!pools.v_q || !pools.k_scale || !pools.v_scale)
                                                                     : (!pools.k_pool || !pools.v_pool)))) {
        std::fprintf(stderr, "qsa_decode_attn: incomplete KV pools\n");
        std::exit(1);
    }
    const int n_chunks = (int) ((cap + CHUNK - 1) / CHUNK);
    float* part_acc = scratch;
    float* part_m = scratch + (size_t) n_chunks * s.n_head * HD;
    float* part_l = part_m + (size_t) n_chunks * s.n_head;
    const float scale = 1.0f / sqrtf((float) HD);
    const dim3 grid((unsigned) n_chunks, (unsigned) s.n_head_kv);
    cudaStream_t st = (cudaStream_t) stream;
    if (kv_mode == 3)
        attn_chunk_kernel<3><<<grid, THREADS, 0, st>>>(q, pools, ids, step, (int) s.n_head_kv, (int) s.page_size,
                                                        scale, part_acc, part_m, part_l, n_chunks);
    else if (kv_mode == 2)
        attn_chunk_kernel<2><<<grid, THREADS, 0, st>>>(q, pools, ids, step, (int) s.n_head_kv, (int) s.page_size,
                                                        scale, part_acc, part_m, part_l, n_chunks);
    else if (kv_mode == 1)
        attn_chunk_kernel<1><<<grid, THREADS, 0, st>>>(q, pools, ids, step, (int) s.n_head_kv, (int) s.page_size,
                                                        scale, part_acc, part_m, part_l, n_chunks);
    else
        attn_chunk_kernel<0><<<grid, THREADS, 0, st>>>(q, pools, ids, step, (int) s.n_head_kv, (int) s.page_size,
                                                        scale, part_acc, part_m, part_l, n_chunks);
    attn_merge_kernel<<<(unsigned) s.n_head, HD, 0, st>>>(part_acc, part_m, part_l, n_chunks, attn);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "qsa_decode_attn: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

}  // namespace strata::kernels
