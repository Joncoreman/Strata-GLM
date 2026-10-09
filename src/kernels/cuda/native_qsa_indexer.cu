// Adapted from llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d:
// src/models/qwen4exp.cpp; ggml/src/ggml-cuda/{set-rows.cu,norm.cu,rope.cu}.
// MIT License
// Copyright (c) 2023-2026 The ggml authors
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/mrope.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace strata::kernels {
namespace {
std::atomic<bool> enabled{false};
constexpr int D = 128, R = 4, ROT = 64, THREADS = 256;
__device__ float warp_sum(float x) {
#pragma unroll
    for (int offset = 16; offset; offset >>= 1)
        x += __shfl_xor_sync(0xffffffffu, x, offset);
    return x;
}
// TAB (#280, STRATA_ROPE_TABLE=1): the angles from the session's float64 table.  The host launches <false> whenever
// no table applies - the default - so the default kernel is 0.1.31's code exactly (the table read is not in it;
// with it merely skipped at run time, the compiled default path changed its results).
template <bool TAB>
__global__ void append(const float* __restrict__ raw, const int32_t* __restrict__ pos_dev,
                        int pos_stride, int n_steps,
                        int pos_base, const float* __restrict__ gamma, float epsilon,
                        float* __restrict__ tail, float* __restrict__ dead,
                        float* __restrict__ pooled, int32_t* __restrict__ block_pos,
                        int max_cells, float theta_scale, float freq_scale, float corr_low,
                        float corr_high, float ext_factor, float mscale,
                        const int32_t* __restrict__ mtab, RopeTab rt) {
    __shared__ float values[D];
    __shared__ float partials[32];
    const int d = threadIdx.x;
    for (int step_idx = 0; step_idx < n_steps; ++step_idx) {
        const int pos = pos_dev[(std::size_t) step_idx * pos_stride];
        if (pos >= 0 && pos < max_cells) {
            const float* raw_step = raw + (std::size_t) step_idx * D;
            const int slot = pos % R;
            float incoming = 0.0f;
            if (d < D) {
                // SET_ROWS stores F16; GET_ROWS expands those exact values to F32.
                incoming = __half2float(__float2half_rn(raw_step[d]));
                if (slot < R - 1) tail[slot * D + d] = incoming;
            }
            if (pos == 0 || slot == R - 1) {
                float mean = 0.0f;
                if (d < D) {
                    // The spare's four gather indices all name cell zero. Completed blocks
                    // use chronological slices; each graph ADD materializes an F32 sum.
                    float sum = pos == 0 ? incoming : tail[d];
#pragma unroll
                    for (int j = 1; j < R; ++j)
                        sum = __fadd_rn(sum, pos == 0 || j == R - 1 ? incoming : tail[j * D + d]);
                    mean = __fmaf_rn(0.25f, sum, 0.0f); // SCALE includes a zero bias.
                }
                float square_sum = 0.0f;
                if (d < D) square_sum += mean * mean;
                square_sum = warp_sum(square_sum);
                const int lane = d % 32;
                if (lane == 0) partials[d / 32] = square_sum;
                __syncthreads();
                square_sum = lane < THREADS / 32 ? partials[lane] : 0.0f;
                square_sum = warp_sum(square_sum);
                const float scale = rsqrtf(square_sum / D + epsilon);
                if (d < D) values[d] = scale * mean * gamma[d];
                __syncthreads();
                if (d < D) {
                    const int b = pos / R;
                    const int rope_pos = pos == 0 ? 0 : pos_base + R * b;
                    float y = values[d];
                    if (d < ROT) {
                        const int pair = d % (ROT / 2);
                        // The spare (pos == 0) keeps its zero angle; under YaRN the mscale magnitude still rides in
                        // through cos(0) - which is exactly what the queries are scaled by, so the top-k is unmoved.
                        float c, s;
                        if (!(TAB && rope_tab_cs(rt, pos == 0 ? 0 : mrope_pos(mtab, rope_pos, pair), pair, c, s))) {
                            const float theta_extrap = (pos == 0 ? 0 : mrope_pos(mtab, rope_pos, pair)) * powf(theta_scale, float(pair));
                            rope_scaled_angle(theta_extrap, freq_scale, corr_low, corr_high, ext_factor, mscale, pair, c, s);
                        }
                        const float a = values[pair], z = values[pair + ROT / 2];
                        y = d < ROT / 2 ? a * c - z * s : a * s + z * c;
                    }
                    pooled[std::size_t(b) * D + d] = y;
                    if (pos == 0) dead[d] = y;
                    else pooled[std::size_t(b + 1) * D + d] = dead[d];
                    if (d == 0 && pos != 0) *block_pos = rope_pos;
                }
            }
        }
        if (n_steps > 1) __syncthreads();
    }
}
// ---- C-2: the batched append.  The pooled key of a completed block b (its last cell pos = 4b+3), computed with
// the single append's arithmetic: keys rounded through F16, summed tail[0]+tail[1]+tail[2]+incoming in that order,
// the RMS norm, gamma, the rotation at pos_base + 4b.  `k(j)` is the key of the block's cell j: a row of this batch
// (cell >= p0) or the tail the previous batch left (cell < p0).  Only the batch's last completed block writes the
// spare after it (every earlier one's is overwritten by the next block in order).
template <bool TAB>
__device__ __forceinline__ float pooled_value(const float* values, int d, int rope_pos, float theta_scale,
                                              float freq_scale, float corr_low, float corr_high,
                                              float ext_factor, float mscale, const int32_t* mtab, bool zero_pos,
                                              const RopeTab& rt) {
    float y = values[d];
    if (d < ROT) {
        const int pair = d % (ROT / 2);
        // The same rotation the single append applies, rope_scaling.hpp's helper shared: under none this is
        // today's cosf/sinf; under linear/YaRN the pooled keys move with the cache's K, and the spare's zero
        // angle carries the mscale magnitude through cos(0) exactly as the single append's does.
        float c, s;
        if (!(TAB && rope_tab_cs(rt, zero_pos ? 0 : mrope_pos(mtab, rope_pos, pair), pair, c, s))) {
            const float theta_extrap = (zero_pos ? 0 : mrope_pos(mtab, rope_pos, pair)) * powf(theta_scale, float(pair));
            rope_scaled_angle(theta_extrap, freq_scale, corr_low, corr_high, ext_factor, mscale, pair, c, s);
        }
        const float a = values[pair], z = values[pair + ROT / 2];
        y = d < ROT / 2 ? a * c - z * s : a * s + z * c;
    }
    return y;
}
__device__ __forceinline__ float norm_scale(float mean, float* partials, int d) {
    float square_sum = d < D ? mean * mean : 0.0f;
    square_sum = warp_sum(square_sum);
    const int lane = d % 32;
    if (lane == 0) partials[d / 32] = square_sum;
    __syncthreads();
    square_sum = lane < THREADS / 32 ? partials[lane] : 0.0f;
    square_sum = warp_sum(square_sum);
    return square_sum;
}
// cell 0 of a sequence: the spare (every gather index names cell 0), written to pooled[0] and dead
template <bool TAB>
__global__ void append_first(const float* __restrict__ raw, const float* __restrict__ gamma, float epsilon,
                             float* __restrict__ dead, float* __restrict__ pooled, float theta_scale,
                             float freq_scale, float corr_low, float corr_high, float ext_factor, float mscale,
                             const int32_t* __restrict__ mtab, RopeTab rt) {
    const int d = threadIdx.x;
    __shared__ float values[D];
    __shared__ float partials[32];
    float mean = 0.0f;
    if (d < D) {
        const float incoming = __half2float(__float2half_rn(raw[d]));
        float sum = incoming;
#pragma unroll
        for (int j = 1; j < R; ++j) sum = __fadd_rn(sum, incoming);
        mean = __fmaf_rn(0.25f, sum, 0.0f);
    }
    const float square_sum = norm_scale(mean, partials, d);
    const float scale = rsqrtf(square_sum / D + epsilon);
    if (d < D) values[d] = scale * mean * gamma[d];
    __syncthreads();
    if (d >= D) return;
    const float y = pooled_value<TAB>(values, d, 0, theta_scale, freq_scale, corr_low, corr_high, ext_factor, mscale,
                                 mtab, true, rt);
    pooled[d] = y;
    dead[d] = y;
}
template <bool TAB>
__global__ void append_blocks(const float* __restrict__ raw, int64_t n, int64_t p0, int pos_base,
                              const float* __restrict__ gamma, float epsilon, const float* __restrict__ tail,
                              const float* __restrict__ dead, float* __restrict__ pooled, int32_t* __restrict__ block_pos,
                              int64_t first_block, int64_t last_block, float theta_scale, float freq_scale,
                              float corr_low, float corr_high, float ext_factor, float mscale,
                              const int32_t* __restrict__ mtab, RopeTab rt) {
    const int64_t b = first_block + blockIdx.x;
    const int d = threadIdx.x;
    __shared__ float values[D];
    __shared__ float partials[32];
    float mean = 0.0f;
    if (d < D) {
        auto key = [&](int j) -> float {
            const int64_t cell = b * R + j;
            return cell >= p0 ? __half2float(__float2half_rn(raw[(cell - p0) * D + d])) : tail[j * D + d];
        };
        float sum = key(0);
#pragma unroll
        for (int j = 1; j < R; ++j) sum = __fadd_rn(sum, key(j));
        mean = __fmaf_rn(0.25f, sum, 0.0f);
    }
    const float square_sum = norm_scale(mean, partials, d);
    const float scale = rsqrtf(square_sum / D + epsilon);
    if (d < D) values[d] = scale * mean * gamma[d];
    __syncthreads();
    if (d >= D) return;
    const int rope_pos = pos_base + R * (int) b;
    pooled[std::size_t(b) * D + d] = pooled_value<TAB>(values, d, rope_pos, theta_scale, freq_scale, corr_low, corr_high,
                                                  ext_factor, mscale, mtab, false, rt);
    if (b == last_block) {
        pooled[std::size_t(b + 1) * D + d] = dead[d];
        if (d == 0) *block_pos = rope_pos;
    }
}
// the tail after the batch: slot s holds the key of the batch's last cell with cell % 4 == s (s < 3), if any
__global__ void append_tail(const float* __restrict__ raw, int64_t n, int64_t p0, float* __restrict__ tail) {
    const int s = blockIdx.x, d = threadIdx.x;
    if (d >= D) return;
    const int64_t last = p0 + n - 1;
    int64_t cell = last - ((last % R) - s + R) % R;       // the last cell <= last with cell % 4 == s
    if (cell < p0) return;
    tail[s * D + d] = __half2float(__float2half_rn(raw[(cell - p0) * D + d]));
}
struct Span { const void* p; std::size_t n; };
void validate(Span s) {
    const auto p = reinterpret_cast<std::uintptr_t>(s.p);
    if (!p || p % 4 || s.n > UINTPTR_MAX - p)
        throw std::invalid_argument("native QSA indexer requires aligned bounded spans");
}
bool overlaps(Span a, Span b) {
    const auto x = reinterpret_cast<std::uintptr_t>(a.p), y = reinterpret_cast<std::uintptr_t>(b.p);
    return x < y + b.n && y < x + a.n;
}
} // namespace

// ---- glm5-next k-pool (see the header) ------------------------------------------------------------------------
namespace {
constexpr int KD = 128;   // indexer head size
constexpr int KP = 4;     // cells per pool
constexpr int KH = 32;    // indexer heads

__global__ void __launch_bounds__(KD) glm5_kpool_append_kernel(const float* __restrict__ raw_k, const float* __restrict__ gate,
                                                               const int32_t* __restrict__ pos, const float* __restrict__ kw,
                                                               const float* __restrict__ kb, const float* __restrict__ ape,
                                                               float eps, float* __restrict__ tail,
                                                               __nv_bfloat16* __restrict__ pooled, int64_t max_cells,
                                                               float* __restrict__ key_out) {
    __shared__ float red[KD / 32];
    const int d = threadIdx.x, lane = d & 31, w = d >> 5;
    const int cell = *pos;
    if (cell < 0 || cell >= max_cells) return;
    // LayerNorm (ggml_norm: mean, then mean of squares of the centred values), then * weight + bias
    const float x = raw_k[d];
    float s = x;
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if (lane == 0) red[w] = s;
    __syncthreads();
    const float mean = (red[0] + red[1] + red[2] + red[3]) / (float) KD;
    __syncthreads();
    const float xc = x - mean;
    float v = xc * xc;
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (lane == 0) red[w] = v;
    __syncthreads();
    const float var = (red[0] + red[1] + red[2] + red[3]) / (float) KD;
    const float key = xc * (1.0f / sqrtf(var + eps)) * kw[d] + kb[d];
    const float g = gate[d];
    if (key_out) key_out[d] = key;
    const int slot = cell % KP;
    if (slot < KP - 1) {
        tail[slot * 2 * KD + d] = key;
        tail[slot * 2 * KD + KD + d] = g;
        return;
    }
    // the block completes: softmax over the 4 members per channel (ggml_soft_max: max, exp, sum, scale)
    float lg[KP], kj[KP];
#pragma unroll
    for (int j = 0; j < KP - 1; ++j) { kj[j] = tail[j * 2 * KD + d]; lg[j] = tail[j * 2 * KD + KD + d] + ape[j * KD + d]; }
    kj[KP - 1] = key; lg[KP - 1] = g + ape[(KP - 1) * KD + d];
    float mx = lg[0];
#pragma unroll
    for (int j = 1; j < KP; ++j) mx = fmaxf(mx, lg[j]);
    float sum = 0.f;
#pragma unroll
    for (int j = 0; j < KP; ++j) { lg[j] = expf(lg[j] - mx); sum += lg[j]; }
    const float inv = 1.0f / sum;
    float p = 0.f;
#pragma unroll
    for (int j = 0; j < KP; ++j) p += (lg[j] * inv) * kj[j];   // ggml_sum_rows(probs * pk), j ascending
    pooled[(size_t) (cell / KP) * KD + d] = __float2bfloat16_rn(p);   // BF16 (2026-10-09), as Infernix keeps its pooled keys
}

// one warp per block: each lane holds 4 of the 128 pooled values; the query heads come from shared memory
__global__ void __launch_bounds__(256) glm5_kpool_scores_kernel(const float* __restrict__ q, const float* __restrict__ w,
                                                                 const __nv_bfloat16* __restrict__ pooled,
                                                                 const int32_t* __restrict__ steps, int64_t max_blocks,
                                                                 float* __restrict__ scores) {
    __shared__ float sq[KH][KD];
    __shared__ float sw[KH];
    const int qi = blockIdx.y;
    q += (size_t) qi * KH * KD;
    w += (size_t) qi * KH;
    scores += (size_t) qi * max_blocks;
    const int n_bid = steps[(size_t) qi * kStepCount + kStepNBid];
    // a CTA wholly past the filled blocks leaves before loading the query: with --max-context 262144 the grid is
    // 8,193 CTAs per token while a short context fills a few hundred (the same scores; the rest are never read)
    if ((int64_t) blockIdx.x * (blockDim.x / 32) > n_bid) return;
    for (int i = threadIdx.x; i < KH * KD; i += blockDim.x) sq[i / KD][i % KD] = q[i];
    if (threadIdx.x < KH) sw[threadIdx.x] = w[threadIdx.x];
    __syncthreads();
    const int lane = threadIdx.x & 31;
    const int b = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);
    if (b > n_bid || b >= max_blocks) return;
    if (b == n_bid) { if (lane == 0) scores[b] = kGlm5TailScore; return; }
    float4 k4;
    {
        const uint2 raw = *reinterpret_cast<const uint2*>(pooled + (size_t) b * KD + lane * 4);   // 4 BF16
        const float2 lo = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.x));
        const float2 hi = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.y));
        k4 = make_float4(lo.x, lo.y, hi.x, hi.y);
    }
    float acc = 0.f;
#pragma unroll 4
    for (int h = 0; h < KH; ++h) {
        const float4 q4 = *reinterpret_cast<const float4*>(&sq[h][lane * 4]);
        float s = k4.x * q4.x + k4.y * q4.y + k4.z * q4.z + k4.w * q4.w;
        for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
        acc = fmaf(fmaxf(s, 0.f), sw[h], acc);
    }
    if (lane == 0) scores[b] = acc;
}
}  // namespace

void glm5_kpool_append(const float* raw_k, const float* gate, const int32_t* relative_pos_device, const float* k_norm,
                       const float* k_norm_b, const float* ape, float epsilon, const Glm5KpoolBuffers& b,
                       int64_t max_cells, void* stream, float* key_out) {
    if (!raw_k || !gate || !relative_pos_device || !k_norm || !k_norm_b || !ape || !b.tail || !b.pooled || !stream)
        throw std::invalid_argument("glm5_kpool_append: missing buffer or stream");
    glm5_kpool_append_kernel<<<1, KD, 0, (cudaStream_t) stream>>>(raw_k, gate, relative_pos_device, k_norm, k_norm_b,
                                                                  ape, epsilon, b.tail, (__nv_bfloat16*) b.pooled,
                                                                  max_cells, key_out);
}

void glm5_kpool_scores(const float* q, const float* w, const uint16_t* pooled, const int32_t* steps, int64_t nq,
                       int64_t max_blocks, float* scores, void* stream, int64_t active_blocks) {
    if (!q || !w || !pooled || !steps || !scores || nq <= 0 || max_blocks <= 0 || !stream)
        throw std::invalid_argument("glm5_kpool_scores: missing buffer or stream");
    const int64_t reach = active_blocks > 0 && active_blocks < max_blocks ? active_blocks : max_blocks;
    const dim3 grid((unsigned) ((reach + 7) / 8), (unsigned) nq);
    glm5_kpool_scores_kernel<<<grid, 256, 0, (cudaStream_t) stream>>>(q, w, (const __nv_bfloat16*) pooled, steps,
                                                                       max_blocks, scores);
}

void native_qsa_indexer_set_enabled(bool value) { enabled.store(value, std::memory_order_relaxed); }
bool native_qsa_indexer_enabled() { return enabled.load(std::memory_order_relaxed); }
void native_qsa_indexer_append_steps(const float* raw, const int32_t* relative_pos_device,
                                     int pos_stride, int n_steps, int32_t pos_base,
                                     const float* gamma, float epsilon, const QsaIndexerBuffers& b,
                                     const QsaShapes& s, int64_t max_cells, const RopeScaling& scaling, void* stream) {
    if (n_steps <= 0) return;
    if (!stream || s.idx_dim != D || s.idx_block != R || s.n_rot != ROT ||
        max_cells < 1 || max_cells > INT32_MAX || pos_base < 0 || pos_base % R ||
        int64_t(pos_base) + max_cells > INT32_MAX || !std::isfinite(epsilon) || epsilon <= 0.0f ||
        rope_scaling_invalid(scaling) != nullptr)
        throw std::invalid_argument("native QSA indexer requires fixed geometry, aligned position base, positive capacity/epsilon, valid frequency/scaling and explicit stream");
    const std::size_t pos_span = (n_steps == 1 || pos_stride <= 0) ? 4 : (std::size_t(n_steps - 1) * pos_stride + 1) * 4;
    const Span spans[] = {{raw,std::size_t(n_steps)*D*4},{relative_pos_device,pos_span},{gamma,D*4},{b.tail,(R-1)*D*4},
        {b.dead,D*4},{b.pooled,std::size_t(max_cells/R+1)*D*4},{b.block_pos,4}};
    for (const auto& span : spans) validate(span);
    for (int i = 0; i < 7; ++i) for (int j = i + 1; j < 7; ++j)
        if (overlaps(spans[i], spans[j])) throw std::invalid_argument("native QSA indexer buffers overlap");
    const float theta_scale = powf((float) scaling.freq_base, -2.0f / ROT);
    const RopeKernelArgs k = scaling.kernel_args(ROT);   // none: the identity constants
    const RopeTab rt = rope_table_for(scaling);
    if (rt.cos != nullptr)
        append<true><<<1,THREADS,0,static_cast<cudaStream_t>(stream)>>>(raw,relative_pos_device,pos_stride,n_steps,pos_base,gamma,epsilon,
            b.tail,b.dead,b.pooled,b.block_pos,int(max_cells),theta_scale,k.freq_scale,
            k.corr_low,k.corr_high,k.ext_factor,k.attn_factor,mrope_table(),rt);
    else
        append<false><<<1,THREADS,0,static_cast<cudaStream_t>(stream)>>>(raw,relative_pos_device,pos_stride,n_steps,pos_base,gamma,epsilon,
            b.tail,b.dead,b.pooled,b.block_pos,int(max_cells),theta_scale,k.freq_scale,
            k.corr_low,k.corr_high,k.ext_factor,k.attn_factor,mrope_table(),rt);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

void native_qsa_indexer_append(const float* raw, const int32_t* relative_pos_device, int32_t pos_base,
                               const float* gamma, float epsilon, const QsaIndexerBuffers& b,
                               const QsaShapes& s, int64_t max_cells, const RopeScaling& scaling, void* stream) {
    native_qsa_indexer_append_steps(raw, relative_pos_device, 0, 1, pos_base, gamma, epsilon, b, s, max_cells, scaling, stream);
}
void native_qsa_indexer_append_batch(const float* raw, int64_t n, int64_t p0, int32_t pos_base, const float* gamma,
                                     float epsilon, const QsaIndexerBuffers& b, const QsaShapes& s, int64_t max_cells,
                                     const RopeScaling& scaling, void* stream) {
    if (n <= 0) return;
    if (!stream || s.idx_dim != D || s.idx_block != R || s.n_rot != ROT || p0 < 0 || p0 + n > max_cells ||
        max_cells > INT32_MAX || pos_base < 0 || pos_base % R || int64_t(pos_base) + max_cells > INT32_MAX ||
        !std::isfinite(epsilon) || epsilon <= 0.0f || rope_scaling_invalid(scaling) != nullptr)
        throw std::invalid_argument("native QSA indexer (batch): bad geometry, positions, parameters or scaling");
    const float theta_scale = powf((float) scaling.freq_base, -2.0f / ROT);
    const RopeKernelArgs k = scaling.kernel_args(ROT);   // none: the identity constants
    const float fsf = k.freq_scale, cl = k.corr_low, ch = k.corr_high, ef = k.ext_factor, ms = k.attn_factor;
    const cudaStream_t st = static_cast<cudaStream_t>(stream);
    const int32_t* mtab = mrope_table();
    const RopeTab rt = rope_table_for(scaling);
    if (p0 == 0) {
        if (rt.cos != nullptr)
            append_first<true><<<1, THREADS, 0, st>>>(raw, gamma, epsilon, b.dead, b.pooled, theta_scale,
                                                      fsf, cl, ch, ef, ms, mtab, rt);
        else
            append_first<false><<<1, THREADS, 0, st>>>(raw, gamma, epsilon, b.dead, b.pooled, theta_scale,
                                                       fsf, cl, ch, ef, ms, mtab, rt);
    }
    // completed blocks: those whose last cell (4b+3) lies in [p0, p0 + n)
    const int64_t first = p0 <= R - 1 ? 0 : (p0 - (R - 1) + R - 1) / R;       // the smallest b with 4b+3 >= p0
    const int64_t hi = p0 + n - 1 >= R - 1 ? (p0 + n - 1 - (R - 1)) / R : -1;   // the largest b with 4b+3 <= p0+n-1
    if (hi >= first && rt.cos != nullptr)
        append_blocks<true><<<(unsigned) (hi - first + 1), THREADS, 0, st>>>(raw, n, p0, pos_base, gamma, epsilon,
            b.tail, b.dead, b.pooled, b.block_pos, first, hi, theta_scale, fsf, cl, ch, ef, ms, mtab, rt);
    else if (hi >= first)
        append_blocks<false><<<(unsigned) (hi - first + 1), THREADS, 0, st>>>(raw, n, p0, pos_base, gamma, epsilon,
            b.tail, b.dead, b.pooled, b.block_pos, first, hi, theta_scale, fsf, cl, ch, ef, ms, mtab, rt);
    append_tail<<<R - 1, D, 0, st>>>(raw, n, p0, b.tail);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
} // namespace strata::kernels
