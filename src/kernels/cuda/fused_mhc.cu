// src/kernels/cuda/fused_mhc.cu - see include/strata/kernels/fused_gr.hpp (glm5-next mHC section).
//
// GLM-5.3-Flash's hyper-connection (mHC) in the shape of Strata's GR read (fused_gr.cu): the previous half's write
// folded into the read, the weights read once for every token of a verify window, and the projection split over
// many blocks that write UNSCALED partial dots (the v3 idea: rms is one scalar per token here, applied afterwards).
//
//   mhc_down (grid SL slices of the 4*n_embd vector):
//       R'   = apply ? out_prev*post_prev[d] + sum_s comb_prev[d,s]*R[s]  : R      (computed on the fly, not stored)
//       part[t][slice][r] = hc_fn[r, slice] . R'[slice]      ss[t][slice] = |R'[slice]|^2
//   mhc_up (one block per token):
//       mixes = (sum_slices part) / rms(R'),  pre/post/comb (Sinkhorn) as llama.cpp b11461 glm5-next.cpp:445-501
//       R_out <- R' (when apply),  block = sum_h pre[h]*R'[h],  y = rms_norm(block) * w_norm   (attn_norm / ffn_norm)
//
// R' cannot be written by mhc_down (comb mixes the four streams: another block may still read the old R), so
// mhc_up writes it; it owns its token's columns, so R_out == R is fine there.
// Checked against tools/glm5_ref (which matches the llama.cpp dump) by tests/cuda/glm5_mhc_test.cu.
#include "strata/kernels/fused_gr.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace strata::kernels {
namespace {

constexpr int HC = 4;
constexpr int MIX = (2 + HC) * HC;          // 24
constexpr int QK = 32;                      // Q8_0 block
constexpr int SLICE = 1024;                 // floats of R' per down block (a quarter of one 4096 stream)
constexpr int DOWN_THREADS = 256;           // 8 warps x 3 rows = 24 rows
constexpr int UP_THREADS = 512;

struct Q8_0 { __half d; int8_t qs[QK]; };
static_assert(sizeof(Q8_0) == 34, "Q8_0 block");

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float sigm(float x) { return 1.f / (1.f + expf(-x)); }

// R' for stream d, column i of token t.
__device__ __forceinline__ float fold(const MhcMulti& m, int t, int d, int i) {
    const float* R = m.R[t];
    if (!m.apply) return R[d * m.n_embd + i];
    const float* c = m.comb_prev + t * HC * HC + d * HC;
    float v = m.out_prev[t][i] * m.post_prev[t * HC + d];
#pragma unroll
    for (int s = 0; s < HC; ++s) v += c[s] * R[s * m.n_embd + i];
    return v;
}

__global__ void __launch_bounds__(DOWN_THREADS) mhc_down_kernel(MhcMulti m) {
    __shared__ float xs[kFusedGrMaxT][SLICE];
    __shared__ float red[kFusedGrMaxT][DOWN_THREADS / 32];
    const int slice = blockIdx.x;
    const int base = slice * SLICE;          // offset in the 4*n_embd vector
    const int d = base / m.n_embd, i0 = base % m.n_embd;
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;

    for (int t = 0; t < m.n_tok; ++t) {
        float ss = 0.f;
        for (int j = threadIdx.x; j < SLICE; j += DOWN_THREADS) {
            const float v = fold(m, t, d, i0 + j);
            xs[t][j] = v;
            ss += v * v;
        }
        ss = warp_sum(ss);
        if (lane == 0) red[t][wid] = ss;
    }
    __syncthreads();
    if (threadIdx.x < m.n_tok) {
        float s = 0.f;
        for (int w = 0; w < DOWN_THREADS / 32; ++w) s += red[threadIdx.x][w];
        m.ss_part[threadIdx.x * m.n_slices + slice] = s;
    }

    // 3 rows per warp; lane = one Q8_0 block of the slice (1024 / 32 = 32 blocks); each weight block read once
    // and used for every token.
    const int nb_row = HC * m.n_embd / QK;
    for (int r = wid; r < MIX; r += DOWN_THREADS / 32) {
        const Q8_0 blk = static_cast<const Q8_0*>(m.fn)[(size_t)r * nb_row + base / QK + lane];
        const float sc = __half2float(blk.d);
        for (int t = 0; t < m.n_tok; ++t) {
            const float* x = xs[t] + lane * QK;
            float acc = 0.f;
#pragma unroll
            for (int k = 0; k < QK; ++k) acc = fmaf((float)blk.qs[k], x[k], acc);
            acc = warp_sum(acc * sc);
            if (lane == 0) m.part[((size_t)t * m.n_slices + slice) * MIX + r] = acc;
        }
    }
}

__global__ void __launch_bounds__(UP_THREADS) mhc_up_kernel(MhcMulti m) {
    __shared__ float coef[HC + HC + HC * HC];   // pre, post, comb[d*HC + s]
    __shared__ float red[UP_THREADS / 32];
    __shared__ float inv_y;
    const int t = blockIdx.x;
    const int n = m.n_embd;
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;

    __shared__ float mixes[MIX + 1];   // [MIX] = sum of squares
    // the slice partials: one thread per row (and one for the squares), each summed in slice order
    if (threadIdx.x <= MIX) {
        float v = 0.f;
        if (threadIdx.x < MIX)
            for (int s = 0; s < m.n_slices; ++s) v += m.part[((size_t)t * m.n_slices + s) * MIX + threadIdx.x];
        else
            for (int s = 0; s < m.n_slices; ++s) v += m.ss_part[t * m.n_slices + s];
        mixes[threadIdx.x] = v;
    }
    __syncthreads();
    if (threadIdx.x < MIX) {
        mixes[threadIdx.x] *= rsqrtf(mixes[MIX] / (float)(HC * n) + m.rms_eps);
        if (m.mixes_out) m.mixes_out[t * MIX + threadIdx.x] = mixes[threadIdx.x];
    }
    __syncthreads();
    // pre/post and the Sinkhorn on warp 0: lane k < 16 holds comb element k = dst + 4*src (llama.cpp's layout).
    // Every sum is taken in llama.cpp's order (eps first, then ascending index) by reading the 4 lanes in turn.
    if (wid == 0) {
        if (lane < HC) {
            coef[lane] = sigm(mixes[lane] * m.scale[0] + m.base[lane]) + m.hc_eps;
            coef[HC + lane] = 2.f * sigm(mixes[HC + lane] * m.scale[1] + m.base[HC + lane]);
        }
        const int k = lane & (HC * HC - 1), d = k % HC, s = k / HC;
        float c = mixes[2 * HC + k] * m.scale[2] + m.base[2 * HC + k];
        float mx = -3.402823466e+38f;
#pragma unroll
        for (int j = 0; j < HC; ++j) mx = fmaxf(mx, __shfl_sync(0xffffffffu, c, s * HC + j));
        c = expf(c - mx);
        float sum = 0.f;
#pragma unroll
        for (int j = 0; j < HC; ++j) sum += __shfl_sync(0xffffffffu, c, s * HC + j);   // softmax over dst
        c = c * (1.f / sum) + m.hc_eps;
        for (int it = 0; it < m.iters; ++it) {
            if (it > 0) {   // rows: per src, over dst
                float r = m.hc_eps;
#pragma unroll
                for (int j = 0; j < HC; ++j) r += __shfl_sync(0xffffffffu, c, s * HC + j);
                c *= 1.f / r;
            }
            float q = m.hc_eps;   // columns: per dst, over src
#pragma unroll
            for (int j = 0; j < HC; ++j) q += __shfl_sync(0xffffffffu, c, d + HC * j);
            c *= 1.f / q;
        }
        if (lane < HC * HC) coef[2 * HC + d * HC + s] = c;
    }
    __syncthreads();

    float* y = m.y[t];
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += UP_THREADS) {
        float r[HC];
#pragma unroll
        for (int d = 0; d < HC; ++d) r[d] = fold(m, t, d, i);
        if (m.apply) {   // this thread owns column i of every stream: all four were read above
#pragma unroll
            for (int d = 0; d < HC; ++d) m.R_out[t][d * n + i] = r[d];
        }
        float b = 0.f;
#pragma unroll
        for (int h = 0; h < HC; ++h) b = fmaf(coef[h], r[h], b);
        y[i] = b;
        ss += b * b;
    }
    ss = warp_sum(ss);
    if (lane == 0) red[wid] = ss;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.f;
        for (int w = 0; w < UP_THREADS / 32; ++w) s += red[w];
        inv_y = rsqrtf(s / (float)n + m.rms_eps);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += UP_THREADS) y[i] = y[i] * inv_y * m.w_norm[i];
    if (threadIdx.x < HC) m.post_out[t * HC + threadIdx.x] = coef[HC + threadIdx.x];
    if (threadIdx.x < HC * HC) m.comb_out[t * HC * HC + threadIdx.x] = coef[2 * HC + threadIdx.x];
    if (m.pre_out && threadIdx.x < HC) m.pre_out[t * HC + threadIdx.x] = coef[threadIdx.x];
}

// Last layer: the plain write (nothing follows to fold it into).  res[t] (in place): [HC][n_embd].
__global__ void mhc_write_kernel(MhcMulti m) {
    const int t = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m.n_embd) return;
    float r[HC];
#pragma unroll
    for (int d = 0; d < HC; ++d) r[d] = fold(m, t, d, i);
#pragma unroll
    for (int d = 0; d < HC; ++d) m.R_out[t][d * m.n_embd + i] = r[d];
}

__global__ void mhc_exit_mean_kernel(MhcMulti m) {
    const int t = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m.n_embd) return;
    const float* R = m.R[t];
    float v = 0.f;
#pragma unroll
    for (int s = 0; s < HC; ++s) v += R[s * m.n_embd + i];
    m.y[t][i] = v * (1.f / HC);
}

}  // namespace

bool fused_mhc_supported(int64_t n_embd, int64_t hc, int64_t mix_rows) {
    return hc == HC && mix_rows == MIX && n_embd > 0 && (HC * n_embd) % SLICE == 0 && n_embd % SLICE == 0;
}

int fused_mhc_slices(int64_t n_embd) { return (int)(HC * n_embd / SLICE); }

void fused_mhc_read_multi(const MhcMulti& m, void* stream) {
    cudaStream_t st = (cudaStream_t)stream;
    mhc_down_kernel<<<m.n_slices, DOWN_THREADS, 0, st>>>(m);
    mhc_up_kernel<<<m.n_tok, UP_THREADS, 0, st>>>(m);
}

void fused_mhc_write_multi(const MhcMulti& m, void* stream) {
    dim3 g((m.n_embd + 255) / 256, m.n_tok);
    mhc_write_kernel<<<g, 256, 0, (cudaStream_t)stream>>>(m);
}

void fused_mhc_exit_mean(const MhcMulti& m, void* stream) {
    dim3 g((m.n_embd + 255) / 256, m.n_tok);
    mhc_exit_mean_kernel<<<g, 256, 0, (cudaStream_t)stream>>>(m);
}

}  // namespace strata::kernels
