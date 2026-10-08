// src/kernels/cuda/glm5_misc.cu - see include/strata/kernels/glm5_misc.hpp.
#include "strata/kernels/glm5_misc.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}

// one warp per output row o, every token: the weight row is read once
__global__ void __launch_bounds__(256) f32_gemv_kernel(const float* __restrict__ w, const float* __restrict__ x,
                                                       float* __restrict__ y, int n_in, int n_out, int n_tok, float scale) {
    const int o = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (o >= n_out) return;
    const float* wr = w + (size_t) o * n_in;
    for (int t = 0; t < n_tok; ++t) {
        const float* xt = x + (size_t) t * n_in;
        float acc = 0.f;
        for (int i = lane; i < n_in; i += 32) acc = fmaf(wr[i], xt[i], acc);
        for (int s = 16; s > 0; s >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, s);
        if (lane == 0) y[(size_t) t * n_out + o] = acc * scale;
    }
}

__global__ void transpose_abd_kernel(const float* __restrict__ src, float* __restrict__ dst, int a, int b, int d) {
    const int ia = blockIdx.y, ib = blockIdx.z;
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < d; k += gridDim.x * blockDim.x)
        dst[((size_t) ib * a + ia) * d + k] = src[((size_t) ia * b + ib) * d + k];
}

__global__ void pack3_kernel(const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
                             float* __restrict__ dst, int n) {
    const int t = blockIdx.y;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        float* o = dst + (size_t) t * 3 * n;
        o[i] = q[(size_t) t * n + i];
        o[n + i] = k[(size_t) t * n + i];
        o[2 * n + i] = v[(size_t) t * n + i];
    }
}

__global__ void latent_store_kernel(const float* __restrict__ kv, uint16_t* __restrict__ pool,
                                    const int32_t* __restrict__ page_table, const int32_t* __restrict__ steps,
                                    int step_count, int page_size, int dim) {
    const int t = blockIdx.x;
    const int cell = steps[t * step_count + 0];
    const long long page = page_table[cell / page_size];
    if (page < 0) return;
    uint16_t* row = pool + (size_t) (page * page_size + cell % page_size) * dim;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) row[i] = __half_as_ushort(__float2half(kv[(size_t) t * dim + i]));
}

}  // namespace

void glm5_f32_gemv(const float* w, const float* x, float* y, int n_in, int n_out, int n_tok, float scale, void* stream) {
    f32_gemv_kernel<<<(unsigned) ((n_out + 7) / 8), 256, 0, (cudaStream_t) stream>>>(w, x, y, n_in, n_out, n_tok, scale);
    check("glm5_f32_gemv");
}

void glm5_transpose_abd(const float* src, float* dst, int a, int b, int d, void* stream) {
    transpose_abd_kernel<<<dim3((unsigned) ((d + 255) / 256), (unsigned) a, (unsigned) b), 256, 0, (cudaStream_t) stream>>>(
        src, dst, a, b, d);
    check("glm5_transpose_abd");
}

void glm5_pack3(const float* q, const float* k, const float* v, float* dst, int n, int n_tok, void* stream) {
    pack3_kernel<<<dim3((unsigned) ((n + 255) / 256), (unsigned) n_tok), 256, 0, (cudaStream_t) stream>>>(q, k, v, dst, n);
    check("glm5_pack3");
}

void glm5_latent_store(const float* kv, uint16_t* pool, const int32_t* page_table, const int32_t* steps, int step_count,
                       int page_size, int dim, int n_tok, void* stream) {
    latent_store_kernel<<<(unsigned) n_tok, 256, 0, (cudaStream_t) stream>>>(kv, pool, page_table, steps, step_count,
                                                                            page_size, dim);
    check("glm5_latent_store");
}

}  // namespace strata::kernels
