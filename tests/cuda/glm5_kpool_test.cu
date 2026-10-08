// glm5_kpool_test <case.bin> <out.bin>: Strata's glm5_kpool_append (5 cells in order, positions from a device step
// buffer as in a captured graph) and glm5_kpool_scores on llama.cpp's layer-3 indexer tensors
// (tools/glm5_ref/kpool_case.py).  Writes keys [T][128], pooled[0] [128], scores [T][max_blocks].
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/qsa.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

using namespace strata::kernels;

static FILE* g_f;
static std::vector<float> rd(size_t n) { std::vector<float> v(n); std::fread(v.data(), 4, n, g_f); return v; }
static float* up(const std::vector<float>& v) { float* d; cudaMalloc(&d, v.size() * 4); cudaMemcpy(d, v.data(), v.size() * 4, cudaMemcpyHostToDevice); return d; }

int main(int argc, char** argv) {
    if (argc < 3) return 2;
    g_f = std::fopen(argv[1], "rb");
    int T, NP; float eps;
    std::fread(&T, 4, 1, g_f); std::fread(&NP, 4, 1, g_f); std::fread(&eps, 4, 1, g_f);
    auto raw = rd(T * 128), gate = rd(T * 128), kw = rd(128), kb = rd(128), ape = rd(512), q = rd((size_t)T * 32 * 128), w = rd(T * 32), pk = rd((size_t)NP * 128);
    cudaStream_t st; cudaStreamCreate(&st);
    float *draw = up(raw), *dgate = up(gate), *dkw = up(kw), *dkb = up(kb), *dape = up(ape);
    Glm5KpoolBuffers b; cudaMalloc(&b.tail, 3 * 256 * 4); cudaMalloc(&b.pooled, 64 * 128 * 4); cudaMemset(b.pooled, 0, 64 * 128 * 4);
    float* dkeys; cudaMalloc(&dkeys, T * 128 * 4);
    int32_t* dpos; cudaMalloc(&dpos, 4);
    for (int t = 0; t < T; ++t) {
        cudaMemcpyAsync(dpos, &t, 4, cudaMemcpyHostToDevice, st);
        glm5_kpool_append(draw + t * 128, dgate + t * 128, dpos, dkw, dkb, dape, eps, b, 4096, st, dkeys + t * 128);
        cudaStreamSynchronize(st);
    }
    // scores with llama.cpp's own q, weights and pooled keys; steps: n_bid = (t+1)/4 completed blocks
    const int MB = NP;
    std::vector<int32_t> steps((size_t)T * kStepCount, 0);
    for (int t = 0; t < T; ++t) steps[(size_t)t * kStepCount + kStepNBid] = (t + 1) / 4;
    int32_t* dsteps; cudaMalloc(&dsteps, steps.size() * 4); cudaMemcpy(dsteps, steps.data(), steps.size() * 4, cudaMemcpyHostToDevice);
    float* dsc; cudaMalloc(&dsc, (size_t)T * MB * 4); cudaMemset(dsc, 0, (size_t)T * MB * 4);
    glm5_kpool_scores(up(q), up(w), up(pk), dsteps, T, MB, dsc, st);
    if (cudaStreamSynchronize(st) != cudaSuccess) { std::printf("cuda error\n"); return 2; }
    std::vector<float> keys(T * 128), pooled0(128), sc((size_t)T * MB);
    cudaMemcpy(keys.data(), dkeys, keys.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(pooled0.data(), b.pooled, 128 * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(sc.data(), dsc, sc.size() * 4, cudaMemcpyDeviceToHost);
    FILE* o = std::fopen(argv[2], "wb");
    std::fwrite(keys.data(), 4, keys.size(), o); std::fwrite(pooled0.data(), 4, 128, o); std::fwrite(sc.data(), 4, sc.size(), o);
    std::fclose(o);
    std::printf("T=%d pools=%d done\n", T, NP);
    return 0;
}
