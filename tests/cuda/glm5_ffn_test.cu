// glm5_ffn_test <case.bin>: Strata's shared_expert_multi with no scalar gate and the SwiGLU clamp (glm5-next's dense
// FFN of layers 0-2 and its shared expert) on real Q8_0 weights and llama.cpp's own inputs/outputs
// (tools/glm5_ref/make_ffn_case.py).  Prints max abs / relative error against llama.cpp.
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/shared_expert.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static FILE* g_f;
static int rdi() { int v; std::fread(&v, 4, 1, g_f); return v; }
template <class T> static std::vector<T> rd(size_t n) { std::vector<T> v(n); std::fread(v.data(), sizeof(T), n, g_f); return v; }
template <class T> static T* up(const std::vector<T>& v) { T* d; cudaMalloc(&d, v.size() * sizeof(T)); cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice); return d; }

int main(int argc, char** argv) {
    if (argc < 2) return 2;
    g_f = std::fopen(argv[1], "rb");
    const int nc = rdi();
    int fail = 0;
    cudaStream_t st; cudaStreamCreate(&st);
    for (int c = 0; c < nc; ++c) {
        const int T = rdi(), n = rdi(), ff = rdi(), ty = rdi(); float lim; std::fread(&lim, 4, 1, g_f);
        std::vector<unsigned char> w[3];
        for (auto& v : w) { const int b = rdi(); v = rd<unsigned char>(b); }
        const auto x = rd<float>((size_t)T * n), want = rd<float>((size_t)T * n);
        strata::kernels::NativeSharedWeights nw;
        nw.gate_type = nw.up_type = nw.down_type = ty;
        nw.gate_data = up(w[0]); nw.up_data = up(w[1]); nw.down_data = up(w[2]);
        cudaMalloc(&nw.q8_1, (size_t)T * (ff > n ? ff : n) * 2);   // q8_1: 36 B per 32 values < 2 B per value
        float *dx = up(x), *dg, *du, *dgs, *dout;
        cudaMalloc(&dg, (size_t)T * ff * 4); cudaMalloc(&du, (size_t)T * ff * 4); cudaMalloc(&dgs, 64); cudaMalloc(&dout, (size_t)T * n * 4);
        strata::kernels::shared_expert_multi(T, dx, nullptr, nw, nullptr, dg, du, dgs, dout, n, ff, st, lim);
        if (cudaStreamSynchronize(st) != cudaSuccess) { std::printf("cuda error\n"); return 2; }
        std::vector<float> got((size_t)T * n);
        cudaMemcpy(got.data(), dout, got.size() * 4, cudaMemcpyDeviceToHost);
        double ma = 0, mr = 0, se = 0, sr = 0;
        for (size_t i = 0; i < got.size(); ++i) {
            const double d = (double)got[i] - want[i];
            ma = std::fmax(ma, std::fabs(d)); mr = std::fmax(mr, std::fabs((double)want[i]));
            se += d * d; sr += (double)want[i] * want[i];
        }
        const double rel = std::sqrt(se / sr);
        const bool ok = rel < 1e-3;
        fail += !ok;
        std::printf("case %d (n_ff %d, limit %g): max|diff| %.3e  max|ref| %.3e  rms rel %.3e  %s\n", c, ff, lim, ma, mr, rel, ok ? "OK" : "FAIL");
    }
    std::printf("%s\n", fail ? "FAILED" : "ALL OK");
    return fail;
}
