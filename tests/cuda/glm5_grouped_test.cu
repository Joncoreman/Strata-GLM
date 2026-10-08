// glm5_grouped_test <shard.gguf> <gu_type> <d_type> <gate_off> <up_off> <down_off> <expert> [n_embd n_ff]
// One expert of a glm5-next GGUF through native_expert_grouped (q8_1 activations, SwiGLU clamp 10) against the same
// expert dequantized to fp32 by iq_dequant_f32 and multiplied in double (2026-10-08: the prompt path's routed experts
// disagreed with the window's CPU experts - which expert formats does the GPU compute right?).  Prints the relative
// L2 error; q8_1 rounding alone gives about 1e-2, a wrong format about 1.
#include "strata/kernels/iq_kernels.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using namespace strata::kernels;

int main(int argc, char** argv) {
    if (argc < 8) { std::fprintf(stderr, "usage: see the header\n"); return 2; }
    const char* path = argv[1];
    const int gu = std::atoi(argv[2]), dt = std::atoi(argv[3]);
    const long long go = std::atoll(argv[4]), uo = std::atoll(argv[5]), dofs = std::atoll(argv[6]);
    const int ex = std::atoi(argv[7]);
    const int64_t N = argc > 8 ? std::atoll(argv[8]) : 4096, F = argc > 9 ? std::atoll(argv[9]) : 2048;
    NativeExpertLayout L = native_expert_layout(gu, dt, N, F);
    L.swiglu_limit = 10.f;
    const size_t gb = L.up_off, db = L.bytes - L.down_off;
    std::vector<uint8_t> blob(L.bytes);
    std::FILE* f = std::fopen(path, "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path); return 1; }
    const long long offs[3] = {go + (long long) ex * (long long) gb, uo + (long long) ex * (long long) gb,
                               dofs + (long long) ex * (long long) db};
    const size_t lens[3] = {gb, gb, db}, at[3] = {0, L.up_off, L.down_off};
    for (int r = 0; r < 3; ++r) {
        _fseeki64(f, offs[r], SEEK_SET);
        if (std::fread(blob.data() + at[r], 1, lens[r], f) != lens[r]) { std::fprintf(stderr, "short read\n"); return 1; }
    }
    std::fclose(f);
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x((size_t) N);
    for (auto& v : x) v = nd(rng);
    // device
    uint8_t* dblob; float *dx, *dout, *dg, *du, *dd; void *xq, *scr; int32_t* plan; unsigned long long* ptr;
    cudaMalloc(&dblob, L.bytes); cudaMemcpy(dblob, blob.data(), L.bytes, cudaMemcpyHostToDevice);
    cudaMalloc(&dx, N * 4); cudaMemcpy(dx, x.data(), N * 4, cudaMemcpyHostToDevice);
    cudaMalloc(&dout, N * 4); cudaMalloc(&xq, (N / 32) * 36 + 64); cudaMalloc(&scr, native_expert_scratch_bytes(1, F));
    cudaMalloc(&plan, 16 * 4); cudaMalloc(&ptr, 8);
    const int32_t hp[6] = {0, 1, 0, 0, 1, 0};   // start[0..1] | dst[0] | tok[0] | count
    // layout: start at 0 (2), dst at 2 (1), tok at 3 (1), count at 4
    const int32_t hplan[5] = {0, 1, 0, 0, 1};
    (void) hp;
    cudaMemcpy(plan, hplan, sizeof hplan, cudaMemcpyHostToDevice);
    const unsigned long long hptr = (unsigned long long) dblob;
    cudaMemcpy(ptr, &hptr, 8, cudaMemcpyHostToDevice);
    quantize_q8_1_rows(dx, 1, N, xq, nullptr);
    native_expert_grouped(L, ptr, plan, plan + 4, plan + 2, plan + 3, 1, 1, xq, scr, dout, nullptr);
    std::vector<float> out((size_t) N);
    cudaMemcpy(out.data(), dout, N * 4, cudaMemcpyDeviceToHost);
    // reference
    cudaMalloc(&dg, F * N * 4); cudaMalloc(&du, F * N * 4); cudaMalloc(&dd, N * F * 4);
    iq_dequant_f32(gu, dblob, F * N, dg, nullptr);
    iq_dequant_f32(gu, dblob + L.up_off, F * N, du, nullptr);
    iq_dequant_f32(dt, dblob + L.down_off, N * F, dd, nullptr);
    std::vector<float> wg((size_t) (F * N)), wu((size_t) (F * N)), wd((size_t) (N * F));
    cudaMemcpy(wg.data(), dg, F * N * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(wu.data(), du, F * N * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(wd.data(), dd, N * F * 4, cudaMemcpyDeviceToHost);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "cuda: %s\n", cudaGetErrorString(e)); return 1; }
    std::vector<double> h((size_t) F);
    for (int64_t r = 0; r < F; ++r) {
        double g = 0, u = 0;
        for (int64_t i = 0; i < N; ++i) { g += (double) wg[(size_t) (r * N + i)] * x[(size_t) i]; u += (double) wu[(size_t) (r * N + i)] * x[(size_t) i]; }
        g = std::fmin(g, 10.0); u = std::fmax(std::fmin(u, 10.0), -10.0);
        h[(size_t) r] = g / (1.0 + std::exp(-g)) * u;
    }
    double num = 0, den = 0;
    for (int64_t r = 0; r < N; ++r) {
        double y = 0;
        for (int64_t i = 0; i < F; ++i) y += (double) wd[(size_t) (r * F + i)] * h[(size_t) i];
        num += (y - out[(size_t) r]) * (y - out[(size_t) r]);
        den += y * y;
    }
    std::printf("types %d/%d expert %d: relative L2 error %.4g (|ref| %.4g)\n", gu, dt, ex, std::sqrt(num / den), std::sqrt(den));
    return 0;
}
