// glm5_mhc_test <case.bin>: Strata's fused_mhc_* (src/kernels/cuda/fused_mhc.cu) on real GLM-5.3-Flash layer-0
// data written by tools/glm5_ref/make_mhc_case.py; prints max abs / relative error of every output, the
// in-place (R_out == R) variant, and the per-call time for one token.
#include "strata/kernels/fused_gr.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using strata::kernels::MhcMulti;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e_)); std::exit(2); } } while (0)

static FILE* g_f;
template <class T> static std::vector<T> rd(size_t n) {
    std::vector<T> v(n);
    if (std::fread(v.data(), sizeof(T), n, g_f) != n) { std::fprintf(stderr, "short read\n"); std::exit(2); }
    return v;
}
static int rdi() { int v; if (std::fread(&v, 4, 1, g_f) != 1) std::exit(2); return v; }
template <class T> static T* up(const std::vector<T>& v) {
    T* d; CK(cudaMalloc(&d, v.size() * sizeof(T))); CK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice)); return d;
}
static float* dev(size_t n) { float* d; CK(cudaMalloc(&d, n * 4)); CK(cudaMemset(d, 0, n * 4)); return d; }
static std::vector<float> down(const float* d, size_t n) { std::vector<float> v(n); CK(cudaMemcpy(v.data(), d, n * 4, cudaMemcpyDeviceToHost)); return v; }

static int g_fail = 0;
static void check(const char* tag, const char* name, const std::vector<float>& got, const std::vector<float>& want, double tol) {
    double ma = 0, mr = 0;
    for (size_t i = 0; i < got.size(); ++i) { ma = std::fmax(ma, std::fabs((double)got[i] - want[i])); mr = std::fmax(mr, std::fabs((double)want[i])); }
    const double rel = ma / (mr > 0 ? mr : 1);
    const bool ok = rel <= tol;
    g_fail += !ok;
    std::printf("%s %-9s n=%-6zu max|diff| %.3e  max|ref| %.3e  rel %.3e  %s\n", tag, name, got.size(), ma, mr, rel, ok ? "OK" : "FAIL");
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: glm5_mhc_test case.bin\n"); return 2; }
    g_f = std::fopen(argv[1], "rb");
    if (!g_f) { std::perror(argv[1]); return 2; }
    const int T = rdi(), n = rdi(), iters = rdi();
    float rms_eps, hc_eps; std::fread(&rms_eps, 4, 1, g_f); std::fread(&hc_eps, 4, 1, g_f);
    const auto R0 = rd<float>((size_t)T * 4 * n);
    if (!strata::kernels::fused_mhc_supported(n, 4, 24)) { std::printf("unsupported geometry\n"); return 1; }
    const int S = strata::kernels::fused_mhc_slices(n);
    float* part = dev((size_t)T * S * 24); float* ssp = dev((size_t)T * S);

    for (int c = 0; c < 2; ++c) {
        const char* tag = c == 0 ? "[A attn, no fold]" : "[B ffn, folded  ]";
        const int apply = rdi();
        const int nb = rdi(); const auto fn = rd<unsigned char>(nb);
        const auto sc = rd<float>(3), bs = rd<float>(24), wn = rd<float>(n), outp = rd<float>((size_t)T * n);
        const auto postp = rd<float>(T * 4), combp = rd<float>(T * 16);
        const auto w_mix = rd<float>(T * 24), w_pre = rd<float>(T * 4), w_post = rd<float>(T * 4), w_comb = rd<float>(T * 16);
        const auto w_y = rd<float>((size_t)T * n), w_R = rd<float>((size_t)T * 4 * n), w_exit = rd<float>((size_t)T * n);

        for (int inplace = 0; inplace < (apply ? 2 : 1); ++inplace) {
            float* dR = up(R0); float* dRo = inplace ? dR : dev((size_t)T * 4 * n);
            float* dOut = up(outp); float* dy = dev((size_t)T * n);
            MhcMulti m;
            m.n_tok = T; m.n_embd = n; m.n_slices = S; m.apply = apply != 0; m.iters = iters; m.rms_eps = rms_eps; m.hc_eps = hc_eps;
            for (int t = 0; t < T; ++t) {
                m.R[t] = dR + (size_t)t * 4 * n; m.R_out[t] = dRo + (size_t)t * 4 * n;
                m.out_prev[t] = dOut + (size_t)t * n; m.y[t] = dy + (size_t)t * n;
            }
            m.post_prev = up(postp); m.comb_prev = up(combp); m.fn = up(fn); m.scale = up(sc); m.base = up(bs); m.w_norm = up(wn);
            m.post_out = dev(T * 4); m.comb_out = dev(T * 16); m.part = part; m.ss_part = ssp;
            m.pre_out = dev(T * 4); m.mixes_out = dev(T * 24);
            strata::kernels::fused_mhc_read_multi(m, nullptr);
            CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
            const double tol = 2e-5;
            check(tag, "mixes", down(m.mixes_out, T * 24), w_mix, tol);
            check(tag, "pre", down(m.pre_out, T * 4), w_pre, tol);
            check(tag, "post", down(m.post_out, T * 4), w_post, tol);
            check(tag, "comb", down(m.comb_out, T * 16), w_comb, tol);
            check(tag, "y", down(dy, (size_t)T * n), w_y, tol);
            if (apply) {
                check(inplace ? "[B in place      ]" : tag, "R'", down(dRo, (size_t)T * 4 * n), w_R, tol);
                // the last layer's plain write + exit, from the unfolded R
                float* dR2 = up(R0); MhcMulti w = m;
                for (int t = 0; t < T; ++t) { w.R[t] = dR2 + (size_t)t * 4 * n; w.R_out[t] = dR2 + (size_t)t * 4 * n; }
                strata::kernels::fused_mhc_write_multi(w, nullptr);
                strata::kernels::fused_mhc_exit_mean(w, nullptr);
                CK(cudaDeviceSynchronize());
                check(tag, "write", down(dR2, (size_t)T * 4 * n), w_R, tol);
                check(tag, "exit", down(dy, (size_t)T * n), w_exit, tol);
            }
            if (c == 1 && inplace == 0) {   // timing: one token (decode) and 8 tokens (a verify window)
                cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
                for (int nt : {1, T}) {
                    MhcMulti mt = m; mt.n_tok = nt; mt.pre_out = nullptr; mt.mixes_out = nullptr;
                    for (int i = 0; i < 50; ++i) strata::kernels::fused_mhc_read_multi(mt, nullptr);
                    CK(cudaEventRecord(e0));
                    for (int i = 0; i < 500; ++i) strata::kernels::fused_mhc_read_multi(mt, nullptr);
                    CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
                    float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
                    std::printf("fused_mhc_read_multi %d token(s): %.2f us/call (2 launches, not in a CUDA graph)\n", nt, ms * 1000.f / 500);
                }
            }
        }
    }
    std::printf("%s\n", g_fail ? "FAILED" : "ALL OK");
    return g_fail ? 1 : 0;
}
