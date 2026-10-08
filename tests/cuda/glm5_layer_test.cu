// glm5_layer_test <bundle.bin>: a whole GLM-5.3-Flash dense KDA layer (layer 0) composed from Strata's kernels, the
// way the verify window will run it, on the real Q8_0 weights; compared with llama.cpp's own layer outputs.
//   mHC read (attn) -> q/k/v Q8_0 MMVQ -> conv + SiLU + L2 (gdn_conv_l2_multi, 24576 channels) -> f_a/f_b, beta,
//   g_a/g_b MMVQ -> kda_gate_beta_multi -> KDA recurrence + RMSNorm x gamma x sigmoid(z) -> attn_output MMVQ ->
//   mHC read (ffn, the attn write folded in) -> dense FFN (shared_expert_multi, no gate, clamp 10) -> mHC write.
// Bundle: tools/glm5_ref/make_layer_bundle.py 0 <full_dump_dir> <bundle.bin>.
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/shared_expert.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

using namespace strata::kernels;

struct Tensor { int type; std::vector<int64_t> ne; std::vector<unsigned char> bytes; void* dev = nullptr; };
static std::map<std::string, Tensor> g_t;

static void load(const char* path) {
    FILE* f = std::fopen(path, "rb");
    int n; std::fread(&n, 4, 1, f);
    for (int i = 0; i < n; ++i) {
        int nl; std::fread(&nl, 4, 1, f); std::string name(nl, '\0'); std::fread(&name[0], 1, nl, f);
        Tensor t; int nd; std::fread(&t.type, 4, 1, f); std::fread(&nd, 4, 1, f);
        t.ne.resize(nd); std::fread(t.ne.data(), 8, nd, f);
        int64_t nb; std::fread(&nb, 8, 1, f); t.bytes.resize(nb); std::fread(t.bytes.data(), 1, nb, f);
        cudaMalloc(&t.dev, nb); cudaMemcpy(t.dev, t.bytes.data(), nb, cudaMemcpyHostToDevice);
        g_t[name] = std::move(t);
    }
    std::fclose(f);
}
static Tensor& T_(const std::string& n) {
    auto it = g_t.find(n);
    if (it == g_t.end()) { std::fprintf(stderr, "missing tensor %s\n", n.c_str()); std::exit(2); }
    return it->second;
}
static float* W(const std::string& n) { return (float*) T_(n).dev; }
static float* dev(size_t n) { float* d; cudaMalloc(&d, n * 4); cudaMemset(d, 0, n * 4); return d; }
static std::vector<float> down(const float* d, size_t n) { std::vector<float> v(n); cudaMemcpy(v.data(), d, n * 4, cudaMemcpyDeviceToHost); return v; }

static int g_fail = 0;
static void cmp(const char* name, const float* d, const std::string& want, double tol) {
    const Tensor& w = T_(want);
    const size_t n = w.bytes.size() / 4;
    const auto got = down(d, n);
    const float* ref = (const float*) w.bytes.data();
    double se = 0, sr = 0, ma = 0;
    for (size_t i = 0; i < n; ++i) { const double e = (double) got[i] - ref[i]; se += e * e; sr += (double) ref[i] * ref[i]; ma = std::fmax(ma, std::fabs(e)); }
    const double rel = std::sqrt(se / (sr > 0 ? sr : 1));
    g_fail += rel > tol;
    std::printf("%-12s vs llama.cpp %-10s n=%-7zu max|diff| %.3e  rms rel %.3e  %s\n", name, want.c_str() + 5, n, ma, rel, rel <= tol ? "OK" : "FAIL");
}

int main(int argc, char** argv) {
    if (argc < 2) return 2;
    load(argv[1]);
    const std::string P = "blk.0.";
    const int N = 4096, HC = 4, H = 64, S = 128, DI = H * S, C = 3 * DI, NFF = 12288, LR = 128;
    const Tensor& Rin = T_("in:R");
    const int T = (int) Rin.ne[2];
    cudaStream_t st; cudaStreamCreate(&st);
    float* R = (float*) Rin.dev;                       // [T][4][N], updated in place
    const int S_ = fused_mhc_slices(N);
    float *part = dev((size_t) T * S_ * 24), *ssp = dev((size_t) T * S_);
    float *postA = dev(T * 4), *combA = dev(T * 16), *postF = dev(T * 4), *combF = dev(T * 16);
    float *x = dev((size_t) T * N), *qkv = dev((size_t) T * C), *h = dev((size_t) T * C), *hist = dev((size_t) C * 3);
    float *fa = dev((size_t) T * LR), *f = dev((size_t) T * DI), *bl = dev(T * H), *ga = dev((size_t) T * LR), *z = dev((size_t) T * DI);
    float *gate = dev((size_t) T * DI), *beta = dev(T * H), *state = dev((size_t) H * S * S), *y = dev((size_t) T * DI);
    float *attn_out = dev((size_t) T * N), *x2 = dev((size_t) T * N), *ffn_out = dev((size_t) T * N);
    float *g_ff = dev((size_t) T * NFF), *u_ff = dev((size_t) T * NFF), *gsc = dev(16);
    void* q8; cudaMalloc(&q8, (size_t) T * NFF * 2);

    // conv weights: GGUF [4, 1, 8192] each (c*4 + tap) -> q|k|v concatenated [24576][4]
    float* convw; cudaMalloc(&convw, (size_t) C * 4 * 4);
    cudaMemcpy(convw, W(P + "ssm_conv1d_q.weight"), (size_t) DI * 16, cudaMemcpyDeviceToDevice);
    cudaMemcpy(convw + (size_t) DI * 4, W(P + "ssm_conv1d_k.weight"), (size_t) DI * 16, cudaMemcpyDeviceToDevice);
    cudaMemcpy(convw + (size_t) 2 * DI * 4, W(P + "ssm_conv1d_v.weight"), (size_t) DI * 16, cudaMemcpyDeviceToDevice);

    auto mhc = [&](const char* which, bool apply, const float* out_prev, const float* post_prev, const float* comb_prev,
                   float* post_out, float* comb_out, const char* norm, float* yout) {
        MhcMulti m;
        m.n_tok = T; m.n_embd = N; m.n_slices = S_; m.apply = apply; m.iters = 20; m.rms_eps = 1e-5f; m.hc_eps = 1e-6f;
        for (int t = 0; t < T; ++t) {
            m.R[t] = R + (size_t) t * HC * N; m.R_out[t] = R + (size_t) t * HC * N;
            m.out_prev[t] = out_prev ? out_prev + (size_t) t * N : nullptr; m.y[t] = yout + (size_t) t * N;
        }
        m.post_prev = post_prev; m.comb_prev = comb_prev;
        const std::string w = P + "hc_" + which + "_";
        m.fn = T_(w + "fn.weight").dev; m.scale = W(w + "scale.weight"); m.base = W(w + "base.weight");
        m.w_norm = W(P + norm); m.post_out = post_out; m.comb_out = comb_out; m.part = part; m.ss_part = ssp;
        fused_mhc_read_multi(m, st);
    };
    auto mmvq = [&](const std::string& wname, const float* in, int n_in, float* out, int n_out) {
        const Tensor& w = T_(wname);
        native_quantize_q8_1(in, q8, n_in, T, st);
        native_mmvq(w.type, w.dev, q8, out, n_in, n_out, T, st);
    };

    // ---- attention side: mHC read, KDA
    mhc("attn", false, nullptr, nullptr, nullptr, postA, combA, "attn_norm.weight", x);
    native_quantize_q8_1(x, q8, N, T, st);   // x quantized once for every projection of the attention side
    {   // q, k, v land in separate [T][DI] buffers, then interleave per token into [T][C]
        float *q = dev((size_t) T * DI), *k = dev((size_t) T * DI), *v = dev((size_t) T * DI);
        native_mmvq(T_(P + "attn_q.weight").type, T_(P + "attn_q.weight").dev, q8, q, N, DI, T, st);
        native_mmvq(T_(P + "attn_k.weight").type, T_(P + "attn_k.weight").dev, q8, k, N, DI, T, st);
        native_mmvq(T_(P + "attn_v.weight").type, T_(P + "attn_v.weight").dev, q8, v, N, DI, T, st);
        for (int t = 0; t < T; ++t) {
            cudaMemcpyAsync(qkv + (size_t) t * C, q + (size_t) t * DI, DI * 4, cudaMemcpyDeviceToDevice, st);
            cudaMemcpyAsync(qkv + (size_t) t * C + DI, k + (size_t) t * DI, DI * 4, cudaMemcpyDeviceToDevice, st);
            cudaMemcpyAsync(qkv + (size_t) t * C + 2 * DI, v + (size_t) t * DI, DI * 4, cudaMemcpyDeviceToDevice, st);
        }
        native_mmvq(T_(P + "ssm_f_a.weight").type, T_(P + "ssm_f_a.weight").dev, q8, fa, N, LR, T, st);
        native_mmvq(T_(P + "ssm_g_a.weight").type, T_(P + "ssm_g_a.weight").dev, q8, ga, N, LR, T, st);
        native_mmvq(T_(P + "ssm_beta.weight").type, T_(P + "ssm_beta.weight").dev, q8, bl, N, H, T, st);
    }
    gdn_conv_l2_multi(hist, qkv, convw, h, C, 2 * H, 1e-6f, T, st, 0);
    mmvq(P + "ssm_f_b.weight", fa, LR, f, DI);
    mmvq(P + "ssm_g_b.weight", ga, LR, z, DI);
    kda_gate_beta_multi(f, W(P + "ssm_dt.bias"), W(P + "ssm_a"), -5.0f, bl, gate, beta, H, T, st);
    gdn_step_norm_multi(state, h, C, gate, beta, z, W(P + "ssm_norm.weight"), 1e-5f, y, H, H, T, nullptr, st, 0, true);
    mmvq(P + "attn_output.weight", y, DI, attn_out, N);
    // ---- FFN side: mHC read with the attention write folded in, dense FFN
    mhc("ffn", true, attn_out, postA, combA, postF, combF, "ffn_norm.weight", x2);
    NativeSharedWeights nw;
    nw.gate_type = T_(P + "ffn_gate.weight").type; nw.up_type = T_(P + "ffn_up.weight").type; nw.down_type = T_(P + "ffn_down.weight").type;
    nw.gate_data = T_(P + "ffn_gate.weight").dev; nw.up_data = T_(P + "ffn_up.weight").dev; nw.down_data = T_(P + "ffn_down.weight").dev;
    nw.q8_1 = q8;
    shared_expert_multi(T, x2, nullptr, nw, nullptr, g_ff, u_ff, gsc, ffn_out, N, NFF, st, 10.0f);
    // ---- the layer's last write (in the engine it is folded into layer 1's first read)
    {
        MhcMulti m;
        m.n_tok = T; m.n_embd = N;
        for (int t = 0; t < T; ++t) {
            m.R[t] = R + (size_t) t * HC * N; m.R_out[t] = R + (size_t) t * HC * N; m.out_prev[t] = ffn_out + (size_t) t * N;
        }
        m.post_prev = postF; m.comb_prev = combF; m.apply = true;
        fused_mhc_write_multi(m, st);
    }
    if (cudaStreamSynchronize(st) != cudaSuccess) { std::printf("cuda error: %s\n", cudaGetErrorString(cudaGetLastError())); return 2; }
    cmp("attn_norm", x, "want:attn_norm", 1e-5);
    {   // the KDA intermediates: v after conv + SiLU (the v third of h), gate, beta, z
        std::vector<float> hv = down(h, (size_t) T * C), vv((size_t) T * DI);
        for (int t = 0; t < T; ++t) for (int i = 0; i < DI; ++i) vv[(size_t) t * DI + i] = hv[(size_t) t * C + 2 * DI + i];
        float* dv = dev(vv.size()); cudaMemcpy(dv, vv.data(), vv.size() * 4, cudaMemcpyHostToDevice);
        cmp("v_conv", dv, "want:kda_v_conv", 2e-3);
    }
    {   // the v projection + conv from llama.cpp's own attn_norm: isolates the mHC's share of the difference
        float* xl = dev((size_t) T * N);
        cudaMemcpy(xl, T_("want:attn_norm").bytes.data(), (size_t) T * N * 4, cudaMemcpyHostToDevice);
        float *vv = dev((size_t) T * DI), *qkv2 = dev((size_t) T * C), *h2 = dev((size_t) T * C), *hist2 = dev((size_t) C * 3);
        native_quantize_q8_1(xl, q8, N, T, st);
        native_mmvq(T_(P + "attn_v.weight").type, T_(P + "attn_v.weight").dev, q8, vv, N, DI, T, st);
        for (int t = 0; t < T; ++t)
            cudaMemcpyAsync(qkv2 + (size_t) t * C + 2 * DI, vv + (size_t) t * DI, DI * 4, cudaMemcpyDeviceToDevice, st);
        gdn_conv_l2_multi(hist2, qkv2, convw, h2, C, 2 * H, 1e-6f, T, st, 0);
        cudaStreamSynchronize(st);
        std::vector<float> hv = down(h2, (size_t) T * C), v2((size_t) T * DI);
        for (int t = 0; t < T; ++t) for (int i = 0; i < DI; ++i) v2[(size_t) t * DI + i] = hv[(size_t) t * C + 2 * DI + i];
        float* dv = dev(v2.size()); cudaMemcpy(dv, v2.data(), v2.size() * 4, cudaMemcpyHostToDevice);
        cmp("v(llama x)", dv, "want:kda_v_conv", 2e-3);
        {
            const float* ref = (const float*) T_("want:kda_v_conv").bytes.data();
            for (int t = 0; t < T; ++t) {
                double se = 0, sr = 0;
                for (int i = 0; i < DI; ++i) { const double e = v2[(size_t) t * DI + i] - ref[(size_t) t * DI + i]; se += e * e; sr += (double) ref[(size_t) t * DI + i] * ref[(size_t) t * DI + i]; }
                std::printf("   v(llama x) token %d: rms rel %.3e\n", t, std::sqrt(se / sr));
            }
            // the raw projection of token 0 against conv weight tap 3 alone: with an empty history, conv = x * w3
            std::vector<float> raw = down(vv, (size_t) DI), w = down(convw + (size_t) 2 * DI * 4, (size_t) DI * 4);
            double se = 0, sr = 0;
            for (int i = 0; i < DI; ++i) {
                const float c = raw[i] * w[i * 4 + 3];
                const float sv = c / (1.0f + std::exp(-c));
                const double e = sv - ref[i]; se += e * e; sr += (double) ref[i] * ref[i];
            }
            std::printf("   token 0 recomputed on the host (x*w3, exact SiLU): rms rel %.3e\n", std::sqrt(se / sr));
        }
        native_quantize_q8_1(x, q8, N, T, st);   // restore: the later projections reuse x's codes
    }
    cmp("gate", gate, "want:kda_g1", 2e-3);
    cmp("beta", beta, "want:kda_beta", 2e-3);
    cmp("z", z, "want:kda_g2", 2e-3);
    cmp("kda_out", attn_out, "want:kda_out", 2e-3);
    {   // the output projection alone, from llama.cpp's own y = kda_normed * sigmoid(kda_g2)
        const float* nm = (const float*) T_("want:kda_normed").bytes.data();
        const float* g2 = (const float*) T_("want:kda_g2").bytes.data();
        std::vector<float> yl((size_t) T * DI);
        for (size_t i = 0; i < yl.size(); ++i) yl[i] = nm[i] * (1.f / (1.f + std::exp(-g2[i])));
        float* dyl = dev(yl.size()); cudaMemcpy(dyl, yl.data(), yl.size() * 4, cudaMemcpyHostToDevice);
        float* o2 = dev((size_t) T * N);
        mmvq(P + "attn_output.weight", dyl, DI, o2, N);
        cudaStreamSynchronize(st);
        cmp("out(llama y)", o2, "want:kda_out", 2e-3);
        // and our y against llama.cpp's y
        float* dyl2 = dyl;
        std::vector<float> ours = down(y, yl.size());
        double se = 0, sr = 0; for (size_t i = 0; i < yl.size(); ++i) { const double e = ours[i] - yl[i]; se += e * e; sr += (double) yl[i] * yl[i]; }
        std::printf("y (ours) vs llama.cpp y: rms rel %.3e\n", std::sqrt(se / sr));
        (void) dyl2;
    }
    cmp("ffn_norm", x2, "want:ffn_norm", 2e-3);
    cmp("ffn_out", ffn_out, "want:ffn_out", 5e-3);
    cmp("l_out", R, "want:l_out", 5e-3);
    std::printf("%s\n", g_fail ? "FAILED" : "ALL OK");
    return g_fail;
}
