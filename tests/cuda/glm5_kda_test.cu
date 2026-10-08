// glm5_kda_test <case.bin>: Strata's KDA branch of gdn_step_norm_multi (verify window, state not written) and
// kda_gate_beta_multi on llama.cpp's own layer-0 tensors (tools/glm5_ref/make_kda_case.py).  Also runs the commit
// form (n_keep = T, state written) and checks it gives the same outputs.
#include "strata/kernels/verify_kernels.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static FILE* g_f;
template <class T> static std::vector<T> rd(size_t n) { std::vector<T> v(n); if (std::fread(v.data(), sizeof(T), n, g_f) != n) std::exit(2); return v; }
static float* up(const std::vector<float>& v) { float* d; cudaMalloc(&d, v.size() * 4); cudaMemcpy(d, v.data(), v.size() * 4, cudaMemcpyHostToDevice); return d; }
static float* dev(size_t n) { float* d; cudaMalloc(&d, n * 4); cudaMemset(d, 0, n * 4); return d; }
static std::vector<float> down(const float* d, size_t n) { std::vector<float> v(n); cudaMemcpy(v.data(), d, n * 4, cudaMemcpyDeviceToHost); return v; }

static int g_fail = 0;
static void check(const char* name, const std::vector<float>& got, const std::vector<float>& want, double tol) {
    double ma = 0, mr = 0, se = 0, sr = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double d = (double)got[i] - want[i];
        ma = std::fmax(ma, std::fabs(d)); mr = std::fmax(mr, std::fabs((double)want[i])); se += d * d; sr += (double)want[i] * want[i];
    }
    const double rel = std::sqrt(se / (sr > 0 ? sr : 1));
    g_fail += rel > tol;
    std::printf("%-26s n=%-7zu max|diff| %.3e  max|ref| %.3e  rms rel %.3e  %s\n", name, got.size(), ma, mr, rel, rel <= tol ? "OK" : "FAIL");
}

int main(int argc, char** argv) {
    if (argc < 2) return 2;
    g_f = std::fopen(argv[1], "rb");
    int T, H, S; float rms_eps, lb;
    std::fread(&T, 4, 1, g_f); std::fread(&H, 4, 1, g_f); std::fread(&S, 4, 1, g_f); std::fread(&rms_eps, 4, 1, g_f); std::fread(&lb, 4, 1, g_f);
    const size_t HS = (size_t)H * S;
    auto h = rd<float>(T * 3 * HS), gate = rd<float>(T * HS), beta = rd<float>(T * H), z = rd<float>(T * HS), gamma = rd<float>(S);
    auto want_o = rd<float>(T * HS), want_y = rd<float>(T * HS), f = rd<float>(T * HS), dtb = rd<float>(HS), A = rd<float>(H);
    auto bl = rd<float>(T * H), want_g = rd<float>(T * HS);
    (void)want_o;

    // the gate/beta kernel
    float *dg = dev(T * HS), *db = dev(T * H);
    strata::kernels::kda_gate_beta_multi(up(f), up(dtb), up(A), lb, up(bl), dg, db, H, T, nullptr);
    cudaDeviceSynchronize();
    check("kda_gate (vs ref on same f)", down(dg, T * HS), want_g, 1e-5);
    std::vector<float> wb(T * H); for (int i = 0; i < T * H; ++i) wb[i] = 1.f / (1.f + std::exp(-bl[i]));
    check("kda_beta (vs ref)", down(db, T * H), wb, 1e-6);

    // the recurrence + norm + output gate, with llama.cpp's own gate/beta (verify form: state read, not written)
    float* dh = up(h); float* dgt = up(gate); float* dbt = up(beta); float* dz = up(z); float* dgam = up(gamma);
    float* state = dev(HS * S); float* dy = dev(T * HS);
    strata::kernels::gdn_step_norm_multi(state, dh, (int)(3 * HS), dgt, dbt, dz, dgam, rms_eps, dy, H, H, T, nullptr, nullptr, 0, true);
    if (cudaDeviceSynchronize() != cudaSuccess) { std::printf("cuda error\n"); return 2; }
    check("KDA y (vs llama.cpp)", down(dy, T * HS), want_y, 1e-4);
    const auto s_after_verify = down(state, HS * S);
    double nz = 0; for (float v : s_after_verify) nz = std::fmax(nz, std::fabs(v));
    std::printf("state after the verify form: max|s| %.3e (must stay 0)\n", nz);
    g_fail += nz != 0;

    // the commit form: n_keep = T writes the state; the outputs must be the same
    int32_t* nk; cudaMalloc(&nk, 4); cudaMemcpy(nk, &T, 4, cudaMemcpyHostToDevice);
    float* dy2 = dev(T * HS);
    strata::kernels::gdn_step_norm_multi(state, dh, (int)(3 * HS), dgt, dbt, dz, dgam, rms_eps, dy2, H, H, T, nk, nullptr, 0, true);
    cudaDeviceSynchronize();
    check("KDA y, commit form", down(dy2, T * HS), down(dy, T * HS), 0.0);
    // replay: from the committed state, the last token alone again must not equal (state advanced) - sanity only
    std::printf("%s\n", g_fail ? "FAILED" : "ALL OK");
    return g_fail;
}
