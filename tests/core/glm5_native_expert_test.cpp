#include "strata/kernels/cpu/native_expert.hpp"
#include "ggml.h"
#include "ggml-cpu.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <vector>

namespace c = strata::kernels::cpu;
static double reference(double g, double u, double limit) {
    if (limit > 0) { if (g > limit) g = limit; if (u > limit) u = limit; if (u < -limit) u = -limit; }
    return g / (1.0 + std::exp(-g)) * u;
}

int main() {
    int passed = 0;
    auto check = [&](bool ok, const std::string& msg) { if (!ok) throw std::runtime_error(msg); ++passed; };
    try {
        // Force eligible fused paths on so a missing clamp guard is visible at nt=3.
#ifdef _WIN32
        _putenv_s("STRATA_KQ256", "1");
        _putenv_s("STRATA_IQ_MT_MIN", "1");
#else
        setenv("STRATA_KQ256", "1", 1);
        setenv("STRATA_IQ_MT_MIN", "1", 1);
#endif
        // Check boundaries, negative gates (must NOT clip to -10), and values outside both limits.
        const float edge[] = {-100, -20, -10.001f, -10, -9.999f, -1, 0, 1, 9.999f, 10, 10.001f, 20, 100};
        for (float g : edge) for (float u : edge) {
            const double want = reference(g, u, 10);
            check(std::abs(c::native_swiglu(g, u, 10) - want) <= 2e-6 * (1 + std::abs(want)), "scalar clamp boundary");
            const float legacy = (g / (1.f + std::exp(-g))) * u;
            const float got = c::native_swiglu(g, u);
            check(std::memcmp(&got, &legacy, sizeof(float)) == 0, "Qwen unclamped arithmetic changed");
        }
        check(std::abs(c::native_swiglu(-20, 20, 10)) < 1e-6, "gate lower bound must remain unclipped");
        std::mt19937 rng(53);
        std::uniform_real_distribution<float> values(-40, 40);
        for (int i = 0; i < 4096; ++i) {
            const float g = values(rng), u = values(rng);
            const double want = reference(g, u, 10);
            check(std::abs(c::native_swiglu(g, u, 10) - want) <= 2e-6 * (1 + std::abs(want)), "random clamp");
        }
        // All expert type pairs actually observed in UD-IQ4_XS, plus the eventual all-Q8_0 target.
        const int pairs[][2] = {{21, 23}, {23, 14}, {21, 14}, {11, 12}, {8, 8}};
        for (const auto& pair : pairs) {
            c::NativeFmt f;
            std::string err;
            check(c::native_fmt(pair[0], pair[1], 4096, 2048, f, err, 10), err);
            const bool q8k = pair[0] != 8;
            check(f.act_bytes == (q8k ? 4672u : 4352u) && f.h_bytes == (q8k ? 2336u : 2176u), "GLM activation sizes");
            check(f.swiglu_limit == 10 && f.act_bytes <= c::kNativeActBytes && f.h_bytes <= c::kNativeHBytes,
                  "GLM activation fits buffers");
            std::vector<float> x(4096, 1.f), h(2048, 1.f);
            // Guard bytes immediately after the *actual* quantized rows, not just after the larger capacities.
            std::vector<uint8_t> ax(f.act_bytes + 64, 0xA5), ah(f.h_bytes + 64, 0xA5);
            c::native_quant_act(f, x.data(), ax.data()); c::native_quant_h(f, h.data(), ah.data());
            check(std::all_of(ax.begin() + f.act_bytes, ax.end(), [](auto b) { return b == 0xA5; }) &&
                  std::all_of(ah.begin() + f.h_bytes, ah.end(), [](auto b) { return b == 0xA5; }), "quantizer guard bytes");
        }
        c::NativeFmt f;
        std::string err;
        check(c::native_fmt(8, 8, 2560, 640, f, err), err);
        check(f.swiglu_limit == 0, "Qwen clamp must default off");
        check(!c::native_fmt(8, 8, 4096, 2048, f, err, -1), "negative clamp accepted");
        check(!c::native_fmt(8, 8, 8192, 2048, f, err), "oversize activation accepted");
        check(!c::native_fmt(8, 8, 4096, 4096, f, err), "oversize intermediate accepted");

        // Exercise the production gate/up entry point, including multi-token dispatch. The scalar oracle
        // applies the formula to ggml's own independent row dots, so quantization does not mask a clamp bug.
        for (int type : {8, 12, 23, 21}) {
        check(c::native_fmt(type, type, 4096, 2048, f, err, 10), err);
        ggml_quantize_init(static_cast<ggml_type>(type));
        std::vector<uint8_t> blob(f.bytes, 0);
        std::vector<float> row(f.n_embd), x(f.n_embd, 1.f);
        const auto quant = ggml_get_type_traits(static_cast<ggml_type>(type))->from_float_ref;
        const auto dot = ggml_get_type_traits_cpu(static_cast<ggml_type>(type))->vec_dot;
        const float gates[] = {-20, -10, 0, 10, 20}, ups[] = {-20, 20, -1, 20, -20};
        for (int r = 0; r < 5; ++r) {
            std::fill(row.begin(), row.end(), gates[r] / float(f.n_embd));
            quant(row.data(), blob.data() + r * f.gu_row, f.n_embd);
            std::fill(row.begin(), row.end(), ups[r] / float(f.n_embd));
            quant(row.data(), blob.data() + f.up_off + r * f.gu_row, f.n_embd);
        }
        std::vector<uint8_t> act(f.act_bytes);
        c::native_quant_act(f, x.data(), act.data());
        const void* acts[3] = {act.data(), act.data(), act.data()};
        std::vector<float> a(7, 12345), b(7, 12345), d(7, 12345);
        float* outputs[3] = {a.data() + 1, b.data() + 1, d.data() + 1};
        for (int nt : {1, 3}) {
            c::native_gu_rows(f, blob.data(), acts, nt, outputs, 0, 5);
            for (int t = 0; t < nt; ++t) for (int r = 0; r < 5; ++r) {
                float g = 0, u = 0;
                dot(int(f.n_embd), &g, 0, blob.data() + r * f.gu_row, 0, acts[t], 0, 1);
                dot(int(f.n_embd), &u, 0, blob.data() + f.up_off + r * f.gu_row, 0, acts[t], 0, 1);
                const double want = reference(g, u, 10);
                check(std::abs(outputs[t][r] - want) <= 2e-6 * (1 + std::abs(want)), "native_gu_rows clamp");
            }
        }
        check(a.front() == 12345 && a.back() == 12345 && b.front() == 12345 && b.back() == 12345 &&
              d.front() == 12345 && d.back() == 12345, "gate/up output guard bytes");
        }

        std::printf("glm5_native_expert_test: %d passed, 0 failed\n", passed);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "glm5_native_expert_test: %d passed, 1 failed: %s\n", passed, e.what()); return 1;
    }
}
