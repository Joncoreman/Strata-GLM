// glm5_expert_layout_test <dir> <n_layers> <n_expert>: expert_layout_load on a GLM native_experts.txt (v5) written by
// tools/glm5_ref/write_glm5_layout.py; prints each layer's NativeFmt (dense layers have no experts).
#include "strata/kernels/cpu/expert_layout.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>

using namespace strata::kernels::cpu;

int main(int argc, char** argv) {
    if (argc < 4) return 2;
    std::string err;
    if (!expert_layout_load(argv[1], std::atoll(argv[2]), std::atoll(argv[3]), err)) {
        std::printf("expert_layout_load FAILED: %s\n", err.c_str());
        return 1;
    }
    const ExpertLayout& L = expert_layout();
    int dense = 0, bad = 0;
    for (int64_t l = 0; l < L.n_layers; ++l) {
        const NativeFmt& f = L.fmt[(size_t) l];
        if (f.gu_type < 0) { ++dense; std::printf("layer %2lld: dense (no experts)\n", (long long) l); continue; }
        bad += f.n_embd != 4096 || f.n_ff != 2048 || f.swiglu_limit != 10.f;
        if (l < 5 || l > L.n_layers - 3)
            std::printf("layer %2lld: gu %d d %d n_embd %lld n_ff %lld swiglu_limit %g blob %llu\n", (long long) l, f.gu_type,
                        f.d_type, (long long) f.n_embd, (long long) f.n_ff, f.swiglu_limit, (unsigned long long) L.bytes[(size_t) l]);
    }
    std::printf("version %d, n_expert %lld, dense layers %d, max blob %llu, total %llu bytes; %s\n", L.version,
                (long long) L.n_expert, dense, (unsigned long long) L.max_blob, (unsigned long long) L.total,
                bad ? "GEOMETRY WRONG" : "ALL OK");
    return bad ? 1 : 0;
}
