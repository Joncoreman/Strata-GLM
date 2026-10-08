#include "strata/core/layout.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include <cmath>
#include <cstdio>
#include <stdexcept>

int main(int argc, char** argv) {
    int passed = 0;
    auto check = [&](bool ok, const char* what) {
        if (!ok) throw std::runtime_error(what);
        ++passed;
    };
    try {
        const strata::core::ModelGeometry q;
        check(q.n_embd == 2560 && q.n_layers == 48 && q.qsa_interval == 4, "Qwen trunk defaults");
        check(q.ssm_state_size == 128 && q.ssm_k_heads == 16 && q.ssm_v_heads == 48 &&
              q.ssm_d_conv == 4 && q.ssm_conv_channels == 10240 && q.ssm_value_dim == 6144, "Qwen GDN defaults");
        check(q.n_head == 24 && q.n_head_kv == 2 && q.head_dim == 256 && q.idx_q_heads == 4 &&
              q.idx_key_dim == 128 && q.hc == 4 && q.hc_lr == 320 && q.n_expert == 512 && q.n_ff == 640,
              "Qwen attention/residual/expert defaults");
        check(q.n_qsa_layers() == 12 && q.n_gdn_layers() == 36, "Qwen layer counts");
        for (int i = 0; i < 48; ++i) check(strata::core::is_qsa_layer(q, i) == (i % 4 == 3), "Qwen layer predicate");
        if (argc == 2) {
            const strata::GgufFile file(argv[1]); // metadata only: no tensor_data calls
            const auto g = strata::core::geometry_from_gguf(file);
            if (g.architecture == "qwen4exp") {
                check(g.n_embd == q.n_embd && g.n_layers == q.n_layers && g.n_ff == q.n_ff &&
                      g.ssm_conv_channels == q.ssm_conv_channels && g.hc_lr == q.hc_lr && g.rms_eps == q.rms_eps,
                      "Qwen metadata must preserve legacy geometry");
                check(g.n_expert == int64_t(file.get("qwen4exp.expert_count")->u) &&
                      g.n_expert_used == int64_t(file.get("qwen4exp.expert_used_count")->u), "Qwen pruned expert overrides");
                std::printf("glm5_geometry_test: %d passed, 0 failed (Qwen GGUF metadata)\n", passed);
                return 0;
            }
            // Expected values independently transcribed from the five-shard UD-IQ4_XS metadata.
            check(g.is_glm5_next() && g.n_embd == 4096 && g.n_layers == 45 && g.n_mtp_layers == 1, "GLM trunk");
            check(g.n_vocab == 154880 && g.n_ctx_train == 1048576, "GLM vocabulary/context");
            check(g.n_head == 64 && g.n_head_kv == 1 && g.head_count_kv.size() == 46, "GLM head array");
            check(g.n_qsa_layers() == 11 && g.n_gdn_layers() == 34 && g.is_sparse_layer(45), "GLM split and MTP");
            check(g.ssm_state_size == 128 && g.ssm_k_heads == 64 && g.ssm_v_heads == 64 &&
                  g.ssm_value_dim == 8192 && g.ssm_conv_channels == 24576 && g.ssm_d_conv == 4, "GLM KDA");
            check(g.q_lora_rank == 1536 && g.kv_lora_rank == 512 && g.mla_key_dim == 256 &&
                  g.mla_value_dim == 256 && g.cache_key_dim == 512 && g.cache_value_dim == 512 && g.rope_dim == 0,
                  "GLM MLA dimensions");
            check(g.idx_q_heads == 32 && g.idx_key_dim == 128 && g.idx_top_k == 2048 && g.idx_kpool == 4,
                  "GLM indexer dimensions");
            check(g.idx_select_tail && g.idx_share_mtp && g.indexer_types.size() == 45, "GLM indexer options");
            check(g.hc == 4 && g.hc_lr == 0 && g.hc_dim() == 16384 && g.hc_sinkhorn_iters == 20, "GLM mHC");
            check(g.rms_eps == 1e-5f && g.norm_eps == 1e-6f && g.hc_eps == 1e-6f && g.kda_gate_lower_bound == -5,
                  "GLM epsilons and decay bound");
            check(g.n_expert == 288 && g.n_expert_used == 8 && g.n_ff == 2048 && g.n_ff_shared == 2048 &&
                  g.n_expert_shared == 1 && g.n_ff_dense == 12288 && g.n_dense_layers == 3, "GLM feed forward");
            check(g.expert_groups == 1 && g.expert_groups_used == 1 && g.expert_gating_func == 2 &&
                  g.expert_weights_norm && g.expert_weights_scale == 2.5f, "GLM router");
            check(g.swiglu_clamp_exp.size() == 46 && g.swiglu_clamp_shared.size() == 46, "GLM clamp arrays");
            for (int i = 0; i < 46; ++i)
                check(g.swiglu_clamp_exp[i] == 10 && g.swiglu_clamp_shared[i] == 10 && g.n_ff_exp_by_layer[i] == 2048,
                      "GLM per-layer feed forward");
            auto irregular = g;
            irregular.head_count_kv[0] = 1;
            irregular.head_count_kv[3] = 0;
            check(irregular.is_sparse_layer(0) && !irregular.is_sparse_layer(3) && irregular.n_qsa_layers() == 11,
                  "GLM predicate follows array, not layer period");
        }
        std::printf("glm5_geometry_test: %d passed, 0 failed%s\n", passed, argc == 2 ? " (GGUF metadata)" : " (Qwen defaults)");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "glm5_geometry_test: %d passed, 1 failed: %s\n", passed, e.what());
        return 1;
    }
}
