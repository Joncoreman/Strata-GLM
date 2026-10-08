// CPU GGUF schema, transcribed from llama.cpp b11461 src/models/glm5-next.cpp:60-185.
// Names checked against its llama-arch.cpp and the complete UD-IQ4_XS tensor directories.
#include "strata/core/layout.hpp"
#include <algorithm>
#include <stdexcept>

namespace strata::core {
bool tensor_shape_matches(const TensorSpec& want, const std::vector<uint64_t>& shape) {
    if (shape == want.shape) return true;
    if (!want.allow_trailing_unit_dim || shape.size() != want.shape.size() + 1 || shape.back() != 1) return false;
    return std::equal(want.shape.begin(), want.shape.end(), shape.begin());
}

std::vector<TensorSpec> glm5_tensor_layout(const ModelGeometry& g) {
    if (!g.is_glm5_next()) throw std::invalid_argument("glm5_tensor_layout requires glm5-next geometry");
    std::vector<TensorSpec> out;
    const auto E = g.n_embd, V = g.n_vocab, H = g.n_head, K = g.ssm_state_size;
    const auto I = g.ssm_value_dim, C = g.ssm_d_conv, M = g.hc * (g.hc + 2);
    const auto Q = g.q_lora_rank, L = g.kv_lora_rank, DK = g.mla_key_dim, DV = g.mla_value_dim;
    const auto IH = g.idx_q_heads, ID = g.idx_key_dim;
    auto add = [&](const std::string& name, std::initializer_list<int64_t> shape, bool optional = false, bool conv = false) {
        TensorSpec t{name, {}, optional, conv};
        for (auto d : shape) {
            if (d <= 0) throw std::invalid_argument("non-positive tensor dimension: " + name);
            t.shape.push_back(uint64_t(d));
        }
        out.push_back(std::move(t));
    };
    add("token_embd.weight", {E, V});
    add("output_norm.weight", {E});
    add("output.weight", {E, V});
    for (int64_t i = 0; i < g.n_layers_all(); ++i) {
        const auto p = "blk." + std::to_string(i) + ".";
        add(p + "attn_norm.weight", {E});
        add(p + "ffn_norm.weight", {E});
        if (i < g.n_layers) {
            for (const char* block : {"hc_attn_", "hc_ffn_"}) {
                add(p + block + "fn.weight", {g.hc_dim(), M});
                add(p + block + "base.weight", {M});
                add(p + block + "scale.weight", {3}); // pre, post, combination: reference tensor contract
            }
        }
        if (!g.is_sparse_layer(i)) {
            for (const char* role : {"q", "k", "v"}) {
                add(p + "ssm_conv1d_" + role + ".weight", {C, 1, I}, false, true);
                add(p + "attn_" + role + ".weight", {E, I});
            }
            add(p + "ssm_f_a.weight", {E, K});
            add(p + "ssm_f_b.weight", {K, I});
            add(p + "ssm_beta.weight", {E, H});
            add(p + "ssm_a", {H});
            add(p + "ssm_dt.bias", {I});
            add(p + "ssm_g_a.weight", {E, K});
            add(p + "ssm_g_b.weight", {K, I});
            add(p + "ssm_norm.weight", {K});
            add(p + "attn_output.weight", {I, E});
        } else {
            add(p + "attn_q_a_norm.weight", {Q});
            add(p + "attn_kv_a_norm.weight", {L});
            add(p + "attn_q_a.weight", {E, Q});
            add(p + "attn_q_b.weight", {Q, H * DK});
            add(p + "attn_kv_a_mqa.weight", {E, L}); // no rotary dimensions in glm5-next
            add(p + "attn_k_b.weight", {DK, L, H});
            add(p + "attn_v_b.weight", {L, DV, H});
            add(p + "attn_output.weight", {H * DV, E});
            const bool optional = i < g.n_layers && g.indexer_types.at(size_t(i)) == 0;
            add(p + "indexer.k_norm.weight", {ID}, optional);
            add(p + "indexer.k_norm.bias", {ID}, optional);
            add(p + "indexer.proj.weight", {E, IH}, optional);
            add(p + "indexer.attn_k.weight", {E, ID}, optional);
            add(p + "indexer.attn_q_b.weight", {Q, IH * ID}, optional);
            add(p + "indexer_compressor_gate.weight", {E, ID}, optional);
            add(p + "indexer_compressor_ape.weight", {ID, g.idx_kpool}, optional);
        }
        if (i < g.n_dense_layers) {
            add(p + "ffn_gate.weight", {E, g.n_ff_dense});
            add(p + "ffn_down.weight", {g.n_ff_dense, E});
            add(p + "ffn_up.weight", {E, g.n_ff_dense});
        } else {
            const auto F = g.n_ff_exp_by_layer.at(size_t(i));
            const auto S = F * g.n_expert_shared; // reference derives shared width per layer
            add(p + "ffn_gate_inp.weight", {E, g.n_expert});
            add(p + "exp_probs_b.bias", {g.n_expert});
            add(p + "ffn_gate_exps.weight", {E, F, g.n_expert});
            add(p + "ffn_up_exps.weight", {E, F, g.n_expert});
            add(p + "ffn_down_exps.weight", {F, E, g.n_expert});
            add(p + "ffn_gate_shexp.weight", {E, S});
            add(p + "ffn_up_shexp.weight", {E, S});
            add(p + "ffn_down_shexp.weight", {S, E});
        }
        if (i >= g.n_layers) {
            add(p + "nextn.eh_proj.weight", {2 * E, E});
            add(p + "nextn.enorm.weight", {E});
            add(p + "nextn.hnorm.weight", {E});
            add(p + "nextn.embed_tokens.weight", {E, V}, true);
            add(p + "nextn.shared_head_head.weight", {E, V}, true);
            add(p + "nextn.shared_head_norm.weight", {E}, true);
        }
    }
    return out;
}
}
