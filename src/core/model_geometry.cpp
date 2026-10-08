#include "strata/core/layout.hpp"
#include "strata/artifact/gguf_reader.hpp"

#include <cmath>
#include <limits>

namespace strata::core {
namespace {
class Metadata {
public:
    explicit Metadata(const GgufFile& file) : file(file) {}
    const MetaValue& get(const char* suffix) const {
        const std::string key = std::string("glm5-next.") + suffix;
        const auto* v = file.get(key);
        if (!v) throw std::runtime_error("geometry: missing " + key);
        return *v;
    }
    double number(const MetaValue& v, const char* key) const {
        if (!v.is_num() || !std::isfinite(v.num()))
            throw std::runtime_error(std::string("geometry: expected finite number: ") + key);
        return v.num();
    }
    int64_t integer(const char* key, int64_t minimum = 1) const {
        const auto& v = get(key);
        if (v.type != MetaType::U32 && v.type != MetaType::I32 &&
            v.type != MetaType::U64 && v.type != MetaType::I64)
            throw std::runtime_error(std::string("geometry: expected integer: ") + key);
        if (v.u > uint64_t((std::numeric_limits<int64_t>::max)()) || int64_t(v.u) < minimum)
            throw std::runtime_error(std::string("geometry: invalid integer: ") + key);
        return int64_t(v.u);
    }
    float real(const char* key) const { return float(number(get(key), key)); }
    bool boolean(const char* key) const {
        const auto& v = get(key);
        if (v.type != MetaType::BOOL) throw std::runtime_error(std::string("geometry: expected bool: ") + key);
        return v.u != 0;
    }
    template<class T> std::vector<T> array(const char* key, int64_t count, bool scalar = false) const {
        const auto& v = get(key);
        auto element = [&](const MetaValue& item) -> T {
            const double x = number(item, key);
            if constexpr (std::numeric_limits<T>::is_integer) {
                if (item.type == MetaType::BOOL || x < 0 || x > double((std::numeric_limits<int32_t>::max)()) ||
                    x != std::floor(x))
                    throw std::runtime_error(std::string("geometry: invalid array integer: ") + key);
            }
            return T(x);
        };
        if (scalar && v.is_num()) return std::vector<T>(size_t(count), element(v));
        if (v.type != MetaType::ARRAY || v.count != uint64_t(count) || v.items.size() != size_t(count))
            throw std::runtime_error(std::string("geometry: wrong array length: ") + key);
        std::vector<T> result;
        for (const auto& item : v.items) result.push_back(element(item));
        return result;
    }
    const GgufFile& file;
};
}

ModelGeometry geometry_from_gguf(const GgufFile& file) {
    ModelGeometry g;
    const auto* arch = file.get("general.architecture");
    if (!arch || arch->type != MetaType::STRING) throw std::runtime_error("geometry: missing general.architecture");
    g.architecture = arch->s;
    if (g.architecture == "qwen4exp") {
        const auto error = check_architecture(file);
        if (!error.empty()) throw std::runtime_error(error);
        if (const auto* v = file.get("qwen4exp.expert_count")) g.n_expert = int64_t(v->u);
        if (const auto* v = file.get("qwen4exp.expert_used_count")) g.n_expert_used = int64_t(v->u);
        return g;
    }
    if (!g.is_glm5_next()) throw std::runtime_error("geometry: unsupported architecture " + g.architecture);
    const Metadata m(file);
    const int64_t total = m.integer("block_count");
    // Resource bound, not a model dimension. Also bounds all per-layer allocations below.
    if (total > 4096) throw std::runtime_error("geometry: implausible block_count");
    g.n_mtp_layers = m.integer("nextn_predict_layers", 0);
    g.n_layers = total - g.n_mtp_layers;
    if (g.n_layers <= 0) throw std::runtime_error("geometry: nextn_predict_layers leaves no main layers");
    g.n_embd = m.integer("embedding_length");
    g.n_vocab = m.integer("vocab_size");
    g.n_ctx_train = m.integer("context_length");
    g.n_head = m.integer("attention.head_count");
    g.head_count_kv = read_head_count_kv(file);
    g.n_head_kv = *std::max_element(g.head_count_kv.begin(), g.head_count_kv.end());
    for (auto n : g.head_count_kv)
        if (n > g.n_head) throw std::runtime_error("geometry: head_count_kv exceeds head_count");
    g.qsa_interval = 0; // GLM uses the metadata array, never a periodic approximation.
    g.q_lora_rank = m.integer("attention.q_lora_rank");
    g.kv_lora_rank = m.integer("attention.kv_lora_rank");
    g.mla_key_dim = m.integer("attention.key_length_mla");
    g.mla_value_dim = m.integer("attention.value_length_mla");
    g.head_dim = g.mla_key_dim;
    g.cache_key_dim = m.integer("attention.key_length");
    g.cache_value_dim = m.integer("attention.value_length");
    g.rope_dim = m.integer("rope.dimension_count", 0);
    if (g.rope_dim != 0) throw std::runtime_error("geometry: glm5-next requires rope.dimension_count = 0");
    g.ssm_state_size = m.integer("kda.head_dim");
    g.ssm_k_heads = g.ssm_v_heads = g.n_head; // glm5-next.cpp:96-98, equal K/V heads
    g.ssm_value_dim = g.ssm_state_size * g.n_head;
    g.ssm_conv_channels = 3 * g.ssm_value_dim; // separate Q, K and V convolutions
    g.ssm_d_conv = m.integer("ssm.conv_kernel");
    g.kda_gate_lower_bound = m.real("kda.gate_lower_bound");
    g.idx_q_heads = m.integer("attention.indexer.head_count");
    g.idx_key_dim = m.integer("attention.indexer.key_length");
    g.idx_top_k = m.integer("attention.indexer.top_k");
    g.idx_kpool = m.integer("attention.indexer.kpool");
    if (g.idx_kpool <= 1 || g.idx_top_k % g.idx_kpool)
        throw std::runtime_error("geometry: invalid indexer kpool/top_k");
    // llama-hparams.h defaults when these optional metadata keys are absent.
    if (file.get("glm5-next.attention.indexer.kpool_select_tail"))
        g.idx_select_tail = m.boolean("attention.indexer.kpool_select_tail");
    g.indexer_types.assign(size_t(g.n_layers), 1);
    if (file.get("glm5-next.attention.indexer.types"))
        g.indexer_types = m.array<int64_t>("attention.indexer.types", g.n_layers, true);
    if (file.get("glm5-next.attention.indexer.index_share_mtp"))
        g.idx_share_mtp = m.boolean("attention.indexer.index_share_mtp"); // recorded only; reference graph absent
    g.hc = m.integer("hyper_connection.count");
    g.hc_lr = 0; // mHC has no Qwen low-rank mixer
    g.hc_sinkhorn_iters = m.integer("hyper_connection.sinkhorn_iterations");
    g.hc_eps = m.real("hyper_connection.epsilon");
    g.rms_eps = m.real("attention.layer_norm_rms_epsilon");
    g.norm_eps = m.real("attention.layer_norm_epsilon");
    if (!(g.rms_eps > 0 && g.norm_eps > 0 && g.hc_eps > 0))
        throw std::runtime_error("geometry: normalization epsilons must be positive");
    g.n_expert = m.integer("expert_count");
    g.n_expert_used = m.integer("expert_used_count");
    if (g.n_expert_used > g.n_expert) throw std::runtime_error("geometry: expert_used_count exceeds expert_count");
    g.n_dense_layers = m.integer("leading_dense_block_count", 0);
    if (g.n_dense_layers > g.n_layers) throw std::runtime_error("geometry: too many leading dense blocks");
    g.n_ff_dense = m.integer("feed_forward_length");
    g.n_ff_exp_by_layer = m.array<int64_t>("expert_feed_forward_length", total, true);
    for (auto n : g.n_ff_exp_by_layer) if (n <= 0) throw std::runtime_error("geometry: invalid expert feed-forward length");
    g.n_ff = g.n_ff_exp_by_layer.front();
    g.n_expert_shared = m.integer("expert_shared_count");
    g.n_ff_shared = m.integer("expert_shared_feed_forward_length");
    g.expert_groups = m.integer("expert_group_count");
    g.expert_groups_used = m.integer("expert_group_used_count");
    g.expert_gating_func = m.integer("expert_gating_func");
    g.expert_weights_scale = m.real("expert_weights_scale");
    g.expert_weights_norm = m.boolean("expert_weights_norm");
    g.swiglu_clamp_exp = m.array<float>("swiglu_clamp_exp", total, true);
    g.swiglu_clamp_shared = m.array<float>("swiglu_clamp_shexp", total, true);
    for (auto limit : g.swiglu_clamp_exp)
        if (!std::isfinite(limit) || limit < 0) throw std::runtime_error("geometry: invalid swiglu_clamp_exp");
    for (auto limit : g.swiglu_clamp_shared)
        if (!std::isfinite(limit) || limit < 0) throw std::runtime_error("geometry: invalid swiglu_clamp_shexp");
    return g;
}
}
