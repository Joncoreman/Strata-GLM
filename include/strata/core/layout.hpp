// include/strata/core/layout.hpp - the pack's LAYOUT, resolved by name and checked against the kernels.
//
// `WeightTable` answers "where are the bytes".  This answers "is this the tensor I think it is", and it is
// the layer where a pack and a kernel disagree - which is the failure this project keeps paying for.  Two
// examples, both real:
//
//   * round 194 sized the indexer key store from `indexer.head_count = 4`, which counts QUERY heads.  The
//     cached key is ONE shared head of 128 (`indexer.k_proj` is [2560, 128]), so the term was 4x too big and
//     the error survived a round because it moved in the direction that TIGHTENS the budget.
//   * `docs/semantics.md` and the tensor manifest agree on every dimension here, but nothing CHECKED that a
//     named tensor had the shape the kernel reading it assumes.
//
// So this header does two things: it names the tensors per layer type, and it asserts every shape the
// kernels depend on.  A mismatch is reported with the tensor name, the shape found and the shape required,
// at LOAD time - not as a wrong number inside a GEMV at token 4000.
#pragma once

#include "strata/core/weights.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace strata { class GgufFile; }

namespace strata::core {

/// The model's geometry, taken from `docs/semantics.md` and the artifact's own metadata.  Every field here
/// is a number a kernel depends on, so a change is a change to a kernel contract and not a tuning knob.
struct ModelGeometry {
    std::string architecture = "qwen4exp";
    int64_t n_embd = 2560;
    int64_t n_layers = 48;
    int64_t qsa_interval = 4;      ///< every 4th layer is full attention: layers 3, 7, ... 47

    // GDN (36 layers)
    int64_t ssm_state_size = 128;
    int64_t ssm_k_heads = 16;
    int64_t ssm_v_heads = 48;
    int64_t ssm_d_conv = 4;
    int64_t ssm_conv_channels = 10240;   ///< 2*128*16 + 128*48
    int64_t ssm_value_dim = 6144;        ///< 128 * 48

    // QSA (12 layers)
    int64_t n_head = 24;
    int64_t n_head_kv = 2;
    int64_t head_dim = 256;
    int64_t idx_q_heads = 4;
    int64_t idx_key_dim = 128;

    // gated residual, on every layer
    int64_t hc = 4;
    int64_t hc_lr = 320;

    // MoE, on every layer
    int64_t n_expert = 512;
    int64_t n_ff = 640;

    // GLM5-Next metadata. Existing Qwen fields/defaults above retain their old meaning.
    int64_t n_mtp_layers = 1, n_vocab = 0, n_ctx_train = 0;
    int64_t n_expert_used = 10, n_dense_layers = 0, n_ff_dense = 0, n_ff_shared = 0;
    int64_t n_expert_shared = 0, expert_groups = 0, expert_groups_used = 0, expert_gating_func = 0;
    int64_t q_lora_rank = 0, kv_lora_rank = 0, mla_key_dim = 0, mla_value_dim = 0;
    int64_t cache_key_dim = 0, cache_value_dim = 0, rope_dim = 0;
    int64_t idx_top_k = 0, idx_kpool = 0, hc_sinkhorn_iters = 0;
    float rms_eps = 1e-6f, norm_eps = 0, hc_eps = 0, kda_gate_lower_bound = 0;
    float expert_weights_scale = 1;
    bool expert_weights_norm = true, idx_select_tail = true, idx_share_mtp = false;
    // Full layer arrays include the NextN blocks; n_layers counts only the main trunk.
    std::vector<int64_t> head_count_kv, n_ff_exp_by_layer, indexer_types;
    std::vector<float> swiglu_clamp_exp, swiglu_clamp_shared;

    bool is_glm5_next() const { return architecture == "glm5-next"; }
    int64_t n_layers_all() const { return n_layers + n_mtp_layers; }
    bool is_sparse_layer(int64_t layer) const {
        if (is_glm5_next()) return head_count_kv.at(static_cast<size_t>(layer)) != 0;
        return layer % qsa_interval == qsa_interval - 1;
    }

    int64_t hc_dim() const { return hc * n_embd; }
    /// `layer % qsa_interval == qsa_interval - 1` is full attention.  Derived, not a second list.
    int64_t n_qsa_layers() const {
        if (!is_glm5_next()) return n_layers / qsa_interval;
        int64_t n = 0;
        for (int64_t i = 0; i < n_layers; ++i) n += is_sparse_layer(i);
        return n;
    }
    int64_t n_gdn_layers() const { return n_layers - n_qsa_layers(); }
};

/// True for the full-attention layers.  `docs/semantics.md` gives this twice over - `full_attention_interval
/// = 4` and an explicit `attention.compress_ratios` array - and this is the first of the two.
inline bool is_qsa_layer(const ModelGeometry& g, int64_t layer) {
    return g.is_sparse_layer(layer);
}

/// Header-only input: never reads weight payloads. Throws a named metadata error on invalid input.
/// Qwen retains its legacy geometry and only applies its existing expert-count overrides.
ModelGeometry geometry_from_gguf(const strata::GgufFile& file);

/// GGUF dimensions (contiguous axis first), including routed experts and the NextN layer.
/// Optional tensors follow llama.cpp's TENSOR_NOT_REQUIRED contract.
struct TensorSpec {
    std::string name;
    std::vector<uint64_t> shape;
    bool optional = false;
    bool allow_trailing_unit_dim = false; // KDA convolutions: [conv,1,inner] or [conv,1,inner,1]
};
std::vector<TensorSpec> glm5_tensor_layout(const ModelGeometry& g);
bool tensor_shape_matches(const TensorSpec& want, const std::vector<uint64_t>& shape);

/// One layer's tensors, resolved by NAME.  `get("attn_qkv.weight")` looks up `blk.<layer>.attn_qkv.weight`
/// and returns null if the pack does not have it - a null is information, because a GDN layer has no
/// `attn_q` and a QSA layer has no `attn_qkv`.
///
/// Holds a reference to the table; the table must outlive it.
class LayerView {
public:
    LayerView(const WeightTable& table, int64_t layer) : table_(&table), layer_(layer) {}

    int64_t layer() const { return layer_; }
    std::string name(const char* suffix) const;
    const WeightRef* get(const char* suffix) const { return table_->find(name(suffix)); }

private:
    const WeightTable* table_;
    int64_t layer_;
};

/// Every shape the kernels depend on, asserted for ONE layer.  Returns false and fills `err` with the first
/// mismatch, naming the tensor, what it has and what is required.
///
/// This is deliberately a separate function from `LayerView`: a caller that only wants the pointers should
/// not pay for the checks, and a caller that wants the checks should get ALL of them rather than the ones
/// its own call site happens to touch.
bool check_layer(const WeightTable& table, const ModelGeometry& g, int64_t layer, std::string& err);

/// `check_layer` over every layer, plus the cross-layer properties: exactly 12 QSA layers at the right
/// indices, every GDN layer having the GDN set and no QSA tensor, and vice versa.  Returns the first failure.
bool check_all(const WeightTable& table, const ModelGeometry& g, std::string& err);

}  // namespace strata::core
