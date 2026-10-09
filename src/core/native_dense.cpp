#include "strata/core/native_dense.hpp"
#include "strata/core/weights.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <climits>
#include <exception>
#include <cstdlib>
#include <limits>
#include <memory>
#include <set>

namespace strata::core {
namespace {
bool eligible(const strata::TensorInfo& tensor, bool include_ple_key) {
    const auto& name = tensor.name;
    if (name.rfind("blk.", 0) != 0) return false;
    // Match the native PLE kernel: Q2_0, IQ3_XXS, IQ4_XS and Q8_0 (UD-Q4_K_XL). Other keys retain the packed BF16
    // fallback.
    if (name == "blk.1.ple_key.weight")
        return include_ple_key && (tensor.type == 42 || tensor.type == 18 || tensor.type == 23 || tensor.type == 8);
    static const char* suffixes[] = {".attn_qkv.weight", ".attn_gate.weight", ".ssm_out.weight",
        ".attn_q.weight", ".attn_k.weight", ".attn_v.weight", ".attn_output.weight",
        ".ffn_gate_shexp.weight", ".ffn_up_shexp.weight", ".ffn_down_shexp.weight",
        // glm5-next (all Q8_0 in its GGUFs; none of these names occurs in qwen4exp, whose BF16 ssm_beta is not an
        // MMVQ type anyway): the dense FFN of layers 0-2, mHC, KDA gates, MLA, the k-pool indexer, NextN
        ".ffn_gate.weight", ".ffn_up.weight", ".ffn_down.weight", ".hc_attn_fn.weight", ".hc_ffn_fn.weight",
        ".ssm_beta.weight", ".ssm_f_a.weight", ".ssm_f_b.weight", ".ssm_g_a.weight", ".ssm_g_b.weight",
        ".attn_q_a.weight", ".attn_q_b.weight", ".attn_kv_a_mqa.weight", ".attn_k_b.weight", ".attn_v_b.weight",
        ".indexer.attn_k.weight", ".indexer.attn_q_b.weight", ".indexer_compressor_gate.weight",
        ".nextn.eh_proj.weight"};
    for (const char* suffix : suffixes) if (name.ends_with(suffix)) return true;
    return false;
}
struct DeviceFree { void operator()(void* p) const { if (p) cudaFree(p); } };
using DevicePtr = std::unique_ptr<void, DeviceFree>;
struct Pending {
    WeightRef* ref;
    int type;
    uint64_t bytes;
    DevicePtr data;
};
}

bool NativeDense::served_names(const std::vector<std::string>& shards, bool include_ple_key,
                               std::set<std::string>& out, std::string& err) {
    try {
        for (const auto& path : shards) {
            strata::GgufFile gguf(path);
            for (const auto& tensor : gguf.tensors())
                if (eligible(tensor, include_ple_key) && strata::kernels::native_mmvq_supported(tensor.type) &&
                    (tensor.shape.size() == 2 || tensor.shape.size() == 3))   // 3: glm5-next's per-head wk_b / wv_b
                    out.insert(tensor.name);
        }
        return true;
    } catch (const std::exception& error) {
        err = std::string("native dense: ") + error.what();
        return false;
    }
}

bool NativeDense::keep_unquantized_ple_key(const std::string& pack_dir, std::set<std::string>& skip,
                                           std::string& err) {
    const std::string key = "blk.1.ple_key.weight";
    if (!skip.count(key)) return true;
    int code_bits = -1;
    if (!WeightTable::index_code_bits(pack_dir, key, code_bits, err)) return false;
    if (code_bits == 0) skip.erase(key);
    return true;
}

NativeDense::~NativeDense() {
    if (scratch_) cudaFree(scratch_);
    for (void* p : weights_) cudaFree(p);
}

bool NativeDense::load(const std::vector<std::string>& shards, WeightTable& table, std::string& err,
                       bool include_ple_key, int64_t layer_lo, int64_t layer_hi,
                       const std::vector<std::pair<int64_t, int64_t>>* skip) {
    if (scratch_ || !weights_.empty()) { err = "native dense: already loaded"; return false; }
    if (shards.empty()) { err = "native dense: at least one GGUF shard is required"; return false; }
    try {
        std::vector<Pending> pending;
        std::set<std::string> seen;
        int max_in = 0;
        uint64_t total = 0;
        uint64_t split_count = 0, split_tensors = 0;
        std::set<uint64_t> split_numbers;
        bool have_architecture = false;
        for (const auto& path : shards) {
            strata::GgufFile gguf(path);
            const auto* count = gguf.get("split.count");
            const auto* number = gguf.get("split.no");
            const auto* tensors = gguf.get("split.tensors.count");
            if (gguf.get("general.architecture")) {
                err = strata::check_architecture(gguf);
                if (!err.empty()) return false;
                have_architecture = true;
                if (count && number && tensors && number->u == 0 && count->u > 1) {
                    split_count = count->u;
                    split_tensors = tensors->u;
                }
            } else if (!have_architecture || !split_count || !count || !number || !tensors ||
                       count->u != split_count || number->u == 0 || number->u >= split_count ||
                       tensors->u != split_tensors) {
                err = "native dense: additional shard must match the architecture-validated first shard's split metadata";
                return false;
            }
            if (number && !split_numbers.insert(number->u).second) {
                err = "native dense: duplicate split shard number"; return false;
            }
            std::vector<uint64_t> offsets;
            for (const auto& tensor : gguf.tensors()) offsets.push_back(tensor.offset);
            std::sort(offsets.begin(), offsets.end());
            if (std::adjacent_find(offsets.begin(), offsets.end()) != offsets.end()) {
                err = "native dense: tensor payload offsets overlap"; return false;
            }
            // Validate every directory span, including tensors we do not upload:
            // an ignored tensor must not overlap the native matrix that follows it.
            const uint64_t payload = gguf.file_size() - gguf.data_start();
            for (const auto& tensor : gguf.tensors()) {
                int block_elements = 0, block_bytes = 0;
                uint64_t elements = 1;
                if (tensor.shape.empty() || !strata::block_geometry(tensor.type, block_elements, block_bytes) ||
                    tensor.shape[0] % (uint64_t) block_elements != 0) {
                    err = "native dense: invalid block geometry " + tensor.name; return false;
                }
                for (uint64_t dimension : tensor.shape) {
                    if (!dimension || elements > (std::numeric_limits<uint64_t>::max)() / dimension) {
                        err = "native dense: invalid tensor extent " + tensor.name; return false;
                    }
                    elements *= dimension;
                }
                const uint64_t blocks = elements / (uint64_t) block_elements;
                if (blocks > (std::numeric_limits<uint64_t>::max)() / (uint64_t) block_bytes) {
                    err = "native dense: tensor byte count overflow " + tensor.name; return false;
                }
                const uint64_t bytes = blocks * (uint64_t) block_bytes;
                if (tensor.offset > payload || bytes > payload - tensor.offset) {
                    err = "native dense: truncated payload " + tensor.name; return false;
                }
                const auto next = std::upper_bound(offsets.begin(), offsets.end(), tensor.offset);
                if (next != offsets.end() && bytes > *next - tensor.offset) {
                    err = "native dense: overlapping payload " + tensor.name; return false;
                }
            }
            for (const auto& tensor : gguf.tensors()) {
                if (!eligible(tensor, include_ple_key)) continue;
                if (layer_hi > 0) {   // MULTINODE: a stage engine uploads only its own layers
                    const long long l = std::atoll(tensor.name.c_str() + 4);   // "blk.<L>."
                    if (l < layer_lo || l >= layer_hi) continue;
                }
                if (skip != nullptr && tensor.name.rfind("blk.", 0) == 0) {   // layers this engine never runs
                    const long long l = std::atoll(tensor.name.c_str() + 4);
                    if (std::any_of(skip->begin(), skip->end(), [l](const auto& r) { return l >= r.first && l < r.second; }))
                        continue;
                }
                if (!seen.insert(tensor.name).second) {
                    err = "native dense: duplicate tensor " + tensor.name; return false;
                }
                auto found = table.table_.find(tensor.name);
                if (found == table.table_.end()) {
                    err = "native dense: tensor absent from canonical table: " + tensor.name; return false;
                }
                auto& ref = found->second;
                if (ref.native_data) { err = "native dense: override already attached"; return false; }
                if (!strata::kernels::native_mmvq_supported(tensor.type)) continue;
                // #326: the pack keeps an unquantized (--compat-bf16) key, which the PLE reads from the arena
                if (tensor.name == "blk.1.ple_key.weight" && !ref.quantized()) continue;
                // a 3-D tensor (glm5-next attn_k_b / attn_v_b: [in, out, head]) is served as one [in, out * head]
                // matrix: head h's rows start at h * out
                const uint64_t rows = tensor.shape.size() == 3 ? tensor.shape[1] * tensor.shape[2]
                                                               : (tensor.shape.size() == 2 ? tensor.shape[1] : 0);
                if (!ref.quantized() || (tensor.shape.size() != 2 && tensor.shape.size() != 3) ||
                    ref.ne0 <= 0 || ref.ne0 > INT_MAX || ref.ne1 <= 0 || ref.ne1 > INT_MAX ||
                    tensor.shape[0] != (uint64_t) ref.ne0 || rows != (uint64_t) ref.ne1) {
                    err = "native dense: incompatible matrix " + tensor.name; return false;
                }
                const auto bytes = strata::kernels::native_mmvq_weight_bytes(
                    tensor.type, (int) ref.ne0, (int) ref.ne1);
                void* allocation = nullptr;
                auto status = cudaMalloc(&allocation, bytes);
                DevicePtr data(allocation);
                if (status == cudaSuccess)
                    status = cudaMemcpy(data.get(), gguf.tensor_data(tensor), bytes, cudaMemcpyHostToDevice);
                if (status != cudaSuccess) {
                    err = "native dense upload " + tensor.name + ": " + cudaGetErrorString(status); return false;
                }
                max_in = (std::max)(max_in, (int) ref.ne0);
                total += bytes;
                pending.push_back(Pending{&ref, (int) tensor.type, bytes, std::move(data)});
            }
        }
        if (pending.empty()) { err = "native dense: no supported GDN/QSA matrices in supplied shards"; return false; }
        void* allocation = nullptr;
        const auto status = cudaMalloc(&allocation, strata::kernels::native_q8_1_bytes(max_in));
        DevicePtr scratch(allocation);
        if (status != cudaSuccess) { err = std::string("native dense scratch: ") + cudaGetErrorString(status); return false; }
        // All checks and allocations finish before publishing any reference.
        weights_.reserve(pending.size());
        for (auto& item : pending) {
            item.ref->native_data = item.data.get();
            item.ref->native_type = item.type;
            item.ref->native_q8_1 = scratch.get();
            weights_.push_back(item.data.release());
        }
        scratch_ = scratch.release();
        bytes_ = total;
        return true;
    } catch (const std::exception& error) {
        err = std::string("native dense: ") + error.what();
        return false;
    }
}
} // namespace strata::core
