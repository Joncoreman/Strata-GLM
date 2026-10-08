// include/strata/core/glm5_mtp.hpp - glm5-next (GLM-5.3-Flash) MTP drafter: the NextN layer (blk.45), 2026-10-08.
//
// As llama.cpp src/models/glm5-next.cpp and vLLM glm5next/common/mtp.py run it: for the token x at cell c+1 and the
// trunk's h at cell c (output_norm of the mean of the 4 streams - the head's input),
//     u = eh_proj([enorm(embed(x)); hnorm(h)])
//     u += DSA(attn_norm(u))          (MLA + k-pool indexer over the drafter's own cache of cells)
//     u += MoE(ffn_norm(u))           (sigmoid router + exp_probs_b, x2.5, shared expert)
//     h' = shared_head_norm(u);  the draft = argmax head(h');  h' is the next chain step's h.
// One residual stream (no hyper-connections).  The catch-up runs every accepted cell through the layer (its K/V and
// indexer keys); a draft chain writes cells beyond them and restores the indexer tail afterwards.
//
// Phase A: the routed experts of blk.45 are not loaded (no expert tier holds them): the MoE is the shared expert
// alone.  The verify window checks every draft, so this lowers the acceptance only, never the output.
#pragma once

#include "strata/core/weights.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {

class NativeHead;
struct ModelGeometry;

class Glm5Mtp {
public:
    Glm5Mtp() = default;
    ~Glm5Mtp();
    Glm5Mtp(const Glm5Mtp&) = delete;
    Glm5Mtp& operator=(const Glm5Mtp&) = delete;

    bool init(const WeightTable& wt, const ModelGeometry& g, const NativeHead* head, int64_t max_cells, int max_t,
              std::string& err);
    /// Catch-up: rows t < n pair h_rows[t] (device, the trunk's h at cell c0 + t) with tokens[t] (host, the token at
    /// cell c0 + t + 1).  Writes the drafter's cache for cells c0 .. c0 + n - 1.  When `draft_from` >= 0, also returns
    /// the first draft from row `draft_from` (its output) in *first and that row's h' becomes the chain's h.
    bool catchup(const float* h_rows, const int32_t* tokens, int n, int64_t c0, int draft_from, int32_t* first,
                 std::string& err);
    /// The chain after catchup(..., draft_from = a, &d1): `n_more` further drafts, cells c_last + 1 ...  The indexer
    /// tail is restored when it returns (the cells it wrote are rewritten by the next catch-up).
    bool chain(int32_t first, int64_t c_last, int n_more, int32_t* out, std::string& err);

    double ms_draft = 0;
    int64_t rounds = 0;

private:
    bool layer(const int32_t* tok_dev, const float* h_in, int n, int64_t c0, std::string& err);
    struct Impl;
    Impl* m_ = nullptr;
};

}  // namespace strata::core
