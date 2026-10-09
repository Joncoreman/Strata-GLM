#pragma once
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/rope_scaling.hpp"

namespace strata::kernels {

// Configure before session construction/capture. Disabled independently of the
// native QSA norm/gate switch; existing captures retain their selected kernels.
void native_qsa_indexer_set_enabled(bool enabled);
bool native_qsa_indexer_enabled();

// Pinned F16 raw-key cache / F32 pooling / weighted RMSNorm / text IMRoPE.
// Fixed single-sequence geometry: idx_dim=128, idx_block=4, n_rot=64.
// Uses the existing buffers: tail[3,128] stores half-rounded raw values expanded
// to F32; dead[128] is the repeated-cell0 spare key; pooled[max_cells/4+1,128]
// contains completed blocks followed by the spare; block_pos[1] changes only on
// completion and records that block's first absolute position.
//
// relative_pos_device is a cell index, advancing 0..max_cells-1 (use kStepPos,
// not the absolute RoPE position vector). The sequence is text,
// contiguous, with nonnegative pos_base divisible by four. Clear state before
// starting a new sequence. Completed block b rotates at pos_base+4*b; the spare
// always rotates at zero, including nonzero sequence bases, as in the oracle.
// Under rope scaling (rope_scaling.hpp) the zero angle keeps the spare's
// rotation an identity; YaRN's mscale rides in through cos(0) on both the pooled
// keys and the indexer queries, so selection is unchanged.
//
// Explicit nonnull stream; no allocation or synchronization. All spans must be
// aligned and disjoint, remain valid through replay, and contain finite values;
// raw values must round to finite F16 and epsilon must be finite and positive.
// max_cells is a positive fixed capacity; pos_base+max_cells must fit int32.
// Out-of-range device positions are ignored defensively, not host-validated.
// This adapter does not alter score accumulation, top-k, or finite 1e9 tail bias.
// perf-review C-2: the appends of cells [p0, p0 + n) at once, leaving the buffers exactly as n calls of the
// single append in order would (the prompt path appends a chunk before any query reads the state). raw [n, 128].
// Host-side positions (not for a captured graph).
void native_qsa_indexer_append_batch(const float* raw, int64_t n, int64_t p0, int32_t pos_base, const float* gamma,
                                     float epsilon, const QsaIndexerBuffers& b, const QsaShapes& s, int64_t max_cells,
                                     const RopeScaling& scaling, void* stream);

void native_qsa_indexer_append(const float* raw, const int32_t* relative_pos_device,
                               int32_t pos_base, const float* gamma, float epsilon,
                               const QsaIndexerBuffers& buffers, const QsaShapes& shapes,
                               int64_t max_cells, const RopeScaling& scaling, void* stream);

void native_qsa_indexer_append_steps(const float* raw, const int32_t* relative_pos_device,
                                     int pos_stride, int n_steps, int32_t pos_base,
                                     const float* gamma, float epsilon,
                                     const QsaIndexerBuffers& buffers, const QsaShapes& shapes,
                                     int64_t max_cells, const RopeScaling& scaling, void* stream);
// ---- glm5-next (GLM-5.3-Flash) k-pool indexer (llama.cpp b11461 glm5-next.cpp:756-859) -------------------------
// Per cell: key = LayerNorm(indexer.attn_k x) * k_norm + k_norm_b (mean-centred, eps layer_norm_epsilon) and gate =
// indexer_compressor_gate x, 128 each.  The first three cells of a block wait in `tail` (key | gate, 3 x 256 floats);
// the fourth completes the block: pooled[b][d] = sum_j softmax_j(gate_j[d] + ape[j][d]) * key_j[d] (no norm, no RoPE).
// The score of block b for a query: sum_h w_h * relu(q_h . pooled[b]) with w = indexer_proj(x) / sqrt(128 * 32)
// (the caller scales), q = indexer_attn_q_b(qr) [32 x 128].  The incomplete tail block n_bid gets kGlm5TailScore so the
// selection (qsa_block_topk) always keeps it; selection width 4 * min(n_bid, 512) + (n_kv - 4 * n_bid).
struct Glm5KpoolBuffers {
    float* tail = nullptr;        ///< [3][2 * 128] key | gate of the block's first three cells
    uint16_t* pooled = nullptr;   ///< [max_cells / 4 + 1][128] BF16 (2026-10-09: F32 before; 369 -> 185 MiB at 262K)
};
inline constexpr float kGlm5TailScore = 1.0e30f;
/// raw_k, gate: [128] each for the cell at *relative_pos_device (kStepPos).  Capturable; no allocation.
void glm5_kpool_append(const float* raw_k, const float* gate, const int32_t* relative_pos_device, const float* k_norm,
                       const float* k_norm_b, const float* ape, float epsilon, const Glm5KpoolBuffers& b,
                       int64_t max_cells, void* stream, float* key_out = nullptr);
/// The prompt path (2026-10-10): cells first_cell .. first_cell + n - 1 (consecutive; raw_k / gate [n][128]) in two
/// launches - the blocks they complete in parallel, then the trailing incomplete block's cells into the tail - each
/// cell's arithmetic glm5_kpool_append's, the same pooled keys and tail.
void glm5_kpool_append_batch(const float* raw_k, const float* gate, int64_t first_cell, int64_t n, const float* k_norm,
                             const float* k_norm_b, const float* ape, float epsilon, const Glm5KpoolBuffers& b,
                             int64_t max_cells, void* stream);
/// The prompt path's scores as a GEMM (2026-10-10): the caller converts q to BF16 (glm5_f32_to_bf16) and computes
/// S [nq * 32][tb] = q . pooled[b0 .. b0 + tb)^T with cuBLAS; this sums them as glm5_kpool_scores does
/// (sum_h w_h relu(S), h ascending; the tail block n_bid gets kGlm5TailScore) into scores [nq][max_blocks].
void glm5_f32_to_bf16(const float* x, uint16_t* y, int64_t n, void* stream);
void glm5_kpool_scores_reduce(const float* S, const float* w, const int32_t* steps, int64_t nq, int64_t b0, int tb,
                              int64_t max_blocks, float* scores, void* stream);
/// scores [nq][max_blocks] for blocks 0..n_bid (steps[kStepNBid]); q [nq][32][128], w [nq][32].  active_blocks > 0: no
/// query of the call has more than this many blocks (n_bid + 1), so the grid stops there (the prompt path; the same scores)
void glm5_kpool_scores(const float* q, const float* w, const uint16_t* pooled, const int32_t* steps, int64_t nq,
                       int64_t max_blocks, float* scores, void* stream, int64_t active_blocks = -1);

} // namespace strata::kernels
