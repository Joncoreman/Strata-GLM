// include/strata/kernels/glm5_misc.hpp - small glm5-next (GLM-5.3-Flash) helpers for the verify window
// (src/kernels/cuda/glm5_misc.cu).  Graph-capturable: no allocation, no synchronization.
#pragma once

#include <cstdint>

namespace strata::kernels {

/// y[t][o] = sum_i W[o][i] * x[t][i] for an F32 weight stored as GGUF [in, out] (row o contiguous): the router
/// (ffn_gate_inp, 4096 -> 288) and the indexer's head weights (indexer.proj, 4096 -> 32).  `scale` multiplies y.
void glm5_f32_gemv(const float* w, const float* x, float* y, int n_in, int n_out, int n_tok, float scale, void* stream);
/// dst[b][a][d] = src[a][b][d]  (A x B blocks of D floats): [token][head][dim] <-> [head][token][dim]
void glm5_transpose_abd(const float* src, float* dst, int a, int b, int d, void* stream);
/// dst[t] = q[t] | k[t] | v[t]  (each n floats): the KDA projections into the conv's 3n channels
void glm5_pack3(const float* q, const float* k, const float* v, float* dst, int n, int n_tok, void* stream);
/// The MLA cache write: token t's latent (dim floats, kv[t]) into fp16 row page_table[c / page] * page + c % page of
/// `pool`, c = steps[t * step_count + 0] (kStepPos, the cell).
void glm5_latent_store(const float* kv, uint16_t* pool, const int32_t* page_table, const int32_t* steps, int step_count,
                       int page_size, int dim, int n_tok, void* stream);
/// The same into a q4_0 pool (dim / 32 ggml blocks of 18 bytes per row; read by mla_decode_attn_batch_q4).
void glm5_latent_store_q4(const float* kv, uint8_t* pool, const int32_t* page_table, const int32_t* steps, int step_count,
                          int page_size, int dim, int n_tok, void* stream);
/// The prompt path's MMQ experts (2026-10-09): in place, row r of `rows` rows of 2 * n_ff gate|up values gets
/// silu(gate) * up in its first n_ff (GLM's clamp when limit > 0: gate <= limit, |up| <= limit).
void glm5_swiglu_rows(float* gu, int64_t rows, int n_ff, float limit, void* stream);
/// dst[idx[r]] = src[r] for r < n_rows (rows of n_embd floats): the prompt path's CPU expert outputs into `parts`
void glm5_copy_rows(const float* src, const int32_t* idx, float* dst, int n_rows, int n_embd, void* stream);
/// dst[0] = a, dst[1] = b on the stream (an MMQ product's bounds)
void glm5_set2(int32_t* dst, int32_t a, int32_t b, void* stream);
/// The prompt path's layer walk (2026-10-09): for u < n_u, acc[u_tok[u]] += sum_{j in [u_off[u], u_off[u+1])}
/// e_w[j] * rows[e_row[j]] (n_embd floats each), the entries in order; distinct u_tok per call.
void glm5_scatter_add(const float* rows, const int32_t* u_tok, const int32_t* u_off, const int32_t* e_row, const float* e_w,
                      int n_u, float* acc, int n_embd, void* stream);

}  // namespace strata::kernels
