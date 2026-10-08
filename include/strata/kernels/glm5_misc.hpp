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

}  // namespace strata::kernels
