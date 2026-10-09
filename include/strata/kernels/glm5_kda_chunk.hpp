// include/strata/kernels/glm5_kda_chunk.hpp - the KDA layers of glm5-next over a prompt's sub-batch at once
// (2026-10-09, the prompt path rewrite stage 3).
//
// The window path runs the KDA recurrence a token at a time inside gdn_step_norm_kda_kernel, a slice of <= 8 tokens a
// launch.  This is the chunked (WY / delta-rule) form over 16-token chunks: a prepare kernel per (chunk, head) and one
// recurrence kernel that walks every chunk of the sub-batch with the state in registers - ported from Infernix
// (https://github.com/Wallawalla47/Infernix, Apache-2.0, src/ops/linear_attention/kimi_delta_attention/chunked/),
// adapted to this engine's buffers: q/k/v read as F32 from the conv output (q and k already L2-normalised), the gate as
// this engine's natural-log decay per key dimension (kda_gate_beta), beta already the sigmoid, the state in this engine's
// [key][head][value] layout, the output F32 (scaled, before the norm).  The conv, the gates and the output norm have
// sub-batch kernels here too, with the window kernels' arithmetic per element.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels {

/// Bytes of the per-(chunk, head) packets for `tokens` tokens and `h_v` value heads.
std::size_t glm5_kda_chunk_workspace_bytes(int h_v, int tokens);

/// The depthwise causal conv (width 4) + SiLU over T tokens; channels [0, DI) from q, [DI, 2 DI) from k, [2 DI, 3 DI)
/// from v (each [T][DI]); the first `qk_heads` 128-channel heads L2-normalised (eps).  hist [3 DI][3] holds the three
/// inputs before token 0 and is left holding the last three (as gdn_conv_l2_multi + gdn_conv_commit).  h [T][3 DI].
void glm5_kda_conv_chunk(float* hist, const float* q, const float* k, const float* v, const float* conv_w, float* h,
                         int DI, int qk_heads, float eps, int T, void* stream);

/// gate[t][head][i] = lb * sigmoid(-(f + dt_b) * A[head]) and beta = sigmoid(beta_logit), as kda_gate_beta_multi.
void glm5_kda_gate_beta_chunk(const float* f, const float* dt_b, const float* ssm_a, float lower_bound,
                              const float* beta_logit, float* gate, float* beta, int h_v, int T, void* stream);

/// The recurrence over T tokens: h [T][C] (q | k | v), gate [T][h_v][128], beta [T][h_v], state [128][h_v][128]
/// (updated in place), y [T][h_v * 128] = o / sqrt(128).  `v_bf16` [T][h_v][128] and `workspace`
/// (glm5_kda_chunk_workspace_bytes) are scratch.  h_k == h_v.
void glm5_kda_chunk(const float* h, int C, const float* gate, const float* beta, float* state, float* y, int h_v,
                    int T, uint16_t* v_bf16, void* workspace, int sm_count, void* stream);

/// y = y * rsqrt(mean(y^2) + eps) * gamma * sigmoid(z) per (token, head), as the window kernel's output.
void glm5_kda_out_norm(float* y, const float* z, const float* gamma, float eps, int h_v, int T, void* stream);

}  // namespace strata::kernels
