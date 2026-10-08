// src/core/verify_glm5.cpp - the verify window for glm5-next (GLM-5.3-Flash), 2026-10-08.
//
// The same window as record_window (verify.cpp): T tokens, one captured graph, the expert doorbell / GPU plan / CPU
// pool / combine machinery of the Qwen path unchanged.  What differs per layer (glm-notes/配線図_decode経路):
//   mHC read (attention side; the previous layer's FFN write folded in)       fused_mhc_read_multi
//   KDA layers: q|k|v Q8_0 -> conv+SiLU+L2 -> f_a/f_b, beta, g_a/g_b -> KDA recurrence + RMSNorm x gamma x
//               sigmoid(z) -> attn_output                                     gdn_*_multi (kda), kda_gate_beta_multi
//   DSA layers: q_a + q_a_norm -> q_b -> wk_b per head (absorb); kv_a_mqa + kv_a_norm -> the fp16 latent cache;
//               the k-pool indexer (append, scores, top-k); MLA attention over the selection; wv_b per head; wo
//   mHC read (FFN side; the attention write folded in)
//   layers 0-2: the dense SwiGLU FFN (no experts: the doorbell only rings); the others: the sigmoid router with
//               its bias and x2.5, the shared expert (no gate), the routed experts through the doorbell
//   the last layer: the plain mHC write, the mean of the 4 streams, output_norm, the head.
// The commit re-runs the KDA recurrence of the kept tokens and re-appends their indexer keys after restoring the
// indexer tail it snapshotted (as the Qwen commit does).
#include "strata/core/verify.hpp"

#include "strata/core/native_head.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/glm5_misc.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_select.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/router_top10.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"
#include "strata/kernels/sampler.hpp"
#include "strata/kernels/shared_expert.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace strata::core {

struct Verifier::Glm5 {
    // geometry (from the GGUF)
    int64_t N = 0, H = 0, S = 0, C = 0, DI = 0;            // n_embd, KDA heads, head dim, conv channels, H*S
    int64_t NH = 0, QLR = 0, KVR = 0, QK = 0, VD = 0;      // MLA heads, q_lora_rank, kv_lora_rank (latent), q/v head dims
    int64_t IH = 0, ID = 0, NFF = 0, NFFD = 0, NE = 0, K = 0, max_cells = 0, max_blocks = 0, cap = 0;
    float rms_eps = 1e-5f, ln_eps = 1e-6f, hc_eps = 1e-6f, lb = -5.f, wscale = 1.f, clamp_exp = 0.f, clamp_sh = 0.f;
    int iters = 20, n_slices = 0;
    std::vector<int64_t> kda_index, dsa_index;            // per layer: its ordinal among its kind, or -1
    // per KDA layer
    float* convw = nullptr;          // [nK][C][4]
    float* gate = nullptr;           // [nK][MT][H][S]
    // per DSA layer
    uint16_t* latent = nullptr;      // [nD][max_cells][KVR] fp16
    float* tail = nullptr;           // [nD][3][2*ID]
    float* pooled = nullptr;         // [nD][max_cells/4 + 1][ID]
    float* tail_snap = nullptr;      // [nD][3][2*ID]
    float* raw_k = nullptr;          // [nD][MT][ID]  the window's indexer keys / gates, for the commit's re-append
    float* raw_g = nullptr;
    int32_t* page_table = nullptr;   // identity, [max_cells / 4]
    // scratch [MT][...]
    float *q = nullptr, *k = nullptr, *v = nullptr, *fa = nullptr, *f = nullptr, *ga = nullptr, *bl = nullptr;
    float *qr = nullptr, *qb = nullptr, *qT = nullptr, *qaT = nullptr, *qa = nullptr, *kv = nullptr, *iq = nullptr,
          *iw = nullptr, *lat = nullptr, *latT = nullptr, *vbT = nullptr, *vb = nullptr, *mla_scratch = nullptr;
    float *part = nullptr, *ssp = nullptr, *postA = nullptr, *combA = nullptr, *postF = nullptr, *combF = nullptr;
    float *dg = nullptr, *du = nullptr, *hmean = nullptr;
    uint8_t* q8 = nullptr;           // q8_1 activations, the widest use (per-head wv_b: 64 x MT columns of 512)
    std::vector<void*> allocs;
    template <typename T> T* dev(size_t n) {
        void* p = nullptr;
        if (cudaMalloc(&p, n * sizeof(T) + 256) != cudaSuccess) { cudaGetLastError(); allocs.push_back(nullptr); return nullptr; }
        cudaMemset(p, 0, n * sizeof(T) + 256);
        allocs.push_back(p);
        return (T*) p;
    }
};

namespace {
const WeightRef* wfind(const WeightTable& wt, int64_t l, const char* suffix, std::string& err) {
    const std::string name = l < 0 ? std::string(suffix) : "blk." + std::to_string(l) + "." + suffix;
    const WeightRef* r = wt.find(name);
    if (r == nullptr && err.empty()) err = "verify glm5: " + name + " is missing";
    return r;
}
// one Q8_0/K-quant native matrix on `ncols` columns of x (q8_1-quantized into `q8` first)
void mmvq(const WeightRef* w, const float* x, int n_in, float* y, int n_out, int ncols, uint8_t* q8, void* cs) {
    strata::kernels::native_quantize_q8_1(x, q8, n_in, ncols, cs);
    strata::kernels::native_mmvq(w->native_type, w->native_data, q8, y, n_in, n_out, ncols, cs);
}
}  // namespace

void Verifier::free_glm5() {
    if (!glm_) return;
    for (void* p : glm_->allocs) cudaFree(p);
    delete glm_;
    glm_ = nullptr;
}

bool Verifier::init_glm5(std::string& err) {
    using namespace strata::kernels;
    const ModelGeometry& g = *g_;
    const WeightTable& wt = *wt_;
    SessionState& ss = *ss_;
    glm_ = new Glm5();
    Glm5& m = *glm_;
    const int64_t MT = max_t_;
    m.N = g.n_embd; m.H = g.ssm_v_heads; m.S = g.ssm_state_size; m.C = g.ssm_conv_channels; m.DI = g.ssm_value_dim;
    m.NH = g.n_head; m.QLR = g.q_lora_rank; m.KVR = g.kv_lora_rank; m.QK = g.mla_key_dim; m.VD = g.mla_value_dim;
    m.IH = g.idx_q_heads; m.ID = g.idx_key_dim; m.NFF = g.n_ff; m.NFFD = g.n_ff_dense; m.NE = g.n_expert; m.K = ss.k;
    m.rms_eps = g.rms_eps; m.ln_eps = g.norm_eps; m.hc_eps = g.hc_eps; m.iters = (int) g.hc_sinkhorn_iters;
    m.lb = g.kda_gate_lower_bound; m.wscale = g.expert_weights_scale;
    m.clamp_exp = g.swiglu_clamp_exp.empty() ? 0.f : g.swiglu_clamp_exp.back();
    m.clamp_sh = g.swiglu_clamp_shared.empty() ? 0.f : g.swiglu_clamp_shared.back();
    m.max_cells = ss.qsa_states[ss.qsa_primary()].max_cells;
    m.max_blocks = m.max_cells / 4 + 2;
    m.cap = cap_;
    m.n_slices = fused_mhc_slices(m.N);
    if (!fused_mhc_supported(m.N, g.hc, (2 + g.hc) * g.hc) || m.S != 128 || m.NH % 16 != 0 || m.KVR != 512 ||
        m.ID != 128 || m.IH != 32 || lb_ != 0 || le_ != g.n_layers) {
        err = "verify glm5: unsupported geometry or a layer split (the window runs the whole trunk)";
        return false;
    }
    // the layers by kind, and their ordinals
    m.kda_index.assign((size_t) g.n_layers, -1);
    m.dsa_index.assign((size_t) g.n_layers, -1);
    int64_t nK = 0, nD = 0;
    for (int64_t l = 0; l < g.n_layers; ++l) (g.is_sparse_layer(l) ? m.dsa_index : m.kda_index)[(size_t) l] = g.is_sparse_layer(l) ? nD++ : nK++;
    if (nK != g.n_gdn_layers() || nD != g.n_qsa_layers()) { err = "verify glm5: layer kinds disagree with the geometry"; return false; }
    // per-layer device state
    m.convw = m.dev<float>((size_t) nK * m.C * 4);
    m.gate = m.dev<float>((size_t) nK * MT * m.H * m.S);
    m.latent = m.dev<uint16_t>((size_t) nD * m.max_cells * m.KVR);
    m.tail = m.dev<float>((size_t) nD * 3 * 2 * m.ID);
    m.pooled = m.dev<float>((size_t) nD * (m.max_cells / 4 + 1) * m.ID);
    m.tail_snap = m.dev<float>((size_t) nD * 3 * 2 * m.ID);
    m.raw_k = m.dev<float>((size_t) nD * MT * m.ID);
    m.raw_g = m.dev<float>((size_t) nD * MT * m.ID);
    m.page_table = m.dev<int32_t>((size_t) (m.max_cells / 4 + 1));
    // the KDA conv weights: GGUF ssm_conv1d_{q,k,v} [4, 1, DI] each (channel-major, tap fastest) -> q|k|v [C][4]
    for (int64_t l = 0; l < g.n_layers; ++l) {
        const int64_t gi = m.kda_index[(size_t) l];
        if (gi < 0) continue;
        const char* names[3] = {"ssm_conv1d_q.weight", "ssm_conv1d_k.weight", "ssm_conv1d_v.weight"};
        for (int r = 0; r < 3; ++r) {
            const WeightRef* w = wfind(wt, l, names[r], err);
            if (!w || !w->data) { if (err.empty()) err = "verify glm5: a conv weight is not in the arena"; return false; }
            cudaMemcpy(m.convw + ((size_t) gi * m.C + (size_t) r * m.DI) * 4, w->data, (size_t) m.DI * 4 * 4,
                       cudaMemcpyDeviceToDevice);
        }
    }
    {
        std::vector<int32_t> pt((size_t) (m.max_cells / 4 + 1));
        for (size_t i = 0; i < pt.size(); ++i) pt[i] = (int32_t) i;
        cudaMemcpy(m.page_table, pt.data(), pt.size() * 4, cudaMemcpyHostToDevice);
    }
    // scratch
    const size_t T = (size_t) MT;
    m.q = m.dev<float>(T * m.DI); m.k = m.dev<float>(T * m.DI); m.v = m.dev<float>(T * m.DI);
    m.fa = m.dev<float>(T * 128); m.f = m.dev<float>(T * m.DI); m.ga = m.dev<float>(T * 128); m.bl = m.dev<float>(T * m.H);
    m.qr = m.dev<float>(T * m.QLR); m.qb = m.dev<float>(T * m.NH * m.QK); m.qT = m.dev<float>(T * m.NH * m.QK);
    m.qaT = m.dev<float>(T * m.NH * m.KVR); m.qa = m.dev<float>(T * m.NH * m.KVR); m.kv = m.dev<float>(T * m.KVR);
    m.iq = m.dev<float>(T * m.IH * m.ID); m.iw = m.dev<float>(T * m.IH);
    m.lat = m.dev<float>(T * m.NH * m.KVR); m.latT = m.dev<float>(T * m.NH * m.KVR);
    m.vbT = m.dev<float>(T * m.NH * m.VD); m.vb = m.dev<float>(T * m.NH * m.VD);
    m.mla_scratch = m.dev<float>(T * mla_decode_attn_scratch_floats(m.cap, m.NH));
    m.part = m.dev<float>(T * m.n_slices * 24); m.ssp = m.dev<float>(T * m.n_slices);
    m.postA = m.dev<float>(T * 4); m.combA = m.dev<float>(T * 16); m.postF = m.dev<float>(T * 4); m.combF = m.dev<float>(T * 16);
    m.dg = m.dev<float>(T * m.NFFD); m.du = m.dev<float>(T * m.NFFD); m.hmean = m.dev<float>(T * m.N);
    m.q8 = m.dev<uint8_t>((size_t) (m.NH * T) * (m.KVR / 32) * 36 + (size_t) T * (m.NFFD / 32) * 36);
    for (void* p : m.allocs) if (p == nullptr) { err = "verify glm5: a device buffer does not fit"; return false; }
    // the small tensors are read as F32 straight from the arena: refuse a pack that stored them narrower
    for (const char* nm : {"blk.0.attn_norm.weight", "blk.0.ssm_dt.bias", "blk.3.ffn_gate_inp.weight", "blk.3.indexer.proj.weight",
                           "blk.0.hc_attn_scale.weight", "output_norm.weight"}) {
        const WeightRef* r = wt.find(nm);
        if (r == nullptr || r->data == nullptr || r->bytes < (uint64_t) r->elements * 4) {
            err = std::string("verify glm5: ") + nm + " is not an F32 tensor in the arena";
            return false;
        }
    }
    std::fprintf(stderr, "strata verify: glm5 geometry N %lld H %lld S %lld C %lld DI %lld NH %lld QLR %lld KVR %lld QK %lld "
                         "VD %lld IH %lld ID %lld NFF %lld NFFD %lld NE %lld K %lld cap %lld eps %g/%g/%g lb %g ws %g clamp %g/%g\n",
                 (long long) m.N, (long long) m.H, (long long) m.S, (long long) m.C, (long long) m.DI, (long long) m.NH,
                 (long long) m.QLR, (long long) m.KVR, (long long) m.QK, (long long) m.VD, (long long) m.IH, (long long) m.ID,
                 (long long) m.NFF, (long long) m.NFFD, (long long) m.NE, (long long) m.K, (long long) m.cap, m.rms_eps,
                 m.ln_eps, m.hc_eps, m.lb, m.wscale, m.clamp_exp, m.clamp_sh);
    std::fprintf(stderr, "strata verify: glm5-next window: %lld KDA + %lld DSA layers, %lld dense, %lld-cell latent cache "
                         "(%.1f MiB)\n", (long long) nK, (long long) nD, (long long) g.n_dense_layers,
                 (long long) m.max_cells, (double) nD * m.max_cells * m.KVR * 2 / 1048576.0);
    return true;
}

bool Verifier::record_window_glm5(int T, cudaStream_t cs, std::string& err) {
    using namespace strata::kernels;
    const ModelGeometry& g = *g_;
    const WeightTable& wt = *wt_;
    SessionState& ss = *ss_;
    Glm5& m = *glm_;
    const int64_t N = m.N, HC = g.hc, K = m.K, MT = max_t_;
    const uint64_t gdn_floats = (uint64_t) m.S * m.H * m.S + (uint64_t) m.C * (g.ssm_d_conv - 1);
    auto Rt = [&](int t) { return R_ + (size_t) t * HC * N; };
    const QsaShapes s = [&] { QsaShapes x = qsa_real_shapes(); x.n_head = g.n_head; x.n_head_kv = g.n_head_kv;
                              x.head_dim = g.head_dim; x.idx_n_head = g.idx_q_heads; x.idx_dim = g.idx_key_dim; return x; }();
    groups_[T] = 1;
    const int tb = 0, te = T, n = T;

    copy_i32_from_mapped(tok_, m_tok_, T, cs);
    copy_i32_from_mapped(step_, m_step_, (int64_t) T * kStepCount, cs);
    if (const NativeEmbed* ne = native_embed()) {
        ne->gather_dev(tok_, T, emb_, cs);
        broadcast_streams(emb_, R_, N, (int) HC, T, cs);
    } else { err = "verify glm5: needs the native token embedding"; return false; }

    auto mhc_read = [&](int64_t l, const char* which, bool apply, const float* post_prev, const float* comb_prev,
                        float* post_out, float* comb_out, const char* norm, float* y) -> bool {
        const std::string w = std::string("hc_") + which + "_";
        const WeightRef *fn = wfind(wt, l, (w + "fn.weight").c_str(), err), *sc = wfind(wt, l, (w + "scale.weight").c_str(), err),
                        *bs = wfind(wt, l, (w + "base.weight").c_str(), err), *wn = wfind(wt, l, norm, err);
        if (!fn || !sc || !bs || !wn || !fn->native_data) { if (err.empty()) err = "verify glm5: mHC weights"; return false; }
        MhcMulti a;
        a.n_tok = n; a.n_embd = (int) N; a.n_slices = m.n_slices; a.apply = apply; a.iters = m.iters;
        a.rms_eps = m.rms_eps; a.hc_eps = m.hc_eps;
        for (int t = 0; t < n; ++t) { a.R[t] = Rt(t); a.R_out[t] = Rt(t); a.out_prev[t] = bo_ + (size_t) t * N; a.y[t] = y + (size_t) t * N; }
        a.post_prev = post_prev; a.comb_prev = comb_prev;
        a.fn = fn->native_data; a.scale = (const float*) sc->data; a.base = (const float*) bs->data; a.w_norm = (const float*) wn->data;
        a.post_out = post_out; a.comb_out = comb_out; a.part = m.part; a.ss_part = m.ssp;
        fused_mhc_read_multi(a, cs);
        return true;
    };

    // ---- pre(l): the attention block and the FFN's input, router and doorbell
    auto pre = [&](int64_t l) -> bool {
        float* xm = mixed_;   // the block input of the attention, then of the FFN
        if (!mhc_read(l, "attn", l > 0, m.postF, m.combF, m.postA, m.combA, "attn_norm.weight", xm)) return false;
        const int64_t gi = m.kda_index[(size_t) l], qi = m.dsa_index[(size_t) l];
        if (gi >= 0) {
            // ======================= KDA =======================
            const WeightRef *wq = wfind(wt, l, "attn_q.weight", err), *wk = wfind(wt, l, "attn_k.weight", err),
                            *wv = wfind(wt, l, "attn_v.weight", err), *wo = wfind(wt, l, "attn_output.weight", err),
                            *wfa = wfind(wt, l, "ssm_f_a.weight", err), *wfb = wfind(wt, l, "ssm_f_b.weight", err),
                            *wga = wfind(wt, l, "ssm_g_a.weight", err), *wgb = wfind(wt, l, "ssm_g_b.weight", err),
                            *wbe = wfind(wt, l, "ssm_beta.weight", err), *wdt = wfind(wt, l, "ssm_dt.bias", err),
                            *wsa = wfind(wt, l, "ssm_a", err), *wnm = wfind(wt, l, "ssm_norm.weight", err);
            if (!wq || !wk || !wv || !wo || !wfa || !wfb || !wga || !wgb || !wbe || !wdt || !wsa || !wnm) return false;
            float* state = ss.gdn_state + (size_t) (gi - ss.gdn_ord0) * gdn_floats;
            float* conv = state + (uint64_t) m.S * m.H * m.S;
            float* qkv = qkv_L_ + (size_t) gi * MT * m.C;
            float* hb = h_L_ + (size_t) gi * MT * m.C;
            float* gate = m.gate + (size_t) gi * MT * m.H * m.S;
            float* beta = beta_L_ + (size_t) gi * MT * m.H;
            native_quantize_q8_1(xm, xq_, (int) N, n, cs);   // x once for the six projections from it
            native_mmvq(wq->native_type, wq->native_data, xq_, m.q, (int) N, (int) m.DI, n, cs);
            native_mmvq(wk->native_type, wk->native_data, xq_, m.k, (int) N, (int) m.DI, n, cs);
            native_mmvq(wv->native_type, wv->native_data, xq_, m.v, (int) N, (int) m.DI, n, cs);
            native_mmvq(wfa->native_type, wfa->native_data, xq_, m.fa, (int) N, 128, n, cs);
            native_mmvq(wga->native_type, wga->native_data, xq_, m.ga, (int) N, 128, n, cs);
            native_mmvq(wbe->native_type, wbe->native_data, xq_, m.bl, (int) N, (int) m.H, n, cs);
            glm5_pack3(m.q, m.k, m.v, qkv, (int) m.DI, n, cs);
            gdn_conv_l2_multi(conv, qkv, m.convw + (size_t) gi * m.C * 4, hb, (int) m.C, (int) (2 * m.H), 1e-6f, n, cs, tb);
            mmvq(wfb, m.fa, 128, m.f, (int) m.DI, n, m.q8, cs);
            mmvq(wgb, m.ga, 128, z_, (int) m.DI, n, m.q8, cs);
            kda_gate_beta_multi(m.f, (const float*) wdt->data, (const float*) wsa->data, m.lb, m.bl, gate, beta,
                                (int) m.H, n, cs);
            gdn_step_norm_multi(state, hb, (int) m.C, gate, beta, z_, (const float*) wnm->data, m.rms_eps, y_, (int) m.H,
                                (int) m.H, te, nullptr, cs, tb, true);
            mmvq(wo, y_, (int) m.DI, bo_, (int) N, n, m.q8, cs);
        } else {
            // ======================= DSA (MLA + k-pool indexer) =======================
            const WeightRef *wqa = wfind(wt, l, "attn_q_a.weight", err), *wqan = wfind(wt, l, "attn_q_a_norm.weight", err),
                            *wqb = wfind(wt, l, "attn_q_b.weight", err), *wkb = wfind(wt, l, "attn_k_b.weight", err),
                            *wvb = wfind(wt, l, "attn_v_b.weight", err), *wkv = wfind(wt, l, "attn_kv_a_mqa.weight", err),
                            *wkvn = wfind(wt, l, "attn_kv_a_norm.weight", err), *wo = wfind(wt, l, "attn_output.weight", err),
                            *wik = wfind(wt, l, "indexer.attn_k.weight", err), *wiqb = wfind(wt, l, "indexer.attn_q_b.weight", err),
                            *wikn = wfind(wt, l, "indexer.k_norm.weight", err), *wikb = wfind(wt, l, "indexer.k_norm.bias", err),
                            *wip = wfind(wt, l, "indexer.proj.weight", err), *wape = wfind(wt, l, "indexer_compressor_ape.weight", err),
                            *wig = wfind(wt, l, "indexer_compressor_gate.weight", err);
            if (!wqa || !wqan || !wqb || !wkb || !wvb || !wkv || !wkvn || !wo || !wik || !wiqb || !wikn || !wikb || !wip ||
                !wape || !wig) return false;
            const int64_t QLR = m.QLR, NH = m.NH, QK = m.QK, KVR = m.KVR, VD = m.VD, ID = m.ID, IH = m.IH;
            uint16_t* lat_pool = m.latent + (size_t) qi * m.max_cells * KVR;
            float* tail = m.tail + (size_t) qi * 3 * 2 * ID;
            float* pooled = m.pooled + (size_t) qi * (m.max_cells / 4 + 1) * ID;
            float* rk = m.raw_k + (size_t) qi * MT * ID;
            float* rg = m.raw_g + (size_t) qi * MT * ID;
            native_quantize_q8_1(xm, xq_, (int) N, n, cs);
            native_mmvq(wqa->native_type, wqa->native_data, xq_, m.qr, (int) N, (int) QLR, n, cs);
            native_mmvq(wkv->native_type, wkv->native_data, xq_, m.kv, (int) N, (int) KVR, n, cs);
            native_mmvq(wik->native_type, wik->native_data, xq_, rk, (int) N, (int) ID, n, cs);
            native_mmvq(wig->native_type, wig->native_data, xq_, rg, (int) N, (int) ID, n, cs);
            native_qsa_rms_norm_weighted(m.qr, (const float*) wqan->data, m.qr, (int) QLR, n, m.rms_eps, cs);
            native_qsa_rms_norm_weighted(m.kv, (const float*) wkvn->data, m.kv, (int) KVR, n, m.rms_eps, cs);
            glm5_latent_store(m.kv, lat_pool, m.page_table, step_, kStepCount, 4, (int) KVR, n, cs);
            // q: q_b, then the absorbed query q_abs[h] = wk_b[h] . q[h] (256 -> 512), head by head
            mmvq(wqb, m.qr, (int) QLR, m.qb, (int) (NH * QK), n, m.q8, cs);
            glm5_transpose_abd(m.qb, m.qT, n, (int) NH, (int) QK, cs);                      // [h][t][QK]
            native_quantize_q8_1(m.qT, m.q8, (int) (QK * NH * n), 1, cs);   // NH*n contiguous columns as one vector
            {
                const size_t row = (size_t) (QK / 32) * 34, qcol = (size_t) (QK / 32) * 36;
                for (int64_t h = 0; h < NH; ++h)
                    native_mmvq(wkb->native_type, (const uint8_t*) wkb->native_data + (size_t) h * KVR * row,
                                m.q8 + (size_t) h * n * qcol, m.qaT + (size_t) h * n * KVR, (int) QK, (int) KVR, n, cs);
            }
            glm5_transpose_abd(m.qaT, m.qa, (int) NH, n, (int) KVR, cs);                    // [t][h][KVR]
            // the indexer: keys and gates of these tokens into the pool (tail snapshot first, for the commit)
            copy_from_mapped(m.tail_snap + (size_t) qi * 3 * 2 * ID, tail, 3 * 2 * ID, cs);
            const Glm5KpoolBuffers kb{tail, pooled};
            for (int t = 0; t < n; ++t)
                glm5_kpool_append(rk + (size_t) t * ID, rg + (size_t) t * ID, step_ + (size_t) t * kStepCount + kStepPos,
                                  (const float*) wikn->data, (const float*) wikb->data, (const float*) wape->data, m.ln_eps,
                                  kb, m.max_cells, cs);
            mmvq(wiqb, m.qr, (int) QLR, m.iq, (int) (IH * ID), n, m.q8, cs);
            glm5_f32_gemv((const float*) wip->data, xm, m.iw, (int) N, (int) IH, n, 1.0f / std::sqrt((float) (ID * IH)), cs);
            glm5_kpool_scores(m.iq, m.iw, pooled, step_, n, max_blocks_, scores_, cs);
            qsa_block_topk(scores_, step_, n, max_blocks_, cap_, s, sel_, cs);
            // MLA over the selection (fp16 latents; scale 1/sqrt(q head dim) as llama.cpp), wv_b per head, then wo
            mla_decode_attn_batch(m.qa, lat_pool, m.page_table, 4, sel_, step_, cap_, (int) NH, 1.0f / std::sqrt((float) QK),
                                  m.mla_scratch, m.lat, n, cs);
            glm5_transpose_abd(m.lat, m.latT, n, (int) NH, (int) KVR, cs);                  // [h][t][KVR]
            native_quantize_q8_1(m.latT, m.q8, (int) (KVR * NH * n), 1, cs);
            {
                const size_t row = (size_t) (KVR / 32) * 34, qcol = (size_t) (KVR / 32) * 36;
                for (int64_t h = 0; h < NH; ++h)
                    native_mmvq(wvb->native_type, (const uint8_t*) wvb->native_data + (size_t) h * VD * row,
                                m.q8 + (size_t) h * n * qcol, m.vbT + (size_t) h * n * VD, (int) KVR, (int) VD, n, cs);
            }
            glm5_transpose_abd(m.vbT, m.vb, (int) NH, n, (int) VD, cs);                     // [t][h][VD]
            mmvq(wo, m.vb, (int) (NH * VD), bo_, (int) N, n, m.q8, cs);
        }
        // ---- the FFN side
        if (!mhc_read(l, "ffn", true, m.postA, m.combA, m.postF, m.combF, "ffn_norm.weight", xm)) return false;
        if (l < g.n_dense_layers) {
            const WeightRef *wg = wfind(wt, l, "ffn_gate.weight", err), *wu = wfind(wt, l, "ffn_up.weight", err),
                            *wd = wfind(wt, l, "ffn_down.weight", err);
            if (!wg || !wu || !wd) return false;
            NativeSharedWeights nw;
            nw.gate_type = wg->native_type; nw.gate_data = wg->native_data; nw.up_type = wu->native_type; nw.up_data = wu->native_data;
            nw.down_type = wd->native_type; nw.down_data = wd->native_data; nw.q8_1 = m.q8;
            shared_expert_multi(n, xm, nullptr, nw, nullptr, m.dg, m.du, sh_g_, bo_, N, m.NFFD, cs, m.clamp_sh);
            doorbell_ring(m_seq_, cs);   // no experts: the host skips this layer's pool and raises its flags
            return true;
        }
        const WeightRef *wr = wfind(wt, l, "ffn_gate_inp.weight", err), *wb = wfind(wt, l, "exp_probs_b.bias", err),
                        *wsg = wfind(wt, l, "ffn_gate_shexp.weight", err), *wsu = wfind(wt, l, "ffn_up_shexp.weight", err),
                        *wsd = wfind(wt, l, "ffn_down_shexp.weight", err);
        if (!wr || !wb || !wsg || !wsu || !wsd) return false;
        glm5_f32_gemv((const float*) wr->data, xm, logits_, (int) N, (int) m.NE, n, 1.0f, cs);
        router_sigmoid_topk(logits_, (const float*) wb->data, n, (int) m.NE, (int) K, m.wscale, ids_, w_, cs);
        doorbell_publish(xm, ids_, w_, (int64_t) n * N, (int64_t) n * K, m_x_, m_ids_, m_w_, m_seq_, cs);
        NativeSharedWeights nsw;
        nsw.gate_type = wsg->native_type; nsw.gate_data = wsg->native_data; nsw.up_type = wsu->native_type;
        nsw.up_data = wsu->native_data; nsw.down_type = wsd->native_type; nsw.down_data = wsd->native_data; nsw.q8_1 = m.q8;
        shared_expert_multi(n, xm, nullptr, nsw, nullptr, sh_gate_, sh_up_, sh_g_, shared_, N, m.NFF, cs, m.clamp_sh);
        quantize_q8_1_rows(xm, n, N, nat_xq_, cs);
        return true;
    };

    // ---- post(l): the routed experts (the Qwen path's machinery), the combine into bo_
    auto post = [&](int64_t l) -> bool {
        if (l < g.n_dense_layers) return true;   // the dense FFN's output is already in bo_
        const uint32_t ring = (uint32_t) (l + 1);
        const int64_t cap = (int64_t) n * K, capx = (int64_t) MT * K;
        int32_t* pl = plan_;
        wait_flag_ge(m_flagA_, ring, cs);
        copy_i32_from_mapped(pl, m_plan_, plan_i32_, cs);
        const int32_t* p_counts = pl;
        const int32_t* p_start = pl + 4;
        const int32_t* p_dst = p_start + capx + 1;
        const int32_t* p_tok = p_dst + capx;
        const int64_t ptr_off = ((4 + (capx + 1) + 2 * capx) + 1) & ~1ll;
        const unsigned long long* p_ptr = (const unsigned long long*) (pl + ptr_off);
        const unsigned long long* p_ptr2 = p_ptr + capx;
        const int32_t* p_start2 = pl + ptr_off + 4 * capx;
        const auto& lay = strata::kernels::cpu::expert_layout();
        const auto& f = lay.fmt[(size_t) l];
        NativeExpertLayout L = native_expert_layout(f.gu_type, f.d_type, f.n_embd, f.n_ff);
        L.swiglu_limit = m.clamp_exp;
        native_expert_grouped(L, p_ptr, p_start, p_counts, p_dst, p_tok, cap, cap, nat_xq_, hit_scratch_, hit_out_, cs, 0);
        wait_flag_ge(m_flagB_, ring, cs);
        if (sink_.pcie_mode == 2) {
            fetch_blobs(p_ptr2, p_counts + 2, staging_, (int64_t) lay.blob_bytes(l), (int) kStagingBlobs, cs);
            rebase_ptrs((unsigned long long*) p_ptr2, p_counts + 2, staging_, (int64_t) lay.blob_bytes(l), cs);
        }
        native_expert_grouped(L, p_ptr2, p_start2, p_counts + 2, p_dst, p_tok, cap, cap, nat_xq_, hit_scratch_, hit_out_, cs,
                              kPcieGroupRows);
        wait_flag_ge(m_flag_, ring, cs);
        copy_rows_from_mapped(parts_, m_ymiss_, (int64_t) n * K, N, p_dst, p_counts + 1, cs);
        moe_hit_add(parts_, hit_out_, p_dst, p_counts + 1, cap, N, cs);
        native_moe_combine_multi(parts_, w_, shared_, bo_, N, K, n, cs);
        return true;
    };

    try {
        for (int64_t l = 0; l < g.n_layers; ++l) {
            if (!pre(l)) return false;
            if (!post(l)) return false;
        }
        // the last layer's FFN write (nothing follows to fold it into), the mean of the streams, output_norm, the head
        MhcMulti w;
        w.n_tok = n; w.n_embd = (int) N; w.apply = true; w.post_prev = m.postF; w.comb_prev = m.combF;
        for (int t = 0; t < n; ++t) { w.R[t] = Rt(t); w.R_out[t] = Rt(t); w.out_prev[t] = bo_ + (size_t) t * N; w.y[t] = head_mixed_ + (size_t) t * N; }
        fused_mhc_write_multi(w, cs);
        fused_mhc_exit_mean(w, cs);
        const WeightRef* on = wfind(wt, -1, "output_norm.weight", err);
        if (!on) return false;
        native_qsa_rms_norm_weighted(head_mixed_, (const float*) on->data, head_mixed_, (int) N, n, m.rms_eps, cs);
        if (head_ == nullptr || !head_->loaded()) { err = "verify glm5: needs the native head (--native)"; return false; }
        native_quantize_q8_1(head_mixed_, xq_, (int) N, T, cs);
        native_mmvq(head_->type(), head_->weights(), xq_, head_logits_, (int) N, (int) n_vocab_, T, cs);
        SamplerParams sp;
        sp.greedy = true;
        sp.temperature = 0.0f;
        sample_tokens(head_logits_, T, (int) n_vocab_, nullptr, 0, sp, m_out_, cs);
    } catch (const std::exception& e) {
        err = std::string("verify glm5: ") + e.what();
        return false;
    }
    return true;
}

bool Verifier::capture_commit_glm5(std::string& err) {
    using namespace strata::kernels;
    const ModelGeometry& g = *g_;
    SessionState& ss = *ss_;
    Glm5& m = *glm_;
    const int64_t MT = max_t_;
    const uint64_t gdn_floats = (uint64_t) m.S * m.H * m.S + (uint64_t) m.C * (g.ssm_d_conv - 1);
    if (cudaStreamBeginCapture(cs_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
        err = "verify glm5: begin commit capture failed";
        return false;
    }
    bool ok = true;
    try {
        copy_i32_from_mapped(commit_, m_commit_, 2 + MT, cs_);
        for (int64_t l = 0; l < g.n_layers && ok; ++l) {
            const int64_t gi = m.kda_index[(size_t) l], qi = m.dsa_index[(size_t) l];
            if (gi >= 0) {
                const WeightRef* wnm = wfind(*wt_, l, "ssm_norm.weight", err);
                if (!wnm) { ok = false; break; }
                float* state = ss.gdn_state + (size_t) (gi - ss.gdn_ord0) * gdn_floats;
                float* conv = state + (uint64_t) m.S * m.H * m.S;
                gdn_conv_commit(conv, qkv_L_ + (size_t) gi * MT * m.C, (int) m.C, commit_, cs_);
                gdn_step_norm_multi(state, h_L_ + (size_t) gi * MT * m.C, (int) m.C, m.gate + (size_t) gi * MT * m.H * m.S,
                                    beta_L_ + (size_t) gi * MT * m.H, z_, (const float*) wnm->data, m.rms_eps, y_dummy_,
                                    (int) m.H, (int) m.H, (int) MT, commit_, cs_, 0, true);
            } else {
                const WeightRef *wikn = wfind(*wt_, l, "indexer.k_norm.weight", err),
                                *wikb = wfind(*wt_, l, "indexer.k_norm.bias", err),
                                *wape = wfind(*wt_, l, "indexer_compressor_ape.weight", err);
                if (!wikn || !wikb || !wape) { ok = false; break; }
                float* tail = m.tail + (size_t) qi * 3 * 2 * m.ID;
                copy_from_mapped(tail, m.tail_snap + (size_t) qi * 3 * 2 * m.ID, 3 * 2 * m.ID, cs_);
                const Glm5KpoolBuffers kb{tail, m.pooled + (size_t) qi * (m.max_cells / 4 + 1) * m.ID};
                // commit_ = [n_keep, n_keep - 1, pos_0 .. pos_{MT-1}]: rows past n_keep carry -1 and are ignored
                for (int64_t t = 0; t < MT; ++t)
                    glm5_kpool_append(m.raw_k + ((size_t) qi * MT + t) * m.ID, m.raw_g + ((size_t) qi * MT + t) * m.ID,
                                      commit_ + 2 + t, (const float*) wikn->data, (const float*) wikb->data,
                                      (const float*) wape->data, m.ln_eps, kb, m.max_cells, cs_);
            }
        }
    } catch (const std::exception& e) {
        err = std::string("verify glm5 commit: ") + e.what();
        ok = false;
    }
    cudaGraph_t graph = nullptr;
    const cudaError_t ce = cudaStreamEndCapture(cs_, &graph);
    if (!ok) {
        if (graph) cudaGraphDestroy(graph);
        return false;
    }
    if (ce != cudaSuccess || cudaGraphInstantiate(&commit_exec_, graph, 0) != cudaSuccess) {
        if (graph) cudaGraphDestroy(graph);
        err = std::string("verify glm5: commit capture: ") + cudaGetErrorString(ce);
        return false;
    }
    cudaGraphDestroy(graph);
    return true;
}

}  // namespace strata::core
