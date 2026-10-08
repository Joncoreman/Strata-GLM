// src/core/glm5_mtp.cpp - see include/strata/core/glm5_mtp.hpp.  The DSA block is verify_glm5.cpp's, on blk.45 and
// the drafter's own latent cache and indexer pool.
#include "strata/core/glm5_mtp.hpp"

#include "strata/core/layout.hpp"
#include "strata/core/native_head.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/glm5_misc.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_select.hpp"
#include "strata/kernels/sampler.hpp"
#include "strata/kernels/shared_expert.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace strata::core {

struct Glm5Mtp::Impl {
    const WeightTable* wt = nullptr;
    const NativeHead* head = nullptr;
    int64_t N = 0, NH = 0, QLR = 0, KVR = 0, QK = 0, VD = 0, IH = 0, ID = 0, NFF = 0, n_vocab = 0;
    int64_t max_cells = 0, max_blocks = 0, cap = 0;
    int MT = 0;
    float rms_eps = 1e-5f, ln_eps = 1e-6f, clamp_sh = 0.f;
    strata::kernels::QsaShapes s;
    cudaStream_t cs = nullptr;
    // weights
    const WeightRef *eh = nullptr, *enorm = nullptr, *hnorm = nullptr, *shn = nullptr, *an = nullptr, *fn = nullptr;
    const WeightRef *wqa = nullptr, *wqan = nullptr, *wqb = nullptr, *wkb = nullptr, *wvb = nullptr, *wkv = nullptr,
                    *wkvn = nullptr, *wo = nullptr, *wik = nullptr, *wiqb = nullptr, *wikn = nullptr, *wikb = nullptr,
                    *wip = nullptr, *wape = nullptr, *wig = nullptr, *wsg = nullptr, *wsu = nullptr, *wsd = nullptr;
    // state
    uint16_t* latent = nullptr;   // [max_cells][KVR] fp16
    float *tail = nullptr, *pooled = nullptr, *tail_snap = nullptr;
    int32_t* page_table = nullptr;
    // scratch [MT][...]
    int32_t *tok = nullptr, *steps = nullptr, *sel = nullptr, *out_tok = nullptr;
    float *emb = nullptr, *eN = nullptr, *hN = nullptr, *cat = nullptr, *u = nullptr, *x = nullptr, *ao = nullptr;
    float *qr = nullptr, *qb = nullptr, *qT = nullptr, *qaT = nullptr, *qa = nullptr, *kv = nullptr, *rk = nullptr,
          *rg = nullptr, *iq = nullptr, *iw = nullptr, *scores = nullptr, *lat = nullptr, *latT = nullptr, *vbT = nullptr,
          *vb = nullptr, *mla_scratch = nullptr, *sg = nullptr, *su = nullptr, *sgs = nullptr, *sh = nullptr,
          *hp = nullptr, *logits = nullptr;
    uint8_t* q8 = nullptr;
    std::vector<int32_t> h_steps;
    std::vector<void*> allocs;
    template <typename T> T* dev(size_t n) {
        void* p = nullptr;
        if (cudaMalloc(&p, n * sizeof(T) + 256) != cudaSuccess) return nullptr;
        cudaMemset(p, 0, n * sizeof(T) + 256);
        allocs.push_back(p);
        return (T*) p;
    }
};

Glm5Mtp::~Glm5Mtp() {
    if (!m_) return;
    if (m_->cs) { cudaStreamSynchronize(m_->cs); cudaStreamDestroy(m_->cs); }
    for (void* p : m_->allocs) cudaFree(p);
    delete m_;
}

bool Glm5Mtp::init(const WeightTable& wt, const ModelGeometry& g, const NativeHead* head, int64_t max_cells, int max_t,
                   std::string& err) {
    using namespace strata::kernels;
    m_ = new Impl();
    Impl& m = *m_;
    m.wt = &wt;
    m.head = head;
    m.N = g.n_embd; m.NH = g.n_head; m.QLR = g.q_lora_rank; m.KVR = g.kv_lora_rank; m.QK = g.mla_key_dim; m.VD = g.mla_value_dim;
    m.IH = g.idx_q_heads; m.ID = g.idx_key_dim; m.NFF = g.n_ff;
    m.rms_eps = g.rms_eps; m.ln_eps = g.norm_eps;
    m.clamp_sh = g.swiglu_clamp_shared.empty() ? 0.f : g.swiglu_clamp_shared.back();
    m.max_cells = max_cells;
    m.max_blocks = max_cells / 4 + 2;
    m.MT = max_t;
    m.s = qsa_real_shapes();
    m.s.n_head = g.n_head; m.s.n_head_kv = g.n_head_kv; m.s.head_dim = g.head_dim; m.s.idx_n_head = g.idx_q_heads;
    m.s.idx_dim = g.idx_key_dim;
    m.cap = qsa_selection_width(kTopkMaxCells, m.s);
    if (head == nullptr || !head->loaded()) { err = "glm5 mtp: needs the native head"; return false; }
    if (native_embed() == nullptr) { err = "glm5 mtp: needs the native token embedding"; return false; }
    const int64_t L = g.n_layers;   // the NextN layer's index (blk.45)
    auto W = [&](const char* suffix) -> const WeightRef* {
        const std::string name = "blk." + std::to_string(L) + "." + suffix;
        const WeightRef* r = wt.find(name);
        if (r == nullptr && err.empty()) err = "glm5 mtp: " + name + " is missing";
        return r;
    };
    m.eh = W("nextn.eh_proj.weight"); m.enorm = W("nextn.enorm.weight"); m.hnorm = W("nextn.hnorm.weight");
    m.shn = W("nextn.shared_head_norm.weight"); m.an = W("attn_norm.weight"); m.fn = W("ffn_norm.weight");
    m.wqa = W("attn_q_a.weight"); m.wqan = W("attn_q_a_norm.weight"); m.wqb = W("attn_q_b.weight");
    m.wkb = W("attn_k_b.weight"); m.wvb = W("attn_v_b.weight"); m.wkv = W("attn_kv_a_mqa.weight");
    m.wkvn = W("attn_kv_a_norm.weight"); m.wo = W("attn_output.weight"); m.wik = W("indexer.attn_k.weight");
    m.wiqb = W("indexer.attn_q_b.weight"); m.wikn = W("indexer.k_norm.weight"); m.wikb = W("indexer.k_norm.bias");
    m.wip = W("indexer.proj.weight"); m.wape = W("indexer_compressor_ape.weight"); m.wig = W("indexer_compressor_gate.weight");
    m.wsg = W("ffn_gate_shexp.weight"); m.wsu = W("ffn_up_shexp.weight"); m.wsd = W("ffn_down_shexp.weight");
    if (!err.empty()) return false;
    for (const WeightRef* q : {m.eh, m.wqa, m.wqb, m.wkb, m.wvb, m.wkv, m.wo, m.wik, m.wiqb, m.wig, m.wsg, m.wsu, m.wsd})
        if (q->native_data == nullptr) { err = "glm5 mtp: a blk.45 projection is not served natively (--native)"; return false; }
    for (const WeightRef* f : {m.enorm, m.hnorm, m.shn, m.an, m.fn, m.wqan, m.wkvn, m.wikn, m.wikb, m.wip, m.wape})
        if (f->data == nullptr || f->bytes < (uint64_t) f->elements * 4) { err = "glm5 mtp: a blk.45 norm is not F32 in the arena"; return false; }
    const WeightRef* wout = wt.find("output.weight");
    m.n_vocab = wout ? wout->ne1 : 0;
    if (m.n_vocab <= 0) { err = "glm5 mtp: output.weight is missing"; return false; }
    if (cudaStreamCreateWithFlags(&m.cs, cudaStreamNonBlocking) != cudaSuccess) { err = "glm5 mtp: stream"; return false; }
    const size_t T = (size_t) m.MT;
    m.latent = m.dev<uint16_t>((size_t) max_cells * m.KVR);
    m.tail = m.dev<float>(3 * 2 * m.ID); m.tail_snap = m.dev<float>(3 * 2 * m.ID);
    m.pooled = m.dev<float>((size_t) (max_cells / 4 + 1) * m.ID);
    m.page_table = m.dev<int32_t>((size_t) (max_cells / 4 + 1));
    m.tok = m.dev<int32_t>(T); m.steps = m.dev<int32_t>(T * kStepCount); m.sel = m.dev<int32_t>(T * m.cap);
    m.out_tok = m.dev<int32_t>(T);
    m.emb = m.dev<float>(T * m.N); m.eN = m.dev<float>(T * m.N); m.hN = m.dev<float>(T * m.N); m.cat = m.dev<float>(T * 2 * m.N);
    m.u = m.dev<float>(T * m.N); m.x = m.dev<float>(T * m.N); m.ao = m.dev<float>(T * m.N);
    m.qr = m.dev<float>(T * m.QLR); m.qb = m.dev<float>(T * m.NH * m.QK); m.qT = m.dev<float>(T * m.NH * m.QK);
    m.qaT = m.dev<float>(T * m.NH * m.KVR); m.qa = m.dev<float>(T * m.NH * m.KVR); m.kv = m.dev<float>(T * m.KVR);
    m.rk = m.dev<float>(T * m.ID); m.rg = m.dev<float>(T * m.ID); m.iq = m.dev<float>(T * m.IH * m.ID); m.iw = m.dev<float>(T * m.IH);
    m.scores = m.dev<float>(T * m.max_blocks);
    m.lat = m.dev<float>(T * m.NH * m.KVR); m.latT = m.dev<float>(T * m.NH * m.KVR);
    m.vbT = m.dev<float>(T * m.NH * m.VD); m.vb = m.dev<float>(T * m.NH * m.VD);
    m.mla_scratch = m.dev<float>(T * mla_decode_attn_scratch_floats(m.cap, m.NH));
    m.sg = m.dev<float>(T * m.NFF); m.su = m.dev<float>(T * m.NFF); m.sgs = m.dev<float>(T + 4); m.sh = m.dev<float>(T * m.N);
    m.hp = m.dev<float>(T * m.N); m.logits = m.dev<float>(T * m.n_vocab);
    m.q8 = m.dev<uint8_t>((size_t) (m.NH * T) * (m.KVR / 32) * 36 + T * (2 * m.N / 32) * 36);
    for (void* p : m.allocs) if (p == nullptr) { err = "glm5 mtp: a device buffer does not fit"; return false; }
    {
        std::vector<int32_t> pt((size_t) (max_cells / 4 + 1));
        for (size_t i = 0; i < pt.size(); ++i) pt[i] = (int32_t) i;
        cudaMemcpy(m.page_table, pt.data(), pt.size() * 4, cudaMemcpyHostToDevice);
    }
    m.h_steps.assign(T * kStepCount, 0);
    std::fprintf(stderr, "strata glm5 mtp: NextN layer blk.%lld bound (shared expert only, phase A), %lld-cell cache\n",
                 (long long) L, (long long) max_cells);
    return true;
}

// The NextN layer over n rows: tokens tok_dev (device) at cells c0+1.., h rows h_in (device) -> m.hp (h') and
// m.out_tok (argmax), the drafter's cache written at cells c0 ..
bool Glm5Mtp::layer(const int32_t* tok_dev, const float* h_in, int n, int64_t c0, std::string& err) {
    using namespace strata::kernels;
    Impl& m = *m_;
    cudaStream_t cs = m.cs;
    const int64_t N = m.N, QLR = m.QLR, NH = m.NH, QK = m.QK, KVR = m.KVR, VD = m.VD, ID = m.ID, IH = m.IH;
    if (n < 1 || n > m.MT || c0 + n > m.max_cells) { err = "glm5 mtp: rows out of range"; return false; }
    for (int t = 0; t < n; ++t) qsa_step_fill(m.h_steps.data() + (size_t) t * kStepCount, c0 + t, m.s);
    cudaMemcpyAsync(m.steps, m.h_steps.data(), (size_t) n * kStepCount * 4, cudaMemcpyHostToDevice, cs);
    try {
        auto mmvq = [&](const WeightRef* w, const float* in, int n_in, float* y, int n_out) {
            native_quantize_q8_1(in, m.q8, n_in, n, cs);
            native_mmvq(w->native_type, w->native_data, m.q8, y, n_in, n_out, n, cs);
        };
        // u = eh_proj([enorm(e); hnorm(h)])
        native_embed()->gather_dev(tok_dev, n, m.emb, cs);
        native_qsa_rms_norm_weighted(m.emb, (const float*) m.enorm->data, m.eN, (int) N, n, m.rms_eps, cs);
        native_qsa_rms_norm_weighted(h_in, (const float*) m.hnorm->data, m.hN, (int) N, n, m.rms_eps, cs);
        cudaMemcpy2DAsync(m.cat, 2 * N * 4, m.eN, N * 4, N * 4, n, cudaMemcpyDeviceToDevice, cs);
        cudaMemcpy2DAsync(m.cat + N, 2 * N * 4, m.hN, N * 4, N * 4, n, cudaMemcpyDeviceToDevice, cs);
        mmvq(m.eh, m.cat, (int) (2 * N), m.u, (int) N);
        // ---- DSA (verify_glm5.cpp's block)
        native_qsa_rms_norm_weighted(m.u, (const float*) m.an->data, m.x, (int) N, n, m.rms_eps, cs);
        native_quantize_q8_1(m.x, m.q8, (int) N, n, cs);
        native_mmvq(m.wqa->native_type, m.wqa->native_data, m.q8, m.qr, (int) N, (int) QLR, n, cs);
        native_mmvq(m.wkv->native_type, m.wkv->native_data, m.q8, m.kv, (int) N, (int) KVR, n, cs);
        native_mmvq(m.wik->native_type, m.wik->native_data, m.q8, m.rk, (int) N, (int) ID, n, cs);
        native_mmvq(m.wig->native_type, m.wig->native_data, m.q8, m.rg, (int) N, (int) ID, n, cs);
        native_qsa_rms_norm_weighted(m.qr, (const float*) m.wqan->data, m.qr, (int) QLR, n, m.rms_eps, cs);
        native_qsa_rms_norm_weighted(m.kv, (const float*) m.wkvn->data, m.kv, (int) KVR, n, m.rms_eps, cs);
        glm5_latent_store(m.kv, m.latent, m.page_table, m.steps, kStepCount, 4, (int) KVR, n, cs);
        mmvq(m.wqb, m.qr, (int) QLR, m.qb, (int) (NH * QK));
        glm5_transpose_abd(m.qb, m.qT, n, (int) NH, (int) QK, cs);
        native_quantize_q8_1(m.qT, m.q8, (int) (QK * NH * n), 1, cs);
        {
            const size_t row = (size_t) (QK / 32) * 34, qcol = (size_t) (QK / 32) * 36;
            for (int64_t h = 0; h < NH; ++h)
                native_mmvq(m.wkb->native_type, (const uint8_t*) m.wkb->native_data + (size_t) h * KVR * row,
                            m.q8 + (size_t) h * n * qcol, m.qaT + (size_t) h * n * KVR, (int) QK, (int) KVR, n, cs);
        }
        glm5_transpose_abd(m.qaT, m.qa, (int) NH, n, (int) KVR, cs);
        const Glm5KpoolBuffers kb{m.tail, m.pooled};
        for (int t = 0; t < n; ++t)
            glm5_kpool_append(m.rk + (size_t) t * ID, m.rg + (size_t) t * ID, m.steps + (size_t) t * kStepCount + kStepPos,
                              (const float*) m.wikn->data, (const float*) m.wikb->data, (const float*) m.wape->data, m.ln_eps,
                              kb, m.max_cells, cs);
        mmvq(m.wiqb, m.qr, (int) QLR, m.iq, (int) (IH * ID));
        glm5_f32_gemv((const float*) m.wip->data, m.x, m.iw, (int) N, (int) IH, n, 1.0f / std::sqrt((float) (ID * IH)), cs);
        glm5_kpool_scores(m.iq, m.iw, m.pooled, m.steps, n, m.max_blocks, m.scores, cs);
        qsa_block_topk(m.scores, m.steps, n, m.max_blocks, m.cap, m.s, m.sel, cs);
        mla_decode_attn_batch(m.qa, m.latent, m.page_table, 4, m.sel, m.steps, m.cap, (int) NH, 1.0f / std::sqrt((float) QK),
                              m.mla_scratch, m.lat, n, cs);
        glm5_transpose_abd(m.lat, m.latT, n, (int) NH, (int) KVR, cs);
        native_quantize_q8_1(m.latT, m.q8, (int) (KVR * NH * n), 1, cs);
        {
            const size_t row = (size_t) (KVR / 32) * 34, qcol = (size_t) (KVR / 32) * 36;
            for (int64_t h = 0; h < NH; ++h)
                native_mmvq(m.wvb->native_type, (const uint8_t*) m.wvb->native_data + (size_t) h * VD * row,
                            m.q8 + (size_t) h * n * qcol, m.vbT + (size_t) h * n * VD, (int) KVR, (int) VD, n, cs);
        }
        glm5_transpose_abd(m.vbT, m.vb, (int) NH, n, (int) VD, cs);
        mmvq(m.wo, m.vb, (int) (NH * VD), m.ao, (int) N);
        add_inplace(m.u, m.ao, (int64_t) n * N, cs);
        // ---- FFN: the shared expert (phase A: no routed experts)
        native_qsa_rms_norm_weighted(m.u, (const float*) m.fn->data, m.x, (int) N, n, m.rms_eps, cs);
        NativeSharedWeights nsw;
        nsw.gate_type = m.wsg->native_type; nsw.gate_data = m.wsg->native_data; nsw.up_type = m.wsu->native_type;
        nsw.up_data = m.wsu->native_data; nsw.down_type = m.wsd->native_type; nsw.down_data = m.wsd->native_data; nsw.q8_1 = m.q8;
        shared_expert_multi(n, m.x, nullptr, nsw, nullptr, m.sg, m.su, m.sgs, m.sh, N, m.NFF, cs, m.clamp_sh);
        add_inplace(m.u, m.sh, (int64_t) n * N, cs);
        // ---- h' and the draft
        native_qsa_rms_norm_weighted(m.u, (const float*) m.shn->data, m.hp, (int) N, n, m.rms_eps, cs);
        native_quantize_q8_1(m.hp, m.q8, (int) N, n, cs);
        native_mmvq(m.head->type(), m.head->weights(), m.q8, m.logits, (int) N, (int) m.n_vocab, n, cs);
        SamplerParams sp;
        sp.greedy = true;
        sp.temperature = 0.0f;
        sample_tokens(m.logits, n, (int) m.n_vocab, nullptr, 0, sp, m.out_tok, cs);
    } catch (const std::exception& e) {
        err = std::string("glm5 mtp: ") + e.what();
        return false;
    }
    return true;
}

bool Glm5Mtp::catchup(const float* h_rows, const int32_t* tokens, int n, int64_t c0, int draft_from, int32_t* first,
                      std::string& err) {
    Impl& m = *m_;
    const auto t0 = std::chrono::steady_clock::now();
    cudaMemcpyAsync(m.tok, tokens, (size_t) n * 4, cudaMemcpyHostToDevice, m.cs);
    if (!layer(m.tok, h_rows, n, c0, err)) return false;
    if (draft_from >= 0) {
        int32_t d = 0;
        cudaMemcpyAsync(&d, m.out_tok + draft_from, 4, cudaMemcpyDeviceToHost, m.cs);
        // the chain's h: row draft_from's h', moved to row 0 of the h buffer the chain reads
        cudaMemcpyAsync(m.hN, m.hp + (size_t) draft_from * m.N, (size_t) m.N * 4, cudaMemcpyDeviceToDevice, m.cs);
        if (cudaStreamSynchronize(m.cs) != cudaSuccess) { err = "glm5 mtp: catch-up failed on the GPU"; return false; }
        *first = d;
    } else if (cudaStreamSynchronize(m.cs) != cudaSuccess) {
        err = "glm5 mtp: catch-up failed on the GPU";
        return false;
    }
    ms_draft += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return true;
}

bool Glm5Mtp::chain(int32_t first, int64_t c_last, int n_more, int32_t* out, std::string& err) {
    using namespace strata::kernels;
    Impl& m = *m_;
    const auto t0 = std::chrono::steady_clock::now();
    if (n_more <= 0) return true;
    copy_from_mapped(m.tail_snap, m.tail, 3 * 2 * m.ID, m.cs);   // the cells written below are rewritten later
    int32_t tok = first;
    for (int j = 0; j < n_more; ++j) {
        cudaMemcpyAsync(m.tok, &tok, 4, cudaMemcpyHostToDevice, m.cs);
        // h: the previous step's h' (kept in m.hN row 0; layer() overwrites hN with hnorm, so pass a copy)
        cudaMemcpyAsync(m.x, m.hN, (size_t) m.N * 4, cudaMemcpyDeviceToDevice, m.cs);
        if (!layer(m.tok, m.x, 1, c_last + 1 + j, err)) return false;
        int32_t d = 0;
        cudaMemcpyAsync(&d, m.out_tok, 4, cudaMemcpyDeviceToHost, m.cs);
        cudaMemcpyAsync(m.hN, m.hp, (size_t) m.N * 4, cudaMemcpyDeviceToDevice, m.cs);
        if (cudaStreamSynchronize(m.cs) != cudaSuccess) { err = "glm5 mtp: draft failed on the GPU"; return false; }
        out[j] = d;
        tok = d;
    }
    copy_from_mapped(m.tail, m.tail_snap, 3 * 2 * m.ID, m.cs);
    cudaStreamSynchronize(m.cs);
    ms_draft += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    ++rounds;
    return true;
}

}  // namespace strata::core
