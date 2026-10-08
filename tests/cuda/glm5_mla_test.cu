// glm5_mla_test <case.bin> <out.bin>: Strata's mla_decode_attn_batch on llama.cpp's layer-3 q_absorbed / kv_cmpr
// (tools/glm5_ref/make_mla_case.py make), every query causal (query t: cells 0..t), page_size 4.  Writes O
// [T, 64, 512] for make_mla_case.py check, and times one query over a 2048-cell selection.
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace strata::kernels;

int main(int argc, char** argv) {
    if (argc < 3) return 2;
    FILE* f = std::fopen(argv[1], "rb");
    int T, H, D; float scale;
    std::fread(&T, 4, 1, f); std::fread(&H, 4, 1, f); std::fread(&D, 4, 1, f); std::fread(&scale, 4, 1, f);
    std::vector<float> q((size_t)T * H * D); std::fread(q.data(), 4, q.size(), f);
    std::vector<uint16_t> kv((size_t)T * D); std::fread(kv.data(), 2, kv.size(), f); std::fclose(f);
    const int page = 4, cap = 64;
    const int n_pages = (T + page - 1) / page;
    std::vector<int32_t> pt(n_pages); for (int i = 0; i < n_pages; ++i) pt[i] = i;
    std::vector<int32_t> ids((size_t)T * cap, 0), steps((size_t)T * kStepCount, 0);
    for (int t = 0; t < T; ++t) {
        for (int c = 0; c <= t; ++c) ids[(size_t)t * cap + c] = c;
        steps[(size_t)t * kStepCount + kStepWidth] = t + 1;
    }
    float *dq, *dout, *scr; uint16_t* dkv; int32_t *dpt, *dids, *dst;
    cudaMalloc(&dq, q.size() * 4); cudaMemcpy(dq, q.data(), q.size() * 4, cudaMemcpyHostToDevice);
    cudaMalloc(&dkv, (size_t)n_pages * page * D * 2); cudaMemset(dkv, 0, (size_t)n_pages * page * D * 2);
    cudaMemcpy(dkv, kv.data(), kv.size() * 2, cudaMemcpyHostToDevice);
    cudaMalloc(&dpt, pt.size() * 4); cudaMemcpy(dpt, pt.data(), pt.size() * 4, cudaMemcpyHostToDevice);
    cudaMalloc(&dids, ids.size() * 4); cudaMemcpy(dids, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice);
    cudaMalloc(&dst, steps.size() * 4); cudaMemcpy(dst, steps.data(), steps.size() * 4, cudaMemcpyHostToDevice);
    const size_t sf = mla_decode_attn_scratch_floats(cap, H);
    cudaMalloc(&scr, sf * T * 4); cudaMalloc(&dout, q.size() * 4);
    mla_decode_attn_batch(dq, dkv, dpt, page, dids, dst, cap, H, scale, scr, dout, T, nullptr);
    if (cudaDeviceSynchronize() != cudaSuccess) { std::printf("cuda error\n"); return 2; }
    std::vector<float> out(q.size()); cudaMemcpy(out.data(), dout, out.size() * 4, cudaMemcpyDeviceToHost);
    f = std::fopen(argv[2], "wb"); std::fwrite(out.data(), 4, out.size(), f); std::fclose(f);

    // timing: one query, 2048 selected cells (the steady-state decode width), random latents
    const int big = 2048, bp = big / page;
    std::vector<int32_t> pt2(bp), ids2(big), st2(kStepCount, 0);
    for (int i = 0; i < bp; ++i) pt2[i] = i;
    for (int i = 0; i < big; ++i) ids2[i] = i;
    st2[kStepWidth] = big;
    uint16_t* dkv2; int32_t *dpt2, *dids2, *dst2; float* scr2;
    cudaMalloc(&dkv2, (size_t)big * D * 2); cudaMemset(dkv2, 0x3c, (size_t)big * D * 2);   // fp16 ~1.0-ish pattern
    cudaMalloc(&dpt2, bp * 4); cudaMemcpy(dpt2, pt2.data(), bp * 4, cudaMemcpyHostToDevice);
    cudaMalloc(&dids2, big * 4); cudaMemcpy(dids2, ids2.data(), big * 4, cudaMemcpyHostToDevice);
    cudaMalloc(&dst2, kStepCount * 4); cudaMemcpy(dst2, st2.data(), kStepCount * 4, cudaMemcpyHostToDevice);
    cudaMalloc(&scr2, mla_decode_attn_scratch_floats(big, H) * 4);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (int i = 0; i < 20; ++i) mla_decode_attn_batch(dq, dkv2, dpt2, page, dids2, dst2, big, H, scale, scr2, dout, 1, nullptr);
    cudaEventRecord(e0);
    for (int i = 0; i < 200; ++i) mla_decode_attn_batch(dq, dkv2, dpt2, page, dids2, dst2, big, H, scale, scr2, dout, 1, nullptr);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms; cudaEventElapsedTime(&ms, e0, e1);
    std::printf("T=%d H=%d D=%d done; one query over %d cells: %.1f us/call\n", T, H, D, big, ms * 1000.f / 200);
    return 0;
}
