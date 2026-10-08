// glm5_router_test <logits.bin> <bias.f32> <k> <w_scale> <out.bin>: Strata's router_sigmoid_topk on llama.cpp's real
// router logits (STRATA_DUMP_DIR full tensor: int32 ne[4] + floats, ne0 = n_expert, ne1 = tokens).
// Writes int32 T, k, then ids[T*k] int32, weights[T*k] f32; tools/glm5_ref/check_router.py compares with llama.cpp.
#include "strata/kernels/router_top10.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 6) { std::fprintf(stderr, "usage: glm5_router_test logits.bin bias.f32 k w_scale out.bin\n"); return 2; }
    FILE* f = std::fopen(argv[1], "rb");
    int ne[4]; std::fread(ne, 4, 4, f);
    const int E = ne[0], T = ne[1];
    std::vector<float> lg((size_t)E * T); std::fread(lg.data(), 4, lg.size(), f); std::fclose(f);
    std::vector<float> bias(E); f = std::fopen(argv[2], "rb"); std::fread(bias.data(), 4, E, f); std::fclose(f);
    const int k = std::atoi(argv[3]); const float ws = (float)std::atof(argv[4]);
    float *dl, *db, *dw; int* di;
    cudaMalloc(&dl, lg.size() * 4); cudaMalloc(&db, E * 4); cudaMalloc(&dw, T * k * 4); cudaMalloc(&di, T * k * 4);
    cudaMemcpy(dl, lg.data(), lg.size() * 4, cudaMemcpyHostToDevice); cudaMemcpy(db, bias.data(), E * 4, cudaMemcpyHostToDevice);
    strata::kernels::router_sigmoid_topk(dl, db, T, E, k, ws, di, dw, nullptr);
    if (cudaDeviceSynchronize() != cudaSuccess) { std::fprintf(stderr, "cuda error\n"); return 2; }
    std::vector<int> ids(T * k); std::vector<float> w(T * k);
    cudaMemcpy(ids.data(), di, T * k * 4, cudaMemcpyDeviceToHost); cudaMemcpy(w.data(), dw, T * k * 4, cudaMemcpyDeviceToHost);
    f = std::fopen(argv[5], "wb"); std::fwrite(&T, 4, 1, f); std::fwrite(&k, 4, 1, f);
    std::fwrite(ids.data(), 4, ids.size(), f); std::fwrite(w.data(), 4, w.size(), f); std::fclose(f);
    std::printf("E=%d T=%d k=%d done\n", E, T, k);
    return 0;
}
