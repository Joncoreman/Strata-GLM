// Opens only the directories/metadata of all shards. Never dereferences tensor_data().
#include "strata/core/layout.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include <filesystem>
#include <cstdio>
#include <set>
#include <sstream>

static std::string shape(const std::vector<uint64_t>& s) {
    std::ostringstream out;
    out << '[';
    for (size_t i = 0; i < s.size(); ++i) out << (i ? "," : "") << s[i];
    out << ']';
    return out.str();
}

int main(int argc, char** argv) {
    try {
        using namespace strata::core;
        if (argc != 2) {
            std::fprintf(stderr, "usage: glm5_header_check <any complete GGUF shard>\n"); return 2;
        }
        const auto paths = strata::gguf_split_paths(argv[1]);
        for (const auto& p : paths)
            if (std::filesystem::exists(p + ".aria2")) throw std::runtime_error("download still active: " + p);
        const strata::GgufModel model(paths);
        const auto g = geometry_from_gguf(model.meta());
        const auto layout = glm5_tensor_layout(g);
        std::set<std::string> unaccounted, spec_names;
        size_t total = 0, matched = 0, missing = 0, mismatched = 0, optional_absent = 0, out_of_bounds = 0;
        for (size_t s = 0; s < model.size(); ++s) {
            const auto& shard = model.shard(s);
            std::printf("shard %zu: %zu tensors; header data_start=%llu\n", s + 1, shard.tensors().size(),
                        (unsigned long long)shard.data_start());
            total += shard.tensors().size();
            for (const auto& t : shard.tensors()) {
                unaccounted.insert(t.name);
                if (!model.in_bounds(t, s)) { ++out_of_bounds; std::printf("OUT_OF_BOUNDS %s\n", t.name.c_str()); }
            }
        }
        for (const auto& want : layout) {
            if (!spec_names.insert(want.name).second) throw std::runtime_error("duplicate schema name " + want.name);
            const auto* t = model.find(want.name);
            if (!t) {
                if (want.optional) { ++optional_absent; std::printf("OPTIONAL_ABSENT %s\n", want.name.c_str()); }
                else { ++missing; std::printf("MISSING %s\n", want.name.c_str()); }
                continue;
            }
            unaccounted.erase(want.name);
            if (!tensor_shape_matches(want, t->shape)) {
                ++mismatched;
                std::printf("SHAPE %s: got %s want %s\n", t->name.c_str(), shape(t->shape).c_str(), shape(want.shape).c_str());
            } else ++matched;
        }
        for (const auto& n : unaccounted) std::printf("LEFTOVER %s\n", n.c_str());
        std::printf("expert types (layer gate up down):\n");
        for (int64_t i = 0; i < g.n_layers_all(); ++i) {
            if (i < g.n_dense_layers) { std::printf("%lld dense (no routed experts)\n", (long long)i); continue; }
            std::printf("%lld", (long long)i);
            for (const char* role : {"gate", "up", "down"}) {
                const auto* t = model.find("blk." + std::to_string(i) + ".ffn_" + role + "_exps.weight");
                std::printf(" %s", t ? t->type_name() : "MISSING");
            }
            std::printf("\n");
        }
        const size_t failed = missing + mismatched + unaccounted.size() + out_of_bounds;
        std::printf("glm5_header_check: total=%zu matched=%zu missing=%zu shape_mismatch=%zu leftovers=%zu "
                    "optional_absent=%zu out_of_bounds=%zu; %zu passed, %zu failed\n",
                    total, matched, missing, mismatched, unaccounted.size(), optional_absent, out_of_bounds, matched, failed);
        return failed ? 1 : 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "glm5_header_check: FAIL: %s\n", e.what()); return 1;
    }
}
