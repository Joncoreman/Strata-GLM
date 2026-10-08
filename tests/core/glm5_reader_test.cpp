#include "gguf_fixture.hpp"
#include <cstdio>
#include <filesystem>
#include <chrono>

int main(int argc, char** argv) {
    int passed = 0;
    auto check = [&](bool ok, const char* msg) { if (!ok) throw std::runtime_error(msg); ++passed; };
    // CTest's build directory (or the caller's working directory), never the system temp directory.
    const auto tmp = std::filesystem::current_path() /
        ("strata_glm5_reader_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".gguf");
    try {
        auto base = [] (const std::string& arch) {
            const auto p = arch + ".";
            return std::vector<fixture::Kv>{fixture::str("general.architecture", arch),
                fixture::u32(p + "block_count", 48), fixture::u32(p + "embedding_length", 2560),
                fixture::u32(p + "expert_count", 512), fixture::u32(p + "expert_used_count", 10),
                fixture::u32(p + "attention.head_count", 24)};
        };
        auto inspect = [&](std::vector<fixture::Kv> kv, bool accepted, size_t array_size = 0) {
            fixture::write(tmp, kv, {});
            const strata::GgufFile file(tmp.string());
            check(strata::check_architecture(file).empty() == accepted, "architecture acceptance");
            if (array_size) check(strata::read_head_count_kv(file).size() == array_size, "full head array retained");
        };
        auto q = base("qwen4exp");
        q.push_back(fixture::u32("qwen4exp.attention.head_count_kv", 2));
        inspect(q, true, 48);
        q[2].u = 4096; inspect(q, false); // legacy Qwen guard still rejects different hidden width
        auto g = base("glm5-next");
        // Synthetic layer count deliberately beyond the old 64-item metadata sample.
        g[1].u = 80;
        std::vector<int32_t> heads(80, 0); heads.back() = 1;
        g.push_back(fixture::i32_array("glm5-next.attention.head_count_kv", heads));
        inspect(g, true, 80);
        g.back().array.pop_back(); inspect(g, false);
        g.back().array.push_back(-1); inspect(g, false);
        g.back().array.back() = 25; inspect(g, false);
        g.back() = fixture::u32("glm5-next.attention.head_count_kv", 1); inspect(g, false);
        g.pop_back(); inspect(g, false);
        inspect(base("unknown"), false);
        if (argc == 2) {
            const strata::GgufFile file(argv[1]);
            check(strata::check_architecture(file).empty(), "real GLM metadata accepted");
            const auto h = strata::read_head_count_kv(file);
            check(h.size() == 46 && h[44] == 0 && h[45] == 1, "real GLM array includes sparse MTP");
        }
        std::filesystem::remove(tmp); // this test's newly created fixture only
        std::printf("glm5_reader_test: %d passed, 0 failed\n", passed);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "glm5_reader_test: %d passed, 1 failed: %s\n", passed, e.what());
        return 1;
    }
}
