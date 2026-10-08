#include "strata/core/layout.hpp"
#include <cstdio>
#include <stdexcept>

int main() {
    using namespace strata::core;
    int passed = 0;
    auto check = [&](bool ok) { if (!ok) throw std::runtime_error("shape contract"); ++passed; };
    try {
        TensorSpec conv{"conv", {4, 1, 8192}, false, true};
        check(tensor_shape_matches(conv, {4, 1, 8192}));
        check(tensor_shape_matches(conv, {4, 1, 8192, 1}));
        check(!tensor_shape_matches(conv, {4, 8192}));
        check(!tensor_shape_matches(conv, {4, 1, 8192, 2}));
        check(!tensor_shape_matches(conv, {4, 1, 4096, 2})); // same element count is insufficient
        TensorSpec mla{"mla", {256, 512, 64}};
        check(tensor_shape_matches(mla, {256, 512, 64}));
        check(!tensor_shape_matches(mla, {512, 256, 64}));
        check(!tensor_shape_matches(mla, {256, 32768})); // head axis must remain explicit
        check(!tensor_shape_matches(mla, {256, 512, 64, 1}));
        bool rejected = false;
        try { glm5_tensor_layout(ModelGeometry{}); } catch (const std::invalid_argument&) { rejected = true; }
        check(rejected);
        std::printf("glm5_layout_test: %d passed, 0 failed\n", passed);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "glm5_layout_test: %d passed, 1 failed: %s\n", passed, e.what()); return 1;
    }
}
