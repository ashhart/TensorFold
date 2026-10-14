// nGEN generator of the XPU prompt GEMMs (int4 weights): one kernel as a zebin (docs/xpu/NAVIGATION.md section 8).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>

#include "w4_prompt.hpp"
#include "w4a8_prompt.hpp"

using namespace w4;

int main(int argc, char **argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s xe2|xe3 64|128 OUT.zebin [b3] [s4] [asplit] [swiglu] [a8|a4]\n", argv[0]);
        return 2;
    }
    const int gsz = std::atoi(argv[2]);
    bool s4 = false, asplit = false, b3 = false, swiglu = false, a8 = false, a4 = false;
    for (int i = 4; i < argc; i++) {
        if (!std::strcmp(argv[i], "s4")) s4 = true;
        if (!std::strcmp(argv[i], "asplit")) asplit = true;
        if (!std::strcmp(argv[i], "b3")) b3 = true;
        if (!std::strcmp(argv[i], "swiglu")) swiglu = true;
        if (!std::strcmp(argv[i], "a8")) a8 = true;
        if (!std::strcmp(argv[i], "a4")) a4 = true;
    }
    std::vector<uint8_t> bin;
    if (a8 || a4)                                      // W4A8 / W4A4 (GPTQ g128): the only option is swiglu (W4A8)
        bin = W4A8Prompt<HW::Xe2>(Product(ProductFamily::GenericXe2, 0, PlatformType::Discrete), swiglu, a4).getBinary();
    else if (std::strcmp(argv[1], "xe2") == 0)
        bin = W4Prompt<HW::Xe2>(Product(ProductFamily::GenericXe2, 0, PlatformType::Discrete), gsz, s4, asplit, b3,
                                swiglu).getBinary();
    else
        bin = W4Prompt<HW::Xe3>(Product(ProductFamily::GenericXe3, 0, PlatformType::Integrated), gsz, s4, asplit, b3,
                                swiglu).getBinary();
    std::ofstream(argv[3], std::ios::binary).write(reinterpret_cast<const char *>(bin.data()), bin.size());
    std::printf("%s: %zu bytes\n", argv[3], bin.size());
    return 0;
}
