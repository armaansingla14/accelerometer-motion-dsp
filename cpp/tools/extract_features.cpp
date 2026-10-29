// Usage: extract_features <windows.txt> <features.csv> [c_ax c_ay c_az c_gz c_wl c_wr]
// Runs WindowProcessor over every window and writes one CSV row per window.
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <vector>

#include "motiondsp/features.hpp"
#include "motiondsp/window_io.hpp"

int main(int argc, char** argv) {
    using namespace motiondsp;
    if (argc != 3 && argc != 9) {
        std::fprintf(stderr, "usage: %s windows.txt features.csv [6 cut-offs]\n", argv[0]);
        return 2;
    }
    try {
        WindowSet ws = load_windows(argv[1]);
        if (ws.channels != kChannels) throw std::runtime_error("expected 6 channels");
        WindowConfig cfg;
        cfg.window = ws.window;
        if (argc == 9)
            for (std::size_t c = 0; c < kChannels; ++c) cfg.lp_cutoffs[c] = std::atof(argv[3 + c]);
        WindowProcessor proc(cfg);
        std::FILE* out = std::fopen(argv[2], "w");
        if (!out) throw std::runtime_error("cannot write output");
        std::vector<double> feat(kFeatures);
        for (std::size_t w = 0; w < ws.count(); ++w) {
            proc.compute(ws.window_ptr(w), ws.bias_ptr(w), feat.data());
            for (std::size_t i = 0; i < kFeatures; ++i)
                std::fprintf(out, i + 1 < kFeatures ? "%.17g," : "%.17g\n", feat[i]);
        }
        std::fclose(out);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
