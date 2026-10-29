// Usage: bench_latency <windows.txt> <out.csv> [reps]
// Times WindowProcessor::compute per window (bias removal, Hampel, 4th-order
// Butterworth on 6 channels, 6 FFTs, 53 features) on the host CPU.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <numeric>
#include <vector>

#include "motiondsp/features.hpp"
#include "motiondsp/window_io.hpp"

int main(int argc, char** argv) {
    using namespace motiondsp;
    using clock = std::chrono::steady_clock;
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s windows.txt out.csv [reps]\n", argv[0]);
        return 2;
    }
    const int reps = argc > 3 ? std::atoi(argv[3]) : 20;
    try {
        WindowSet ws = load_windows(argv[1]);
        WindowConfig cfg;
        cfg.window = ws.window;
        WindowProcessor proc(cfg);
        std::vector<double> feat(kFeatures);
        double sink = 0.0;
        for (std::size_t w = 0; w < std::min<std::size_t>(ws.count(), 200); ++w) {
            proc.compute(ws.window_ptr(w), ws.bias_ptr(w), feat.data());
            sink += feat[0];
        }
        std::vector<double> us;
        us.reserve(ws.count() * static_cast<std::size_t>(reps));
        for (int r = 0; r < reps; ++r) {
            for (std::size_t w = 0; w < ws.count(); ++w) {
                const auto t0 = clock::now();
                proc.compute(ws.window_ptr(w), ws.bias_ptr(w), feat.data());
                const auto t1 = clock::now();
                sink += feat[kFeatures - 1];
                us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
            }
        }
        std::sort(us.begin(), us.end());
        auto pct = [&](double p) { return us[static_cast<std::size_t>(p * static_cast<double>(us.size() - 1))]; };
        const double mean = std::accumulate(us.begin(), us.end(), 0.0) / static_cast<double>(us.size());
        const double samples_per_s = static_cast<double>(ws.window * kChannels) / (mean * 1e-6);
        std::FILE* out = std::fopen(argv[2], "w");
        if (!out) throw std::runtime_error("cannot write output");
        std::fprintf(out, "windows,window_len,channels,calls,mean_us,p50_us,p99_us,max_us,channel_samples_per_s\n");
        std::fprintf(out, "%zu,%zu,%zu,%zu,%.3f,%.3f,%.3f,%.3f,%.0f\n", ws.count(), ws.window, kChannels,
                     us.size(), mean, pct(0.5), pct(0.99), us.back(), samples_per_s);
        std::fclose(out);
        std::printf("calls=%zu mean=%.2fus p50=%.2fus p99=%.2fus max=%.2fus (checksum %.3g)\n", us.size(), mean,
                    pct(0.5), pct(0.99), us.back(), sink);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
