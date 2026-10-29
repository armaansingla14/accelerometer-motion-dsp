#pragma once
// Per-window feature extraction (mirror of python/motiondsp/pipeline.py).

#include <array>
#include <cstddef>
#include <vector>

#include "motiondsp/fft.hpp"
#include "motiondsp/filters.hpp"

namespace motiondsp {

constexpr std::size_t kChannels = 6;  // ax, ay, az, gz, wl, wr
constexpr std::size_t kBands = 4;
constexpr std::size_t kPerChannel = 3 + kBands + 1;
constexpr std::size_t kCross = 5;
constexpr std::size_t kFeatures = kChannels * kPerChannel + kCross;

struct WindowConfig {
    double fs = 100.0;
    std::size_t window = 128;
    int lp_order = 4;
    std::array<double, kChannels> lp_cutoffs{10.0, 10.0, 20.0, 10.0, 15.0, 15.0};
    std::size_t hampel_len = 7;
    double hampel_k = 3.0;
    double track_width = 0.30;
    std::array<double, kBands + 1> band_edges{0.0, 3.0, 8.0, 20.0, 50.01};
};

// Processes one window of interleaved samples (row-major, window x 6):
// bias removal -> Hampel despiking -> Butterworth low-pass (steady-state
// initialised) -> time-domain, spectral and cross-sensor features.
// All buffers are allocated in the constructor; compute() does not allocate.
class WindowProcessor {
public:
    explicit WindowProcessor(const WindowConfig& cfg);
    void compute(const double* samples, const double* bias, double* features);
    const WindowConfig& config() const { return cfg_; }

private:
    WindowConfig cfg_;
    std::array<BiquadCascade, kChannels> lp_;
    PowerSpectrum spectrum_;
    std::vector<double> chan_, despiked_, filtered_, power_, scratch_;
    std::vector<double> filt_all_;  // window x channels, channel-major
};

}  // namespace motiondsp
