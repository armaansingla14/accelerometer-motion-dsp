#pragma once
// Plain-text exchange format for analysis windows shared with the Python side.
//
//   <n_windows> <window_len> <channels>
//   then per window: one line of <channels> bias values followed by
//   <window_len> lines of <channels> samples.

#include <cstddef>
#include <string>
#include <vector>

namespace motiondsp {

struct WindowSet {
    std::size_t window = 0;
    std::size_t channels = 0;
    std::vector<double> bias;     // n_windows x channels
    std::vector<double> samples;  // n_windows x window x channels
    std::size_t count() const { return channels ? bias.size() / channels : 0; }
    const double* window_ptr(std::size_t i) const { return &samples[i * window * channels]; }
    const double* bias_ptr(std::size_t i) const { return &bias[i * channels]; }
};

WindowSet load_windows(const std::string& path);

}  // namespace motiondsp
