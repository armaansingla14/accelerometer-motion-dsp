#include "motiondsp/features.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace motiondsp {

WindowProcessor::WindowProcessor(const WindowConfig& cfg)
    : cfg_(cfg),
      spectrum_(cfg.window),
      chan_(cfg.window),
      despiked_(cfg.window),
      filtered_(cfg.window),
      power_(cfg.window / 2 + 1),
      scratch_(2 * cfg.window),
      filt_all_(cfg.window * kChannels) {
    if (cfg.window < 8) throw std::invalid_argument("window too short");
    for (std::size_t c = 0; c < kChannels; ++c)
        lp_[c] = BiquadCascade::butterworth_lowpass(cfg.lp_order, cfg.lp_cutoffs[c], cfg.fs);
}

void WindowProcessor::compute(const double* samples, const double* bias, double* out) {
    const std::size_t n = cfg_.window;
    const double df = cfg_.fs / static_cast<double>(n);
    double* f = out;
    for (std::size_t c = 0; c < kChannels; ++c) {
        for (std::size_t i = 0; i < n; ++i) chan_[i] = samples[i * kChannels + c] - bias[c];
        hampel(chan_.data(), despiked_.data(), n, cfg_.hampel_len, cfg_.hampel_k, scratch_.data());

        BiquadCascade& lp = lp_[c];
        lp.reset_steady(despiked_[0]);
        lp.process_block(despiked_.data(), filtered_.data(), n);
        std::copy(filtered_.begin(), filtered_.end(), filt_all_.begin() + static_cast<std::ptrdiff_t>(c * n));

        double sum = 0.0, mn = filtered_[0], mx = filtered_[0];
        for (double v : filtered_) {
            sum += v;
            mn = std::min(mn, v);
            mx = std::max(mx, v);
        }
        const double mean = sum / static_cast<double>(n);
        double var = 0.0;
        for (double v : filtered_) var += (v - mean) * (v - mean);
        *f++ = mean;
        *f++ = std::sqrt(var / static_cast<double>(n));
        *f++ = mx - mn;

        spectrum_.compute(despiked_.data(), power_);
        for (std::size_t b = 0; b < kBands; ++b) {
            double e = 0.0;
            for (std::size_t k = 0; k < power_.size(); ++k) {
                const double fk = static_cast<double>(k) * df;
                if (fk >= cfg_.band_edges[b] && fk < cfg_.band_edges[b + 1]) e += power_[k];
            }
            *f++ = std::log10(e + 1e-9);
        }
        std::size_t best = 1;
        for (std::size_t k = 2; k < power_.size(); ++k)
            if (power_[k] > power_[best]) best = k;
        *f++ = static_cast<double>(best) * df;
    }

    const double* ax = &filt_all_[0 * n];
    const double* ay = &filt_all_[1 * n];
    const double* gz = &filt_all_[3 * n];
    const double* wl = &filt_all_[4 * n];
    const double* wr = &filt_all_[5 * n];
    double wdiff = 0.0, wbar = 0.0, axm = 0.0, yaw = 0.0, lat = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double wb = 0.5 * (wl[i] + wr[i]);
        wdiff += wr[i] - wl[i];
        wbar += wb;
        axm += ax[i];
        yaw += gz[i] - (wr[i] - wl[i]) / cfg_.track_width;
        lat += ay[i] - wb * gz[i];
    }
    const double inv = 1.0 / static_cast<double>(n);
    const double duration = static_cast<double>(n - 1) / cfg_.fs;
    const double wheel_accel = (0.5 * (wl[n - 1] + wr[n - 1]) - 0.5 * (wl[0] + wr[0])) / duration;
    *f++ = wdiff * inv;
    *f++ = wbar * inv;
    *f++ = axm * inv - wheel_accel;
    *f++ = yaw * inv;
    *f++ = lat * inv;
}

}  // namespace motiondsp
