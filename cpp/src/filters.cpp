#include "motiondsp/filters.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace motiondsp {

namespace {
constexpr double kPi = 3.14159265358979323846;
}

MovingAverage::MovingAverage(std::size_t n) : n_(n), buf_(n, 0.0) {
    if (n == 0) throw std::invalid_argument("MovingAverage length must be >= 1");
}

double MovingAverage::process(double x) {
    if (count_ == n_) sum_ -= buf_[head_];
    else ++count_;
    buf_[head_] = x;
    sum_ += x;
    head_ = (head_ + 1) % n_;
    return sum_ / static_cast<double>(count_);
}

void MovingAverage::reset() {
    std::fill(buf_.begin(), buf_.end(), 0.0);
    head_ = count_ = 0;
    sum_ = 0.0;
}

Ema::Ema(double fc, double fs) : alpha_(alpha_for(fc, fs)) {}

double Ema::alpha_for(double fc, double fs) { return 1.0 - std::exp(-2.0 * kPi * fc / fs); }

double Ema::process(double x) {
    if (!primed_) {
        y_ = x;
        primed_ = true;
        return y_;
    }
    y_ += alpha_ * (x - y_);
    return y_;
}

BiquadCascade::BiquadCascade(std::vector<Biquad> sections)
    : sec_(std::move(sections)), z1_(sec_.size(), 0.0), z2_(sec_.size(), 0.0) {}

BiquadCascade BiquadCascade::butterworth_lowpass(int order, double fc, double fs) {
    if (order < 2 || order % 2 != 0) throw std::invalid_argument("order must be even and >= 2");
    if (!(fc > 0.0 && fc < fs / 2.0)) throw std::invalid_argument("cut-off must be in (0, fs/2)");
    const double w0 = 2.0 * kPi * fc / fs;
    const double cw = std::cos(w0), sw = std::sin(w0);
    std::vector<Biquad> secs;
    for (int k = 0; k < order / 2; ++k) {
        const double theta = kPi * (2.0 * k + 1.0) / (2.0 * order);
        const double q = 1.0 / (2.0 * std::cos(theta));
        const double alpha = sw / (2.0 * q);
        const double a0 = 1.0 + alpha;
        Biquad b{};
        b.b0 = (1.0 - cw) / 2.0 / a0;
        b.b1 = (1.0 - cw) / a0;
        b.b2 = b.b0;
        b.a1 = -2.0 * cw / a0;
        b.a2 = (1.0 - alpha) / a0;
        secs.push_back(b);
    }
    return BiquadCascade(std::move(secs));
}

double BiquadCascade::process(double x) {
    double v = x;
    for (std::size_t i = 0; i < sec_.size(); ++i) {
        const Biquad& s = sec_[i];
        const double y = s.b0 * v + z1_[i];
        z1_[i] = s.b1 * v - s.a1 * y + z2_[i];
        z2_[i] = s.b2 * v - s.a2 * y;
        v = y;
    }
    return v;
}

void BiquadCascade::process_block(const double* in, double* out, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) out[i] = process(in[i]);
}

void BiquadCascade::reset() {
    std::fill(z1_.begin(), z1_.end(), 0.0);
    std::fill(z2_.begin(), z2_.end(), 0.0);
}

void BiquadCascade::reset_steady(double x0) {
    double u = x0;
    for (std::size_t i = 0; i < sec_.size(); ++i) {
        const Biquad& s = sec_[i];
        const double gain = (s.b0 + s.b1 + s.b2) / (1.0 + s.a1 + s.a2);
        const double y = gain * u;
        z2_[i] = s.b2 * u - s.a2 * y;
        z1_[i] = s.b1 * u - s.a1 * y + z2_[i];
        u = y;
    }
}

namespace {

double window_median(const double* in, std::size_t len, std::ptrdiff_t center, std::size_t n) {
    std::array<double, kMaxMedianLen> w{};
    const std::ptrdiff_t h = static_cast<std::ptrdiff_t>(n / 2);
    const std::ptrdiff_t last = static_cast<std::ptrdiff_t>(len) - 1;
    for (std::ptrdiff_t j = -h; j <= h; ++j) {
        const std::ptrdiff_t idx = std::clamp(center + j, std::ptrdiff_t{0}, last);
        w[static_cast<std::size_t>(j + h)] = in[idx];
    }
    auto mid = w.begin() + static_cast<std::ptrdiff_t>(n / 2);
    std::nth_element(w.begin(), mid, w.begin() + static_cast<std::ptrdiff_t>(n));
    return *mid;
}

void check_len(std::size_t n) {
    if (n % 2 == 0 || n > kMaxMedianLen) throw std::invalid_argument("median length must be odd and <= 31");
}

}  // namespace

void median_filter(const double* in, double* out, std::size_t len, std::size_t n) {
    check_len(n);
    for (std::size_t i = 0; i < len; ++i)
        out[i] = window_median(in, len, static_cast<std::ptrdiff_t>(i), n);
}

void hampel(const double* in, double* out, std::size_t len, std::size_t n, double k, double* scratch) {
    check_len(n);
    double* med = scratch;
    double* dev = scratch + len;
    median_filter(in, med, len, n);
    for (std::size_t i = 0; i < len; ++i) dev[i] = std::fabs(in[i] - med[i]);
    for (std::size_t i = 0; i < len; ++i) {
        const double mad = window_median(dev, len, static_cast<std::ptrdiff_t>(i), n);
        const double thresh = k * 1.4826 * mad;
        out[i] = dev[i] > thresh ? med[i] : in[i];
    }
}

void hampel(const double* in, double* out, std::size_t len, std::size_t n, double k) {
    std::vector<double> scratch(2 * len);
    hampel(in, out, len, n, k, scratch.data());
}

}  // namespace motiondsp
