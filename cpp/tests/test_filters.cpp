#include <cmath>
#include <complex>
#include <random>
#include <vector>

#include "motiondsp/filters.hpp"
#include "test_framework.hpp"

using namespace motiondsp;

namespace {
constexpr double kPi = 3.14159265358979323846;

double cascade_gain(const BiquadCascade& c, double f, double fs) {
    const std::complex<double> z = std::polar(1.0, -2.0 * kPi * f / fs);  // z^-1
    std::complex<double> h{1.0, 0.0};
    for (const auto& s : c.sections())
        h *= (s.b0 + s.b1 * z + s.b2 * z * z) / (1.0 + s.a1 * z + s.a2 * z * z);
    return std::abs(h);
}
}  // namespace

TEST(moving_average_startup_and_steady) {
    MovingAverage ma(4);
    CHECK_NEAR(ma.process(4.0), 4.0, 1e-12);
    CHECK_NEAR(ma.process(2.0), 3.0, 1e-12);
    ma.process(0.0);
    CHECK_NEAR(ma.process(2.0), 2.0, 1e-12);
    CHECK_NEAR(ma.process(6.0), 2.5, 1e-12);  // (2+0+2+6)/4
    CHECK_THROWS(MovingAverage(0));
}

TEST(ema_converges_and_has_3db_cutoff) {
    Ema e(5.0, 100.0);
    e.reset(0.0);
    double y = 0;
    for (int i = 0; i < 1000; ++i) y = e.process(1.0);
    CHECK_NEAR(y, 1.0, 1e-9);
    const double a = e.alpha();
    const std::complex<double> z = std::polar(1.0, -2.0 * kPi * 5.0 / 100.0);
    const double g = std::abs(a / (1.0 - (1.0 - a) * z));
    CHECK_NEAR(20 * std::log10(g), -3.0, 0.4);
}

TEST(butterworth_magnitude_response) {
    for (int order : {2, 4, 6}) {
        auto c = BiquadCascade::butterworth_lowpass(order, 10.0, 100.0);
        CHECK(c.sections().size() == static_cast<std::size_t>(order / 2));
        CHECK_NEAR(cascade_gain(c, 0.0, 100.0), 1.0, 1e-12);
        CHECK_NEAR(20 * std::log10(cascade_gain(c, 10.0, 100.0)), -3.0103, 0.01);
        // Butterworth: |H(f)|^2 = 1 / (1 + (tan(pi f/fs)/tan(pi fc/fs))^(2N))
        const double r = std::tan(kPi * 25.0 / 100.0) / std::tan(kPi * 10.0 / 100.0);
        CHECK_NEAR(cascade_gain(c, 25.0, 100.0), 1.0 / std::sqrt(1.0 + std::pow(r, 2 * order)), 1e-9);
    }
    CHECK_THROWS(BiquadCascade::butterworth_lowpass(3, 10.0, 100.0));
    CHECK_THROWS(BiquadCascade::butterworth_lowpass(4, 60.0, 100.0));
}

TEST(biquad_steady_state_reset) {
    auto c = BiquadCascade::butterworth_lowpass(4, 8.0, 100.0);
    c.reset_steady(9.81);
    for (int i = 0; i < 50; ++i) CHECK_NEAR(c.process(9.81), 9.81, 1e-9);
    c.reset();
    CHECK(std::fabs(c.process(9.81)) < 1.0);  // zero state: start-up transient
}

TEST(biquad_attenuates_high_frequency) {
    auto c = BiquadCascade::butterworth_lowpass(4, 5.0, 100.0);
    double peak = 0;
    for (int i = 0; i < 400; ++i) {
        const double y = c.process(std::sin(2 * kPi * 30.0 * i / 100.0));
        if (i > 200) peak = std::max(peak, std::fabs(y));
    }
    CHECK(peak < 0.01);
}

TEST(hampel_removes_isolated_spikes) {
    std::mt19937 rng(1);
    std::normal_distribution<double> noise(0.0, 0.05);
    std::vector<double> x(200), clean(200), y(200);
    for (int i = 0; i < 200; ++i) {
        clean[i] = std::sin(2 * kPi * i / 100.0);
        x[i] = clean[i] + noise(rng);
    }
    x[40] += 8.0;
    x[150] -= 6.0;
    hampel(x.data(), y.data(), x.size(), 7, 3.0);
    CHECK(std::fabs(y[40] - clean[40]) < 0.3);
    CHECK(std::fabs(y[150] - clean[150]) < 0.3);
    std::vector<double> s(2 * x.size());
    std::vector<double> y2(200);
    hampel(x.data(), y2.data(), x.size(), 7, 3.0, s.data());
    for (int i = 0; i < 200; ++i) CHECK(y[i] == y2[i]);
    CHECK_THROWS(hampel(x.data(), y.data(), x.size(), 6, 3.0));
}

TEST(median_filter_edges) {
    const std::vector<double> x{5, 1, 9, 2, 2};
    std::vector<double> y(5);
    median_filter(x.data(), y.data(), x.size(), 3);
    CHECK_NEAR(y[0], 5, 0);  // window {5,5,1}
    CHECK_NEAR(y[1], 5, 0);
    CHECK_NEAR(y[2], 2, 0);
    CHECK_NEAR(y[4], 2, 0);
}
