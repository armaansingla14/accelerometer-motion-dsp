#include <cmath>
#include <random>
#include <vector>

#include "motiondsp/features.hpp"
#include "test_framework.hpp"

using namespace motiondsp;

namespace {
std::vector<double> make_window(std::size_t n, double v, double w, double vib_hz, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> nd(0.0, 0.02);
    std::vector<double> s(n * kChannels);
    for (std::size_t i = 0; i < n; ++i) {
        const double t = static_cast<double>(i) / 100.0;
        double* row = &s[i * kChannels];
        row[0] = nd(rng);
        row[1] = v * w + nd(rng);
        row[2] = 9.81 + 0.3 * std::sin(2 * 3.14159265358979323846 * vib_hz * t) + nd(rng);
        row[3] = w + nd(rng);
        row[4] = v - w * 0.15;
        row[5] = v + w * 0.15;
    }
    return s;
}
}  // namespace

TEST(feature_count_and_finiteness) {
    CHECK(kFeatures == 53);
    WindowConfig cfg;
    WindowProcessor proc(cfg);
    auto s = make_window(cfg.window, 0.6, 0.0, 15.0, 1);
    std::vector<double> bias(kChannels, 0.0), f(kFeatures);
    proc.compute(s.data(), bias.data(), f.data());
    for (double v : f) CHECK(std::isfinite(v));
}

TEST(turn_features_reflect_kinematics) {
    WindowConfig cfg;
    WindowProcessor proc(cfg);
    const double v = 0.5, w = 1.2;
    auto s = make_window(cfg.window, v, w, 15.0, 2);
    std::vector<double> bias(kChannels, 0.0), f(kFeatures);
    proc.compute(s.data(), bias.data(), f.data());
    const std::size_t cross = kChannels * kPerChannel;
    CHECK_NEAR(f[cross + 0], w * cfg.track_width, 1e-6);  // wr - wl
    CHECK_NEAR(f[cross + 1], v, 1e-6);                    // mean wheel speed
    CHECK_NEAR(f[cross + 3], 0.0, 0.02);                  // gyro agrees with encoders
    CHECK_NEAR(f[cross + 4], 0.0, 0.03);                  // lateral accel = v * w
    CHECK_NEAR(f[3 * kPerChannel + 0], w, 0.02);          // gz mean
}

TEST(vibration_frequency_detected_on_z_axis) {
    WindowConfig cfg;
    WindowProcessor proc(cfg);
    auto s = make_window(cfg.window, 0.6, 0.0, 18.75, 3);  // 18.75 Hz = bin 24
    std::vector<double> bias(kChannels, 0.0), f(kFeatures);
    proc.compute(s.data(), bias.data(), f.data());
    CHECK_NEAR(f[2 * kPerChannel + kPerChannel - 1], 18.75, 1e-9);
}

TEST(bias_is_subtracted) {
    WindowConfig cfg;
    WindowProcessor proc(cfg);
    auto s = make_window(cfg.window, 0.0, 0.0, 15.0, 4);
    std::vector<double> bias{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, f0(kFeatures), f1(kFeatures);
    proc.compute(s.data(), bias.data(), f0.data());
    for (std::size_t i = 0; i < cfg.window; ++i) s[i * kChannels + 3] += 0.05;
    bias[3] = 0.05;
    proc.compute(s.data(), bias.data(), f1.data());
    for (std::size_t i = 0; i < kFeatures; ++i) CHECK_NEAR(f0[i], f1[i], 1e-9);
}
