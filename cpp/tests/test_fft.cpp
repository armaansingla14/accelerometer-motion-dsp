#include <cmath>
#include <random>
#include <vector>

#include "motiondsp/fft.hpp"
#include "test_framework.hpp"

using namespace motiondsp;

TEST(fft_matches_dft) {
    std::mt19937 rng(7);
    std::normal_distribution<double> nd;
    for (std::size_t n : {1u, 2u, 4u, 16u, 128u, 512u}) {
        std::vector<cplx> x(n);
        for (auto& v : x) v = cplx(nd(rng), nd(rng));
        auto ref = dft(x);
        FftPlan plan(n);
        plan.forward(x);
        double err = 0;
        for (std::size_t k = 0; k < n; ++k) err = std::max(err, std::abs(x[k] - ref[k]));
        CHECK(err < 1e-9 * static_cast<double>(n));
    }
    CHECK_THROWS(FftPlan(12));
    CHECK(!is_power_of_two(0) && is_power_of_two(64) && !is_power_of_two(96));
}

TEST(power_spectrum_peak_and_parseval) {
    const std::size_t n = 128;
    const double fs = 100.0, f0 = 25.0;
    std::vector<double> x(n);
    for (std::size_t i = 0; i < n; ++i) x[i] = 2.0 * std::sin(2 * 3.14159265358979323846 * f0 * i / fs);
    PowerSpectrum ps(n);
    std::vector<double> p;
    ps.compute(x.data(), p);
    CHECK(p.size() == n / 2 + 1);
    std::size_t best = 1;
    for (std::size_t k = 2; k < p.size(); ++k)
        if (p[k] > p[best]) best = k;
    CHECK_NEAR(best * fs / n, f0, 1e-12);
    double total = 0, ms = 0;
    for (double v : p) total += v;
    for (double v : x) ms += v * v;
    CHECK_NEAR(total / n, ms / n, 0.02 * ms / n);
}

TEST(hann_window_shape) {
    auto w = hann(8);
    CHECK_NEAR(w[0], 0.0, 1e-15);
    CHECK_NEAR(w[4], 1.0, 1e-15);
    CHECK_NEAR(w[2], w[6], 1e-15);
}
