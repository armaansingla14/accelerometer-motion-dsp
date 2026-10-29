#include "motiondsp/fft.hpp"

#include <cmath>
#include <stdexcept>

namespace motiondsp {

namespace {
constexpr double kPi = 3.14159265358979323846;
}

bool is_power_of_two(std::size_t n) { return n != 0 && (n & (n - 1)) == 0; }

std::vector<cplx> dft(const std::vector<cplx>& x) {
    const std::size_t n = x.size();
    std::vector<cplx> out(n);
    for (std::size_t k = 0; k < n; ++k) {
        cplx acc{0.0, 0.0};
        for (std::size_t t = 0; t < n; ++t) {
            const double ang = -2.0 * kPi * static_cast<double>((k * t) % n) / static_cast<double>(n);
            acc += x[t] * cplx(std::cos(ang), std::sin(ang));
        }
        out[k] = acc;
    }
    return out;
}

FftPlan::FftPlan(std::size_t n) : n_(n), rev_(n), twiddle_(n / 2) {
    if (!is_power_of_two(n)) throw std::invalid_argument("FFT length must be a power of two");
    std::size_t bits = 0;
    while ((std::size_t{1} << bits) < n) ++bits;
    for (std::size_t i = 0; i < n; ++i) {
        std::size_t r = 0;
        for (std::size_t b = 0; b < bits; ++b)
            if (i & (std::size_t{1} << b)) r |= std::size_t{1} << (bits - 1 - b);
        rev_[i] = r;
    }
    for (std::size_t i = 0; i < n / 2; ++i) {
        const double ang = -2.0 * kPi * static_cast<double>(i) / static_cast<double>(n);
        twiddle_[i] = cplx(std::cos(ang), std::sin(ang));
    }
}

void FftPlan::forward(std::vector<cplx>& a) const {
    if (a.size() != n_) throw std::invalid_argument("FFT input has wrong length");
    for (std::size_t i = 0; i < n_; ++i)
        if (i < rev_[i]) std::swap(a[i], a[rev_[i]]);
    for (std::size_t size = 2; size <= n_; size <<= 1) {
        const std::size_t half = size / 2;
        const std::size_t step = n_ / size;
        for (std::size_t start = 0; start < n_; start += size) {
            for (std::size_t j = 0; j < half; ++j) {
                const cplx t = a[start + j + half] * twiddle_[j * step];
                const cplx u = a[start + j];
                a[start + j] = u + t;
                a[start + j + half] = u - t;
            }
        }
    }
}

std::vector<double> hann(std::size_t n) {
    std::vector<double> w(n);
    for (std::size_t i = 0; i < n; ++i)
        w[i] = 0.5 - 0.5 * std::cos(2.0 * kPi * static_cast<double>(i) / static_cast<double>(n));
    return w;
}

PowerSpectrum::PowerSpectrum(std::size_t n) : n_(n), plan_(n), window_(hann(n)), wsum2_(0.0), buf_(n) {
    for (double w : window_) wsum2_ += w * w;
}

void PowerSpectrum::compute(const double* x, std::vector<double>& power) {
    double mean = 0.0;
    for (std::size_t i = 0; i < n_; ++i) mean += x[i];
    mean /= static_cast<double>(n_);
    for (std::size_t i = 0; i < n_; ++i) buf_[i] = cplx((x[i] - mean) * window_[i], 0.0);
    plan_.forward(buf_);
    const std::size_t nb = bins();
    power.resize(nb);
    for (std::size_t k = 0; k < nb; ++k) {
        double p = std::norm(buf_[k]) / wsum2_;
        if (k != 0 && k != nb - 1) p *= 2.0;
        power[k] = p;
    }
}

}  // namespace motiondsp
