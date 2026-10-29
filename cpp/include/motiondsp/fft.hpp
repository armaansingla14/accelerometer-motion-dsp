#pragma once
// DFT / radix-2 FFT and power spectrum helpers.

#include <complex>
#include <cstddef>
#include <vector>

namespace motiondsp {

using cplx = std::complex<double>;

// O(N^2) reference transform.
std::vector<cplx> dft(const std::vector<cplx>& x);

// Iterative in-place radix-2 Cooley-Tukey FFT with precomputed twiddles and
// bit-reversal table. Construct once per window length and reuse.
class FftPlan {
public:
    explicit FftPlan(std::size_t n);
    void forward(std::vector<cplx>& data) const;
    std::size_t size() const { return n_; }

private:
    std::size_t n_;
    std::vector<std::size_t> rev_;
    std::vector<cplx> twiddle_;
};

bool is_power_of_two(std::size_t n);

// Periodic Hann window.
std::vector<double> hann(std::size_t n);

// One-sided power spectrum of a Hann-windowed, mean-removed real frame.
// Output has n/2 + 1 bins with bin spacing fs / n.
class PowerSpectrum {
public:
    explicit PowerSpectrum(std::size_t n);
    void compute(const double* x, std::vector<double>& power);
    std::size_t bins() const { return n_ / 2 + 1; }

private:
    std::size_t n_;
    FftPlan plan_;
    std::vector<double> window_;
    double wsum2_;
    std::vector<cplx> buf_;
};

}  // namespace motiondsp
