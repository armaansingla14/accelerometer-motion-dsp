#pragma once
// Causal filters for streaming IMU / encoder samples.

#include <cstddef>
#include <vector>

namespace motiondsp {

// Causal boxcar average. Before n samples have been seen it averages the
// samples available so far (matches python/motiondsp/filters.moving_average).
class MovingAverage {
public:
    explicit MovingAverage(std::size_t n);
    double process(double x);
    void reset();
    std::size_t length() const { return n_; }

private:
    std::size_t n_;
    std::vector<double> buf_;
    std::size_t head_ = 0;
    std::size_t count_ = 0;
    double sum_ = 0.0;
};

// One-pole low-pass: y[n] = y[n-1] + alpha * (x[n] - y[n-1]).
class Ema {
public:
    Ema(double fc, double fs);
    explicit Ema(double alpha) : alpha_(alpha) {}
    static double alpha_for(double fc, double fs);
    double process(double x);
    void reset(double y0) { y_ = y0; primed_ = true; }
    double alpha() const { return alpha_; }

private:
    double alpha_;
    double y_ = 0.0;
    bool primed_ = false;
};

struct Biquad {
    double b0, b1, b2, a1, a2;  // a0 normalised to 1
};

// Cascade of biquads in transposed direct form II.
class BiquadCascade {
public:
    BiquadCascade() = default;
    explicit BiquadCascade(std::vector<Biquad> sections);

    // Even-order Butterworth low-pass built from RBJ biquads.
    static BiquadCascade butterworth_lowpass(int order, double fc, double fs);

    double process(double x);
    void process_block(const double* in, double* out, std::size_t n);
    // Zero state.
    void reset();
    // State equal to the steady-state response to a constant input x0.
    void reset_steady(double x0);
    const std::vector<Biquad>& sections() const { return sec_; }

private:
    std::vector<Biquad> sec_;
    std::vector<double> z1_, z2_;
};

// Hampel spike removal on a block (centered window of length n, edge
// replication). Samples deviating more than k * 1.4826 * MAD from the local
// median are replaced by the median. n must be odd and <= kMaxMedianLen.
// `scratch` must hold 2 * len doubles; the overload without it allocates.
constexpr std::size_t kMaxMedianLen = 31;
void hampel(const double* in, double* out, std::size_t len, std::size_t n, double k,
            double* scratch);
void hampel(const double* in, double* out, std::size_t len, std::size_t n, double k);

// Running median (n odd), edge-replicated, centered.
void median_filter(const double* in, double* out, std::size_t len, std::size_t n);

}  // namespace motiondsp
