# accelerometer-motion-dsp

Signal processing and motion classification for a small differential-drive
robot using a 3-axis accelerometer, a yaw-rate gyro and two wheel encoders.
The project has two halves:

- **Python (NumPy / SciPy / scikit-learn)**: a synthetic dataset generator,
  reference filters, DFT/FFT, feature extraction, classifiers and the
  experiment harness.
- **C++17 library (`cpp/`)**: the on-robot part. Streaming filters, an FFT
  plan, and a `WindowProcessor` that turns one window of raw samples into the
  same 53 features as the Python reference (checked to ~1e-13).

All data in this repository is synthetic. No physical robot or IMU was used.

## Features

- Synthetic multi-class motion recordings: `idle`, `straight`, `turn`, `bump`,
  `slip`. Kinematics come from commanded velocities through an actuator lag;
  sensor channels are derived physically (longitudinal accel = dv/dt,
  centripetal accel = v*omega, wheel speeds from track width, speed-dependent
  drivetrain vibration, damped chassis transients for bumps, wheel spin and
  fishtailing for slip). A clean and a noisy copy of every signal are kept so
  filters can be scored against ground truth.
- Sensor error model: white noise, per-session constant bias, sparse spikes,
  encoder quantisation and timing jitter.
- Bias calibration from the stationary period at the start of each session.
- Filters: causal moving average, one-pole EMA, even-order Butterworth
  low-pass designed as RBJ biquads (validated against `scipy.signal.butter`),
  steady-state filter initialisation, Hampel spike removal, running median,
  and a zero-phase (forward-backward) offline reference.
- O(N^2) DFT and an iterative radix-2 FFT (Python and C++), Hann-windowed
  one-sided power spectrum, band energies, dominant frequency.
- 53 window features: per-channel mean / std / peak-to-peak on the filtered
  signal, four log band energies and the dominant frequency on the despiked
  signal, plus five cross-sensor consistency features (e.g. gyro yaw rate vs
  encoder-derived yaw rate, accelerometer vs wheel acceleration).
- Random forest and logistic-regression classifiers, compared with a no-DSP
  baseline (raw per-channel mean and std), using a session-level train/test
  split.
- C++ `WindowProcessor` that preallocates all buffers, a feature-extraction CLI
  used for the parity check, and a latency benchmark.

## Layout

```
python/motiondsp/
  synth.py        synthetic sessions (clean + noisy) with per-sample labels
  filters.py      moving average, EMA, Butterworth biquads, Hampel, median
  spectral.py     DFT, radix-2 FFT, power spectrum, band features
  pipeline.py     calibration, stream denoising, windowing, features
  classify.py     models, session split, evaluation
  cpp_bridge.py   window file format and C++ tool wrappers
python/tests/     pytest suite (incl. C++/Python feature parity)
cpp/
  include/motiondsp/  filters.hpp fft.hpp features.hpp window_io.hpp
  src/                implementations
  tests/              ctest suite (small built-in test framework)
  tools/extract_features.cpp
  bench/bench_latency.cpp
scripts/
  build_cpp.sh        build + ctest inside the devtools Docker image
  run_experiments.py  full experiment, writes results/
results/              CSV outputs and summary.md from the last run
```

## How to run

```bash
# C++ library, tools and tests (CMake + Ninja in Docker)
./scripts/build_cpp.sh

# Python tests (the parity test is skipped if the C++ tools are not built)
python -m pytest -q

# Full experiment (about 70 s on a desktop-class CPU)
python scripts/run_experiments.py --sessions 120 --seed 0
```

The Python code needs `numpy`, `scipy` and `scikit-learn`.

## Results

Produced by `python scripts/run_experiments.py --sessions 120 --seed 0`
(full output in `results/summary.md`). Synthetic data only.

**Dataset.** 120 sessions, 636,686 time steps at 100 Hz (106 minutes of
simulated motion), i.e. 3,820,116 channel samples. Sessions are split 60
train / 24 filter-tuning / 36 test.

**Noise reduction.** RMSE between the processed signal and the clean ground
truth on the 36 test sessions; the cut-off or window length of each method is
chosen per channel on the tuning sessions. Reduction is `1 - RMSE_filtered /
RMSE_raw`.

| method | ax | ay | az | gz | wl | wr | mean |
|---|---|---|---|---|---|---|---|
| bias calibration only | 6.4% | 8.4% | 8.5% | 5.5% | 0.0% | 0.0% | 4.8% |
| Hampel only | 15.0% | 18.1% | 13.7% | 15.7% | 5.5% | 5.9% | 12.3% |
| moving average | 41.3% | 74.7% | 23.1% | 52.2% | 34.3% | 33.8% | 43.3% |
| EMA (one-pole) | 48.7% | 75.4% | 33.4% | 53.5% | 39.7% | 39.4% | 48.3% |
| Hampel + EMA | 47.6% | 75.2% | 31.8% | 54.3% | 36.5% | 37.2% | 47.1% |
| Butterworth (4th order) | 22.1% | 74.3% | 13.8% | 41.3% | 17.4% | 17.0% | 31.0% |
| Hampel + Butterworth | 26.5% | 73.7% | 13.7% | 42.5% | 18.5% | 18.2% | 32.2% |
| zero-phase Butterworth (offline, non-causal) | 53.5% | 81.1% | 45.5% | 74.4% | 52.7% | 52.3% | 59.9% |

Observations: among causal filters the one-pole EMA gives the lowest error
(48.3% mean RMSE reduction). The 4th-order Butterworth rejects noise better
but its larger group delay lags fast transitions, and that lag dominates the
error; the non-causal zero-phase version shows what is lost to delay. On these
signals, with spikes at 0.2% probability, the Hampel stage on its own removes
12% of the error, but once a low-pass follows it the gain is marginal
(+1.2 points before the Butterworth, -1.2 points before the EMA).

**Classification.** 1.28 s windows (128 samples, 50% overlap) with at least 90%
of samples from one class; 5,745 train and 2,457 test windows.

| model | features | accuracy | macro F1 |
|---|---|---|---|
| random forest | 53 DSP features | 92.8% | 0.920 |
| random forest | 12 raw mean/std (baseline) | 91.4% | 0.904 |
| logistic regression | 53 DSP features | 80.8% | 0.799 |
| logistic regression | 12 raw mean/std (baseline) | 75.8% | 0.749 |

The DSP features add 1.4 points over the raw-statistics baseline for the
random forest and 5.0 points for logistic regression. Most remaining errors
are `bump` windows predicted as `straight` (143 of 487 test bump windows):
bumps are sparse transients, so many windows labelled `bump` contain no bump.

**C++ latency.** `bench_latency` on 2,457 test windows x 20 repetitions
(49,140 calls), single thread, Release build, run on the development
machine's x86 CPU under WSL2:

| metric | value |
|---|---|
| mean per-window latency (6 channels: bias, Hampel, Butterworth, FFT, 53 features) | 179 us |
| p50 / p99 / max | 170 / 302 / 2849 us |
| max abs difference vs Python features | 3.5e-13 |

The processing time is small compared with the other terms of the response
time. Group delay at 1 Hz of the feature low-pass filters is 18-40 ms, the
Hampel look-ahead is 30 ms, and a new decision is produced every 640 ms (hop)
from 1.28 s of data. The window length, not the computation, sets how fast
the classifier reacts to a change of motion.

## Limitations

- Everything is simulated. The sensor and vehicle models are simple and the
  noise levels were chosen by hand; real IMUs add effects not modelled here
  (temperature drift, cross-axis sensitivity, mounting resonance, clipping).
- Labels are per segment, so short events (bumps) produce partly mislabelled
  windows.
- The latency figure is for an x86 desktop CPU; a microcontroller target was
  not measured.
- The C++ side computes features only; the classifier runs in Python.

## License

MIT, see `LICENSE`.
