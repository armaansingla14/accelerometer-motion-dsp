# Experiment summary

Synthetic data: 120 sessions, seed 0, fs = 100 Hz.

## Dataset

- time steps: 636686 (106.1 min of robot motion)
- channel samples (x6 channels): 3820116
- sessions: 60 train / 24 filter-tuning / 36 test

## Noise reduction on test sessions (RMSE vs clean signal)

Per-channel parameters selected on the tuning sessions.

| method | ax | ay | az | gz | wl | wr | mean |
|---|---|---|---|---|---|---|---|
| calib | 6.4% | 8.4% | 8.5% | 5.5% | 0.0% | 0.0% | 4.8% |
| hampel | 15.0% | 18.1% | 13.7% | 15.7% | 5.5% | 5.9% | 12.3% |
| ma | 41.3% | 74.7% | 23.1% | 52.2% | 34.3% | 33.8% | 43.3% |
| ema | 48.7% | 75.4% | 33.4% | 53.5% | 39.7% | 39.4% | 48.3% |
| hampel+ema | 47.6% | 75.2% | 31.8% | 54.3% | 36.5% | 37.2% | 47.1% |
| butter | 22.1% | 74.3% | 13.8% | 41.3% | 17.4% | 17.0% | 31.0% |
| hampel+butter | 26.5% | 73.7% | 13.7% | 42.5% | 18.5% | 18.2% | 32.2% |
| zerophase | 53.5% | 81.1% | 45.5% | 74.4% | 52.7% | 52.3% | 59.9% |

Raw RMSE per channel: ax=0.3882, ay=0.3956, az=0.3967, gz=0.0659, wl=0.0203, wr=0.0202

## Classification (windows of 128 samples = 1.28 s, hop 64)

Windows: 8202 (5745 train, 2457 test, split by session). Python feature extraction took 40.4 s.

| model | features | n features | accuracy | macro F1 |
|---|---|---|---|---|
| rf | dsp | 53 | 92.8% | 0.920 |
| rf | baseline_raw_stats | 12 | 91.4% | 0.904 |
| logreg | dsp | 53 | 80.8% | 0.799 |
| logreg | baseline_raw_stats | 12 | 75.8% | 0.749 |

Confusion matrix (random forest, DSP features), rows = true class:

| | idle | straight | turn | bump | slip |
|---|---|---|---|---|---|
| idle | 592 | 1 | 0 | 0 | 0 |
| straight | 1 | 368 | 3 | 22 | 1 |
| turn | 0 | 0 | 524 | 0 | 0 |
| bump | 1 | 143 | 1 | 340 | 2 |
| slip | 0 | 0 | 2 | 0 | 456 |

## Latency

- Butterworth group delay at 1 Hz for the feature cut-offs [10.0, 10.0, 20.0, 10.0, 15.0, 15.0] Hz: 40.4 ms, 40.4 ms, 18.0 ms, 40.4 ms, 25.7 ms, 25.7 ms
- Hampel look-ahead: 3 samples = 30 ms
- Decision interval (hop): 640 ms; window span 1280 ms

## C++ WindowProcessor (host CPU, single thread, -O3 Release)

- max |C++ - Python| over 2457 test-window feature vectors: 3.46e-13
- timed calls: 49140
- per-window latency: mean 179.1 us, p50 169.8 us, p99 302.0 us, max 2848.6 us
- throughput: 4.29e+06 channel-samples/s

Total experiment runtime: 69 s.
