"""Calibration, denoising, windowing and feature extraction.

Two processing modes exist:

* **stream** - whole-session causal filtering, used to score noise reduction
  against the clean ground truth (``denoise_stream``).
* **window** - each analysis window is processed independently, exactly as
  the C++ ``WindowProcessor`` does on the robot (``window_features``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from . import filters, spectral
from .synth import CHANNELS, G, Session

EXPECTED_AT_REST = np.array([0.0, 0.0, G, 0.0])  # ax, ay, az, gz


@dataclass(frozen=True)
class PipelineConfig:
    fs: float = 100.0
    window: int = 128
    hop: int = 64
    lp_order: int = 4
    # Per-channel low-pass cut-offs [Hz] for (ax, ay, az, gz, wl, wr). The
    # defaults are overwritten by the validation sweep in run_experiments.py.
    lp_cutoffs: tuple = (10.0, 10.0, 20.0, 10.0, 15.0, 15.0)
    hampel_len: int = 7
    hampel_k: float = 3.0
    purity: float = 0.9
    track_width: float = 0.30


def estimate_bias(raw: np.ndarray, calib_samples: int) -> np.ndarray:
    """Per-channel bias from the leading stationary period (IMU channels only)."""
    bias = np.zeros(raw.shape[1])
    bias[:4] = raw[:calib_samples, :4].mean(axis=0) - EXPECTED_AT_REST
    return bias


def _per_channel(value, n: int) -> list:
    if np.isscalar(value):
        return [value] * n
    value = list(value)
    if len(value) != n:
        raise ValueError("expected one parameter per channel")
    return value


METHODS = ("calib", "hampel", "ma", "ema", "butter", "hampel+ma", "hampel+ema",
           "hampel+butter", "zerophase")


def denoise_stream(raw: np.ndarray, bias: np.ndarray, cfg: PipelineConfig,
                   method: str = "hampel+butter", param=None) -> np.ndarray:
    """Denoise a whole recording with the selected method.

    Methods are causal except ``zerophase`` (forward-backward Butterworth),
    which needs the whole recording and serves as an offline reference.
    A ``hampel+`` prefix runs the Hampel spike removal (``hampel_len // 2``
    samples of look-ahead) before the smoothing stage.

    ``param`` is the per-channel (or scalar) tuning knob: window length for
    ``ma``, cut-off in Hz for the IIR filters. ``None`` uses ``cfg.lp_cutoffs``.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}")
    x = raw - bias
    nch = x.shape[1]
    if method.startswith("hampel"):
        x = filters.hampel(x, cfg.hampel_len, cfg.hampel_k)
        method = method.partition("+")[2]
    if method in ("calib", ""):
        return x
    out = np.empty_like(x)
    if method == "ma":
        for c, n in enumerate(_per_channel(5 if param is None else param, nch)):
            out[:, c] = filters.moving_average(x[:, c], int(n))
        return out
    cut = _per_channel(cfg.lp_cutoffs if param is None else param, nch)
    for c, fc in enumerate(cut):
        if method == "ema":
            out[:, c] = filters.ema(x[:, c], filters.ema_alpha(fc, cfg.fs))
            continue
        sos = filters.butter_lowpass_sos(cfg.lp_order, fc, cfg.fs)
        if method == "butter":
            out[:, c] = filters.sosfilt_steady(sos, x[:, c])
        else:  # zerophase
            out[:, c] = signal.sosfiltfilt(sos, x[:, c])
    return out


def rmse_per_channel(est: np.ndarray, clean: np.ndarray, skip: int = 0) -> np.ndarray:
    d = est[skip:] - clean[skip:]
    return np.sqrt(np.mean(d ** 2, axis=0))


def iter_windows(n: int, window: int, hop: int):
    for start in range(0, n - window + 1, hop):
        yield start


def window_label(labels: np.ndarray, purity: float) -> int:
    counts = np.bincount(labels)
    k = int(np.argmax(counts))
    return k if counts[k] / labels.size >= purity else -1


N_BANDS = len(spectral.DEFAULT_BANDS)
PER_CHANNEL = ("mean", "std", "ptp") + tuple(f"logband{i}" for i in range(N_BANDS)) + ("domfreq",)
CROSS = ("wdiff_mean", "wbar_mean", "accel_mismatch", "yaw_mismatch", "lat_mismatch")
FEATURE_NAMES = tuple(f"{c}_{f}" for c in CHANNELS for f in PER_CHANNEL) + CROSS
BASELINE_NAMES = tuple(f"{c}_{f}" for c in CHANNELS for f in ("mean", "std"))


def preprocess_window(raw_win: np.ndarray, bias: np.ndarray, cfg: PipelineConfig):
    """Return (despiked, filtered) versions of one window, processed standalone."""
    x = raw_win - bias
    d = filters.hampel(x, cfg.hampel_len, cfg.hampel_k)
    y = np.empty_like(d)
    for c, fc in enumerate(cfg.lp_cutoffs):
        sos = filters.butter_lowpass_sos(cfg.lp_order, fc, cfg.fs)
        y[:, c] = filters.sosfilt_steady(sos, d[:, c])
    return d, y


def window_features(raw_win: np.ndarray, bias: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    d, y = preprocess_window(raw_win, bias, cfg)
    feats = []
    for c in range(y.shape[1]):
        yc = y[:, c]
        f, p = spectral.power_spectrum(d[:, c], cfg.fs)
        bands = spectral.band_energies(f, p)
        feats.extend([yc.mean(), yc.std(), yc.max() - yc.min()])
        feats.extend(np.log10(bands + 1e-9))
        feats.append(spectral.dominant_frequency(f, p))
    ax, ay, gz, wl, wr = y[:, 0], y[:, 1], y[:, 3], y[:, 4], y[:, 5]
    wbar = 0.5 * (wl + wr)
    duration = (y.shape[0] - 1) / cfg.fs
    wheel_accel = (wbar[-1] - wbar[0]) / duration
    feats.append(np.mean(wr - wl))
    feats.append(np.mean(wbar))
    feats.append(np.mean(ax) - wheel_accel)
    feats.append(np.mean(gz - (wr - wl) / cfg.track_width))
    feats.append(np.mean(ay - wbar * gz))
    return np.asarray(feats)


def baseline_features(raw_win: np.ndarray) -> np.ndarray:
    """No-DSP baseline: raw per-channel mean and standard deviation."""
    return np.concatenate([raw_win.mean(axis=0), raw_win.std(axis=0)])


def build_feature_table(sessions: list[Session], cfg: PipelineConfig):
    """Window every session. Returns X_dsp, X_base, y, groups, raw windows, biases."""
    X, Xb, y, groups, wins, biases = [], [], [], [], [], []
    for si, s in enumerate(sessions):
        bias = estimate_bias(s.noisy, s.calib_samples)
        for start in iter_windows(s.n, cfg.window, cfg.hop):
            lab = window_label(s.labels[start:start + cfg.window], cfg.purity)
            if lab < 0:
                continue
            w = s.noisy[start:start + cfg.window]
            X.append(window_features(w, bias, cfg))
            Xb.append(baseline_features(w))
            y.append(lab)
            groups.append(si)
            wins.append(w)
            biases.append(bias)
    return (np.array(X), np.array(Xb), np.array(y), np.array(groups),
            np.array(wins), np.array(biases))
