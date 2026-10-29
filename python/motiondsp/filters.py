"""Causal filters used by the motion pipeline.

The filter designs here mirror ``cpp/src/filters.cpp`` exactly so that the
Python reference and the C++ library produce the same output (checked in the
tests through the C++ feature tool).
"""

from __future__ import annotations

import numpy as np
from scipy import signal


def moving_average(x: np.ndarray, n: int) -> np.ndarray:
    """Causal boxcar average of length ``n`` (shorter at the start)."""
    if n < 1:
        raise ValueError("n must be >= 1")
    x = np.asarray(x, dtype=float)
    c = np.cumsum(np.insert(x, 0, 0.0, axis=0), axis=0)
    out = np.empty_like(x)
    k = min(n, x.shape[0])
    out[:k] = c[1:k + 1] / np.arange(1, k + 1).reshape((-1,) + (1,) * (x.ndim - 1))
    out[k:] = (c[k + 1:] - c[1:x.shape[0] - k + 1]) / n
    return out


def ema_alpha(fc: float, fs: float) -> float:
    """Smoothing factor of a one-pole low-pass with -3 dB cut-off ``fc``."""
    return 1.0 - np.exp(-2.0 * np.pi * fc / fs)


def ema(x: np.ndarray, alpha: float, y0: float | None = None) -> np.ndarray:
    """First-order IIR low-pass y[n] = y[n-1] + alpha (x[n] - y[n-1])."""
    x = np.asarray(x, dtype=float)
    b = [alpha]
    a = [1.0, -(1.0 - alpha)]
    zi = None
    if y0 is None:
        y0 = x[0]
    zi = np.atleast_1d((1.0 - alpha) * y0)
    y, _ = signal.lfilter(b, a, x, zi=zi)
    return y


def butter_lowpass_sos(order: int, fc: float, fs: float) -> np.ndarray:
    """Butterworth low-pass as cascaded RBJ biquads.

    For an even order N the analog Butterworth poles split into N/2 conjugate
    pairs with quality factors Q_k = 1 / (2 cos(theta_k)),
    theta_k = pi (2k + 1) / (2N). Each pair maps to one bilinear-transformed
    low-pass biquad with frequency pre-warping, which reproduces
    ``scipy.signal.butter`` to numerical precision.
    """
    if order < 2 or order % 2:
        raise ValueError("order must be an even number >= 2")
    if not 0 < fc < fs / 2:
        raise ValueError("cut-off must be in (0, fs/2)")
    w0 = 2 * np.pi * fc / fs
    cw, sw = np.cos(w0), np.sin(w0)
    sos = []
    for k in range(order // 2):
        theta = np.pi * (2 * k + 1) / (2 * order)
        q = 1.0 / (2.0 * np.cos(theta))
        alpha = sw / (2 * q)
        a0 = 1 + alpha
        b = np.array([(1 - cw) / 2, 1 - cw, (1 - cw) / 2]) / a0
        a = np.array([1.0, -2 * cw / a0, (1 - alpha) / a0])
        sos.append(np.concatenate([b, a]))
    return np.array(sos)


def sosfilt_df2t(sos: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Reference transposed direct-form II biquad cascade (pure Python loop).

    Slow; used to validate against scipy and to document the exact recursion
    implemented in C++.
    """
    y = np.asarray(x, dtype=float).copy()
    for b0, b1, b2, _, a1, a2 in sos:
        z1 = z2 = 0.0
        out = np.empty_like(y)
        for i, xi in enumerate(y):
            yi = b0 * xi + z1
            z1 = b1 * xi - a1 * yi + z2
            z2 = b2 * xi - a2 * yi
            out[i] = yi
        y = out
    return y


def sosfilt_steady(sos: np.ndarray, x: np.ndarray, axis: int = 0) -> np.ndarray:
    """Biquad cascade with state initialised to the steady state of x[0].

    Avoids the start-up transient that a zero initial state produces on
    signals with a large DC component (e.g. gravity on the z axis).
    """
    x = np.asarray(x, dtype=float)
    zi = signal.sosfilt_zi(sos)
    x0 = np.take(x, [0], axis=axis)
    shape = [sos.shape[0], 2] + [1] * (x.ndim - 1)
    zi = zi.reshape(shape)
    if x.ndim > 1:
        x0 = np.moveaxis(x0, axis, 0)  # (1, rest...)
        zi = zi * x0[np.newaxis]
        zi = zi.reshape([sos.shape[0], 2] + list(x0.shape[1:]))
        xm = np.moveaxis(x, axis, 0)
        y, _ = signal.sosfilt(sos, xm, axis=0, zi=zi)
        return np.moveaxis(y, 0, axis)
    y, _ = signal.sosfilt(sos, x, zi=zi * x0[0])
    return y


def median3(x: np.ndarray) -> np.ndarray:
    """Causal-friendly median-of-3 (one-sample look-ahead) for spike removal."""
    return median_filter(x, 3)


def median_filter(x: np.ndarray, n: int) -> np.ndarray:
    """Centered running median with edge replication (n odd)."""
    if n % 2 == 0:
        raise ValueError("median length must be odd")
    x = np.asarray(x, dtype=float)
    h = n // 2
    pad = [(h, h)] + [(0, 0)] * (x.ndim - 1)
    xp = np.pad(x, pad, mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(xp, n, axis=0)
    return np.median(win, axis=-1)


def hampel(x: np.ndarray, n: int = 7, k: float = 3.0) -> np.ndarray:
    """Hampel identifier: replace samples further than k*MAD from the local
    median by that median. Leaves non-outlier samples untouched, so it removes
    spikes without smoothing the rest of the signal."""
    x = np.asarray(x, dtype=float)
    med = median_filter(x, n)
    h = n // 2
    pad = [(h, h)] + [(0, 0)] * (x.ndim - 1)
    dev = np.abs(x - med)
    devp = np.pad(dev, pad, mode="edge")
    mad = np.median(np.lib.stride_tricks.sliding_window_view(devp, n, axis=0), axis=-1)
    thresh = k * 1.4826 * mad
    out = x.copy()
    mask = dev > thresh
    out[mask] = med[mask]
    return out


def group_delay_samples(sos: np.ndarray, f: float, fs: float) -> float:
    """Group delay of a biquad cascade at frequency f (in samples)."""
    w, gd = signal.group_delay(signal.sos2tf(sos), w=[f], fs=fs)
    return float(gd[0])
