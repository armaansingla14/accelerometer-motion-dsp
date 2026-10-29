"""DFT / FFT implementations and spectral features.

``dft`` is the O(N^2) definition, ``fft_radix2`` an iterative in-place
Cooley-Tukey transform (the same algorithm as ``cpp/src/fft.cpp``). Both are
tested against ``numpy.fft``.
"""

from __future__ import annotations

import numpy as np


def dft(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=complex)
    n = x.shape[0]
    k = np.arange(n)
    w = np.exp(-2j * np.pi * np.outer(k, k) / n)
    return w @ x


def _bit_reverse_indices(n: int) -> np.ndarray:
    bits = n.bit_length() - 1
    idx = np.arange(n)
    rev = np.zeros(n, dtype=np.int64)
    for b in range(bits):
        rev |= ((idx >> b) & 1) << (bits - 1 - b)
    return rev


def fft_radix2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=complex)
    n = x.shape[0]
    if n & (n - 1):
        raise ValueError("length must be a power of two")
    a = x[_bit_reverse_indices(n)].copy()
    size = 2
    while size <= n:
        half = size // 2
        tw = np.exp(-2j * np.pi * np.arange(half) / size)
        a = a.reshape(-1, size)
        top = a[:, :half].copy()
        bot = a[:, half:] * tw
        a[:, :half] = top + bot
        a[:, half:] = top - bot
        a = a.reshape(-1)
        size *= 2
    return a


def hann(n: int) -> np.ndarray:
    """Periodic Hann window (matches the C++ implementation)."""
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)


def power_spectrum(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """One-sided power spectrum of a Hann-windowed, mean-removed frame."""
    n = x.shape[0]
    w = hann(n)
    xf = fft_radix2((x - x.mean()) * w)
    p = np.abs(xf[: n // 2 + 1]) ** 2 / np.sum(w ** 2)
    p[1:-1] *= 2
    f = np.arange(n // 2 + 1) * fs / n
    return f, p


DEFAULT_BANDS = ((0.0, 3.0), (3.0, 8.0), (8.0, 20.0), (20.0, 50.01))


def band_energies(f: np.ndarray, p: np.ndarray, bands=DEFAULT_BANDS) -> np.ndarray:
    return np.array([p[(f >= lo) & (f < hi)].sum() for lo, hi in bands])


def dominant_frequency(f: np.ndarray, p: np.ndarray) -> float:
    return float(f[1:][np.argmax(p[1:])])


def spectral_centroid(f: np.ndarray, p: np.ndarray) -> float:
    s = p.sum()
    return float((f * p).sum() / s) if s > 0 else 0.0
