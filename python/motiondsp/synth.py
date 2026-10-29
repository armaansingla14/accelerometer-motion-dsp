"""Synthetic multi-class motion dataset for a differential-drive robot.

Each *session* is a continuous recording made of consecutive motion segments.
The robot body follows commanded linear/angular velocities through a
first-order actuator lag, and sensor channels are derived from the resulting
kinematics:

    ax  longitudinal acceleration      [m/s^2]
    ay  lateral (centripetal) accel.   [m/s^2]
    az  vertical acceleration (incl. g)[m/s^2]
    gz  yaw rate                       [rad/s]
    wl  left wheel speed (encoder)     [m/s]
    wr  right wheel speed (encoder)    [m/s]

For every session two versions of the signals are produced: ``clean`` (true
physical signal, including real vibration and bump transients) and ``noisy``
(what the sensors report: white noise, constant bias, sparse spikes and
encoder quantisation). This lets filters be scored against ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

CLASSES = ("idle", "straight", "turn", "bump", "slip")
CHANNELS = ("ax", "ay", "az", "gz", "wl", "wr")
G = 9.81


@dataclass
class SensorNoise:
    accel_sigma: float = 0.35      # m/s^2 white noise (MEMS at ~50 Hz BW)
    gyro_sigma: float = 0.06       # rad/s
    accel_bias_sigma: float = 0.15 # per-session constant bias
    gyro_bias_sigma: float = 0.02
    spike_prob: float = 0.002      # probability of a spike per sample/channel
    spike_scale: float = 6.0       # spike amplitude in units of sigma
    encoder_res: float = 0.01      # wheel speed quantisation step [m/s]
    encoder_sigma: float = 0.02    # m/s jitter from tick timing


@dataclass
class SessionConfig:
    fs: float = 100.0
    seg_min_s: float = 3.0
    seg_max_s: float = 7.0
    n_segments: int = 10
    lead_idle_s: float = 3.0       # stationary period used for bias calibration
    track_width: float = 0.30      # wheel separation [m]
    actuator_tau: float = 0.25     # first-order lag of the drive [s]
    noise: SensorNoise = field(default_factory=SensorNoise)


@dataclass
class Session:
    fs: float
    clean: np.ndarray   # (n, 6)
    noisy: np.ndarray   # (n, 6)
    labels: np.ndarray  # (n,) int class index
    calib_samples: int  # leading stationary samples

    @property
    def n(self) -> int:
        return self.labels.shape[0]


def _first_order(cmd: np.ndarray, tau: float, fs: float, x0: float = 0.0) -> np.ndarray:
    a = np.exp(-1.0 / (tau * fs))
    out = np.empty_like(cmd)
    x = x0
    for i, c in enumerate(cmd):
        x = a * x + (1.0 - a) * c
        out[i] = x
    return out


def _bump_transient(n: int, fs: float, rng: np.random.Generator) -> np.ndarray:
    """Sum of a few damped oscillations (chassis hitting small obstacles)."""
    out = np.zeros(n)
    t = np.arange(n) / fs
    n_bumps = rng.integers(2, 6)
    for _ in range(n_bumps):
        t0 = rng.uniform(0.2, max(0.3, n / fs - 0.5))
        f0 = rng.uniform(8.0, 16.0)
        zeta = rng.uniform(0.08, 0.2)
        amp = rng.uniform(2.0, 6.0)
        tt = t - t0
        mask = tt >= 0
        out[mask] += amp * np.exp(-zeta * 2 * np.pi * f0 * tt[mask]) * np.sin(2 * np.pi * f0 * tt[mask])
    return out


def generate_session(cfg: SessionConfig, rng: np.random.Generator) -> Session:
    fs = cfg.fs
    lens, kinds = [], []
    lens.append(int(cfg.lead_idle_s * fs))
    kinds.append(0)
    for _ in range(cfg.n_segments):
        lens.append(int(rng.uniform(cfg.seg_min_s, cfg.seg_max_s) * fs))
        kinds.append(int(rng.integers(0, len(CLASSES))))
    n = int(sum(lens))

    v_cmd = np.zeros(n)
    w_cmd = np.zeros(n)
    slip_ratio = np.zeros(n)
    bump = np.zeros(n)
    pitch_ax = np.zeros(n)
    labels = np.zeros(n, dtype=np.int64)

    i = 0
    for L, k in zip(lens, kinds):
        sl = slice(i, i + L)
        labels[sl] = k
        name = CLASSES[k]
        if name == "straight":
            v_cmd[sl] = rng.uniform(0.3, 1.2)
            w_cmd[sl] = rng.normal(0, 0.03)
        elif name == "turn":
            v_cmd[sl] = rng.uniform(0.2, 0.8)
            w_cmd[sl] = rng.choice([-1, 1]) * rng.uniform(0.6, 1.6)
        elif name == "bump":
            v_cmd[sl] = rng.uniform(0.3, 1.0)
            w_cmd[sl] = rng.normal(0, 0.05)
            b = _bump_transient(L, fs, rng)
            bump[sl] = b
            pitch_ax[sl] = 0.35 * b * rng.uniform(0.5, 1.0)
        elif name == "slip":
            v_cmd[sl] = rng.uniform(0.3, 1.0)
            w_cmd[sl] = 0.0
            t = np.arange(L) / fs
            base = rng.uniform(0.4, 1.0)
            wobble = 0.3 * np.sin(2 * np.pi * rng.uniform(0.5, 2.0) * t + rng.uniform(0, 2 * np.pi))
            slip_ratio[sl] = np.clip(base + wobble, 0.1, None)
        i += L

    v = _first_order(v_cmd, cfg.actuator_tau, fs)
    w = _first_order(w_cmd, cfg.actuator_tau, fs)
    # During slip the body barely follows the command; add fishtailing yaw.
    slip_mask = slip_ratio > 0
    if slip_mask.any():
        v = np.where(slip_mask, v / (1.0 + slip_ratio), v)
        fish = np.zeros(n)
        idx = np.flatnonzero(slip_mask)
        fish[idx] = 0.25 * np.sin(2 * np.pi * 1.3 * idx / fs) * slip_ratio[idx]
        w = w + fish
    v = _first_order(v, 0.05, fs)  # smooth the slip on/off steps

    ax = np.gradient(v) * fs + pitch_ax
    ay = v * w
    v_wheel_mean = v * (1.0 + slip_ratio)
    wl = v_wheel_mean - w * cfg.track_width / 2
    wr = v_wheel_mean + w * cfg.track_width / 2

    # Drivetrain vibration: frequency proportional to wheel speed.
    t = np.arange(n) / fs
    wheel_speed = np.abs(v_wheel_mean)
    vib_f = 14.0 * wheel_speed + 2.0
    phase = 2 * np.pi * np.cumsum(vib_f) / fs
    vib_amp = 0.25 * np.clip(wheel_speed / 0.5, 0, 2)
    vib = vib_amp * np.sin(phase)
    az = G + bump + vib
    ax = ax + 0.4 * vib
    gz = w

    clean = np.stack([ax, ay, az, gz, wl, wr], axis=1)

    nz = cfg.noise
    noisy = clean.copy()
    sig = np.array([nz.accel_sigma] * 3 + [nz.gyro_sigma])
    noisy[:, :4] += rng.normal(0, 1, (n, 4)) * sig
    bias = np.concatenate([rng.normal(0, nz.accel_bias_sigma, 3), rng.normal(0, nz.gyro_bias_sigma, 1)])
    noisy[:, :4] += bias
    spikes = rng.random((n, 4)) < nz.spike_prob
    noisy[:, :4] += spikes * rng.choice([-1, 1], (n, 4)) * nz.spike_scale * sig
    enc = noisy[:, 4:] + rng.normal(0, nz.encoder_sigma, (n, 2))
    noisy[:, 4:] = np.round(enc / nz.encoder_res) * nz.encoder_res

    return Session(fs=fs, clean=clean, noisy=noisy, labels=labels,
                   calib_samples=int(cfg.lead_idle_s * fs))


def generate_dataset(n_sessions: int, seed: int = 0, cfg: SessionConfig | None = None) -> list[Session]:
    cfg = cfg or SessionConfig()
    rng = np.random.default_rng(seed)
    return [generate_session(cfg, rng) for _ in range(n_sessions)]
