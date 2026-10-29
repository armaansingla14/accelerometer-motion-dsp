"""Exchange windows with the C++ tools (cpp/build/extract_features, bench_latency)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
BUILD = REPO / "cpp" / "build"


def write_windows(path: Path, windows: np.ndarray, biases: np.ndarray) -> None:
    """Write windows (n, window, ch) and biases (n, ch) in the C++ text format."""
    n, wlen, ch = windows.shape
    with open(path, "w") as fh:
        fh.write(f"{n} {wlen} {ch}\n")
        for w, b in zip(windows, biases):
            fh.write(" ".join(f"{v:.17g}" for v in b) + "\n")
            np.savetxt(fh, w, fmt="%.17g")


def tool(name: str) -> Path | None:
    p = BUILD / name
    return p if p.exists() else None


def run_extract_features(windows_txt: Path, out_csv: Path, cutoffs=None) -> np.ndarray:
    exe = tool("extract_features")
    if exe is None:
        raise FileNotFoundError("build the C++ library first (scripts/build_cpp.sh)")
    args = [str(exe), str(windows_txt), str(out_csv)]
    if cutoffs is not None:
        args += [f"{c:.17g}" for c in cutoffs]
    subprocess.run(args, check=True)
    return np.loadtxt(out_csv, delimiter=",", ndmin=2)
