#!/usr/bin/env python
"""End-to-end experiment: dataset -> denoising -> classification -> C++ timing.

Writes results/*.csv and results/summary.md. All numbers come from synthetic
data produced by motiondsp.synth with the seed given on the command line.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from motiondsp import classify, cpp_bridge, filters, pipeline as P, synth  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"

GRIDS = {
    "calib": [None],
    "hampel": [None],
    "ma": [3, 5, 7, 9, 15],
    "ema": [2, 3, 5, 8, 12, 16, 20, 30],
    "hampel+ema": [2, 3, 5, 8, 12, 16, 20, 30],
    "butter": [2, 3, 5, 8, 12, 16, 20, 30],
    "hampel+butter": [2, 3, 5, 8, 12, 16, 20, 30],
    "zerophase": [2, 3, 5, 8, 12, 16, 20, 30],
}


def split_sessions(n: int, seed: int):
    idx = np.random.default_rng(seed).permutation(n)
    n_tune, n_test = int(0.2 * n), int(0.3 * n)
    return idx[:n_tune], idx[n_tune:n_tune + n_test], idx[n_tune + n_test:]


def sq_errors(sessions, method, param, cfg):
    """Sum of squared errors per channel (filtered and raw) and sample count."""
    err = np.zeros(6)
    raw = np.zeros(6)
    n = 0
    for s in sessions:
        b = P.estimate_bias(s.noisy, s.calib_samples)
        est = P.denoise_stream(s.noisy, b, cfg, method, param)
        err += np.sum((est - s.clean) ** 2, axis=0)
        raw += np.sum((s.noisy - s.clean) ** 2, axis=0)
        n += s.n
    return err, raw, n


def noise_experiment(tune, test, cfg):
    rows = []
    for method, grid in GRIDS.items():
        # choose the per-channel parameter on the tuning sessions only
        tune_err = {p: sq_errors(tune, method, p, cfg)[0] for p in grid}
        best = [min(grid, key=lambda p: tune_err[p][c] if p is not None else 0) for c in range(6)]
        param = None if best[0] is None else best
        err, raw, n = sq_errors(test, method, param, cfg)
        rmse_f, rmse_r = np.sqrt(err / n), np.sqrt(raw / n)
        for c, ch in enumerate(synth.CHANNELS):
            rows.append(dict(method=method, channel=ch,
                             param="" if param is None else param[c],
                             rmse_raw=rmse_r[c], rmse_filtered=rmse_f[c],
                             reduction_pct=100 * (1 - rmse_f[c] / rmse_r[c])))
    return rows


def write_csv(path: Path, rows: list[dict]):
    keys = list(rows[0].keys())
    with open(path, "w") as fh:
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(f"{r[k]:.6g}" if isinstance(r[k], (float, np.floating)) else str(r[k])
                              for k in keys) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bench-reps", type=int, default=20)
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    cfg = P.PipelineConfig()
    md = ["# Experiment summary", "",
          f"Synthetic data: {args.sessions} sessions, seed {args.seed}, fs = {cfg.fs:.0f} Hz.", ""]

    t0 = time.time()
    sessions = synth.generate_dataset(args.sessions, seed=args.seed)
    n_samples = sum(s.n for s in sessions)
    tune_i, test_i, train_i = split_sessions(len(sessions), args.seed)
    tune = [sessions[i] for i in tune_i]
    test = [sessions[i] for i in test_i]
    md += ["## Dataset", "",
           f"- time steps: {n_samples} ({n_samples / cfg.fs / 60:.1f} min of robot motion)",
           f"- channel samples (x6 channels): {6 * n_samples}",
           f"- sessions: {len(train_i)} train / {len(tune_i)} filter-tuning / {len(test_i)} test", ""]

    # ---- noise reduction ------------------------------------------------
    rows = noise_experiment(tune, test, cfg)
    write_csv(RESULTS / "noise_reduction.csv", rows)
    md += ["## Noise reduction on test sessions (RMSE vs clean signal)", "",
           "Per-channel parameters selected on the tuning sessions.", "",
           "| method | " + " | ".join(synth.CHANNELS) + " | mean |",
           "|---" * 8 + "|"]
    for method in GRIDS:
        r = [x for x in rows if x["method"] == method]
        red = [x["reduction_pct"] for x in r]
        md.append(f"| {method} | " + " | ".join(f"{v:.1f}%" for v in red) + f" | {np.mean(red):.1f}% |")
    md.append("")
    r = [x for x in rows if x["method"] == "calib"]
    md.append("Raw RMSE per channel: " + ", ".join(f"{x['channel']}={x['rmse_raw']:.4f}" for x in r))
    md.append("")

    # ---- classification ---------------------------------------------------
    t1 = time.time()
    X, Xb, y, g, wins, biases = P.build_feature_table(sessions, cfg)
    feat_time = time.time() - t1
    test_set = set(test_i.tolist())
    te = np.array([gi in test_set for gi in g])
    tr = ~te
    cls_rows = []
    best_conf = None
    for model in ("rf", "logreg"):
        for name, F in (("dsp", X), ("baseline_raw_stats", Xb)):
            res = classify.evaluate(classify.make_model(model, args.seed), F[tr], y[tr], F[te], y[te],
                                    len(synth.CLASSES))
            cls_rows.append(dict(model=model, features=name, n_features=F.shape[1],
                                 train_windows=int(tr.sum()), test_windows=int(te.sum()),
                                 accuracy=res.accuracy, macro_f1=res.macro_f1))
            if model == "rf" and name == "dsp":
                best_conf = res.confusion
    write_csv(RESULTS / "classification.csv", cls_rows)
    with open(RESULTS / "confusion_rf_dsp.csv", "w") as fh:
        fh.write("true\\pred," + ",".join(synth.CLASSES) + "\n")
        for k, row in enumerate(best_conf):
            fh.write(synth.CLASSES[k] + "," + ",".join(str(v) for v in row) + "\n")
    md += ["## Classification (windows of "
           f"{cfg.window} samples = {cfg.window / cfg.fs:.2f} s, hop {cfg.hop})", "",
           f"Windows: {len(y)} ({int(tr.sum())} train, {int(te.sum())} test, split by session). "
           f"Python feature extraction took {feat_time:.1f} s.", "",
           "| model | features | n features | accuracy | macro F1 |", "|---|---|---|---|---|"]
    for c in cls_rows:
        md.append(f"| {c['model']} | {c['features']} | {c['n_features']} | "
                  f"{100 * c['accuracy']:.1f}% | {c['macro_f1']:.3f} |")
    md += ["", "Confusion matrix (random forest, DSP features), rows = true class:", "",
           "| | " + " | ".join(synth.CLASSES) + " |", "|---" * 6 + "|"]
    for k, row in enumerate(best_conf):
        md.append(f"| {synth.CLASSES[k]} | " + " | ".join(str(v) for v in row) + " |")
    md.append("")

    # ---- latency budget -----------------------------------------------------
    gd = [filters.group_delay_samples(filters.butter_lowpass_sos(cfg.lp_order, fc, cfg.fs), 1.0, cfg.fs)
          for fc in cfg.lp_cutoffs]
    hampel_lookahead = cfg.hampel_len // 2
    md += ["## Latency", "",
           f"- Butterworth group delay at 1 Hz for the feature cut-offs {list(cfg.lp_cutoffs)} Hz: "
           + ", ".join(f"{1000 * d / cfg.fs:.1f} ms" for d in gd),
           f"- Hampel look-ahead: {hampel_lookahead} samples = {1000 * hampel_lookahead / cfg.fs:.0f} ms",
           f"- Decision interval (hop): {1000 * cfg.hop / cfg.fs:.0f} ms; window span "
           f"{1000 * cfg.window / cfg.fs:.0f} ms", ""]

    # ---- C++ parity and timing -----------------------------------------------
    if cpp_bridge.tool("extract_features") and cpp_bridge.tool("bench_latency"):
        with tempfile.TemporaryDirectory() as td:
            txt = Path(td) / "windows.txt"
            sel = np.flatnonzero(te)
            cpp_bridge.write_windows(txt, wins[sel], biases[sel])
            Xc = cpp_bridge.run_extract_features(txt, Path(td) / "f.csv", cfg.lp_cutoffs)
            max_diff = float(np.max(np.abs(Xc - X[sel])))
            import subprocess
            subprocess.run([str(cpp_bridge.tool("bench_latency")), str(txt),
                            str(RESULTS / "cpp_latency.csv"), str(args.bench_reps)], check=True)
        lat = np.genfromtxt(RESULTS / "cpp_latency.csv", delimiter=",", names=True)
        md += ["## C++ WindowProcessor (host CPU, single thread, -O3 Release)", "",
               f"- max |C++ - Python| over {len(sel)} test-window feature vectors: {max_diff:.2e}",
               f"- timed calls: {int(lat['calls'])}",
               f"- per-window latency: mean {lat['mean_us']:.1f} us, p50 {lat['p50_us']:.1f} us, "
               f"p99 {lat['p99_us']:.1f} us, max {lat['max_us']:.1f} us",
               f"- throughput: {lat['channel_samples_per_s']:.3g} channel-samples/s", ""]
    else:
        md += ["C++ tools not built; run scripts/build_cpp.sh to add parity and timing results.", ""]

    md.append(f"Total experiment runtime: {time.time() - t0:.0f} s.")
    (RESULTS / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
