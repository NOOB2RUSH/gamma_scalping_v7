"""Identify persistent regime shifts in rolling ATM-IV change correlations.

The input is the pair-level CSV emitted by ``etf_atm_iv_change_corr.py``.
For every eligible observation, the statistic compares the mean rolling
correlation over the preceding and following windows.  Candidates must retain
the direction of the move over the following persistence window; nearby
candidates are collapsed to the largest absolute shift.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "output" / "research"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--input-prefix", default=None, help="Correlation-output timestamp.")
    parser.add_argument("--comparison-window", type=int, default=60)
    parser.add_argument("--persistence-window", type=int, default=20)
    parser.add_argument("--min-absolute-shift", type=float, default=0.12)
    parser.add_argument("--min-separation", type=int, default=90)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_INPUT_DIR)
    return parser.parse_args()


def detect_shifts(series, comparison_window, persistence_window, min_shift, min_separation):
    values = series.to_numpy()
    candidates = []
    for index in range(comparison_window, len(series) - comparison_window):
        before = values[index - comparison_window : index]
        after = values[index : index + comparison_window]
        shift = after.mean() - before.mean()
        persistent_shift = values[index : index + persistence_window].mean() - before.mean()
        if (
            abs(shift) >= min_shift
            and shift * persistent_shift > 0
            and abs(persistent_shift) >= min_shift * 0.58
        ):
            candidates.append((index, shift, before.mean(), after.mean(), persistent_shift))

    selected = []
    for candidate in sorted(candidates, key=lambda item: abs(item[1]), reverse=True):
        if all(abs(candidate[0] - existing[0]) >= min_separation for existing in selected):
            selected.append(candidate)
    return sorted(selected)


def main():
    args = parse_args()
    pattern = "*_atm_iv_log_change_corr_60d.csv"
    paths = sorted(args.input_dir.glob(pattern))
    if args.input_prefix:
        paths = [path for path in paths if path.name.startswith(args.input_prefix)]
    if not paths:
        raise FileNotFoundError(f"No inputs matching {pattern} in {args.input_dir}")

    rows = []
    for path in paths:
        frame = pd.read_csv(path, encoding="utf-8-sig")
        corr_column = next(column for column in frame if column.startswith("atm_iv_log_change_corr_"))
        corr = pd.Series(pd.to_numeric(frame[corr_column], errors="coerce").to_numpy(), index=pd.to_datetime(frame["date"])).dropna()
        pair = corr_column.removeprefix("atm_iv_log_change_corr_").removesuffix("_60d").replace("_vs_", "/")
        for index, shift, before, after, persistent_shift in detect_shifts(
            corr, args.comparison_window, args.persistence_window,
            args.min_absolute_shift, args.min_separation,
        ):
            rows.append({
                "pair": pair,
                "shift_date": corr.index[index].strftime("%Y-%m-%d"),
                "direction": "up" if shift > 0 else "down",
                "corr_before_mean": before,
                "corr_after_mean": after,
                "corr_mean_shift": shift,
                "persistence_shift": persistent_shift,
                "valid_correlation_observations": len(corr),
                "data_first_date": corr.index[0].strftime("%Y-%m-%d"),
                "data_last_date": corr.index[-1].strftime("%Y-%m-%d"),
                "source_csv": str(path),
            })

    output = pd.DataFrame(rows).sort_values(["pair", "shift_date"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output_dir / f"{timestamp}_atm_iv_corr_regime_shifts.csv"
    output.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(output_path)
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
