from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_BACKTEST_ROOT = PROJECT_ROOT / "output" / "backtest"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "research"
HEDGE_TYPE = "option_delta_hedge_combination"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Measure gamma/vega/theta side effects of option delta hedge "
            "combination trades."
        )
    )
    parser.add_argument(
        "--trades",
        type=Path,
        default=None,
        help="Path to a backtest trades.csv. Defaults to latest output/backtest/*/trades.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for impact CSV outputs.",
    )
    return parser.parse_args()


def _latest_trades_path():
    candidates = sorted(
        DEFAULT_BACKTEST_ROOT.glob("*/trades.csv"),
        key=lambda path: path.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def _read_trades(path):
    return pd.read_csv(path, encoding="utf-8-sig")


def build_impact_frame(trades):
    if trades.empty or "type" not in trades.columns:
        return pd.DataFrame()
    rows = trades.loc[trades["type"].astype(str).eq(HEDGE_TYPE)].copy()
    if rows.empty:
        return rows

    required = [
        "residual_delta_before_option_hedge",
        "delta_effect",
        "gamma_effect",
        "vega_effect",
        "theta_effect",
        "projected_option_delta",
        "projected_account_delta",
    ]
    missing = [column for column in required if column not in rows.columns]
    if missing:
        raise ValueError(
            "trades.csv is missing option hedge impact columns: "
            + ", ".join(missing)
            + ". Re-run the backtest after the Greek-impact fields were added."
        )

    for column in required:
        rows[column] = pd.to_numeric(rows[column], errors="coerce")

    before_abs = rows["residual_delta_before_option_hedge"].abs()
    option_after_abs = rows["projected_option_delta"].abs()
    combined_after_abs = rows["projected_account_delta"].abs()
    delta_improvement = before_abs - option_after_abs
    denominator = delta_improvement.abs()
    denominator = denominator.where(denominator > 1e-12)

    result = pd.DataFrame(
        {
            "date": rows.get("date"),
            "residual_delta_before": rows["residual_delta_before_option_hedge"],
            "delta_effect": rows["delta_effect"],
            "projected_delta_after_option": rows["projected_option_delta"],
            "projected_delta_after_option_and_etf": rows["projected_account_delta"],
            "delta_abs_before": before_abs,
            "delta_abs_after_option": option_after_abs,
            "delta_abs_after_option_and_etf": combined_after_abs,
            "delta_abs_improvement_by_option": delta_improvement,
            "gamma_effect": rows["gamma_effect"],
            "vega_effect": rows["vega_effect"],
            "theta_effect": rows["theta_effect"],
            "abs_gamma_effect": rows["gamma_effect"].abs(),
            "abs_vega_effect": rows["vega_effect"].abs(),
            "abs_theta_effect": rows["theta_effect"].abs(),
            "abs_gamma_per_delta_improvement": rows["gamma_effect"].abs() / denominator,
            "abs_vega_per_delta_improvement": rows["vega_effect"].abs() / denominator,
            "abs_theta_per_delta_improvement": rows["theta_effect"].abs() / denominator,
            "close_call_code": rows.get("close_call_code"),
            "close_call_qty": rows.get("close_call_qty"),
            "open_call_code": rows.get("open_call_code"),
            "open_call_qty": rows.get("open_call_qty"),
            "etf_buy_qty": rows.get("etf_buy_qty"),
            "delta_neutral_achieved": rows.get("delta_neutral_achieved"),
        }
    )
    return result.reset_index(drop=True)


def summarize_impact(frame):
    if frame.empty:
        return pd.DataFrame(
            [{"metric": "option_delta_hedge_combination_count", "value": 0}]
        )
    metrics = {
        "option_delta_hedge_combination_count": len(frame),
        "delta_abs_improvement_sum": frame["delta_abs_improvement_by_option"].sum(),
        "delta_abs_improvement_mean": frame["delta_abs_improvement_by_option"].mean(),
    }
    for greek in ["gamma", "vega", "theta"]:
        abs_col = f"abs_{greek}_effect"
        ratio_col = f"abs_{greek}_per_delta_improvement"
        series = pd.to_numeric(frame[abs_col], errors="coerce").dropna()
        ratio = pd.to_numeric(frame[ratio_col], errors="coerce").dropna()
        metrics[f"{abs_col}_sum"] = series.sum()
        metrics[f"{abs_col}_mean"] = series.mean()
        metrics[f"{abs_col}_median"] = series.median()
        metrics[f"{abs_col}_p95"] = series.quantile(0.95) if not series.empty else pd.NA
        metrics[f"{ratio_col}_mean"] = ratio.mean()
        metrics[f"{ratio_col}_median"] = ratio.median()
        metrics[f"{ratio_col}_p95"] = ratio.quantile(0.95) if not ratio.empty else pd.NA
    return pd.DataFrame(
        [{"metric": key, "value": value} for key, value in metrics.items()]
    )


def main():
    args = parse_args()
    trades_path = args.trades or _latest_trades_path()
    if trades_path is None:
        raise FileNotFoundError("No output/backtest/*/trades.csv found.")
    trades_path = Path(trades_path)
    trades = _read_trades(trades_path)
    impact = build_impact_frame(trades)
    summary = summarize_impact(impact)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = trades_path.parent.name
    impact_path = args.output_dir / f"{stem}_option_delta_hedge_greek_impact.csv"
    summary_path = args.output_dir / f"{stem}_option_delta_hedge_greek_impact_summary.csv"
    impact.to_csv(impact_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print(f"trades={trades_path}")
    print(f"events={len(impact)}")
    print(f"impact_csv={impact_path}")
    print(f"summary_csv={summary_path}")
    if not summary.empty:
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
