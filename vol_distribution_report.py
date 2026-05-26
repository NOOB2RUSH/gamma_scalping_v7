from pathlib import Path
import argparse
from dataclasses import replace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import core


CONFIG = core.config.CONFIG


def parse_args():
    parser = argparse.ArgumentParser(description="生成波动率分布报告。")
    parser.add_argument(
        "--product",
        choices=core.config.available_products(),
        default=CONFIG.data.product,
        help="交易品种配置，默认使用 50ETF。",
    )
    parser.add_argument("--start", default=None, help="报告开始日期。")
    parser.add_argument("--end", default=None, help="报告结束日期。")
    return parser.parse_args()


def sync_config(config):
    global CONFIG
    CONFIG = config
    core.config.CONFIG = config
    core.vol_engine.CONFIG = config
    core.strategy.CONFIG = config
    core.backtester.CONFIG = config
    core.position.CONFIG = config
    core.hedge.CONFIG = config
    core.analytics.CONFIG = config


def select_runtime_config(args):
    selected_config = core.config.load_config(args.product)
    backtest_updates = {
        key: value
        for key, value in {"start": args.start, "end": args.end}.items()
        if value is not None
    }
    if backtest_updates:
        selected_config = replace(
            selected_config,
            backtest=replace(selected_config.backtest, **backtest_updates),
        )
    return selected_config


def make_output_dir():
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(CONFIG.report.output_root) / f"vol_distribution_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_features():
    etf_by_date = core.data_loader.load_etf_series(
        CONFIG.backtest.start,
        CONFIG.backtest.end,
    )
    opt_by_date = core.data_loader.load_opt_series(
        CONFIG.backtest.start,
        CONFIG.backtest.end,
    )
    trading_calendar = core.data_loader.load_etf_trading_calendar()
    enriched_opt_by_date = core.cache.get_enriched_option_chains(
        etf_by_date,
        opt_by_date,
        trading_calendar,
        CONFIG.backtest.start,
        CONFIG.backtest.end,
    )
    return core.cache.get_vol_features(
        etf_by_date,
        opt_by_date,
        trading_calendar,
        enriched_opt_by_date,
        CONFIG.backtest.start,
        CONFIG.backtest.end,
    )


def calc_distribution_summary(features):
    rows = []
    cols = ["atm_iv", "yz_hv60", "atm_iv_percentile"]
    for col in cols:
        if col not in features.columns:
            continue
        series = features[col].dropna()
        if series.empty:
            continue
        rows.append(
            {
                "metric": col,
                "count": int(series.count()),
                "missing": int(features[col].isna().sum()),
                "mean": series.mean(),
                "std": series.std(),
                "min": series.min(),
                "p01": series.quantile(0.01),
                "p05": series.quantile(0.05),
                "p10": series.quantile(0.10),
                "p25": series.quantile(0.25),
                "p50": series.quantile(0.50),
                "p75": series.quantile(0.75),
                "p90": series.quantile(0.90),
                "p95": series.quantile(0.95),
                "p99": series.quantile(0.99),
                "max": series.max(),
            }
        )
    return pd.DataFrame(rows)


def add_full_sample_percentiles(features):
    report_df = features.copy()
    if "atm_iv" in report_df.columns:
        report_df["atm_iv_full_sample_percentile"] = report_df["atm_iv"].rank(pct=True)
    if "yz_hv60" in report_df.columns:
        report_df["yz_hv60_full_sample_percentile"] = report_df["yz_hv60"].rank(
            pct=True
        )
    return report_df


def calc_percentile_bucket_summary(features):
    rows = []
    bucket_cols = [
        "atm_iv_full_sample_percentile",
        "yz_hv60_full_sample_percentile",
        "atm_iv_percentile",
    ]
    bins = [i / 10 for i in range(11)]
    labels = [f"{int(left * 100)}%-{int(right * 100)}%" for left, right in zip(bins, bins[1:])]

    for col in bucket_cols:
        if col not in features.columns:
            continue
        series = features[col].dropna()
        if series.empty:
            continue
        buckets = pd.cut(series, bins=bins, labels=labels, include_lowest=True)
        counts = buckets.value_counts(sort=False)
        for bucket, count in counts.items():
            rows.append(
                {
                    "metric": col,
                    "bucket": str(bucket),
                    "days": int(count),
                    "share": count / len(series),
                }
            )
    return pd.DataFrame(rows)


def calc_quantile_curve(features):
    rows = []
    quantiles = [i / 100 for i in range(101)]
    for col in ["atm_iv", "yz_hv60"]:
        if col not in features.columns:
            continue
        series = features[col].dropna()
        if series.empty:
            continue
        for quantile in quantiles:
            rows.append(
                {
                    "metric": col,
                    "percentile": quantile,
                    "value": series.quantile(quantile),
                }
            )
    return pd.DataFrame(rows)


def plot_distribution(features, output_path):
    fig, axes = plt.subplots(3, 2, figsize=(18, 18))

    axes[0, 0].plot(features.index, features["atm_iv"], label="ATM IV", color="tab:red")
    if "yz_hv60" in features.columns:
        axes[0, 0].plot(
            features.index,
            features["yz_hv60"],
            label="YZ HV60",
            color="tab:blue",
            alpha=0.8,
        )
    axes[0, 0].set_title("ATM IV vs HV")
    axes[0, 0].set_ylabel("Volatility")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].hist(
        features["atm_iv"].dropna(),
        bins=50,
        alpha=0.65,
        label="ATM IV",
        color="tab:red",
    )
    if "yz_hv60" in features.columns:
        axes[0, 1].hist(
            features["yz_hv60"].dropna(),
            bins=50,
            alpha=0.55,
            label="YZ HV60",
            color="tab:blue",
        )
    axes[0, 1].set_title("Volatility Distribution")
    axes[0, 1].set_xlabel("Volatility")
    axes[0, 1].set_ylabel("Days")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    quantile_grid = [i / 100 for i in range(101)]
    for col, label, color in [
        ("atm_iv", "ATM IV", "tab:red"),
        ("yz_hv60", "YZ HV60", "tab:blue"),
    ]:
        if col not in features.columns:
            continue
        series = features[col].dropna()
        if series.empty:
            continue
        quantile_values = [series.quantile(q) for q in quantile_grid]
        axes[1, 0].plot(
            quantile_grid,
            quantile_values,
            label=label,
            color=color,
            linewidth=1.8,
        )
    for level in [0.10, 0.25, 0.50, 0.75, 0.90]:
        axes[1, 0].axvline(level, color="gray", linestyle="--", linewidth=0.8, alpha=0.45)
    axes[1, 0].set_title("Percentile -> Absolute Volatility")
    axes[1, 0].set_xlabel("Full-Sample Percentile")
    axes[1, 0].set_ylabel("Volatility")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    for col, label, color in [
        ("atm_iv_full_sample_percentile", "ATM IV Full-Sample Percentile", "tab:red"),
        ("yz_hv60_full_sample_percentile", "HV60 Full-Sample Percentile", "tab:blue"),
    ]:
        if col in features.columns:
            axes[1, 1].hist(
                features[col].dropna(),
                bins=20,
                alpha=0.6,
                label=label,
                color=color,
            )
    axes[1, 1].set_title("Full-Sample Percentile Distribution")
    axes[1, 1].set_xlabel("Percentile")
    axes[1, 1].set_ylabel("Days")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    if "atm_iv_percentile" in features.columns:
        axes[2, 0].plot(
            features.index,
            features["atm_iv_percentile"],
            label=f"ATM IV Rolling Percentile ({CONFIG.vol.atm_iv_percentile_window}D)",
            color="tab:purple",
        )
        axes[2, 1].hist(
            features["atm_iv_percentile"].dropna(),
            bins=20,
            color="tab:purple",
            alpha=0.75,
        )
    axes[2, 0].set_title("ATM IV Rolling Percentile Time Series")
    axes[2, 0].set_ylabel("Percentile")
    axes[2, 0].set_ylim(-0.02, 1.02)
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].legend()

    axes[2, 1].set_title("ATM IV Rolling Percentile Distribution")
    axes[2, 1].set_xlabel("Percentile")
    axes[2, 1].set_ylabel("Days")
    axes[2, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_quantile_curve(quantile_df, output_path):
    fig, ax = plt.subplots(figsize=(14, 8))
    labels = {
        "atm_iv": "ATM IV",
        "yz_hv60": "YZ HV60",
    }
    colors = {
        "atm_iv": "tab:red",
        "yz_hv60": "tab:blue",
    }
    for metric, group in quantile_df.groupby("metric"):
        ax.plot(
            group["percentile"],
            group["value"],
            label=labels.get(metric, metric),
            color=colors.get(metric),
            linewidth=2,
        )

    for level in [0.10, 0.25, 0.50, 0.75, 0.90]:
        ax.axvline(level, color="gray", linestyle="--", linewidth=0.8, alpha=0.45)
    ax.set_title("Volatility Quantile Curve")
    ax.set_xlabel("Full-Sample Percentile")
    ax.set_ylabel("Absolute Volatility")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_text_report(output_dir, summary_df, bucket_df, features):
    lines = [
        "=== 波动率分布报告 ===",
        f"区间: {features.index.min().date()} -> {features.index.max().date()}",
        f"交易日数: {len(features)}",
        f"ATM IV 百分位滚动窗口: {CONFIG.vol.atm_iv_percentile_window}",
        "",
        "=== 数值分布 ===",
    ]
    for _, row in summary_df.iterrows():
        lines.append(
            f"{row['metric']}: count={row['count']}, missing={row['missing']}, "
            f"mean={row['mean']:.4f}, p10={row['p10']:.4f}, "
            f"p50={row['p50']:.4f}, p90={row['p90']:.4f}, max={row['max']:.4f}"
        )

    lines.extend(["", "=== 百分位桶分布 ==="])
    for metric, group in bucket_df.groupby("metric"):
        lines.append(metric)
        for _, row in group.iterrows():
            lines.append(
                f"  {row['bucket']}: days={row['days']}, share={row['share']:.2%}"
            )

    (output_dir / "vol_distribution_report.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8-sig",
    )


def main():
    args = parse_args()
    sync_config(select_runtime_config(args))
    print(
        "报告品种: "
        f"{CONFIG.data.product}, "
        f"数据区间: {CONFIG.backtest.start} - {CONFIG.backtest.end}",
        flush=True,
    )

    features = load_features()
    features = add_full_sample_percentiles(features)
    output_dir = make_output_dir()

    summary_df = calc_distribution_summary(features)
    bucket_df = calc_percentile_bucket_summary(features)
    quantile_df = calc_quantile_curve(features)

    features.to_csv(output_dir / "vol_distribution_features.csv", encoding="utf-8-sig")
    summary_df.to_csv(
        output_dir / "vol_distribution_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    bucket_df.to_csv(
        output_dir / "vol_percentile_bucket_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    quantile_df.to_csv(
        output_dir / "vol_quantile_curve.csv",
        index=False,
        encoding="utf-8-sig",
    )
    save_text_report(output_dir, summary_df, bucket_df, features)
    plot_distribution(features, output_dir / "vol_distribution.png")
    plot_quantile_curve(quantile_df, output_dir / "vol_quantile_curve.png")

    print(f"vol distribution report: {output_dir}")


if __name__ == "__main__":
    main()
