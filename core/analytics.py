from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd

from .config import CONFIG


def _iv_observation_mode():
    return getattr(CONFIG.vol, "iv_observation_mode", "legacy")


def _product_label():
    product = str(CONFIG.data.product).lower()
    labels = {
        "50etf": "50ETF (510050)",
        "300etf": "300ETF (510300)",
        "500etf": "500ETF (510500)",
        "kc50etf": "STAR 50 ETF (588000)",
        "zz1000": "ZZ1000 Index Option (MO)",
    }
    return labels.get(product, str(CONFIG.data.product))


def plot_vol_features(
    features_df,
    backtest_df=None,
    absolute_backtest_df=None,
    percentile_backtest_df=None,
    no_delta_hedge_df=None,
    strategy_label=None,
    output_path=None,
    show=True,
):

    required_cols = {"close", "atm_iv"}
    missing = required_cols - set(features_df.columns)
    if missing:
        raise ValueError(f"features_df missing columns:{missing}")

    has_atm_pool_volume = "atm_pool_total_volume" in features_df.columns
    has_atm_volume = has_atm_pool_volume or "atm_total_volume" in features_df.columns
    ax_nav = None
    ax_drawdown = None
    ax_volume = None
    if backtest_df is not None and has_atm_volume:
        fig, (ax_price, ax_nav, ax_drawdown, ax_volume) = plt.subplots(
            4,
            1,
            figsize=(28, 26),
            sharex=True,
            gridspec_kw={"height_ratios": [2, 1, 1, 1]},
        )
    elif backtest_df is not None:
        fig, (ax_price, ax_nav, ax_drawdown) = plt.subplots(
            3,
            1,
            figsize=(28, 22),
            sharex=True,
            gridspec_kw={"height_ratios": [2, 1, 1]},
        )
    elif has_atm_volume:
        fig, (ax_price, ax_volume) = plt.subplots(
            2,
            1,
            figsize=(28, 18),
            sharex=True,
            gridspec_kw={"height_ratios": [2, 1]},
        )
    else:
        fig, ax_price = plt.subplots(figsize=(28, 14))

    ax_price.plot(
        features_df.index,
        features_df["close"],
        label=f"{_product_label()} Close",
        color="black",
        linewidth=1.5,
    )
    ax_price.set_ylabel(("Underlying Price"))
    if backtest_df is None and ax_volume is None:
        ax_price.set_xlabel("Date")

    ax_price.grid(True, alpha=0.25)
    ax_vol = ax_price.twinx()
    ax_percentile = None

    ax_vol.plot(
        features_df.index,
        features_df["atm_iv"],
        label="ATM IV",
        color="tab:red",
        linewidth=1.5,
    )
    observation_mode = _iv_observation_mode()
    signal_iv_col = "atm_iv"
    if observation_mode == "surface_percentile" and "signal_iv" in features_df.columns:
        signal_iv_col = "signal_iv"
    elif (
        observation_mode == "legacy"
        and getattr(CONFIG.vol, "surface_atm_iv_enabled", False)
        and "signal_iv" in features_df.columns
    ):
        signal_iv_col = "signal_iv"
    if signal_iv_col != "atm_iv":
        ax_vol.plot(
            features_df.index,
            features_df[signal_iv_col],
            label=f"Signal IV ({CONFIG.vol.surface_atm_target_dte}D Surface ATM)",
            color="tab:orange",
            linewidth=1.4,
            alpha=0.9,
        )

    long_mode = getattr(CONFIG.strategy, "long_signal_mode", "absolute")
    if CONFIG.strategy.enable_long_straddle and long_mode == "absolute":
        ax_vol.axhline(
            CONFIG.strategy.long_open_iv_threshold,
            label=f"Long Open IV Threshold {CONFIG.strategy.long_open_iv_threshold:.2%}",
            color="tab:green",
            linewidth=1.0,
            linestyle="--",
            alpha=0.75,
        )
        ax_vol.axhline(
            CONFIG.strategy.long_close_iv_threshold,
            label=f"Long Close IV Threshold {CONFIG.strategy.long_close_iv_threshold:.2%}",
            color="tab:orange",
            linewidth=1.0,
            linestyle="--",
            alpha=0.75,
        )
    short_mode = getattr(CONFIG.strategy, "short_signal_mode", "absolute")
    if CONFIG.strategy.enable_short_straddle and short_mode == "absolute":
        ax_vol.axhline(
            CONFIG.strategy.short_open_iv_threshold,
            label=(
                "Short Open IV Threshold "
                f"{CONFIG.strategy.short_open_iv_threshold:.2%}"
            ),
            color="tab:purple",
            linewidth=1.0,
            linestyle="--",
            alpha=0.75,
        )
        ax_vol.axhline(
            CONFIG.strategy.short_close_iv_threshold,
            label=(
                "Short Close IV Threshold "
                f"{CONFIG.strategy.short_close_iv_threshold:.2%}"
            ),
            color="tab:brown",
            linewidth=1.0,
            linestyle="--",
            alpha=0.75,
        )
    percentile_col = (
        "signal_iv_percentile"
        if observation_mode == "surface_percentile"
        and "signal_iv_percentile" in features_df.columns
        else "atm_iv_percentile"
    )
    show_percentile = (
        observation_mode != "simple_atm_absolute"
        and (
            (
                CONFIG.strategy.enable_short_straddle
                and short_mode == "percentile"
            )
            or (
                CONFIG.strategy.enable_long_straddle
                and long_mode == "percentile"
            )
        )
        and percentile_col in features_df.columns
    )
    if show_percentile:
        ax_percentile = ax_price.twinx()
        ax_percentile.spines["right"].set_position(("axes", 1.06))
        ax_percentile.plot(
            features_df.index,
            features_df[percentile_col],
            label="Signal IV Percentile",
            color="tab:purple",
            linewidth=1.1,
            alpha=0.8,
        )
        if CONFIG.strategy.enable_long_straddle and long_mode == "percentile":
            ax_percentile.axhline(
                CONFIG.strategy.long_open_iv_percentile_threshold,
                label=(
                    "Long Open IV Percentile "
                    f"{CONFIG.strategy.long_open_iv_percentile_threshold:.0%}"
                ),
                color="tab:green",
                linewidth=1.0,
                linestyle="--",
                alpha=0.75,
            )
            ax_percentile.axhline(
                CONFIG.strategy.long_close_iv_percentile_threshold,
                label=(
                    "Long Close IV Percentile "
                    f"{CONFIG.strategy.long_close_iv_percentile_threshold:.0%}"
                ),
                color="tab:orange",
                linewidth=1.0,
                linestyle="--",
                alpha=0.75,
            )
        if CONFIG.strategy.enable_short_straddle and short_mode == "percentile":
            ax_percentile.axhline(
                CONFIG.strategy.short_open_iv_percentile_threshold,
                label=(
                    "Short Open IV Percentile "
                    f"{CONFIG.strategy.short_open_iv_percentile_threshold:.0%}"
                ),
                color="tab:purple",
                linewidth=1.0,
                linestyle="--",
                alpha=0.75,
            )
            ax_percentile.axhline(
                CONFIG.strategy.short_close_iv_percentile_threshold,
                label=(
                    "Short Close IV Percentile "
                    f"{CONFIG.strategy.short_close_iv_percentile_threshold:.0%}"
                ),
                color="tab:brown",
                linewidth=1.0,
                linestyle="--",
                alpha=0.75,
            )
        ax_percentile.set_ylabel("Signal IV Percentile")
        ax_percentile.set_ylim(-0.02, 1.02)

    if "yz_hv20" in features_df.columns:
        ax_vol.plot(
            features_df.index,
            features_df["yz_hv20"],
            label="HV 20",
            color="tab:blue",
            linewidth=1.2,
            linestyle="--",
        )

    if "yz_hv5" in features_df.columns:
        ax_vol.plot(
            features_df.index,
            features_df["yz_hv5"],
            label="HV 5",
            color="tab:green",
            linewidth=1.0,
            linestyle=":",
        )

    if "yz_hv60" in features_df.columns:
        ax_vol.plot(
            features_df.index,
            features_df["yz_hv60"],
            label="HV 60",
            color="tab:purple",
            linewidth=1.0,
            linestyle="-.",
        )

    ax_vol.set_ylabel("Annualized Volatility")

    lines_1, labels_1 = ax_price.get_legend_handles_labels()
    lines_2, labels_2 = ax_vol.get_legend_handles_labels()
    lines_3, labels_3 = (
        ax_percentile.get_legend_handles_labels()
        if ax_percentile is not None
        else ([], [])
    )

    ax_price.legend(
        lines_1 + lines_2 + lines_3,
        labels_1 + labels_2 + labels_3,
        loc="upper right",
    )

    fig.suptitle(f"{_product_label()} - Underlying Price, ATM IV and HV")

    if backtest_df is not None:
        initial_cash = float(CONFIG.backtest.initial_cash)
        if initial_cash <= 0:
            raise ValueError("initial_cash must be positive to plot cumulative return")

        def cumulative_return(frame):
            return (
                pd.to_numeric(frame["nav"], errors="coerce")
                .div(initial_cash)
                .sub(1.0)
            )

        if strategy_label is None:
            strategy_label = "Cumulative Return"
            if (
                getattr(CONFIG.strategy, "short_signal_mode", None) == "absolute"
                and not getattr(CONFIG.strategy, "enable_delta_hedge", True)
            ):
                strategy_label = "Cumulative Return (Absolute Naked)"
        ax_nav.plot(
            backtest_df.index,
            cumulative_return(backtest_df),
            label=strategy_label,
            color="black",
            linewidth=1.5,
        )
        if absolute_backtest_df is not None and "nav" in absolute_backtest_df.columns:
            ax_nav.plot(
                absolute_backtest_df.index,
                cumulative_return(absolute_backtest_df),
                label="Absolute Signal Cumulative Return",
                color="tab:orange",
                linewidth=1.3,
                linestyle="-.",
            )
        if percentile_backtest_df is not None and "nav" in percentile_backtest_df.columns:
            ax_nav.plot(
                percentile_backtest_df.index,
                cumulative_return(percentile_backtest_df),
                label=(
                    "Percentile Signal Cumulative Return "
                    f"({CONFIG.strategy.short_open_iv_percentile_threshold:.0%}/"
                    f"{CONFIG.strategy.short_close_iv_percentile_threshold:.0%} Naked)"
                ),
                color="tab:green",
                linewidth=1.3,
                linestyle=":",
            )
        if no_delta_hedge_df is not None and "nav" in no_delta_hedge_df.columns:
            ax_nav.plot(
                no_delta_hedge_df.index,
                cumulative_return(no_delta_hedge_df),
                label="Absolute Naked Vega Short Cumulative Return",
                color="tab:gray",
                linewidth=1.3,
                linestyle=(0, (3, 1, 1, 1)),
            )

        ax_nav.axhline(
            0.0,
            color="gray",
            linestyle="--",
            linewidth=1,
            label="Initial Capital (0%)",
        )

        ax_nav.set_ylabel("Cumulative Return")
        ax_nav.yaxis.set_major_formatter(PercentFormatter(1.0))
        if ax_drawdown is None and ax_volume is None:
            ax_nav.set_xlabel("Date")
        ax_nav.grid(True, alpha=0.3)
        ax_nav.legend()

    if ax_drawdown is not None:
        drawdown = backtest_df["nav"] / backtest_df["nav"].cummax() - 1
        ax_drawdown.plot(
            backtest_df.index,
            drawdown,
            label="Drawdown",
            color="tab:red",
            linewidth=1.2,
        )
        ax_drawdown.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax_drawdown.set_ylabel("Drawdown")
        if ax_volume is None:
            ax_drawdown.set_xlabel("Date")
        ax_drawdown.grid(True, alpha=0.3)
        ax_drawdown.legend(loc="lower left")

    if ax_volume is not None:
        if has_atm_pool_volume:
            volume_col = "atm_pool_total_volume"
            volume_label = "ATM Nearby Pool Call+Put Volume"
        else:
            volume_col = "atm_total_volume"
            volume_label = "ATM Call+Put Volume"
        ax_volume.plot(
            features_df.index,
            features_df[volume_col],
            label=volume_label,
            color="tab:blue",
            linewidth=0.8,
            alpha=0.35,
        )
        rolling_volume = features_df[volume_col].rolling(10, min_periods=3).mean()
        ax_volume.plot(
            features_df.index,
            rolling_volume,
            label=f"{volume_label} 10D MA",
            color="tab:blue",
            linewidth=1.8,
        )
        if "atm_pool_min_leg_volume" in features_df.columns:
            ax_volume.plot(
                features_df.index,
                features_df["atm_pool_min_leg_volume"],
                label="ATM Nearby Pool Weaker Leg Volume",
                color="tab:orange",
                linewidth=1.0,
                alpha=0.85,
            )
        ax_volume.set_ylabel("ATM Option Volume")
        ax_volume.set_xlabel("Date")
        ax_volume.grid(True, alpha=0.3)
        ax_volume.legend(loc="upper right")

    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_cumulative_greeks_pnl(backtest_df, output_path=None, show=True):
    pnl_cols = ["delta_pnl", "gamma_pnl", "vega_pnl", "theta_pnl", "greeks_pnl"]
    missing = set(pnl_cols) - set(backtest_df.columns)
    if missing:
        raise ValueError(f"backtest_df missing columns:{missing}")

    daily_pnl = backtest_df[pnl_cols].apply(pd.to_numeric, errors="coerce")
    cum_pnl = daily_pnl.cumsum()
    cumulative_theta_gamma_pnl = (
        daily_pnl["theta_pnl"] + daily_pnl["gamma_pnl"]
    ).cumsum()
    fig, ax = plt.subplots(figsize=(28, 14))

    ax.plot(cum_pnl.index, cum_pnl["delta_pnl"], label="Delta PnL", linewidth=1.2)
    ax.plot(
        cum_pnl.index,
        cumulative_theta_gamma_pnl,
        label="Theta + Gamma PnL",
        linewidth=1.4,
    )
    ax.plot(cum_pnl.index, cum_pnl["vega_pnl"], label="Vega PnL", linewidth=1.2)
    ax.plot(cum_pnl.index, cum_pnl["greeks_pnl"], label="Total Greeks PnL", color="black", linewidth=1.8)

    ax.axhline(0, color="gray", linewidth=1, linestyle="--")
    ax.set_title(f"{_product_label()} - Cumulative Greeks PnL")
    ax.set_xlabel("Date")
    ax.set_ylabel("PnL")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_cumulative_actual_vs_greeks_pnl(backtest_df, output_path=None, show=True):
    """对比手续费前真实累计 PnL 和 Greeks 解释累计 PnL。"""
    pnl_cols = [
        "daily_nav_pnl",
        "daily_nav_pnl_before_fee",
        "greeks_pnl",
        "greeks_unexplained_pnl_before_fee",
    ]
    missing = set(pnl_cols) - set(backtest_df.columns)
    if missing:
        raise ValueError(f"backtest_df missing columns:{missing}")

    cum_actual_pnl = (
        pd.to_numeric(backtest_df["daily_nav_pnl_before_fee"], errors="coerce")
        .fillna(0.0)
        .cumsum()
    )
    cum_actual_after_fee_pnl = (
        pd.to_numeric(backtest_df["daily_nav_pnl"], errors="coerce")
        .fillna(0.0)
        .cumsum()
    )
    cum_greeks_pnl = (
        pd.to_numeric(backtest_df["greeks_pnl"], errors="coerce").fillna(0.0).cumsum()
    )
    cum_unexplained_pnl = (
        pd.to_numeric(
            backtest_df["greeks_unexplained_pnl_before_fee"], errors="coerce"
        )
        .fillna(0.0)
        .cumsum()
    )

    fig, ax = plt.subplots(figsize=(28, 14))
    ax.plot(
        cum_actual_pnl.index,
        cum_actual_pnl,
        label="Actual Cumulative PnL Before Fees",
        color="black",
        linewidth=1.8,
    )
    ax.plot(
        cum_actual_after_fee_pnl.index,
        cum_actual_after_fee_pnl,
        label="Actual Cumulative PnL After Fees",
        color="gray",
        linewidth=1.0,
        linestyle=":",
    )
    ax.plot(
        cum_greeks_pnl.index,
        cum_greeks_pnl,
        label="Greeks Cumulative PnL",
        color="tab:blue",
        linewidth=1.5,
    )
    ax.plot(
        cum_unexplained_pnl.index,
        cum_unexplained_pnl,
        label="Unexplained Cumulative PnL",
        color="tab:red",
        linewidth=1.2,
        linestyle="--",
    )
    exact_column = (
        "full_revaluation_accounted_pnl"
        if "full_revaluation_accounted_pnl" in backtest_df.columns
        else "full_revaluation_greeks_pnl"
    )
    if exact_column in backtest_df.columns:
        cum_full_revaluation = (
            pd.to_numeric(
                backtest_df[exact_column], errors="coerce"
            )
            .fillna(0.0)
            .cumsum()
        )
        ax.plot(
            cum_full_revaluation.index,
            cum_full_revaluation,
            label="Full-Revaluation Attribution (Exact)",
            color="tab:green",
            linewidth=1.3,
            linestyle="-.",
        )

    ax.axhline(0, color="gray", linewidth=1, linestyle="--")
    ax.set_title(
        f"{_product_label()} - Cumulative Actual PnL Before Fees vs Greeks PnL"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("PnL")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig
