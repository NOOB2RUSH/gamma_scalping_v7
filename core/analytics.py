from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_vol_features(features_df, backtest_df=None, output_path=None, show=True):

    required_cols = {"close", "atm_iv"}
    missing = required_cols - set(features_df.columns)
    if missing:
        raise ValueError(f"features_df missing columns:{missing}")

    if backtest_df is None:
        fig, ax_price = plt.subplots(figsize=(28, 14))
    else:
        fig, (ax_price, ax_nav) = plt.subplots(
            2,
            1,
            figsize=(28, 18),
            sharex=True,
            gridspec_kw={"height_ratios": [2, 1]},
        )

    ax_price.plot(
        features_df.index,
        features_df["close"],
        label="ETF Close",
        color="black",
        linewidth=1.5,
    )
    ax_price.set_ylabel(("ETF Price"))
    if backtest_df is None:
        ax_price.set_xlabel("Date")

    ax_price.grid(True, alpha=0.25)
    ax_vol = ax_price.twinx()

    ax_vol.plot(
        features_df.index,
        features_df["atm_iv"],
        label="ATM IV",
        color="tab:red",
        linewidth=1.5,
    )

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

    ax_price.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")

    fig.suptitle(("ETF Price, ATM IV and HV"))

    if backtest_df is not None:
        ax_nav.plot(
            backtest_df.index,
            backtest_df["nav"],
            label="NAV",
            color="black",
            linewidth=1.5,
        )

        ax_nav.axhline(
            backtest_df["nav"].iloc[0],
            color="gray",
            linestyle="--",
            linewidth=1,
            label="Initial NAV",
        )

        ax_nav.set_ylabel("NAV")
        ax_nav.set_xlabel("Date")
        ax_nav.grid(True, alpha=0.3)
        ax_nav.legend()

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

    cum_pnl = backtest_df[pnl_cols].cumsum()
    fig, ax = plt.subplots(figsize=(28, 14))

    ax.plot(cum_pnl.index, cum_pnl["delta_pnl"], label="Delta PnL", linewidth=1.2)
    ax.plot(cum_pnl.index, cum_pnl["gamma_pnl"], label="Gamma PnL", linewidth=1.2)
    ax.plot(cum_pnl.index, cum_pnl["vega_pnl"], label="Vega PnL", linewidth=1.2)
    ax.plot(cum_pnl.index, cum_pnl["theta_pnl"], label="Theta PnL", linewidth=1.2)
    ax.plot(cum_pnl.index, cum_pnl["greeks_pnl"], label="Total Greeks PnL", color="black", linewidth=1.8)

    ax.axhline(0, color="gray", linewidth=1, linestyle="--")
    ax.set_title("Cumulative Greeks PnL")
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
