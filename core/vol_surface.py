from __future__ import annotations

from dataclasses import dataclass, field
from math import erf, exp, log, sqrt

import numpy as np
import pandas as pd


EPS = 1e-12


@dataclass(frozen=True)
class SurfaceFilters:
    min_dte: int = 7
    min_volume: float = 1.0
    max_spread_pct: float | None = 0.50
    min_abs_delta: float | None = 0.05
    max_abs_delta: float | None = 0.95
    min_iv: float = 0.0001
    max_iv: float = 5.0


@dataclass(frozen=True)
class SurfaceConfig:
    standard_dtes: tuple[int, ...] = (30, 60, 90)
    annual_days: int = 252
    risk_free_rate: float = 0.0
    filters: SurfaceFilters = field(default_factory=SurfaceFilters)


def _normal_cdf(value):
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))


def black76_price(flag, forward, strike, ttm, sigma, r=0.0):
    """Black-76 option price for futures options."""
    if ttm <= 0 or sigma <= 0 or forward <= 0 or strike <= 0:
        return np.nan

    discount = exp(-r * ttm)
    vol_sqrt_t = sigma * sqrt(ttm)
    d1 = (log(forward / strike) + 0.5 * sigma * sigma * ttm) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    flag = str(flag).lower()
    if flag == "c":
        return discount * (forward * _normal_cdf(d1) - strike * _normal_cdf(d2))
    if flag == "p":
        return discount * (strike * _normal_cdf(-d2) - forward * _normal_cdf(-d1))
    return np.nan


def implied_vol_black76(
    price,
    forward,
    strike,
    ttm,
    flag,
    r=0.0,
    min_sigma=1e-6,
    max_sigma=5.0,
    tol=1e-8,
    max_iter=100,
):
    """Invert Black-76 with bisection. Returns NaN when the quote is invalid."""
    try:
        price = float(price)
        forward = float(forward)
        strike = float(strike)
        ttm = float(ttm)
    except (TypeError, ValueError):
        return np.nan

    if price <= 0 or forward <= 0 or strike <= 0 or ttm <= 0:
        return np.nan

    flag = str(flag).lower()
    discount = exp(-r * ttm)
    if flag == "c":
        intrinsic = discount * max(forward - strike, 0.0)
    elif flag == "p":
        intrinsic = discount * max(strike - forward, 0.0)
    else:
        return np.nan

    if price < intrinsic - 1e-8:
        return np.nan

    low = min_sigma
    high = max_sigma
    high_price = black76_price(flag, forward, strike, ttm, high, r=r)
    while pd.notna(high_price) and high_price < price and high < 20.0:
        high *= 2.0
        high_price = black76_price(flag, forward, strike, ttm, high, r=r)
    if pd.isna(high_price) or high_price < price:
        return np.nan

    for _ in range(max_iter):
        mid = (low + high) / 2.0
        model_price = black76_price(flag, forward, strike, ttm, mid, r=r)
        if pd.isna(model_price):
            return np.nan
        if abs(model_price - price) <= tol:
            return mid
        if model_price < price:
            low = mid
        else:
            high = mid

    return (low + high) / 2.0


def _ensure_dte(chain_df, annual_days):
    df = chain_df.copy()
    if "dte" not in df.columns:
        if not {"date", "maturity_date"}.issubset(df.columns):
            raise ValueError("prepare_iv_points requires dte or date+maturity_date")
        df["date"] = pd.to_datetime(df["date"])
        df["maturity_date"] = pd.to_datetime(df["maturity_date"])
        df["dte"] = (df["maturity_date"] - df["date"]).dt.days
    if "ttm" not in df.columns:
        df["ttm"] = pd.to_numeric(df["dte"], errors="coerce") / annual_days
    return df


def _pick_forward(df, forward_col=None):
    if forward_col is not None:
        if forward_col not in df.columns:
            raise ValueError(f"forward_col not found: {forward_col}")
        return pd.to_numeric(df[forward_col], errors="coerce")
    for col in ("pricing_spot", "underlying_close", "forward", "future_close", "spot"):
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
    raise ValueError("No forward/underlying price column found for volatility surface")


def prepare_iv_points(
    chain_df,
    annual_days=252,
    r=0.0,
    filters: SurfaceFilters | None = None,
    forward_col=None,
    prefer_existing_iv=False,
):
    """Build clean (K, T, IV) points from one day's option chain.

    The function uses bid/ask mid prices and Black-76 IV by default. When the
    input only has close-as-bid/ask, the resulting IV is still quote dependent
    and should be filtered by volume/spread downstream.
    """
    filters = filters or SurfaceFilters()
    df = _ensure_dte(chain_df, annual_days)
    df = df.copy()
    df["option_type"] = df["option_type"].astype(str).str.lower()
    df["strike_price"] = pd.to_numeric(df["strike_price"], errors="coerce")
    df["forward"] = _pick_forward(df, forward_col=forward_col)
    df["bid"] = pd.to_numeric(df["bid"], errors="coerce")
    df["ask"] = pd.to_numeric(df["ask"], errors="coerce")
    df["volume"] = pd.to_numeric(df.get("volume", 0), errors="coerce").fillna(0.0)
    df["mid"] = (df["bid"] + df["ask"]) / 2.0
    df["spread"] = df["ask"] - df["bid"]
    df["spread_pct"] = df["spread"] / df["mid"].where(df["mid"].abs() > EPS)

    valid = (
        df["option_type"].isin(["c", "p"])
        & (df["strike_price"] > 0)
        & (df["forward"] > 0)
        & (df["ttm"] > 0)
        & (df["dte"] >= filters.min_dte)
        & (df["bid"] > 0)
        & (df["ask"] >= df["bid"])
        & (df["mid"] > 0)
        & (df["volume"] >= filters.min_volume)
    )
    if filters.max_spread_pct is not None:
        valid &= df["spread_pct"].fillna(np.inf) <= filters.max_spread_pct
    if "delta" in df.columns and (
        filters.min_abs_delta is not None or filters.max_abs_delta is not None
    ):
        abs_delta = pd.to_numeric(df["delta"], errors="coerce").abs()
        if filters.min_abs_delta is not None:
            valid &= abs_delta >= filters.min_abs_delta
        if filters.max_abs_delta is not None:
            valid &= abs_delta <= filters.max_abs_delta

    df = df.loc[valid].copy()
    if df.empty:
        return _empty_points()

    if prefer_existing_iv and "iv" in df.columns:
        df["surface_iv"] = pd.to_numeric(df["iv"], errors="coerce")
    else:
        df["surface_iv"] = [
            implied_vol_black76(
                row.mid,
                row.forward,
                row.strike_price,
                row.ttm,
                row.option_type,
                r=r,
            )
            for row in df.itertuples(index=False)
        ]

    df["surface_iv"] = pd.to_numeric(df["surface_iv"], errors="coerce")
    df = df[
        df["surface_iv"].between(filters.min_iv, filters.max_iv, inclusive="both")
    ].copy()
    if df.empty:
        return _empty_points()

    df["log_moneyness"] = np.log(df["strike_price"] / df["forward"])
    df["total_variance"] = df["surface_iv"] ** 2 * df["ttm"]
    df["quote_weight"] = df["volume"] / (df["spread_pct"].abs().fillna(0.0) + 0.01)

    keep_cols = [
        "date",
        "order_book_id",
        "underlying_order_book_id",
        "maturity_date",
        "option_type",
        "strike_price",
        "forward",
        "log_moneyness",
        "dte",
        "ttm",
        "bid",
        "ask",
        "mid",
        "spread",
        "spread_pct",
        "volume",
        "surface_iv",
        "total_variance",
        "quote_weight",
    ]
    return df[[col for col in keep_cols if col in df.columns]].sort_values(
        ["maturity_date", "strike_price", "option_type"]
    )


def _empty_points():
    return pd.DataFrame(
        columns=[
            "date",
            "order_book_id",
            "underlying_order_book_id",
            "maturity_date",
            "option_type",
            "strike_price",
            "forward",
            "log_moneyness",
            "dte",
            "ttm",
            "surface_iv",
            "total_variance",
            "quote_weight",
        ]
    )


def collapse_call_put_points(points):
    """Average call/put quotes at the same expiry and strike into one surface point."""
    if points.empty:
        return points.copy()

    group_cols = [
        col
        for col in [
            "date",
            "maturity_date",
            "underlying_order_book_id",
            "strike_price",
            "forward",
            "dte",
            "ttm",
        ]
        if col in points.columns
    ]
    rows = []
    for keys, group in points.groupby(group_cols, dropna=False):
        weights = pd.to_numeric(group["quote_weight"], errors="coerce").fillna(0.0)
        if weights.sum() <= 0:
            weights = pd.Series(1.0, index=group.index)
        total_variance = np.average(group["total_variance"], weights=weights)
        surface_iv = sqrt(max(total_variance / float(group["ttm"].iloc[0]), 0.0))
        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        row.update(
            {
                "log_moneyness": np.average(group["log_moneyness"], weights=weights),
                "surface_iv": surface_iv,
                "total_variance": total_variance,
                "quote_weight": weights.sum(),
                "volume": group["volume"].sum(),
                "quote_count": len(group),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["maturity_date", "strike_price"])


def select_atm_term_points(points):
    """Pick the nearest log-moneyness point for each expiry/underlying."""
    if points.empty:
        return points.copy()
    group_cols = [
        col for col in ["maturity_date", "underlying_order_book_id"] if col in points.columns
    ]
    rows = []
    for _, group in points.groupby(group_cols, dropna=False):
        idx = group["log_moneyness"].abs().idxmin()
        rows.append(points.loc[idx])
    return pd.DataFrame(rows).sort_values("dte").reset_index(drop=True)


def interpolate_total_variance(
    term_points,
    target_dte,
    allow_extrapolate=False,
    extrapolate_mode="linear",
    annual_days=252,
):
    """Interpolate in total variance, then convert back to fixed-tenor IV."""
    if term_points.empty:
        return {
            "target_dte": target_dte,
            "surface_iv": np.nan,
            "total_variance": np.nan,
            "method": "empty",
        }

    curve = (
        term_points[["dte", "total_variance"]]
        .dropna()
        .groupby("dte", as_index=False)["total_variance"]
        .mean()
        .sort_values("dte")
    )
    if curve.empty:
        return {
            "target_dte": target_dte,
            "surface_iv": np.nan,
            "total_variance": np.nan,
            "method": "empty",
        }

    dtes = curve["dte"].to_numpy(dtype=float)
    variances = curve["total_variance"].to_numpy(dtype=float)
    target_dte = float(target_dte)
    target_ttm = target_dte / float(annual_days)

    exact = np.where(np.isclose(dtes, target_dte))[0]
    if len(exact) > 0:
        total_variance = float(variances[exact[0]])
        method = "exact"
    elif target_dte < dtes.min() or target_dte > dtes.max():
        if not allow_extrapolate or len(dtes) < 2:
            return {
                "target_dte": target_dte,
                "surface_iv": np.nan,
                "total_variance": np.nan,
                "method": "outside_range",
            }
        if extrapolate_mode == "nearest":
            nearest = int(np.argmin(np.abs(dtes - target_dte)))
            source_ttm = dtes[nearest] / float(annual_days)
            source_iv = (
                sqrt(max(float(variances[nearest]) / source_ttm, 0.0))
                if source_ttm > 0
                else np.nan
            )
            total_variance = source_iv * source_iv * target_ttm
            method = "nearest"
        else:
            if target_dte < dtes.min():
                left, right = 0, 1
            else:
                left, right = len(dtes) - 2, len(dtes) - 1
            total_variance = _linear_interpolate(
                target_dte,
                dtes[left],
                dtes[right],
                variances[left],
                variances[right],
            )
            method = "extrapolated"
    else:
        right = int(np.searchsorted(dtes, target_dte, side="right"))
        left = right - 1
        total_variance = _linear_interpolate(
            target_dte,
            dtes[left],
            dtes[right],
            variances[left],
            variances[right],
        )
        method = "interpolated"

    total_variance = max(float(total_variance), 0.0)
    surface_iv = sqrt(total_variance / target_ttm) if target_ttm > 0 else np.nan
    return {
        "target_dte": target_dte,
        "surface_iv": surface_iv,
        "total_variance": total_variance,
        "method": method,
    }


def _linear_interpolate(x, x1, x2, y1, y2):
    if np.isclose(x1, x2):
        return float((y1 + y2) / 2.0)
    return float(y1 + (y2 - y1) * (x - x1) / (x2 - x1))


def fixed_tenor_atm_iv(
    points,
    target_dte,
    allow_extrapolate=False,
    extrapolate_mode="linear",
    annual_days=252,
):
    collapsed = collapse_call_put_points(points)
    atm_terms = select_atm_term_points(collapsed)
    return interpolate_total_variance(
        atm_terms,
        target_dte,
        allow_extrapolate=allow_extrapolate,
        extrapolate_mode=extrapolate_mode,
        annual_days=annual_days,
    )


def _default_k_grid(points, num=31, mode="intersection"):
    maturity_ranges = points.groupby("maturity_date")["log_moneyness"].agg(["min", "max"])
    if mode == "union":
        low = maturity_ranges["min"].min()
        high = maturity_ranges["max"].max()
    else:
        low = maturity_ranges["min"].max()
        high = maturity_ranges["max"].min()
    if pd.isna(low) or pd.isna(high) or low >= high:
        low = points["log_moneyness"].quantile(0.10)
        high = points["log_moneyness"].quantile(0.90)
    if pd.isna(low) or pd.isna(high) or low >= high:
        return np.array([0.0])
    return np.linspace(float(low), float(high), num)


def build_fixed_tenor_smile(
    points,
    target_dte,
    k_grid=None,
    k_grid_mode="intersection",
    allow_term_extrapolate=False,
    term_extrapolate_mode="linear",
    annual_days=252,
):
    """Map observed smiles to a fixed tenor by interpolating total variance."""
    collapsed = collapse_call_put_points(points)
    if collapsed.empty:
        return pd.DataFrame(
            columns=["target_dte", "log_moneyness", "total_variance", "surface_iv"]
        )

    if k_grid is None:
        k_grid = _default_k_grid(collapsed, mode=k_grid_mode)
    k_grid = np.asarray(k_grid, dtype=float)

    maturity_rows = []
    for maturity, group in collapsed.groupby("maturity_date"):
        group = group.sort_values("log_moneyness")
        if len(group) < 2:
            continue
        k = group["log_moneyness"].to_numpy(dtype=float)
        w = group["total_variance"].to_numpy(dtype=float)
        valid_grid = k_grid[(k_grid >= k.min()) & (k_grid <= k.max())]
        if len(valid_grid) == 0:
            continue
        interp_w = np.interp(valid_grid, k, w)
        for grid_k, total_variance in zip(valid_grid, interp_w):
            maturity_rows.append(
                {
                    "maturity_date": maturity,
                    "dte": float(group["dte"].iloc[0]),
                    "log_moneyness": float(grid_k),
                    "total_variance": float(total_variance),
                }
            )
    maturity_smiles = pd.DataFrame(maturity_rows)
    if maturity_smiles.empty:
        return pd.DataFrame(
            columns=["target_dte", "log_moneyness", "total_variance", "surface_iv"]
        )

    rows = []
    for grid_k, group in maturity_smiles.groupby("log_moneyness"):
        interpolation = interpolate_total_variance(
            group,
            target_dte,
            allow_extrapolate=allow_term_extrapolate,
            extrapolate_mode=term_extrapolate_mode,
            annual_days=annual_days,
        )
        if pd.isna(interpolation["surface_iv"]):
            continue
        rows.append(
            {
                "target_dte": float(target_dte),
                "log_moneyness": float(grid_k),
                "total_variance": interpolation["total_variance"],
                "surface_iv": interpolation["surface_iv"],
                "method": interpolation["method"],
            }
        )
    return pd.DataFrame(rows).sort_values("log_moneyness")


def svi_total_variance(k, a, b, rho, m, sigma):
    k = np.asarray(k, dtype=float)
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma**2))


def fit_svi_smile(smile_df, weights=None, annual_days=252):
    """Fit SVI total variance for one fixed-tenor smile.

    SciPy is optional for the project; this function raises ImportError when it
    is unavailable so callers can fall back to the raw interpolated smile.
    """
    try:
        from scipy.optimize import least_squares
    except ImportError as exc:
        raise ImportError("fit_svi_smile requires scipy") from exc

    df = smile_df.dropna(subset=["log_moneyness", "total_variance"]).copy()
    if len(df) < 5:
        raise ValueError("SVI fit requires at least five smile points")

    k = df["log_moneyness"].to_numpy(dtype=float)
    w = df["total_variance"].to_numpy(dtype=float)
    if weights is None:
        sqrt_weights = np.ones_like(w)
    else:
        sqrt_weights = np.sqrt(np.asarray(weights, dtype=float))

    initial = np.array([max(w.min() * 0.5, 1e-6), 0.1, 0.0, 0.0, 0.1])
    lower = np.array([0.0, 1e-8, -0.999, -2.0, 1e-5])
    upper = np.array([np.inf, np.inf, 0.999, 2.0, 5.0])

    def residual(params):
        return (svi_total_variance(k, *params) - w) * sqrt_weights

    result = least_squares(
        residual,
        initial,
        bounds=(lower, upper),
        max_nfev=5000,
    )
    params = {
        "a": result.x[0],
        "b": result.x[1],
        "rho": result.x[2],
        "m": result.x[3],
        "sigma": result.x[4],
        "success": bool(result.success),
        "cost": float(result.cost),
    }
    fitted = df.copy()
    fitted["svi_total_variance"] = svi_total_variance(k, *result.x)
    fitted["svi_iv"] = np.sqrt(
        fitted["svi_total_variance"] / (fitted["target_dte"] / float(annual_days))
    )
    return params, fitted


def build_daily_surface(
    chain_df,
    config: SurfaceConfig | None = None,
    standard_dtes: tuple[int, ...] | None = None,
    k_grid=None,
    k_grid_mode="intersection",
    forward_col=None,
    allow_term_extrapolate=False,
    term_extrapolate_mode="linear",
    fit_svi=False,
):
    """Build fixed-tenor ATM IVs and smiles for one trading day."""
    config = config or SurfaceConfig()
    standard_dtes = standard_dtes or config.standard_dtes
    points = prepare_iv_points(
        chain_df,
        annual_days=config.annual_days,
        r=config.risk_free_rate,
        filters=config.filters,
        forward_col=forward_col,
    )

    atm_rows = []
    smiles = {}
    svi_fits = {}
    for dte in standard_dtes:
        atm_rows.append(
            fixed_tenor_atm_iv(
                points,
                dte,
                allow_extrapolate=allow_term_extrapolate,
                extrapolate_mode=term_extrapolate_mode,
                annual_days=config.annual_days,
            )
        )
        smile = build_fixed_tenor_smile(
            points,
            dte,
            k_grid=k_grid,
            k_grid_mode=k_grid_mode,
            allow_term_extrapolate=allow_term_extrapolate,
            term_extrapolate_mode=term_extrapolate_mode,
            annual_days=config.annual_days,
        )
        smiles[dte] = smile
        if fit_svi and not smile.empty:
            try:
                svi_fits[dte] = fit_svi_smile(smile, annual_days=config.annual_days)
            except (ImportError, ValueError):
                svi_fits[dte] = None

    return {
        "points": points,
        "atm": pd.DataFrame(atm_rows),
        "smiles": smiles,
        "svi": svi_fits,
    }


def surface_grid_from_smiles(smiles):
    rows = []
    for dte, smile in smiles.items():
        if smile is None or smile.empty:
            continue
        df = smile.copy()
        df["target_dte"] = float(dte)
        rows.append(df)
    if not rows:
        return pd.DataFrame(
            columns=["target_dte", "log_moneyness", "total_variance", "surface_iv"]
        )
    return pd.concat(rows, ignore_index=True).sort_values(
        ["target_dte", "log_moneyness"]
    )


def plot_vol_surface(
    surface,
    output_path,
    title=None,
    include_raw_points=True,
    raw_point_max_dte=None,
    invert_log_moneyness_axis=False,
    invert_dte_axis=False,
    show=False,
):
    """Plot fixed-tenor IV surface to a PNG file."""
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = surface_grid_from_smiles(surface.get("smiles", {}))
    if grid.empty:
        raise ValueError("vol surface grid is empty")

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    pivot = grid.pivot_table(
        index="target_dte",
        columns="log_moneyness",
        values="surface_iv",
    ).sort_index()
    x_values = pivot.columns.to_numpy(dtype=float)
    y_values = pivot.index.to_numpy(dtype=float)
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    z_grid = pivot.to_numpy(dtype=float)

    surface_plot = ax.plot_surface(
        x_grid,
        y_grid,
        z_grid,
        cmap="viridis",
        edgecolor="none",
        linewidth=0,
        antialiased=True,
        alpha=0.88,
    )
    fig.colorbar(surface_plot, ax=ax, shrink=0.65, pad=0.10, label="IV")

    if include_raw_points and "points" in surface:
        points = surface["points"]
        if points is not None and not points.empty:
            raw = points.dropna(subset=["log_moneyness", "dte", "surface_iv"])
            if raw_point_max_dte is not None:
                raw = raw[raw["dte"] <= raw_point_max_dte]
            if not raw.empty:
                ax.scatter(
                    raw["log_moneyness"],
                    raw["dte"],
                    raw["surface_iv"],
                    color="black",
                    s=10,
                    alpha=0.35,
                    label="Raw IV points",
                )

    ax.set_xlabel("log(K/F)")
    ax.set_ylabel("DTE")
    ax.set_zlabel("Implied Volatility")
    ax.set_zlim(bottom=0)
    if invert_log_moneyness_axis:
        ax.invert_xaxis()
    if invert_dte_axis:
        ax.invert_yaxis()
    ax.view_init(elev=24, azim=-135)
    if title:
        ax.set_title(title)
    if include_raw_points:
        ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return output_path


__all__ = [
    "SurfaceConfig",
    "SurfaceFilters",
    "black76_price",
    "implied_vol_black76",
    "prepare_iv_points",
    "collapse_call_put_points",
    "select_atm_term_points",
    "interpolate_total_variance",
    "fixed_tenor_atm_iv",
    "build_fixed_tenor_smile",
    "svi_total_variance",
    "fit_svi_smile",
    "build_daily_surface",
    "surface_grid_from_smiles",
    "plot_vol_surface",
]
