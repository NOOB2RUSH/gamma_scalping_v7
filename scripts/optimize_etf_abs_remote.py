from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import core
import optimize_params


PARAM_NAMES = [
    "strategy.long_open_iv_threshold",
    "strategy.long_close_iv_threshold",
    "strategy.short_open_iv_threshold",
    "strategy.short_close_iv_threshold",
    "strategy.short_open_pullback_iv_threshold",
    "backtest.long_qty",
    "backtest.short_qty",
    "strategy.short_stop_loss_enabled",
    "strategy.short_daily_loss_aum_threshold",
    "strategy.delta_hedge_tolerance_ratio",
    "strategy.short_volume_spike_multiplier",
    "strategy.roll_dte_threshold",
    "strategy.short_cooldown_after_long_iv_high_exit_days",
    "strategy.roll_cooldown_days",
]

FIXED_PARAMS = {
    "strategy.enable_long_straddle": True,
    "strategy.enable_short_straddle": True,
    "strategy.short_signal_mode": "absolute",
    "strategy.enable_delta_hedge": True,
    "strategy.short_volume_spike_exit_enabled": True,
}

COARSE_SPACE = {
    "strategy.long_open_iv_threshold": (0.080, 0.180, 0.005),
    "strategy.long_close_iv_threshold": (0.160, 0.380, 0.005),
    "strategy.short_open_iv_threshold": (0.140, 0.340, 0.005),
    "strategy.short_close_iv_threshold": (0.080, 0.240, 0.005),
    "strategy.short_open_pullback_iv_threshold": (0.200, 0.560, 0.010),
    "backtest.long_qty": (5, 80, 5),
    "backtest.short_qty": (5, 80, 5),
    "strategy.short_stop_loss_enabled": [False, True],
    "strategy.short_daily_loss_aum_threshold": (-0.030, -0.005, 0.001),
    "strategy.delta_hedge_tolerance_ratio": (0.050, 0.200, 0.010),
    "strategy.short_volume_spike_multiplier": (1.200, 2.500, 0.100),
    "strategy.roll_dte_threshold": (3, 12, 1),
    "strategy.short_cooldown_after_long_iv_high_exit_days": (0, 12, 1),
    "strategy.roll_cooldown_days": (0, 12, 1),
}

FINE_RADII = [
    {
        "strategy.long_open_iv_threshold": 0.020,
        "strategy.long_close_iv_threshold": 0.040,
        "strategy.short_open_iv_threshold": 0.030,
        "strategy.short_close_iv_threshold": 0.030,
        "strategy.short_open_pullback_iv_threshold": 0.080,
        "backtest.long_qty": 15,
        "backtest.short_qty": 15,
        "strategy.short_daily_loss_aum_threshold": 0.006,
        "strategy.delta_hedge_tolerance_ratio": 0.030,
        "strategy.short_volume_spike_multiplier": 0.300,
        "strategy.roll_dte_threshold": 3,
        "strategy.short_cooldown_after_long_iv_high_exit_days": 4,
        "strategy.roll_cooldown_days": 4,
    },
    {
        "strategy.long_open_iv_threshold": 0.010,
        "strategy.long_close_iv_threshold": 0.020,
        "strategy.short_open_iv_threshold": 0.015,
        "strategy.short_close_iv_threshold": 0.015,
        "strategy.short_open_pullback_iv_threshold": 0.040,
        "backtest.long_qty": 10,
        "backtest.short_qty": 10,
        "strategy.short_daily_loss_aum_threshold": 0.003,
        "strategy.delta_hedge_tolerance_ratio": 0.020,
        "strategy.short_volume_spike_multiplier": 0.200,
        "strategy.roll_dte_threshold": 2,
        "strategy.short_cooldown_after_long_iv_high_exit_days": 2,
        "strategy.roll_cooldown_days": 2,
    },
    {
        "strategy.long_open_iv_threshold": 0.005,
        "strategy.long_close_iv_threshold": 0.010,
        "strategy.short_open_iv_threshold": 0.010,
        "strategy.short_close_iv_threshold": 0.010,
        "strategy.short_open_pullback_iv_threshold": 0.020,
        "backtest.long_qty": 5,
        "backtest.short_qty": 5,
        "strategy.short_daily_loss_aum_threshold": 0.001,
        "strategy.delta_hedge_tolerance_ratio": 0.010,
        "strategy.short_volume_spike_multiplier": 0.100,
        "strategy.roll_dte_threshold": 1,
        "strategy.short_cooldown_after_long_iv_high_exit_days": 1,
        "strategy.roll_cooldown_days": 1,
    },
]

FLOAT_BOUNDS = {}
INT_BOUNDS = {}
for _key, _spec in COARSE_SPACE.items():
    if not isinstance(_spec, tuple):
        continue
    _low, _high, _step = _spec
    if isinstance(_low, float):
        FLOAT_BOUNDS[_key] = _spec
    elif isinstance(_low, int):
        INT_BOUNDS[_key] = _spec

WORKER_BASE_CONFIG = None
WORKER_ETF_BY_DATE = None
WORKER_OPT_BY_DATE = None
WORKER_HEDGE_BY_DATE = None
WORKER_TRADING_CALENDAR = None
WORKER_ENRICHED_OPT_BY_DATE = None
WORKER_FEATURES = None


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Random coarse scan plus progressive fine scan for ETF absolute IV "
            "thresholds, quantities, stop loss, and cooldowns."
        )
    )
    parser.add_argument("--product", default="300etf", choices=core.config.available_products())
    parser.add_argument("--stage", choices=["coarse", "fine", "all"], default="all")
    parser.add_argument("--objective", choices=["nav_drawdown", "nav_sharpe", "final_nav", "sharpe"], default="nav_drawdown")
    parser.add_argument("--coarse-samples", type=int, default=1500)
    parser.add_argument("--fine-samples-per-round", type=int, default=1200)
    parser.add_argument("--fine-rounds", type=int, default=3)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--min-held-days", type=int, default=120)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--seed", type=int, default=20260529)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--coarse-result", default=None)
    parser.add_argument(
        "--param-table",
        default=None,
        help="CSV containing exact PARAM_NAMES columns to use for the coarse scan.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-warm-cache", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=1)
    return parser.parse_args()


def timestamp():
    return pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")


def make_output_dir(args):
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("output") / "backtest" / (
            f"optimize_{args.product}_abs_remote_{timestamp()}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def snap_float(value, step):
    return round(round(value / step) * step, 6)


def random_from_range(rng, low, high, step):
    count = int(round((high - low) / step))
    return low + step * rng.randint(0, count)


def cast_param_value(key, value):
    if key in INT_BOUNDS:
        return int(value)
    if key == "strategy.short_stop_loss_enabled":
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y"}
        return bool(value)
    if key in FLOAT_BOUNDS:
        return round(float(value), 6)
    return value


def normalize_param_set(param_set):
    normalized = {**FIXED_PARAMS}
    for key in PARAM_NAMES:
        normalized[key] = cast_param_value(key, param_set[key])
    return normalized


def is_valid_param_set(param_set):
    if not optimize_params.is_valid_param_set(param_set):
        return False
    long_open = param_set["strategy.long_open_iv_threshold"]
    long_close = param_set["strategy.long_close_iv_threshold"]
    short_open = param_set["strategy.short_open_iv_threshold"]
    short_close = param_set["strategy.short_close_iv_threshold"]
    short_pullback = param_set["strategy.short_open_pullback_iv_threshold"]
    if long_close - long_open < 0.020:
        return False
    if short_open - short_close < 0.020:
        return False
    if short_pullback - short_open < 0.020:
        return False
    if not param_set["strategy.short_stop_loss_enabled"]:
        return True
    return param_set["strategy.short_daily_loss_aum_threshold"] < 0


def param_identity(param_set):
    payload = {key: param_set[key] for key in sorted(param_set)}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def random_coarse_param_set(rng):
    raw = {}
    for key, spec in COARSE_SPACE.items():
        if isinstance(spec, list):
            raw[key] = rng.choice(spec)
        else:
            raw[key] = random_from_range(rng, *spec)
    return normalize_param_set(raw)


def generate_unique(generator, count, seed, max_attempt_factor=80):
    rng = random.Random(seed)
    param_sets = []
    seen = set()
    max_attempts = max(count * max_attempt_factor, 1000)
    attempts = 0
    while len(param_sets) < count and attempts < max_attempts:
        attempts += 1
        param_set = generator(rng)
        if not is_valid_param_set(param_set):
            continue
        identity = param_identity(param_set)
        if identity in seen:
            continue
        seen.add(identity)
        param_sets.append(param_set)
    if len(param_sets) < count:
        print(
            f"Only generated {len(param_sets)} valid unique parameter sets "
            f"after {attempts} attempts.",
            flush=True,
        )
    return param_sets


def generate_coarse_param_sets(count, seed):
    return generate_unique(random_coarse_param_set, count, seed)


def load_param_table(path):
    frame = pd.read_csv(path)
    missing = [key for key in PARAM_NAMES if key not in frame.columns]
    if missing:
        raise ValueError(f"Parameter table missing columns: {missing}")
    param_sets = []
    seen = set()
    for _, row in frame.iterrows():
        raw = {key: row[key] for key in PARAM_NAMES}
        param_set = normalize_param_set(raw)
        if not is_valid_param_set(param_set):
            continue
        identity = param_identity(param_set)
        if identity in seen:
            continue
        seen.add(identity)
        param_sets.append(param_set)
    if not param_sets:
        raise ValueError(f"No valid parameter rows in table: {path}")
    return param_sets


def clamp(value, low, high):
    return max(low, min(high, value))


def sample_near_center(rng, center, radius_by_key):
    raw = {}
    for key in PARAM_NAMES:
        if key == "strategy.short_stop_loss_enabled":
            if rng.random() < 0.85:
                raw[key] = bool(center[key])
            else:
                raw[key] = not bool(center[key])
            continue

        if key in FLOAT_BOUNDS:
            low, high, step = FLOAT_BOUNDS[key]
            radius = radius_by_key[key]
            value = rng.uniform(float(center[key]) - radius, float(center[key]) + radius)
            raw[key] = snap_float(clamp(value, low, high), step)
            continue

        low, high, step = INT_BOUNDS[key]
        radius = int(radius_by_key[key])
        value = int(center[key]) + rng.randrange(-radius, radius + 1, step)
        raw[key] = int(clamp(value, low, high))

    return normalize_param_set(raw)


def generate_fine_param_sets(seed_rows, count, seed, round_index):
    if seed_rows.empty:
        raise ValueError("No seed rows available for fine scan.")
    radius_by_key = FINE_RADII[min(round_index, len(FINE_RADII) - 1)]
    centers = []
    for _, row in seed_rows.iterrows():
        center = {}
        for key in PARAM_NAMES:
            if key not in row:
                raise ValueError(f"Fine seed is missing column: {key}")
            center[key] = cast_param_value(key, row[key])
        centers.append(center)

    def generator(rng):
        center = rng.choice(centers)
        return sample_near_center(rng, center, radius_by_key)

    return generate_unique(generator, count, seed)


def sort_results(df, objective, initial_cash):
    df = optimize_params.ensure_nav_drawdown_score(df)
    df = optimize_params.ensure_nav_sharpe_score(df)
    df = optimize_params.ensure_cash_usage_metrics(df, initial_cash)
    return optimize_params.sort_result_df(df, objective)


def successful_rows(df, min_held_days):
    if df.empty:
        return df
    if "error" in df.columns:
        df = df[df["error"].fillna("").eq("")]
    return optimize_params.filter_by_min_held_days(df, min_held_days)


def select_seed_rows(result_paths, objective, top_n, min_held_days, initial_cash):
    frames = []
    for path in result_paths:
        if path and Path(path).exists():
            frames.append(pd.read_csv(path))
    if not frames:
        raise ValueError("No result CSV found for fine scan seeds.")
    df = pd.concat(frames, ignore_index=True)
    df = successful_rows(df, min_held_days)
    df = sort_results(df, objective, initial_cash)
    return df.head(top_n)


def load_product_config(product, start=None, end=None):
    product_config = core.config.load_config(product)
    updates = {
        key: value
        for key, value in {"start": start, "end": end}.items()
        if value is not None
    }
    if updates:
        product_config = replace(
            product_config,
            backtest=replace(product_config.backtest, **updates),
        )
    return optimize_params.apply_param_set(product_config, FIXED_PARAMS)


def load_market_data(start, end):
    etf_by_date = core.data_loader.load_etf_series(start, end)
    opt_by_date = core.data_loader.load_opt_series(start, end)
    hedge_by_date = core.data_loader.load_hedge_series(start, end)
    opt_by_date = core.data_loader.attach_underlying_prices(opt_by_date, hedge_by_date)
    trading_calendar = core.data_loader.load_etf_trading_calendar()
    return etf_by_date, opt_by_date, hedge_by_date, trading_calendar


def warm_base_cache(base_config, start, end):
    print("Warming base enriched-chain and feature caches...", flush=True)
    etf_by_date, opt_by_date, hedge_by_date, trading_calendar = load_market_data(start, end)
    enriched = core.cache.get_enriched_option_chains(
        etf_by_date,
        opt_by_date,
        trading_calendar,
        start,
        end,
    )
    core.cache.get_vol_features(
        etf_by_date,
        opt_by_date,
        trading_calendar,
        enriched,
        start,
        end,
    )
    print(
        f"Cache warm complete: {base_config.data.product}, {start} - {end}, "
        f"ETF days={len(etf_by_date)}, option days={len(opt_by_date)}",
        flush=True,
    )


def worker_init(product, start, end):
    global WORKER_BASE_CONFIG
    global WORKER_ETF_BY_DATE
    global WORKER_OPT_BY_DATE
    global WORKER_HEDGE_BY_DATE
    global WORKER_TRADING_CALENDAR
    global WORKER_ENRICHED_OPT_BY_DATE
    global WORKER_FEATURES

    WORKER_BASE_CONFIG = load_product_config(product, start, end)
    WORKER_ETF_BY_DATE, WORKER_OPT_BY_DATE, WORKER_HEDGE_BY_DATE, WORKER_TRADING_CALENDAR = (
        load_market_data(start, end)
    )
    WORKER_ENRICHED_OPT_BY_DATE = core.cache.get_enriched_option_chains(
        WORKER_ETF_BY_DATE,
        WORKER_OPT_BY_DATE,
        WORKER_TRADING_CALENDAR,
        start,
        end,
    )
    WORKER_FEATURES = core.cache.get_vol_features(
        WORKER_ETF_BY_DATE,
        WORKER_OPT_BY_DATE,
        WORKER_TRADING_CALENDAR,
        WORKER_ENRICHED_OPT_BY_DATE,
        start,
        end,
    )


def worker_run(param_set):
    try:
        config = optimize_params.apply_param_set(WORKER_BASE_CONFIG, param_set)
        signals = core.strategy.build_signals(WORKER_FEATURES)
        daily_pnl, trades = core.backtester.run_backtest(
            WORKER_ETF_BY_DATE,
            WORKER_OPT_BY_DATE,
            signals,
            trading_calendar=WORKER_TRADING_CALENDAR,
            enriched_opt_by_date=WORKER_ENRICHED_OPT_BY_DATE,
            hedge_by_date=WORKER_HEDGE_BY_DATE,
        )
        row = optimize_params.summarize_result(
            param_set,
            daily_pnl,
            trades,
            config.backtest.initial_cash,
        )
        row["error"] = ""
        return row
    except Exception as exc:
        row = {**param_set, "error": repr(exc)}
        return row


def read_done_identities(output_path):
    if not output_path.exists():
        return set(), []
    df = pd.read_csv(output_path)
    rows = df.to_dict("records")
    done = set()
    for row in rows:
        if all(key in row for key in PARAM_NAMES):
            param_set = {key: cast_param_value(key, row[key]) for key in PARAM_NAMES}
            param_set = normalize_param_set(param_set)
            done.add(param_identity(param_set))
    return done, rows


def write_best_snippet(result_df, output_dir, stage_name, objective, product):
    if result_df.empty:
        return
    best = result_df.iloc[0].to_dict()
    typed_best = {key: cast_param_value(key, best[key]) for key in PARAM_NAMES}
    lines = [
        f"# Best {product} params from {stage_name}, objective={objective}",
        f"# final_nav={best.get('final_nav')}",
        f"# total_return={best.get('total_return')}",
        f"# max_drawdown_pct={best.get('max_drawdown_pct')}",
        f"long_open_iv_threshold={typed_best['strategy.long_open_iv_threshold']:.3f},",
        f"long_close_iv_threshold={typed_best['strategy.long_close_iv_threshold']:.3f},",
        f"short_signal_mode=\"absolute\",",
        f"short_open_iv_threshold={typed_best['strategy.short_open_iv_threshold']:.3f},",
        f"short_close_iv_threshold={typed_best['strategy.short_close_iv_threshold']:.3f},",
        f"short_open_pullback_iv_threshold={typed_best['strategy.short_open_pullback_iv_threshold']:.3f},",
        f"short_stop_loss_enabled={typed_best['strategy.short_stop_loss_enabled']},",
        f"short_daily_loss_aum_threshold={typed_best['strategy.short_daily_loss_aum_threshold']:.3f},",
        f"delta_hedge_tolerance_ratio={typed_best['strategy.delta_hedge_tolerance_ratio']:.3f},",
        f"short_volume_spike_multiplier={typed_best['strategy.short_volume_spike_multiplier']:.3f},",
        f"roll_dte_threshold={typed_best['strategy.roll_dte_threshold']},",
        f"short_cooldown_after_long_iv_high_exit_days={typed_best['strategy.short_cooldown_after_long_iv_high_exit_days']},",
        f"roll_cooldown_days={typed_best['strategy.roll_cooldown_days']},",
        "",
        f"long_qty={typed_best['backtest.long_qty']},",
        f"short_qty={typed_best['backtest.short_qty']},",
    ]
    (output_dir / f"best_config_snippet_{stage_name}.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def run_stage(stage_name, param_sets, output_path, args, base_config, output_dir):
    print(f"\n===== {stage_name} =====", flush=True)
    print(f"Output: {output_path}", flush=True)
    print(f"Workers: {args.workers}", flush=True)
    print(f"Requested parameter sets: {len(param_sets)}", flush=True)

    existing_rows = []
    done = set()
    if args.resume:
        done, existing_rows = read_done_identities(output_path)
        param_sets = [
            param_set
            for param_set in param_sets
            if param_identity(param_set) not in done
        ]
        print(f"Resume: {len(done)} done, {len(param_sets)} remaining.", flush=True)

    rows = list(existing_rows)
    start_time = time.perf_counter()
    completed = 0

    if args.workers <= 1:
        worker_init(args.product, base_config.backtest.start, base_config.backtest.end)
        for param_set in param_sets:
            result_row = worker_run(param_set)
            rows.append(result_row)
            completed += 1
            rows = flush_rows(rows, output_path, args.objective, base_config, output_dir, stage_name)
            print_progress(completed, len(param_sets), start_time, result_row)
    else:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=worker_init,
            initargs=(args.product, base_config.backtest.start, base_config.backtest.end),
        ) as pool:
            futures = [
                pool.submit(worker_run, param_set)
                for param_set in param_sets
            ]
            for future in as_completed(futures):
                result_row = future.result()
                rows.append(result_row)
                completed += 1
                if completed % max(args.chunk_size, 1) == 0 or completed == len(futures):
                    rows = flush_rows(rows, output_path, args.objective, base_config, output_dir, stage_name)
                    print_progress(completed, len(param_sets), start_time, result_row)

    rows = flush_rows(rows, output_path, args.objective, base_config, output_dir, stage_name)
    result_df = sort_results(
        pd.DataFrame(rows),
        args.objective,
        base_config.backtest.initial_cash,
    )
    print_top(result_df, stage_name)
    return output_path


def flush_rows(rows, output_path, objective, base_config, output_dir, stage_name):
    result_df = pd.DataFrame(rows)
    result_df = sort_results(result_df, objective, base_config.backtest.initial_cash)
    result_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    write_best_snippet(
        result_df[result_df["error"].fillna("").eq("")],
        output_dir,
        stage_name,
        objective,
        base_config.data.product,
    )
    return result_df.to_dict("records")


def print_progress(completed, total, start_time, row):
    elapsed = time.perf_counter() - start_time
    avg = elapsed / max(completed, 1)
    eta_min = avg * max(total - completed, 0) / 60
    if row.get("error"):
        status = f"error={row['error']}"
    else:
        status = (
            f"nav={row.get('final_nav', math.nan):,.2f}, "
            f"ret={row.get('total_return', math.nan):.2%}, "
            f"dd={row.get('max_drawdown_pct', math.nan):.2%}, "
            f"sharpe={row.get('sharpe_ratio', math.nan):.2f}"
        )
    print(f"[{completed}/{total}] {status}, eta={eta_min:.1f}m", flush=True)


def print_top(result_df, stage_name):
    display_cols = PARAM_NAMES + [
        "final_nav",
        "total_return",
        "max_drawdown_pct",
        "nav_drawdown_score",
        "sharpe_ratio",
        "cash_negative_days",
        "open_count",
        "roll_count",
        "held_days",
        "long_held_days",
        "short_held_days",
        "error",
    ]
    display_cols = [col for col in display_cols if col in result_df.columns]
    print(f"\n===== {stage_name} top 10 =====", flush=True)
    print(result_df[display_cols].head(10).to_string(index=False), flush=True)


def write_manifest(args, output_dir, base_config):
    manifest = {
        "args": vars(args),
        "product": base_config.data.product,
        "start": base_config.backtest.start,
        "end": base_config.backtest.end,
        "fixed_params": FIXED_PARAMS,
        "param_names": PARAM_NAMES,
        "coarse_space": COARSE_SPACE,
        "fine_radii": FINE_RADII,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def main():
    args = parse_args()
    output_dir = make_output_dir(args)
    base_config = load_product_config(args.product, args.start, args.end)
    write_manifest(args, output_dir, base_config)

    print(
        f"ETF absolute-IV remote optimization script\n"
        f"Product={base_config.data.product}, "
        f"range={base_config.backtest.start}-{base_config.backtest.end}, "
        f"output_dir={output_dir}",
        flush=True,
    )

    if not args.no_warm_cache:
        warm_base_cache(base_config, base_config.backtest.start, base_config.backtest.end)

    result_paths = []
    if args.stage in {"coarse", "all"}:
        coarse_sets = (
            load_param_table(args.param_table)
            if args.param_table
            else generate_coarse_param_sets(args.coarse_samples, args.seed)
        )
        coarse_path = output_dir / "coarse.csv"
        result_paths.append(
            run_stage("coarse", coarse_sets, coarse_path, args, base_config, output_dir)
        )
    elif args.coarse_result:
        result_paths.append(Path(args.coarse_result))

    if args.stage in {"fine", "all"}:
        if args.stage == "fine" and not args.coarse_result:
            raise ValueError("--stage fine requires --coarse-result")
        for round_index in range(max(args.fine_rounds, 1)):
            seed_rows = select_seed_rows(
                result_paths,
                args.objective,
                args.top_n,
                args.min_held_days,
                base_config.backtest.initial_cash,
            )
            fine_sets = generate_fine_param_sets(
                seed_rows,
                args.fine_samples_per_round,
                args.seed + 1000 + round_index,
                round_index,
            )
            fine_path = output_dir / f"fine_round{round_index + 1}.csv"
            result_paths.append(
                run_stage(
                    f"fine_round{round_index + 1}",
                    fine_sets,
                    fine_path,
                    args,
                    base_config,
                    output_dir,
                )
            )

    print("\nDone. Result files:", flush=True)
    for path in result_paths:
        print(f"  {path}", flush=True)
    print(f"Best snippets: {output_dir / 'best_config_snippet_*.txt'}", flush=True)


if __name__ == "__main__":
    main()
