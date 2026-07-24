from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from core import backtester, config, strategy
from core.backtest_strategies import create_strategy
from core.live import account, replay, signal_engine


PRODUCTS = ("50etf", "300etf", "500etf", "kc50etf")


def test_live_straddle_uses_all_four_live_product_configs_without_override():
    for product in PRODUCTS:
        runtime = config.load_config(product)
        plugin = create_strategy("live_straddle", runtime)

        assert plugin.config is runtime
        assert runtime.backtest.long_qty == 10
        assert runtime.backtest.short_qty == 10
        assert runtime.vol.atm_target_dte_min == 3
        assert runtime.vol.atm_target_dte_max == 35
        assert plugin.roll_dte_threshold == 3
        assert plugin.delta_hedge_tolerance_ratio == 0.0
        assert plugin.delta_residual_abs_tolerance == 5_000.0
        assert plugin.allow_etf_short_hedge is False
        assert plugin.enable_atm_straddle_rebalance is True
        assert plugin.force_liquidate_adjusted_options is True
        assert plugin.persist_model_entry_volume_baseline is False
        assert len(plugin.metadata()["effective_config_sha256"]) == 64


def test_live_straddle_signal_is_the_shared_live_signal_with_explicit_config():
    live_config = config.load_config("300etf")
    conflicting = replace(
        config.load_config("50etf"),
        strategy=replace(
            config.load_config("50etf").strategy,
            short_open_iv_threshold=0.99,
        ),
    )
    features = pd.DataFrame(
        {
            "atm_iv": [0.10, 0.30],
            "atm_iv_percentile": [0.1, 0.9],
            "signal_iv": [0.10, 0.30],
            "signal_iv_percentile": [0.1, 0.9],
            "yz_hv60": [0.08, 0.12],
        },
        index=pd.to_datetime(["2026-07-20", "2026-07-21"]),
    )
    plugin = create_strategy("live_straddle", live_config)

    with patch.object(strategy, "CONFIG", conflicting):
        expected = strategy.build_signals(features, config=live_config)
        actual = plugin.build_signals(features)

    pd.testing.assert_frame_equal(actual, expected)
    assert pd.isna(actual.iloc[0]["prev_signal_iv"])
    assert actual.iloc[1]["prev_signal_iv"] == 0.10


def test_live_signal_wrapper_delegates_to_explicit_context_without_writes():
    runtime = config.load_config("kc50etf")
    live_account = account.AccountState(product="kc50etf", cash=123.0)
    market = {"sentinel": object()}
    expected = {"plan_status": "NO_ACTION"}

    with (
        patch.object(signal_engine, "load_product_config", return_value=runtime),
        patch.object(signal_engine.account_store, "load_account", return_value=live_account),
        patch.object(signal_engine.portfolio_account, "shared_cash", return_value=456.0),
        patch.object(signal_engine, "_load_market_context", return_value=market),
        patch.object(
            signal_engine,
            "generate_signal_from_context",
            return_value=expected,
        ) as explicit,
        patch.object(signal_engine.account_store, "save_account") as save,
    ):
        result = signal_engine.generate_signal("kc50etf", quote_snapshot={})

    assert result is expected
    assert live_account.cash == 456.0
    save.assert_not_called()
    assert explicit.call_args.kwargs["live_account"] is live_account
    assert explicit.call_args.kwargs["market"] is market


def test_snapshot_discovery_deduplicates_same_immutable_quote(tmp_path):
    product = "kc50etf"
    signal_dir = tmp_path / "output" / product
    quote_dir = tmp_path / "quotes" / product / "quotes" / "20260722"
    signal_dir.mkdir(parents=True)
    quote_dir.mkdir(parents=True)
    (quote_dir / "145000_etf.parquet").touch()
    (quote_dir / "145000_option_chain.parquet").touch()
    payload = {
        "product": product,
        "date": "2026-07-22",
        "feature": {"close": 1.95},
        "account": {"product": product},
        "quote_snapshot": {
            "snapshot_stamp": "20260722_145000",
            "quote_date": "2026-07-22",
        },
    }
    for stamp in ("145010", "145020"):
        (signal_dir / f"20260722_{stamp}_signal.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    events = replay.discover_signal_events(
        product,
        output_root=tmp_path / "output",
        quote_root=tmp_path / "quotes",
    )

    assert len(events) == 1
    assert events[0].snapshot_stamp == "20260722_145000"


def test_replay_account_conversion_is_detached_from_source_payload():
    payload = {
        "product": "300etf",
        "cash": 1_000_000,
        "positions": {
            "long": None,
            "short": {
                "call_code": "CALL",
                "put_code": "PUT",
                "call_qty": 10,
                "put_qty": 10,
            },
        },
        "hedge": {"qty": 1000, "entry_price": 4.8, "margin": 4800},
    }

    detached = replay.account_from_dict(payload)
    detached.positions["short"]["call_qty"] = 99
    detached.hedge.qty = 0

    assert payload["positions"]["short"]["call_qty"] == 10
    assert payload["hedge"]["qty"] == 1000


def test_replay_effective_cooldown_state_is_advanced_without_touching_source():
    replay_account = account.AccountState(product="300etf")
    replay._apply_effective_strategy_state(
        replay_account,
        {
            "short_entry_cooldown_left": 2,
            "short_entry_cooldown_total_days": 3,
            "short_entry_cooldown_started_date": "2026-07-20",
        },
    )

    assert replay_account.strategy_state.short_entry_cooldown_left == 2
    assert replay_account.strategy_state.short_entry_cooldown_total_days == 3
    assert replay_account.strategy_state.short_entry_cooldown_started_date == "2026-07-20"


def test_standard_backtest_live_plugin_uses_absolute_5000_delta_deadband():
    runtime = config.load_config("kc50etf")
    plugin = create_strategy("live_straddle", runtime)
    engine = backtester.BacktestEngine.__new__(backtester.BacktestEngine)
    engine.config = {
        "enable_delta_hedge": True,
        "delta_hedge_tolerance_ratio": 0.0,
        "allow_etf_short_hedge": False,
    }
    engine.strategy_plugin = plugin
    engine.hedge_by_date = None
    state = backtester.BacktestState(
        cash=1_000_000.0,
        positions={
            "long": {
                "side": "long",
                "call_code": "CALL",
                "put_code": "PUT",
                "call_qty": 10,
                "put_qty": 10,
                "contract_multiplier": 10_000,
            },
            "short": None,
        },
    )
    date = pd.Timestamp("2026-07-22")
    day = engine._new_day(date, 1.9, {}, pd.DataFrame())
    atm = {"underlying_order_book_id": "588000.XSHG"}

    with (
        patch.object(backtester.vol_engine, "select_atm_from_chain", return_value=atm),
        patch.object(
            signal_engine,
            "_atm_straddle_shape_rebalance_item",
            return_value=None,
        ),
        patch.object(
            signal_engine,
            "_delta_hedge_capacity_reduction_item",
            return_value=None,
        ),
        patch.object(engine, "_execute_etf_target") as execute,
    ):
        inside = {**backtester.empty_greeks(), "delta": -4_999.0}
        engine._hedge_to(date, 1.9, state, day, inside)
        execute.assert_not_called()

        outside = {**backtester.empty_greeks(), "delta": -5_001.0}
        engine._hedge_to(date, 1.9, state, day, outside)

    assert execute.call_count == 1
    assert execute.call_args.args[5] == 5_000


def test_standard_backtest_adjusted_option_exit_closes_every_side_and_etf():
    runtime = config.load_config("50etf")
    plugin = create_strategy("live_straddle", runtime)
    engine = backtester.BacktestEngine.__new__(backtester.BacktestEngine)
    engine.config = {"etf_fee_rate": 0.0}
    engine.strategy_plugin = plugin
    state = backtester.BacktestState(
        cash=1_000_000.0,
        positions={
            "long": {"side": "long", "call_qty": 1, "put_qty": 1},
            "short": {"side": "short", "call_qty": 2, "put_qty": 2},
        },
        hedge_etf_qty=1_000,
    )
    day = engine._new_day(
        pd.Timestamp("2026-07-22"),
        2.8,
        {},
        pd.DataFrame(),
    )
    call_row = pd.Series({"mid": 0.1, "contract_multiplier": 10_000})
    put_row = pd.Series({"mid": 0.1, "contract_multiplier": 10_000})

    def close_trade(date, cash, position, call, put, trades, **kwargs):
        trades.append({"date": date, "type": "close", "side": position["side"]})
        return cash + 1.0, 0.0

    def close_etf(*args, **kwargs):
        state.hedge_etf_qty = 0.0

    with (
        patch(
            "core.live.signal_engine._dividend_adjustments_for_positions",
            return_value=[{"code": "ADJUSTED"}],
        ),
        patch.object(engine, "_get_position_rows", return_value=(call_row, put_row)),
        patch(
            "core.live.signal_engine._position_with_current_contract_terms",
            side_effect=lambda position, *args: (position, []),
        ),
        patch.object(strategy, "calc_position_greeks", return_value=backtester.empty_greeks()),
        patch.object(backtester.opt_position, "close_trade", side_effect=close_trade),
        patch.object(engine, "_add_new_option_fees"),
        patch.object(engine, "_execute_etf_target", side_effect=close_etf) as execute_etf,
    ):
        handled = engine._handle_adjusted_option_liquidation(day, state)

    assert handled is True
    assert state.positions == {"long": None, "short": None}
    assert state.hedge_etf_qty == 0.0
    assert day["skip_new_entry_by_side"] == {"long", "short"}
    assert day["defer_delta_hedge"] is True
    assert execute_etf.call_args.args[5] == 0.0


def test_replay_executes_only_one_netted_etf_target_and_charges_one_fee():
    runtime = config.load_config("kc50etf")
    replay_account = account.AccountState(product="kc50etf", cash=1_000_000.0)
    plan = {
        "spot": 2.0,
        "advice": [
            {
                "priority": "action",
                "action": "DELTA_HEDGE",
                "underlying_order_book_id": "588000.XSHG",
                "trade_etf_qty": 10_000,
                "current_hedge_qty": 0,
                "target_hedge_qty": 10_000,
                "estimated_price": 2.0,
            },
            {
                "priority": "action",
                "action": "FINAL_DELTA_HEDGE",
                "underlying_order_book_id": "588000.XSHG",
                "trade_etf_qty": -4_000,
                "current_hedge_qty": 10_000,
                "target_hedge_qty": 6_000,
                "estimated_price": 2.0,
            },
        ],
    }

    trades = replay.execute_plan_in_memory(
        replay_account,
        plan,
        pd.DataFrame(),
        runtime,
        pd.Timestamp("2026-07-22 14:50:00"),
    )

    assert len(trades) == 1
    assert trades[0]["trade_qty"] == 6_000
    assert trades[0]["target_qty"] == 6_000
    assert replay_account.hedge.qty == 6_000
    expected_fee = 6_000 * 2.0 * runtime.backtest.etf_fee_rate
    assert trades[0]["fee"] == expected_fee
    assert replay_account.cash == 1_000_000 - 12_000 - expected_fee


def test_standard_backtest_live_delta_plan_nets_intermediate_etf_targets_once():
    runtime = config.load_config("kc50etf")
    plugin = create_strategy("live_straddle", runtime)
    engine = backtester.BacktestEngine.__new__(backtester.BacktestEngine)
    engine.config = {
        "enable_delta_hedge": True,
        "allow_etf_short_hedge": False,
    }
    engine.strategy_plugin = plugin
    state = backtester.BacktestState(
        cash=1_000_000.0,
        positions={"long": {"side": "long"}, "short": None},
        hedge_etf_qty=10_000,
    )
    day = engine._new_day(
        pd.Timestamp("2026-07-22"),
        2.0,
        {},
        pd.DataFrame(),
    )
    day["delta_hedge_triggered_before_option_actions"] = True
    items = [
        {
            "priority": "action",
            "action": "FINAL_DELTA_HEDGE",
            "underlying_order_book_id": "588000.XSHG",
            "trade_etf_qty": -10_000,
            "current_hedge_qty": 10_000,
            "target_hedge_qty": 0,
            "estimated_price": 2.0,
        },
        {
            "priority": "action",
            "action": "FINAL_ATM_STRADDLE_DELTA_REBALANCE",
            "underlying_order_book_id": "588000.XSHG",
            "trade_etf_qty": 6_000,
            "current_hedge_qty": 0,
            "target_hedge_qty": 6_000,
            "estimated_price": 2.0,
        },
    ]

    with (
        patch.object(backtester.vol_engine, "select_atm_from_chain", return_value={}),
        patch.object(
            signal_engine,
            "_delta_hedge_plan",
            return_value=items,
        ) as delta_plan,
        patch.object(engine, "_update_day_aggregates"),
        patch.object(engine, "_execute_etf_target") as execute,
    ):
        engine._execute_live_delta_plan(
            day["date"],
            day["spot"],
            state,
            day,
            {**backtester.empty_greeks(), "delta": 10_000.0},
        )

    assert execute.call_count == 1
    assert execute.call_args.args[5] == 6_000
    assert delta_plan.call_args.kwargs["force_to_zero"] is True


def test_standard_backtest_live_roll_does_not_trade_temporary_etf_close():
    runtime = config.load_config("300etf")
    plugin = create_strategy("live_straddle", runtime)
    engine = backtester.BacktestEngine.__new__(backtester.BacktestEngine)
    engine.strategy_plugin = plugin
    day = {"date": pd.Timestamp("2026-07-22"), "spot": 4.8}
    state = backtester.BacktestState(cash=1_000_000.0, hedge_etf_qty=10_000)

    with patch.object(engine, "_execute_etf_target") as execute:
        engine._close_hedge_before_roll(day, state, "strike")

    execute.assert_not_called()
    assert state.hedge_etf_qty == 10_000


def test_standard_backtest_live_entry_state_matches_broker_volume_fields():
    runtime = config.load_config("50etf")
    engine = backtester.BacktestEngine.__new__(backtester.BacktestEngine)
    engine.strategy_plugin = create_strategy("live_straddle", runtime)
    position = {
        "entry_call_volume": 100,
        "entry_put_volume": 200,
        "entry_total_volume": 300,
    }

    engine._apply_entry_state_contract(position)

    assert position == {
        "entry_call_volume": None,
        "entry_put_volume": None,
        "entry_total_volume": None,
    }
