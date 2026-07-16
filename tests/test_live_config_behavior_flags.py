import core


def test_live_etf_products_use_atm_rebalance_instead_of_short_etf():
    for product in ("50etf", "300etf", "500etf", "kc50etf"):
        strategy = core.config.load_config(product).strategy
        assert strategy.enable_delta_hedge is True
        assert strategy.delta_hedge_tolerance_ratio == 0.0
        assert strategy.allow_etf_short_hedge is False
        assert strategy.enable_atm_straddle_rebalance is True


def test_live_etf_short_stop_loss_uses_negative_three_percent_aum():
    strategy = core.config.load_config("50etf").strategy
    assert strategy.short_stop_loss_enabled is True
    for product in ("50etf", "300etf", "500etf", "kc50etf"):
        strategy = core.config.load_config(product).strategy
        assert strategy.short_daily_loss_aum_threshold == -0.030


def test_live_etf_products_search_near_month_when_atm_volume_is_low():
    for product in ("50etf", "300etf", "500etf", "kc50etf"):
        assert (
            core.config.load_config(product).vol.atm_low_volume_search_near_month
            is True
        )


def test_live_etf_products_use_three_to_thirty_five_day_atm_window():
    for product in ("50etf", "300etf", "500etf", "kc50etf"):
        config = core.config.load_config(product)
        vol = config.vol
        assert vol.atm_target_dte_min == 3
        assert vol.atm_target_dte_max == 35
        assert config.strategy.roll_dte_threshold == vol.atm_target_dte_min
