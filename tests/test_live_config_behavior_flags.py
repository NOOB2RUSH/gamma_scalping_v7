import core


def test_live_etf_products_use_option_hedges_instead_of_short_etf():
    for product in ("50etf", "300etf", "500etf", "kc50etf"):
        strategy = core.config.load_config(product).strategy
        assert strategy.enable_delta_hedge is True
        assert strategy.allow_etf_short_hedge is False
        assert strategy.enable_option_delta_hedge is True
        assert strategy.option_delta_hedge_combination_enabled is True


def test_live_etf_short_stop_loss_uses_negative_one_point_five_percent_aum():
    strategy = core.config.load_config("50etf").strategy
    assert strategy.short_stop_loss_enabled is True
    for product in ("50etf", "300etf", "500etf", "kc50etf"):
        strategy = core.config.load_config(product).strategy
        assert strategy.short_daily_loss_aum_threshold == -0.015


def test_live_etf_products_search_near_month_when_atm_volume_is_low():
    for product in ("50etf", "300etf", "500etf", "kc50etf"):
        assert (
            core.config.load_config(product).vol.atm_low_volume_search_near_month
            is True
        )
