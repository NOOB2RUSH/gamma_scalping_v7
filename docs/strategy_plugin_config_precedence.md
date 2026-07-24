# 策略插件配置优先级

回测配置分为两层：

1. `core/configs/config_<product>.py`：品种公共配置，供数据、live 和没有专属覆盖的回测插件使用。
2. `core/backtest_strategies/configs/<strategy_id>.py`：策略插件按品种声明的稀疏覆盖。

最终运行配置按以下顺序合并，右侧优先级更高：

```text
品种公共配置 < 策略插件专属配置 < 命令行临时覆盖
```

插件专属配置只需要写与公共配置不同的字段。没有写出的字段自动继承公共配置；显式写出的 `None` 也会被视为有效覆盖值。

当前 `original_atm_iv_straddle` 的专属配置位于：

```text
core/backtest_strategies/configs/original_atm_iv_straddle.py
```

当前开发策略 `dynamic_atm_iv_straddle` 使用一份独立复制的配置：

```text
core/backtest_strategies/configs/dynamic_atm_iv_straddle.py
```

它目前只在原始配置基础上覆盖四个品种的固定多空目标张数：50ETF `35/35`、300ETF `20/20`、500ETF `10/10`、科创50ETF `40/40`。其余策略和波动率参数必须与 `original_atm_iv_straddle` 保持一致，后续功能改进再只修改该开发策略的专属配置。

例如300ETF原始策略只覆盖自己的Long开关、绝对IV阈值、对冲行为、目标DTE和展期阈值；初始资金、张数、手续费、数据路径、合约乘数等继续继承300ETF公共配置。

主回测入口 `run.py` 会自动完成合并。研究脚本若需要同一行为，应显式调用：

```python
from core import config
from core.backtest_strategies import resolve_strategy_config

common = config.load_config("300etf")
effective = resolve_strategy_config(common, "original_atm_iv_straddle")
```

随后可以通过 `dataclasses.replace` 对 `effective` 应用本次研究参数。不要把调优后的插件参数写回品种公共配置。

每次正式回测保存的 `runtime_config.json` 是插件构造后的最终有效配置；`strategy_metadata.json` 保存策略身份和行为说明。

`live_straddle` 不注册独立参数覆盖模块，默认直接继承对应品种公共配置，以保证当前 live 和历史镜像读取同一参数源。日期与初始资金仍可作为回测场景覆盖；若命令行覆盖 IV、DTE 等策略字段，该次结果不再是默认 live 参数口径，必须以 `runtime_config.json` 和 `effective_config_sha256` 为准。
