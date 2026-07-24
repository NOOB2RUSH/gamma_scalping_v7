# 回测策略插件架构

## 1. 目标与边界

本架构用于把“策略决策”与“回测执行、账户和数据基础设施”分离，使不同策略可以复用同一套历史数据、特征缓存、成交模拟、资金管理和报告系统。

当前边界如下：

- 仅用于历史回测，不修改、不依赖 `core/live/`；
- 每次回测只选择一个策略插件；
- 每次只回测一个品种，使用该品种独立的初始现金和账户状态；
- 成交继续使用回测引擎提供的理想日线收盘价；
- 暂不支持多策略共享资金、策略间净额、信号仲裁和组合级风险预算；
- 旧插件 `iv_straddle_v1` 是早期 live IV 决策的回测侧副本，不会自动同步；需要当前 live 行为时使用共享实现的 `live_straddle`。

插件只表达“想做什么”，不直接改现金、持仓或成交记录。所有交易状态变更必须通过 `BacktestEngine` 完成。

## 2. 总体分层

```text
品种配置 + 历史数据
        ↓
统一数据加载 / 期权链 enrich / 波动率特征缓存
        ↓ features_df
单个策略插件：生成信号、目标仓位、平仓原因和策略状态
        ↓ signals_df + strategy hooks
BacktestEngine：逐日估值、开平仓、roll、调仓、资金与风控执行
        ↓
ETF Delta 对冲、手续费、保证金、净值与 Greeks PnL
        ↓
日度报告、成交记录、运行配置、策略元数据
```

主要目录和文件：

| 位置 | 职责 |
|---|---|
| `core/backtest_strategies/base.py` | 定义所有回测策略必须遵守的接口 |
| `core/backtest_strategies/__init__.py` | 策略注册表、可用策略 ID 和工厂函数 |
| `core/backtest_strategies/original_atm_iv_straddle.py` | 原始 ATM IV 跨式基准策略 |
| `core/backtest_strategies/dynamic_atm_iv_straddle.py` | 从原始策略复制、供后续动态仓位改进的独立策略 |
| `core/backtest_strategies/iv_straddle_v1.py` | 静态仓位 IV 跨式插件 |
| `core/backtest_strategies/dynamic_position_straddle.py` | 动态仓位跨式插件 |
| `core/backtest_strategies/dynamic_position_config.py` | 动态仓位插件专属参数 |
| `core/backtester.py` | 与具体策略无关的账户、交易和逐日执行器 |
| `core/cache.py`、`core/vol_engine.py` | 共享数据 enrich、IV/Greeks 和特征缓存 |
| `run.py` | 命令行入口、选择插件、组织回测并保存结果 |

## 3. 职责划分

### 3.1 策略插件负责

- 将统一特征表转换为完整信号表；
- 决定多空跨式是否允许开仓；
- 决定新开仓、已有仓位和 roll 后的目标对数；
- 返回策略平仓原因；
- 定义短仓开仓 regime，并保证 roll 后能够继承；
- 定义策略专属状态，例如冲击后等待回落；
- 声明是否启用共享的 Delta 对冲、ATM 腿再平衡等能力；
- 返回可序列化的策略元数据，确保结果可复现。

### 3.2 回测引擎负责

- 按交易日推进账户状态并读取当日期权链；
- 解析当前持仓合约、选择 ATM 合约和 roll 目标；
- 把插件给出的信号和目标仓位转换成实际模拟成交；
- 按理想收盘价执行开仓、平仓、roll 和等量增减仓；
- 维护现金、最低现金储备、期权保证金、ETF 保证金和手续费；
- 执行共享的 ATM 跨式腿再平衡与 ETF Delta 对冲算法；
- 处理缺失期权链、缺失持仓合约和到期结算等数据异常；
- 记录日度净值、持仓、成交、Greeks PnL 和数据预警。

### 3.3 共享配置与数据层负责

- `core/configs/config_<product>.py` 提供品种数据目录、合约乘数、交易费用和通用策略参数；
- enrich 和特征缓存只计算市场事实，不包含某个插件的持仓决策；
- 同一次回测中，所有插件读取相同口径的 `features_df` 和 enriched option chain；
- 插件可以增加派生信号列，但不得回写或污染共享缓存对象。

## 4. 策略插件契约

所有插件继承 `BacktestStrategy`。接口返回的是决策，不是订单。

| 接口 | 含义 | 默认行为 |
|---|---|---|
| `build_signals(features_df)` | 生成回测逐日使用的完整信号表 | 必须实现 |
| `entry_target_qty(feature_row, max_qty, side)` | 新开仓时每腿目标张数；无开仓意图返回 0 | 必须实现 |
| `existing_position_target_qty(feature_row, side)` | 已有仓位的目标跨式对数 | 默认 `None`，即不做策略调仓 |
| `roll_target_qty(feature_row, max_qty, side, current_qty)` | roll 后新合约的目标对数 | 默认复用 `entry_target_qty` |
| `get_close_reason(feature_row, position_dte)` | 多头仓位平仓原因 | 必须实现 |
| `get_short_close_reason(feature_row, position_dte, position)` | 空头仓位平仓原因 | 必须实现 |
| `short_entry_regime(feature_row)` | 写入新短仓的开仓模式 | 必须实现 |
| `default_short_entry_regime` | 老仓位缺少 regime 时的回退值 | 必须实现 |
| `is_short_daily_loss_stop(daily_pnl, aum)` | 短仓单日止损判断 | 必须实现 |
| `has_short_volume_spike(position, call_row, put_row)` | 短仓成交量放大退出判断 | 必须实现 |
| `roll_dte_threshold` | 共享 roll 执行器使用的 DTE 阈值 | 必须实现 |
| `short_cooldown_after_long_iv_high_exit_days` | 多头高 IV 退出后的短仓冷却期 | 必须实现 |
| `enable_delta_hedge` | 是否调用共享 ETF Delta 对冲器 | 必须实现 |
| `delta_hedge_tolerance_ratio` | 共享对冲器使用的标准化 Delta 容忍度 | 必须实现 |
| `allow_etf_short_hedge` | ETF 对冲仓是否可以为负 | 必须实现 |
| `enable_atm_straddle_rebalance` | 是否调用共享 ATM 腿再平衡算法 | 必须实现 |
| `metadata()` | 写入结果目录的策略身份和参数 | 默认只返回 `strategy_id` |

几个容易混淆的返回值：

- `entry_target_qty == 0`：当前不允许开仓；
- `existing_position_target_qty is None`：已有仓位保持策略对数不变；
- `existing_position_target_qty == 当前策略对数`：目标没有变化，不成交；
- `roll_target_qty` 与 `entry_target_qty` 分开，是为了允许等待状态下“禁止新增，但已有仓位仍按原对数正常 roll”。

`strategy_pair_qty` 是策略仓位阶梯的基准对数。ATM 腿再平衡可能使 call/put 实际张数不相等；动态调仓在两腿上等量增减，因此保留已经形成的两腿张数差。

## 5. 注册与选择机制

每个策略使用稳定且唯一的 `strategy_id`。当前注册表位于 `core/backtest_strategies/__init__.py`：

```python
_STRATEGY_TYPES = {
    "dynamic_atm_iv_straddle": DynamicAtmIvStraddleStrategy,
    "dynamic_position_straddle": DynamicPositionStraddleStrategy,
    "iv_straddle_v1": IvStraddleV1Strategy,
    "live_straddle": LiveStraddleStrategy,
    "original_atm_iv_straddle": OriginalAtmIvStraddleStrategy,
}
```

`create_strategy(strategy_id, config)` 根据 ID 创建本次回测唯一的插件实例。实例在整个回测期间保持不变，因此插件可以用信号列或实例状态表达跨日行为，但必须保证逐日计算不使用未来数据。

`run.py` 的 `--strategy` 参数直接来自注册表；未注册的 ID 会在启动阶段失败，不会静默回退。若底层程序调用 `run_backtest()` 时没有传插件，则为了兼容旧调用默认使用 `iv_straddle_v1`。

## 6. 一次回测的运行顺序

### 6.1 启动阶段

1. `run.py` 读取 `--product` 对应的品种配置，并应用日期、初始资金等命令行覆盖；
2. 同步运行时配置到现有回测模块；
3. 加载 ETF、对冲标的和期权链；
4. 读取或构建 enriched option chain 与统一波动率特征；
5. 用 `create_strategy()` 创建指定插件；
6. 调用一次 `plugin.build_signals(features)`，得到完整逐日信号表；
7. 把同一个插件实例和信号表交给 `BacktestEngine`。

### 6.2 单个交易日

回测引擎按以下优先级处理：

1. 使用当日收盘价和期权链给已有仓位估值；
2. 处理数据缺失、保证金或现金容量等账户级约束；
3. 对已有仓位依次检查：
   - 成交量放大退出；
   - 短仓单日止损；
   - 插件返回的 IV/到期等平仓原因；
   - 通用 DTE 或行权价偏离 roll；
   - 插件返回的已有仓位目标对数；
   - 若均未触发则继续持有；
4. 只有账户当前没有任何期权仓位时，才根据多头、空头开仓信号尝试开仓；
5. 执行共享 ATM 腿再平衡和最终 ETF Delta 对冲；
6. 更新现金、保证金、持仓末值和策略状态；
7. 记录日度结果与成交流水。

平仓优先于 roll。某日一旦触发平仓，该方向会加入当日禁止重开集合，不会在同一收盘价立即平仓再开仓。

当前引擎虽然保存 `long`、`short` 两个仓位槽位，但开仓前要求账户不存在任何期权仓位，因此一次回测同一时点只持有一个方向的策略仓位。

### 6.3 时间和成交口径

- 信号使用交易日 `t` 收盘时可见的特征；
- 与昨日比较时必须通过 `shift(1)` 等方式只读取前一交易日；
- 模拟成交也使用 `t` 日理想收盘价；
- 这是当前日线回测的统一假设，不代表 live 可以在完整收盘信息出现后仍以同一收盘价成交；
- 新插件不得引用 `t+1` 数据生成 `t` 日信号。

## 7. 共享能力与策略开关

插件可以声明是否使用共享能力，但不应复制其执行代码：

| 能力 | 插件决定 | 引擎执行 |
|---|---|---|
| ETF Delta 对冲 | 是否启用、容忍度、是否允许ETF空仓 | 目标数量、成交、手续费、保证金和流水 |
| ATM 腿再平衡 | 是否启用 | call/put 调整数量、成交和对冲衔接 |
| Roll | DTE阈值、roll目标对数 | 目标合约选择、旧仓平仓、新仓开仓 |
| 动态调仓 | 已有仓位目标对数 | 两腿等量增减、现金和保证金校验 |
| 短仓止损/成交量退出 | 是否触发 | 平仓成交和当日禁止重开 |
| 资金容量限制 | 无策略专属实现 | 共享现金储备、保证金和占用率控制 |

命令行参数 `--dynamic-position-control` 指的是共享账户资金占用控制，不是选择“动态仓位跨式策略”。选择动态策略必须显式使用：

```powershell
python run.py --product 300etf --strategy dynamic_position_straddle
```

两者可以同时启用，但职责不同：

```powershell
python run.py --product 300etf --strategy dynamic_position_straddle --dynamic-position-control
```

此时插件先给出期望对数，引擎再根据共享资金占用上限向下裁剪可成交数量。

## 8. 当前插件

### 8.1 初始基准与当前开发策略

- `original_atm_iv_straddle`：只使用绝对 ATM IV、固定仓位和日度 Delta 对冲的只读初始基准；
- `dynamic_atm_iv_straddle`：当前有效的开发策略，元数据状态为 `active_development`。

两者当前除策略身份和生命周期元数据外，交易行为完全一致。后续功能只逐步加入 `dynamic_atm_iv_straddle`，始终保留 `original_atm_iv_straddle` 用于基准对照。

### 8.2 `iv_straddle_v1`

这是默认插件，也是当前 live IV 跨式决策行为的回测侧副本，支持：

- ATM IV、曲面 IV 和历史分位数观察模式；
- 多空跨式开平仓；
- 绝对 IV、IV 分位数、低 IV/HV spread 等短仓 regime；
- 极端高 IV 回落确认；
- 独立多空平仓阈值、临近到期退出；
- 短仓日亏损止损、成交量放大退出和冷却期；
- 共享 roll、ATM 腿再平衡及 ETF Delta 对冲。

它使用品种配置中的 `long_qty`、`short_qty` 作为静态目标仓位。复制 live 行为并不意味着后续自动同步；任何回测侧行为变化都必须在插件及其测试中显式完成。

### 8.3 `[已过期] dynamic_position_straddle`

这是旧版动态仓位实验原型，不再作为新策略的演进基础。创建该插件时会发出弃用警告，回测元数据写入 `strategy_status=deprecated`，并指向替代策略 `dynamic_atm_iv_straddle`。保留它仅用于复现历史实验和参数扫描；不要把其结果与当前开发策略混用。

该插件继承 `iv_straddle_v1` 的通用跨式行为，覆盖绝对 IV 信号、动态目标仓位和冲击等待状态。当前只配置 `300etf`：

- `Pmin=8`、`Pmax=24`；
- 卖出侧：`IVshort=15.5%` 到 `IVmax=35%`，`Nshort=10`；
- 买入侧：`IVlong=8%` 到 `IVmin=5%`，`Nlong=10`；
- 档位和目标张数均向下取整；
- ATM IV 单日上升至少 3 个波动率百分点，或标的收盘绝对对数收益达到 3%，触发统一风险冲击；
- 风险冲击后进入等待状态，直到首次出现 `ATM IV(t) < ATM IV(t-1)`；
- 等待期间禁止新开 short，已有 short 不执行动态档位调整；
- 等待期间若发生 roll，保留当前 `strategy_pair_qty`，而不是按当日 IV 扩仓；
- 独立平仓、止损和成交量退出仍具有更高优先级；
- ATM 腿再平衡属于共享执行行为，仍可能改变两腿张数差。

动态参数位于 `core/backtest_strategies/dynamic_position_config.py`。未配置的品种选择该插件时会直接报错。

## 9. 使用方式

### 9.1 默认静态 IV 跨式

```powershell
python run.py --product 300etf --strategy iv_straddle_v1
```

省略 `--strategy` 时默认也是 `iv_straddle_v1`：

```powershell
python run.py --product 300etf
```

### 9.2 动态仓位跨式

新动态策略基线（当前与原始策略行为相同）：

```powershell
python run.py --product 300etf --strategy dynamic_atm_iv_straddle
```

旧动态仓位原型（已过期，仅用于复现历史实验）：

```powershell
python run.py --product 300etf --strategy dynamic_position_straddle
```

### 9.3 指定回测区间和资金

```powershell
python run.py `
  --product 300etf `
  --strategy dynamic_position_straddle `
  --start 2020-01-01 `
  --end 2025-12-31 `
  --initial-cash 10000000
```

### 9.4 查看可用策略

```powershell
python -c "from core.backtest_strategies import available_strategy_ids; print(available_strategy_ids())"
```

### 9.5 研究脚本中的调用

研究脚本应复用与 `run.py` 相同的顺序：

1. 加载并同步品种配置；
2. 只加载一次数据和共享特征；
3. 为每组实验创建新的插件实例；
4. 修改该实例的研究参数；
5. 重新调用 `build_signals()`；
6. 每组使用独立的回测账户状态运行；
7. 保存参数和策略元数据。

不得在不同参数组合之间复用已经执行过的 `BacktestState` 或可变持仓对象。

## 10. 输出与可复现性

每次回测在 `output/backtest/<timestamp>/` 保存：

| 文件 | 内容 |
|---|---|
| `runtime_config.json` | 本次实际使用的完整品种和回测配置 |
| `strategy_metadata.json` | 策略 ID、名称及插件专属参数 |
| `daily_feature_position.csv` | 日度市场特征、仓位、净值和 PnL |
| `trades.csv` | 开平仓、roll、调仓、腿再平衡和ETF对冲流水 |
| `backtest_summary.csv/.txt` | 收益、夏普、回撤、保证金和 Greeks 汇总 |
| `*.png` | 波动率与PnL诊断图 |

比较两次结果时必须同时核对 `runtime_config.json` 和 `strategy_metadata.json`。只看目录时间或总收益，无法确认品种配置、策略插件和插件参数是否一致。

## 11. 新增策略插件的标准流程

1. 在 `core/backtest_strategies/` 新建独立模块；
2. 继承 `BacktestStrategy`，或在行为确实是其特化时继承已有插件；
3. 定义稳定且唯一的 `strategy_id`；
4. 实现完整接口，并明确 `None`、`0` 和目标数量的语义；
5. 若有专属参数，放在插件自己的配置模块，不向 live 配置注入策略分支；
6. 在 `_STRATEGY_TYPES` 注册插件；
7. 为信号、开平仓、roll、调仓、状态机和元数据增加单元测试；
8. 用短区间烟雾回测检查成交流水，再运行完整样本；
9. 检查输出中的策略元数据与实际参数一致；
10. 更新本文档的“当前插件”章节。

设计约束：

- 不允许在 `BacktestEngine` 中按 `strategy_id` 增加条件分支；
- 新的纯策略差异应由插件 hook 表达；
- 只有多个策略都会复用的执行能力，才应抽象为新的通用接口和引擎实现；
- 普通插件不得调用 live 存储或信号入口，也不得修改 live 账户状态；`live_straddle` 只能使用显式传入模拟状态的只读目标契约，禁止调用真实账户读写函数；
- 信号必须无前视，元数据必须可 JSON 序列化；
- 每次回测仍只能选择一个插件。

## 12. 验证

插件架构和当前两个插件的核心测试：

```powershell
python -m pytest `
  tests/test_backtest_strategy_plugins.py `
  tests/test_dynamic_position_straddle.py `
  tests/test_dynamic_position_short_parameter_scan.py `
  -q
```

新增插件至少应覆盖：

- 注册和工厂创建；
- 信号表字段及无前视比较；
- 多空开仓和平仓原因；
- 已有仓位目标数量；
- roll 数量和状态继承；
- 资金不足或容量裁剪时的行为；
- Delta 对冲和 ATM 腿再平衡开关；
- `strategy_metadata.json` 所需元数据。

## 13. 后续扩展

若未来需要多策略组合，不应让 `BacktestEngine` 直接遍历多个插件。建议在插件与执行器之间新增组合分配层：

```text
多个策略插件
      ↓ 各自的目标仓位/交易意图（带 strategy_id）
组合分配与冲突仲裁层
      ↓ 聚合后的账户级目标
现有 BacktestEngine 执行、资金和报告层
```

组合层需要额外解决共享现金、保证金预算、同合约净额、策略优先级、PnL归属和组合级风险限制；这些能力不属于当前单策略插件接口。

## 14. `live_straddle` 当前 live 镜像

`live_straddle` 是当前 live 策略的标准历史回测入口：

```powershell
python run.py --product 300etf --strategy live_straddle
```

它不保存独立的策略参数副本，而是直接使用品种公共配置，并把完整有效配置、配置 SHA-256 和策略源文件 SHA-256 写入策略元数据。信号判断显式调用与 live 相同的 `core.strategy` 实现；ATM 期权腿求解、Delta 计划、ETF 净额化和调整合约强平使用传入模拟状态的只读 live 目标契约，不读取或修改真实 shadow account。

插件仍只表达策略能力；现金、持仓、手续费和成交继续由 `BacktestEngine` 管理。`BacktestEngine` 不允许根据 `strategy_id` 分支，只能依据通用能力属性执行，例如绝对 Delta 容忍值和调整合约强平。

不可变快照重放和理论/实际对比不通过标准日线数据入口运行，详见 [`live_straddle_replay.md`](live_straddle_replay.md)。
