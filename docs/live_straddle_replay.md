# live_straddle 历史回测、快照重放与实盘对比

## 目标与口径

`live_straddle` 用于回答两个不同问题：

1. 当前 live 策略逻辑在一段完整历史样本上的机械执行表现；
2. 当前策略在已经保存的 live 不可变快照上执行时，与券商实际成交和盈亏的差异。

这两个结果不能混用。标准历史回测从指定初始资金和空仓开始；快照重放从第一个保存信号内嵌的真实账户状态或用户指定的账户 JSON 开始。

所有结果均表示“当前策略版本的反事实结果”，不承诺复现旧日期当时使用的旧代码。旧信号本身仍是历史审计事实。

## 单一事实来源

- 信号判断由 `core.strategy` 实现。live 默认使用同步后的运行时配置，`live_straddle` 显式传入同一份有效品种配置。
- 滚仓、调整合约强平、ATM 期权腿求解和 Delta 目标使用当前 live 执行契约。
- 回测和重放只拥有各自的模拟账户状态；不得调用 `save_account()`、`record_fill()` 或写入 `state/live/<product>/account.sqlite`。
- live 入口仍负责读取真实 shadow account 和行情快照；共享决策入口 `generate_signal_from_context()` 只根据显式上下文生成只读计划。

因此，策略判断和目标状态只有一份实现，成交与账户状态仍由各自执行器负责。

## 标准历史回测

```powershell
python run.py `
  --product kc50etf `
  --strategy live_straddle `
  --start 2023-01-01 `
  --end 2026-07-22 `
  --initial-cash 1000000
```

支持 `50etf`、`300etf`、`500etf` 和 `kc50etf`。插件默认直接使用品种公共配置，不设置另一份 live 参数副本。回测目录中的 `runtime_config.json` 保存完整有效配置，`strategy_metadata.json` 保存配置 SHA-256 指纹。

日期与初始资金是回测场景输入。若通过命令行修改 IV、DTE 等策略参数，结果不再代表默认 live 参数，应以保存的运行配置为准。

## 同快照重放

默认从第一个可用信号 JSON 内嵌的账户状态开始：

```powershell
python scripts/research/live_straddle_replay.py `
  --product kc50etf `
  --start 20260609 `
  --end 20260722
```

也可以显式指定一个账户 JSON，或指定包含 `account` 字段的历史 signal JSON：

```powershell
python scripts/research/live_straddle_replay.py `
  --product kc50etf `
  --start 20260720 `
  --end 20260722 `
  --initial-account output/live/kc50etf/20260720_144926_signal.json
```

重放事件来自 `output/live/<product>/*_signal.json` 中引用的不可变 `snapshot_stamp`。实际 parquet 路径根据品种、日期和时间戳重新解析，避免旧机器用户名造成绝对路径失效。完全相同的不可变快照只执行一次。

重放不会读取真实 SQLite 账户，不会保存 fill，也不会更新 live feature history。券商账户数据库只在生成实际对比报告时以 SQLite `mode=ro` 打开。

## 理论与实际对比

默认重放命令同时生成：

- `snapshot_replay_daily.csv`：当前策略理论重放净值和单日盈亏；
- `snapshot_replay_trades.csv`：按快照中间价模拟的期权和ETF成交；
- `snapshot_replay_plans.json`：每个快照的完整当前策略计划；
- `strategy_metadata.json`：配置、策略源文件和隔离保证；
- `snapshot_input_manifest.json`：每个信号 JSON 与实际读取行情快照的 SHA-256；
- `live_straddle_comparison.csv/.xlsx/.json/.md`：理论与券商实际对比。

对比报告拆分以下内容：

- 理论与实际净盈亏差；
- 同日、同合约、同方向、可匹配数量的执行滑点；
- 理论与实际手续费差；
- 当日匹配、延迟成交、未执行计划和计划外成交的独立数量与参考名义金额；
- 理论与实际收盘期权/ETF仓位是否不同；
- 扣除可可靠计量项目后的未解释残差。

未执行、延迟执行和持仓差异只有在能够可靠取得反事实价格路径时才能分配货币 PnL。当前报告对无法可靠定价的部分明确标为 `present_not_monetarily_valued`，并保留在未解释残差中，不使用人为分摊制造解释能力。

区间首日的理论口径从首个保存快照开始，而券商“单日盈亏”通常还包含该快照之前的时段。报告将首日标为 `interval_start_partial_theoretical_day`，保留明细但不纳入“可比整日”汇总。若需要首日也完整可比，应把初始账户状态和行情检查点放在前一交易日最后一个快照。

## 一致性边界

在相同配置、相同市场快照和相同账户状态下，目标是保持以下决策一致：

- 开仓、平仓和滚仓目标；
- 无合适滚仓合约时的全平及原因；
- 调整合约/除息当日的期权和ETF全平；
- 期权腿结构平衡；
- ETF动作净值化和最终微调；
- Delta 绝对值超过 5,000 时触发控制；触发后先完成期权动作，再以账户 Delta 归零为 ETF 微调目标。5,000 只用于触发，不是调整后的残余目标；
- 单日止损、成交量退出和冷却期；
- 资金、保证金和流动性约束；
- 最终期权数量、ETF目标及预计账户 Delta。

历史回测仍按日线参考中间价模拟成交，实盘由人工和券商成交决定。成交时间、滑点、拒单、漏单和实际费用属于执行差异，不属于策略决策不一致。

当前券商持仓导入不会保存信号时点的期权成交量基线。因此 `live_straddle` 的模拟开仓同样不保留模型侧基线，避免回测产生 live 不可能触发的成交量放大退出；若未来 live 成交确认开始可靠保存该字段，应同时切换这一状态契约并增加迁移测试。

## 完成与回归门槛

修改共享策略逻辑时必须同时验证：

1. 四个品种的 live 配置字段与 `live_straddle` 有效配置一致；
2. `core.strategy` 默认 live 调用与显式插件调用产生相同信号；
3. 标准历史回测可以完成并保存策略身份和参数指纹；
4. 快照重放不触碰真实账户或 feature history；
5. 开平仓、滚仓、调整合约、期权腿平衡和 Delta 阈值场景通过；
6. 对比报告中的成交、手续费、持仓差异和残差可以逐行核对；
7. 当前 live 信号、报告、导入、对账和执行适配器测试没有非预期变化。
