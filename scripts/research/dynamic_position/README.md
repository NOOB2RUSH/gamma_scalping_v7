# 动态仓位研究

本目录存放动态仓位研究脚本；报告文件统一输出到
`output/research/dynamic_position/`（该目录为运行产物，不纳入版本控制）。

## ATM IV 日度状态统计

运行：

```powershell
python scripts/research/dynamic_position/iv_daily_state_report.py --product kc50etf
```

对每一个交易日 `t`，仅以该日与前一个有效交易日 `t-1` 的 ATM IV 比较：

- `IV_t / IV_t-1 - 1 >= 4%`：`iv_up`
- `IV_t / IV_t-1 - 1 <= -4%`：`iv_down`
- 其余：`flat`

首个样本日、当前 ATM IV 缺失、或前一交易日 ATM IV 缺失，均记为
`unclassified`，不计入三种状态的占比。

默认上下阈值均为 4%；如需敏感性分析，仍可通过 `--up-threshold` 和
`--down-threshold` 覆盖。

## 绝对 ATM IV 水平与日度状态

运行：

```powershell
python scripts/research/dynamic_position/iv_level_state_report.py --product kc50etf
```

默认按前一交易日的绝对 ATM IV 分为低 IV（`<=22%`）、中性（`22%` 至 `30%`）和
高 IV（`>=30%`），并计算其对当日 `iv_down`、`flat`、`iv_up` 的预测条件概率。
日度状态使用当日相对前一交易日的 IV 变动，且默认阈值为 `±4%`。可通过
`--low-iv-threshold` 和 `--high-iv-threshold` 调整绝对 IV 分界。

## 平滑条件概率曲线

运行：

```powershell
python scripts/research/dynamic_position/iv_level_state_curve.py --product kc50etf
```

图以前一交易日收盘的绝对 ATM IV 为横轴，并用高斯核加权计算当日
`P(iv_down | IV_{t-1})`、`P(flat | IV_{t-1})`、`P(iv_up | IV_{t-1})` 三条平滑曲线。默认带宽为 3 个 IV 百分点
（`--bandwidth 0.03`）；横轴默认截取 1% 至 99% 样本分位，以免极少数异常 IV
水平主导图形范围。图中同时叠加 30 个绝对 IV 区间的样本量柱状图，使用右侧
纵轴；可通过 `--histogram-bins` 调整柱数。

## 今日 ATM IV 与次日实现波动率

运行：

```powershell
python scripts/research/dynamic_position/iv_level_next_day_realized_vol_curve.py --product kc50etf
```

以交易日 `t` 收盘时的 ATM IV 为横轴，以次一交易日的标的收盘到收盘绝对对数收益的年化值
`sqrt(252) * |log(Close(t+1) / Close(t))|` 为实现波动率。脚本用高斯核平滑计算
`E[annualized realized vol(t+1) | ATM IV(t)]`，并叠加
`E[annualized realized / ATM IV | ATM IV(t)]` 的比率曲线及 IV 分箱样本量柱状图。最后一个没有下一交易日
收盘价的 IV 样本会自动剔除。脚本还会额外输出一幅 `ATM IV − 年化实现波动率` 的绝对差图；正值表示
当日 IV 高于次日实现波动率。

## ATM IV 相邻交易日绝对变化分布

运行：

```powershell
python scripts/research/dynamic_position/iv_daily_absolute_change_distribution.py --product 300etf
```

以相邻两个交易日的 ATM IV 绝对值差
`|ATM IV(t) - ATM IV(t-1)|` 统计经验分布。输出日度样本、描述统计、分位数阈值、
固定阈值尾部事件频率和分布图。阈值同时使用小数和“波动率百分点”表示；例如
`0.03` 等于 3 个波动率百分点。报告还统计每个阈值以上的向上冲高事件中，次一
交易日 ATM IV 回落的比例，用于研究 `IVspike` 和一日回落确认规则。

## 动态仓位short侧参数扫描

运行默认第一阶段网格：

```powershell
python scripts/research/dynamic_position/dynamic_position_short_parameter_scan.py --product 300etf
```

脚本只加载一次历史数据和期权特征，在内存中循环回测 `Pmin`、`Pmax`、`IVmax`、
`IVspike` 与 `Nshort` 的笛卡尔积。默认保持 `Pmin=8`、`Nshort=10`、
`IVspike=3` 个波动率百分点，扫描 `Pmax={18,20,22,24}` 与
`IVmax={35%,40%,45%}`。报告同时记录全样本收益、夏普、最大回撤、2024年冲击
窗口回撤、最差年度、保证金、平均仓位和手续费，并逐组写入CSV检查点。每组回测
使用独立的历史数据副本，避免上一组执行过程中的可变状态污染后续组合。
