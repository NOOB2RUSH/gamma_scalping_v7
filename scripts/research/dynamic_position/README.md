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
