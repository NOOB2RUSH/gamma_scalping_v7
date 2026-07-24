# 四品种 Gamma 暴露研究

研究脚本：`scripts/research/gamma_exposure_report.py`。

该脚本独立于策略插件，不改变开仓、平仓、仓位或对冲逻辑。默认读取 50ETF、300ETF、500ETF、科创50ETF 四个品种各自配置的完整历史区间，并使用与回测一致的 ATM 选择及 IV/Greeks 缓存。

## 计算口径

研究单位为一对多头 ATM 跨式，即一张 call 加一张 put。每条腿使用历史数据中的实际合约单位：

```text
call cash gamma = call gamma × call multiplier × ETF close²
put cash gamma  = put gamma  × put multiplier  × ETF close²
pair cash gamma = call cash gamma + put cash gamma
```

交易日 `t` 收盘持有该暴露，ETF 下一交易日的收盘对数收益为：

```text
r(t+1) = log(close(t+1) / close(t))
long gamma PnL ≈ 0.5 × pair cash gamma(t) × r(t+1)²
```

空头跨式的 Gamma PnL 是上述结果的相反数。该近似只衡量 Gamma 曲率项，不包含 Delta、Theta、Vega、手续费或更高阶 Greeks。

“典型日波动”同时报告日对数收益绝对值的中位数、75%、90%、95%、99%分位数和 RMS。由于 Gamma PnL 与 `r²` 成正比，RMS 是最直接对应平均平方变动的波动尺度。

## 运行

```powershell
python scripts/research/gamma_exposure_report.py
```

指定统一研究区间或部分品种：

```powershell
python scripts/research/gamma_exposure_report.py `
  --products 50etf 300etf `
  --start 20200101 `
  --end 20251231
```

默认输出到 `output/research/gamma_exposure/`：

- `*_daily.csv`：逐日 ATM 合约、Gamma、Cash Gamma、下一日收益及近似 Gamma PnL；
- `*_summary.csv`：各品种 Gamma、Cash Gamma和日波动统计；
- `*_scenarios.csv`：以中位数 Cash Gamma 映射典型波动情景的多空 Gamma PnL；
- `*_metadata.json`：公式、时间对齐和单位说明；
- `*_comparison.png`：四品种 Cash Gamma 与典型日波动对比图。
