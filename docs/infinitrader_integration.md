# InfiniTrader PythonGO 接入

本接入把本项目 live 信号编译成无限易 PythonGO 订单。策略核心仍在
`core.live.signal_engine`，无限易侧只负责生成计划、提交订单、记录回报。

## 一键流程

本地系统下发完整流程：

```powershell
python scripts\live\infinitrader_pipeline.py --product 300etf --source akshare
```

这条命令会：

1. 拉取最新 AKShare 快照。
2. 根据快照生成 live signal 和本地报告。
3. 编译无限易订单计划。
4. 写入 `output/live/<product>/infinitrader/pending_command.json`。
5. 等待无限易 PythonGO 策略读取该命令。

如果要等待无限易执行完成并同步本地账户：

```powershell
python scripts\live\infinitrader_pipeline.py `
  --product 300etf `
  --source akshare `
  --wait-executed `
  --sync-account `
  --write-account-report
```

首次联调建议只做本地账户同步预演：

```powershell
python scripts\live\infinitrader_pipeline.py `
  --product 300etf `
  --source akshare `
  --wait-executed `
  --sync-account `
  --sync-dry-run
```

## 本地预览

先在项目目录运行：

```powershell
python scripts\live\infinitrader_plan.py --product 300etf --source snapshot --date latest
```

也可以基于已生成的信号 JSON 预览：

```powershell
python scripts\live\infinitrader_plan.py --product 300etf --signal-json output\live\300etf\20260623_144727_signal.json
```

输出文件位于 `output/live/<product>/*_infinitrader_plan.json`。

## 无限易 PythonGO

1. 把 `integrations/infinitrader/gamma_scalping_auto.py` 复制到无限易的
   `pyStrategy/self_strategy` 目录。
2. 在无限易中加载 `GammaScalpingAuto`。
3. 参数设置：
   - `project_root`: 本项目目录，例如 `C:\Users\交易员\strategy\gamma_scalping_v7`
   - `product`: `300etf`、`50etf`、`500etf` 或 `kc50etf`
   - `account_id`: 本地影子账户 ID，默认 `default`
   - `quote_source`: 建议先用 `snapshot`
   - `quote_date`: 默认 `latest`
   - `command_file`: 留空时自动读取
     `output/live/<product>/infinitrader/pending_command.json`
   - `investor`: 无限易报单账号
   - `dry_run`: 初次必须保持 `True`
4. dry-run 验证无误后，把 `dry_run` 改为 `False` 执行模拟下单。

运行日志写入：

```text
output/live/<product>/infinitrader/
```

其中 `plan_*.json` 是订单计划，`events_YYYYMMDD.jsonl` 是提交、委托、成交回报。
本地 pipeline 写入的命令文件会先放在 `pending_command.json`；无限易提交完订单后
归档为 `executed_<run_id>.json`。

## 当前边界

- 当前版本提交订单并记录无限易回报。
- `--sync-account` 会根据本地 command 中的策略 advice 聚合成本地 shadow account
  fill；成交回报字段后续可以进一步用于替换实际成交价。
- 下单默认使用限价 GFD，期权订单带开平标志，ETF 订单不带开平标志。
- 不要同时手工确认同一批成交，避免本地账户重复记账。
