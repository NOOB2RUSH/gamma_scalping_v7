# 项目记忆

## Delta 口径

- 期权腿 delta：`long_delta_pnl + short_delta_pnl`，不含 ETF 对冲。
- 账户总 delta：`long_delta_pnl + short_delta_pnl + hedge_delta_pnl`，即输出字段 `delta_pnl`。
- 对外展示和结论默认使用账户总 delta；期权腿 delta 仅用于内部调试，并须明确标注。
- 开启 delta hedge 后，收盘通过 ETF 将 `account_delta = option_delta + hedge_etf_qty` 调至接近 0。

## 账户报告盈亏口径

- 保持“持仓盈亏”和“交易盈亏”字段名称不变。
- `持仓盈亏 = 收盘剩余昨仓数量 × (今收 - 昨收) × 多空方向 × 合约乘数`；不含日内新仓及已平掉的昨仓。
- `交易盈亏 = 实际平仓数量 × (平仓价格 - 当时加权平均持仓成本) × 多空方向 × 合约乘数`；只由平仓产生。
- 同方向加仓更新加权平均持仓成本；部分平仓不改变剩余仓位成本。

## 新品种数据源接入流程

- 固定流程：验证数据源 -> 小样本检查 -> 完整下载 -> 独立配置 -> 小范围回测 -> 全范围回测。
- 数据至少覆盖标的价格、期权代码、日期、到期日、行权价、call/put、期权价格和成交量；估算字段必须注明局限。
- 上交所 ETF 期权下载统一使用 `scripts/download/build_data.py --product <product>`；交易所或数据协议不同的品种可保留独立下载器。数据放入 `data/<product>/`，品种配置放入 `core/configs/`。
- 品种专属参数不得写入公共入口；修改公共模块后必须复核现有品种。
- 验证 ATM、DTE、IV/Greeks、成交量、保证金、手续费、对冲标的、现金和输出品种是否正确。

## 远端交互约定

- PowerShell 读取本文档时使用 UTF-8：`Get-Content -Path memory.md -Encoding UTF8`。
- 唯一有效的 live 账户和账户数据位于远端；账户分析以远端为准，不使用本机账户数据。
- 远端：`yangziqi@172.16.128.67`；项目目录：`/home/yangziqi/strategy/gamma_scalping_v7`。
- 当前 PowerShell 无 `rsync`，使用 `ssh` 查看远端、使用 `scp` 传输文件。
- 所有远端操作仅限 `/home/yangziqi/`，不使用 `sudo`，不操作系统目录、其他用户文件或进程。
- 临时文件使用 `~/tmp/` 或项目目录，不使用系统 `/tmp`，使用后清理。
- 扫描结果需检查列名、行数、错误数、排序首行和关键指标，不能仅依据日志尾部判断。
