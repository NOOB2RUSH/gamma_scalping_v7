# 项目记忆

## Delta 口径

- 始终区分“账户总 delta”和“期权腿 delta”。
- 期权腿 delta 指 `long_delta_pnl + short_delta_pnl`，只反映期权仓位自身的 delta 暴露，不包含 ETF 对冲。
- 账户总 delta 指 `long_delta_pnl + short_delta_pnl + hedge_delta_pnl`，也就是输出字段 `delta_pnl`。
- 开启 delta hedge 后，每日收盘会用 ETF 对冲把 `account_delta = option_delta + hedge_etf_qty` 调到接近 0；因此次日按上一日 EOD 口径计算的账户总 `delta_pnl` 应接近 0。
- 对外展示、图表、summary、结论分析中，默认使用账户总 delta / `delta_pnl`。不要把“期权腿 delta PnL”说成策略的 delta 损益。
- 如果为了调试需要展示期权腿 delta，必须明确标注为“内部调试 / 仅期权腿 / 不含 ETF hedge”，不能和账户总 delta 混用。

## 新品种数据源接入流程

- 接入新标的时，固定按“获取新数据源 -> 整理配置文件 -> 接入回测验证”的顺序推进，不要直接把临时代码混进主流程。
- 获取新数据源：
  - 先用点状数据确认数据源是否覆盖目标品种、历史范围、字段含义和更新稳定性。
  - 再下载小范围样本，检查是否至少包含回测需要的标的收盘价、期权代码、日期、到期日、行权价、call/put、收盘价或可用 mid、成交量或成交量近似字段。
  - 最后扩展到完整历史范围，下载脚本放入 `scripts/download/<product>/`，原始数据和整理后的 parquet/csv 放入 `data/<product>/`，避免不同品种混放。
  - 如果成交量不是交易所逐合约真实成交量，而是估算、活跃度或按总量缩放后的结果，必须在下载脚本和对应 config 注释中说明局限性。
- 整理配置文件：
  - 每个品种使用独立配置文件，并统一放在 `core/configs/` 下，例如 `core/configs/config_50etf.py`、`core/configs/config_zz1000.py`、`core/configs/config_500etf.py`。
  - 品种专属的数据路径、合约乘数、回测区间、ATM DTE、成交量过滤、delta hedge 标的路径、默认策略参数都放在该品种配置文件里。
  - `core/config.py` 只负责选择和加载品种配置，不把某个品种的策略参数写死在公共入口里。
- 接入回测验证：
  - 先确认 `python run.py --product <product>` 能读取正确数据路径，并在图表、summary、输出文件中标明品种名称。
  - 跑通小范围回测，检查 ATM 合约选择、DTE、行权价间隔、成交量、IV/Greeks、保证金、手续费、delta hedge 标的是否合理。
  - 再跑全范围回测，检查是否有缺链、异常 IV、现金为负、成交量预警、异常回撤和输出图表错用其他品种数据。
  - 新品种调参只修改该品种配置或扫描脚本中的显式方向，不能影响已经调好的其他品种；如果改到公共模块，必须二次检查 50ETF/ZZ1000/500ETF 是否行为仍一致。

## 远端交互约定

- 每次新对话开始处理本项目时，先读取 `memory.md`：在 PowerShell 中用 `Get-Content -Path memory.md -Encoding UTF8`，不要用默认编码读，否则中文会显示乱码。
- 默认远端：`yangziqi@172.16.128.67`；默认远端项目目录：`/home/yangziqi/strategy/gamma_scalping_v7`。
- 本仓库有 `sync_remote.py`，但它依赖 `rsync`。当前 Windows PowerShell 环境没有 `rsync`，直接跑 `python sync_remote.py --mode pull-output` 会失败；除非切到 Git Bash/WSL 或安装了 `rsync`，不要反复尝试。
- 在当前 PowerShell 环境下，优先用 `ssh` 查看远端、用 `scp` 拉文件：
  - 查看远端 output 最新目录：`ssh yangziqi@172.16.128.67 "cd /home/yangziqi/strategy/gamma_scalping_v7 && find output -maxdepth 1 -type d -printf '%TY-%Tm-%Td %TH:%TM %p\n' | sort | tail -30"`
  - 查看远端 output 最新文件：`ssh yangziqi@172.16.128.67 "cd /home/yangziqi/strategy/gamma_scalping_v7 && find output -maxdepth 2 -type f | sort | tail -80"`
  - 拉取扫描结果到本地：`scp yangziqi@172.16.128.67:/home/yangziqi/strategy/gamma_scalping_v7/output/<file> .\output\`
- 本地向远端传文件时，不要传到系统 `/tmp`。临时包传到 `yangziqi` 用户家目录下的 `~/tmp/`，或直接传到远端策略目录 `/home/yangziqi/strategy/gamma_scalping_v7/`；用完后清理对应位置的临时包。
- 所有远端操作必须只作用于 `yangziqi` 用户空间：ssh/scp 使用 `yangziqi@172.16.128.67`；读写、解压、清理、启动/停止任务只在 `/home/yangziqi/` 下进行；不要使用 `sudo`，不要改系统目录，不要操作其他用户文件或进程。
- 500ETF 扫描结果通常在远端 `output/optimize_coarse_*.csv`、`output/optimize_fine_*.csv` 和 `output/*500etf*_scan*.log`。读取时先看最新 `optimize_fine_*.csv` 的列名确认方向：`500etf_wide` 通常只有 short 参数；`500etf_full_defense` 会有 long/short 阈值和 `strategy.roll_cooldown_days`。
- 拉回扫描 CSV 后，先检查行数、`error` 非空数量、排序首行和关键指标，不要只凭日志尾部判断最终结果。
