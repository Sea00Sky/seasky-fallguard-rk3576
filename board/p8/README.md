# p8 · 板端验收与部署

运行入口在 `../scripts/`，本目录放验收脚本、systemd unit 与推流部署配置。
历史里那些一次性的压测与诊断脚本（`_bench_*`、`_diag_*`、`s2a_bench.py` 等）
没有随仓发布。

## 对账与验收

| 文件 | 用途 |
|---|---|
| `regress.py` / `regress.sh` | 逐窗对账：`selftest` / `run` / `diff` / `det` |
| `multi_eval.py` | 多人管线验收：`selftest` / `all` / `sweep` / `isolation`；支持 `--window-frames`、`--stitch-bridge`、`--pregate`、`--cooldown-break-*` |
| `verify_window_frames.py` | 窗口按帧数定义之后，窗口边界、概率、覆盖率与旧口径一致 |
| `verify_stitch_bridge.py` | 身份桥接：有共享帧的真拼接优先，跨人误并为 0 |
| `verify_cooldown_break.py` | 强窗破冷却：默认关闭与显式给 0 时逐事件相同，放行只发生在强窗 |
| `verify_pregate.py` | 几何前置闸门：事件级完全相同，影子采样口径正确 |
| `verify_async_parity.py` | 同步与异步（后台线程打分）结果逐字段等价 |
| `verify_*.py` | 采集、共享内存、网关、状态机、RTSP、画布等模块的板上验收 |
| `ann_bridge.py` | 演示用的 RTSP 标注桥 |

每个 `verify_*.py` 都能在 PC 上自跑，不需要板子。对账产物写到 `results/regress/`
下按运行号分的目录里，目录由脚本自己建；实测输出不入库。

## systemd

| 文件 | 用途 |
|---|---|
| `fallguard-sensor-multi.service` | 生产感知入口（含判定开关与三处锁频） |
| `fallguard-agent.service` | 配置代理 |
| `fallguard-snap-upload.service` | 快照上云 |
| `fallguard-daily-restart.service` / `.timer` | 每日低峰重启 |
| `install_daily_restart.sh` | 安装重启定时器 |

## 推流部署

- `cloud/`：云端 mediamtx 配置、SRT 出口 unit 与环境变量模板
- `stream/`：板端本地 RTSP 推流 unit 与环境变量模板
