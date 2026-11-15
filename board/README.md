# board · RK3576 板端

## 运行入口

| 脚本 | 用途 |
|---|---|
| `scripts/run_sensor_multi.py` | 生产入口：相机采集 → 多人管线 → 共享内存 / Web 直播 |
| `scripts/run_stream.py` | 本地 RTSP 推流 |
| `scripts/run_cloud.py` | SRT 云出口 |
| `scripts/config_agent.py` | 长轮询云端下发的判定策略 |
| `scripts/snapshot_uploader.py` | 快照上云 |
| `scripts/retime_videos_fps.py` | 语料帧率重定时（只改容器时间轴，不解码不重编码） |
| `scripts/run_video.py` | 离线逐窗评测，单视频或全量 |

## 模块

- `src/capture/`：GStreamer / OpenCV 采集
- `src/backends/`：RKNN 后端，两套实现（rknnlite 与直调 librknnrt 的零拷贝）
- `src/multi/`：多人跟踪、判定、叠加、事件状态机
- `src/stream/`：RTSP / SRT 推流
- `src/gate.py`、`src/tracker.py`、`src/stitch.py`、`src/pose_filter.py`：覆盖率门、
  跟踪、换号拼接、关键点净化

## 判定开关

生产取值写在 `p8/fallguard-sensor-multi.service` 的 `Environment=` 里，
不要写进 `runtime/multi.env`（那是云端下发、会被整份覆盖的文件），也不要加 drop-in
（drop-in 的 `Environment=` 会盖掉 `EnvironmentFile=`）。

| 环境变量 | 作用 | 代码默认 |
|---|---|---|
| `FG_MULTI_WINDOW_FRAMES` | 评分窗口按帧数定义，生产 60 帧 | `0`，按秒算 |
| `FG_MULTI_STITCH_BRIDGE` | 身份桥接，补跌倒瞬间换号造成的零共享帧对接 | `0` |
| `FG_MULTI_PREGATE` / `_SAMPLE` | 几何前置闸门，以及每 N 个被闸窗口抽样打分留读数 | `0` / `0` |
| `FG_MULTI_COOLDOWN_BREAK_PROB` / `_WINDOWS` | 强窗破冷却，连续 N 窗不低于该概率时放行一次确认 | `0` / `2` |

这几个键都不进 `MultiConfig.as_dict()`，所以云端的配置指纹不受影响，云端界面也
看不到它们，只能靠部署记录追溯。四项开关的实测与理由见
`../docs/判定逻辑与实测.md`。

## 部署路径约定

仓库里的 unit 和脚本按下面这套路径写默认值，接线到别处时用环境变量覆盖：

| 位置 | 约定路径 | 覆盖变量 |
|---|---|---|
| 板端代码 | `/opt/fallguard/board` | `FG_BOARD_DIR` |
| 部署模块 | `/opt/fallguard/deploy` | `FG_DEPLOY_DIR` |
| 网关 | `/opt/fallguard/gateway` | — |
| 网关落盘 | `/var/lib/fallguard` | 网关配置里的 `data_dir` |
| 服务账号 | `fallguard` | unit 里的 `User=` / `Group=` |

unit 里的 `User=` 必须存在同名账号；Python 依赖装在哪个账号下，就以哪个账号跑，
否则会 `ModuleNotFoundError`。

## 验收与部署

`p8/` 放验收脚本与 systemd unit，见 `p8/README.md`；模型契约、版本清单、部署步骤
与回滚见 `../docs/BOARD_DEPLOYMENT.md`。
