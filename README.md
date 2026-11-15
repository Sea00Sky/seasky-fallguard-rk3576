# 衡护 HENGUARD · RK3576 医院病区跌倒检测系统

不把推断交给云端 GPU：跌倒判定整条链跑在院内一台 RK3576 板卡上，现役配置（零拷贝后端）
1 人 30 fps，从取帧到出事件都在板上完成；事件经 POSIX 共享内存交给一个多线程 C 网关，
再上报云端，云端只负责任值班、告警处置与追溯。检测是 YOLO26n-pose（出人体框与 COCO-17
关键点的姿态模型），判定是 ST-GCN++（骨骼时序分类）对 60 帧窗口打分。

状态：实验原型，2026-11 定稿后不再继续开发，不保证生产可用。
许可 AGPL-3.0：把它作为网络服务对外提供时，你的改动也要按同一许可开放；商用集成前先读 `LICENSE`。
平台：只跑在 RK3576 上（Linux/ARM64，systemd 托管），要一路 MIPI 摄像头；没有板卡就只能读
代码，跑不起来。板端与云端是 Python 3，网关是 C，没有 Windows 版本。

取代码：`git clone https://github.com/Sea00Sky/seasky-fallguard-rk3576.git`。仓内一律用
`fallguard` 前缀（服务名、`FG_*` 环境变量），仓库名里的 `seasky-` 是账号前缀。

An end-to-end fall detection system for hospital wards that keeps inference on the device.
YOLO26n-pose and ST-GCN++ run on the RK3576 NPU at 30 fps; events cross a POSIX shared-memory
IPC into a multithreaded C gateway, and reach a FastAPI/PostgreSQL service that closes the
nurse workflow (alert, acknowledge, arrive, dispose, audit).

## 判定链路

从一帧画面到一条事件，这条链上每一道门的位置与代价：

```
每帧 640×480 ──> YOLO26n-pose（NPU, FP16）              框 + COCO-17 关键点
                     │
                     ▼
                IoU 跟踪（阈值 0.30，轨迹丢 30 帧后放弃）    每条轨迹一个环形缓冲
                     │                                      [2] 允许 1–4 帧无共享帧的对接
                     ▼
                评分窗：连续 60 帧（30 fps 下 2.0 s），步长 15 帧   [1] 按帧定义，不按秒
                     │
       覆盖率 ≥ 0.70 ─┴─ 不过 ──> 丢弃（插值补出来的帧不算检出）
                     │
     [4] 几何前置闸门 ─┴─ 明显没横过来 ──> 直接判非跌倒，不进模型
                     │
                     ▼
                ST-GCN++（NPU, FP16，输入 (1,1,64,17,3)：批/人/帧/关节/通道）──> 跌倒概率
                     │
                     ▼
                两道否决：几何（窗内框宽高比 max < 1.80 且框底位移 < 25 px）
                         时序（窗末尾框高比 > 0.81）
                     │
                     ▼
                状态机：连续 2 窗概率 ≥ 0.5 确认 · 连续 4 窗低分撤回 · 概率 ≥ 0.65 强确认不再撤回
                        30 s 冷却内，[3] 连续 2 窗概率 ≥ 0.9 时放行一次
                     │
                     ▼
                事件 ──POSIX 共享内存──> C 网关 ──> SQLite ──> 云端
```

评分窗按帧数定义（生产 60 帧），不按秒。这一条是被数据逼出来的：语料采集实际是 20 fps
而容器帧率被写成 30 fps，按「2.0 秒」取窗只拿到 40 帧，与训练窗口（48–60 源帧）对不上，
详见 `reference/README_video_fps.md`。窗口进模型前会重采样到 64 帧（源项目定死的输入
长度），所以图里 60 帧窗、模型输入 64 帧，两者不矛盾。

用检测框而不是关键点做几何否决，是因为老年跌倒多为直背下坠，躯干角与宽高比本身的
分辨力低，而倒地时关键点大面积丢置信度，检测框反而更稳。两个模型的结构与训练见
`training/README.md`。

这条链上有四项改动，都是从实测失败里长出来的：两项抬召回，一项补结构缺陷，一项省算力。
后两项在这批语料上事件级结果逐项不变。样本只有 19 条，这只能说明看不见回归，不是无回归
的证明。

| 现象（证据） | 做法 | 结果 |
|---|---|---|
| [1] 语料实际 20 fps、容器标 30 fps，按「2.0 秒」取窗只拿到 40 帧，与训练窗口错位 | 窗口按帧定义，生产固定 60 帧 | 事件召回 6/17 → 12/17，有跌录像 5/9 → 9/9，无跌倒录像的误报仍只有 1 条 |
| [2] 跌倒瞬间框形变，IoU 掉到 0.155，低于跟踪阈值 0.30，跟踪器当场换号，同一个人被拆成两段轨迹 | 身份桥接：允许 1–4 帧无共享帧的对接把链接回来 | 19 条轨迹流接回 3 条、0 条跨人误接；事件级逐项不变——这 19 条里 60 帧窗先把问题掩盖了。留着它是为了补「跌倒打断跟踪身份」这个结构缺陷，代价是只在 1–2 人场景验过 |
| [3] 30 秒冷却会吞掉同一个人几秒之后的第二次真跌 | 强窗破冷却：连续两个评分窗 ≥ 0.9 时放行一次确认 | 事件召回 12/17 → 14/17，没有新增非目标告警 |
| [4] ST-GCN 每窗一次 NPU 推理，是这条链上的耗时大头 | 几何前置闸门：明显没横过来的窗直接判非跌倒，不进模型 | 板端现网语料 ST-GCN 调用减少 56%，离线两套语料 64.5% 与 70.9%；事件级结果在这批语料上一字不变 |

四项在代码里默认全关，生产由 systemd 打开（`board/p8/fallguard-sensor-multi.service`）。
默认关是刻意的：关掉时代码路径与改动前逐位相同，离线基线不受影响；开关是
`FG_MULTI_WINDOW_FRAMES`、`FG_MULTI_STITCH_BRIDGE`、`FG_MULTI_COOLDOWN_BREAK_PROB` 与
`FG_MULTI_COOLDOWN_BREAK_WINDOWS`、`FG_MULTI_PREGATE`。

## 与别的做法比

通常会被拿来对比的是两路做法，各差在哪：

可穿戴（手环、胸牌）：要把设备发到每个老人身上，日常还有充电与摘戴的维护；这套东西不
依赖被看护对象佩戴任何东西，只看板卡上自带的一路 MIPI 摄像头（IMX415）。

云端集中推断：要把多路视频压上云、占服务器 GPU；这套东西的判定在板上做完，不占服务器
的 GPU，云只收事件与快照。

它做不到的事一并写在这里：一台板卡只带一路摄像头，卫生间、被帘子挡住的床位这类它
看不到的地方无能为力；俯卧、躺卧姿态在当前训练数据上会系统性漏检，有检出的帧只占
44–47%，覆盖率落回 0.38 到 0.53，过不了 0.70 那道门——拿 MCFD 数据集的 6 段 30 fps
片段（含 7 个可评估事件）跑同一套生产配置是事件 0/7。板端与 PC 用同源 ONNX 逐项一致（逐条对账数字在
`docs/判定逻辑与实测.md` 第九节），所以这不是部署或管线的缺陷，是训练数据域不匹配，
要靠补目标域素材解决，调参没用。

## 系统结构

```
IMX415 ──ISP──┬── video43 (selfpath, 640×480) ──> 检测进程
              │        YOLO26n-pose + ST-GCN++ + 跟踪 + 判定
              │                    │
              │                    ├── POSIX shm ──> C 网关 ──> SQLite ──> 云端
              │                    └── 标注画布页
              └── video42 (mainpath, 1080p) ──> RTSP / SRT ──> 云端 mediamtx
```

ISP 一路出两个节点：selfpath 降到 640×480 给检测，mainpath 保 1080p 给推流。三条数据流
是检测流（事件、状态、快照）、视频流（局域网 RTSP、广域 SRT）、配置流（云端长轮询下发
判定策略）。云端不主动连接板端，连接一律由板端发起。

## 实测

板端（Kickpi K7C，RK3576）：

- 现役配置是零拷贝后端（ctypes 直调 librknnrt）：检测段 24.8 ms，全链 30 fps（1 人；相机
  本身上限就是 30，跑满即此数）。
  24.8 ms 只是检测这一段的耗时，不是整帧——30 fps 的整帧预算是 33.33 ms，其余留给预处理、
  ST-GCN 与编码。
- 换之前走 rknnlite 后端：全链 22.95 fps（画面 1 人）/ 22.93 fps（3 人），含编码与 ST-GCN。
- NPU / DDR / CPU 全部锁 performance；不锁时只有 13.2 fps，锁上之后 21 fps 的帧率门槛
  才有余量。这个门槛来自覆盖率闸门 0.70 × 相机 30 fps。
- 视频流那条（与检测链是两条流）：RTSP 推 1080p 时 30.24 fps、3.65 Mbps、端到端 p50
  52 ms（不含相机曝光）。

判定。清单（`reference/videos.csv`）里是 19 条病区录像，其中一条只有 3 帧、不可用，实际
进入评估的是 18 条，真实采集 20 fps，生产配置：

| 配置 | 事件召回 | 有跌视频 | 无跌倒录像（误报） | 窗口级召回 |
|---|---:|---:|---:|---:|
| 40 帧窗口 | 6/17 | 5/9 | 1/7 | 13/17 |
| 窗口改为固定 60 帧 | 12/17 | 9/9 | 1/7 | 14/17 |
| 再加身份桥接 | 12/17 | 9/9 | 1/7 | 14/17 |
| 再加强窗破冷却 | 14/17 | 9/9 | 1/7 | 14/17 |

这几个分母各按各的口径，不要相加：18 条录像里 7 条无跌倒（清单
`reference/per_video_nofall.csv`）、11 条含跌倒，含跌倒的 11 条里只有 9 条有可评估段
——这就是 9 + 7 = 16 与 18 之间差的 2 条；可评估跌倒段 17 个（标注段共 19 段）；候选窗
915 个。误报那列是 7 条无跌倒录像里报出告警的条数，越少越好，四行都是误报 1 条。事件
召回按段计，窗口级是同一批段上更松的判据，所以数字总是更高；四个指标的精确定义见
`docs/判定逻辑与实测.md` 第八节。最后一行为当前生产配置，共确认 28 条事件、撤回 25 条
——事件条数与 17 个段不是同一个分母——没有新增非目标告警。

这 19 条语料规模不大，1 到 2 段的差异不足以当趋势看。

早期口径把容器帧率当 30 fps 计时，与上表按真实 20 fps 重建的候选窗不是同一批——同样
写 40 帧窗口，那边是 7/17、这边是 6/17。那个口径不再作为性能结论，只用于验证「改动
前后逐窗概率一致」。

云端：只读回归 674 项通过、0 失败、3 跳过；越权矩阵 137/137；页面 p95（loopback）6–15 ms；
30 并发 194 ms；CSP 真浏览器检查 17 页 0 违规。这些数来自云端那轮的现场检查，逐项记录与
复核口径在 `docs/云端与服务.md`，仓内没有可一键复跑的脚本。

## 快速开始

两个前提，缺一个都跑不起来：板子是 RK3576（Linux/ARM64），以及从瑞芯微 SDK 装上
librknnrt 2.3.x 并核对版本——这一步不能跳，核对命令与已验证的版本组合见
`docs/BOARD_DEPLOYMENT.md`。

先跑通管线，不接共享内存也不开直播（只验证判定链本身）：

```bash
cd board
python3 -m pip install -r requirements.txt
python3 scripts/run_sensor_multi.py --no-shm --web-port 0 --seconds 30
```

接上共享内存与直播页，就是生产入口：

```bash
export FG_DEPLOY_DIR=$(cd ../deploy && pwd)   # run_sensor_multi.py 由此导入 shm_adapter / web_stream
python3 scripts/run_sensor_multi.py --web-port <端口>
```

网关：

```bash
cd gateway/edge_gateway
make
cp config.json.example config.json    # 按现场填写 http_port / data_dir / push_url
./fallguard-gateway --config config.json
```

网关从 `/fallguard_shm` 读感知进程写入的事件，落 SQLite，并提供 `/api/status`、
`/api/events`、`/api/snap` 三个只读接口。直播画面不进网关（网关不做视频转发），
契约与线程划分见 `gateway/README.md`。生产由 `gateway/edge_gateway/systemd/`
下的 unit 托管。

云端：

```bash
cd web
python3 -m pip install -r requirements.txt
createdb fallguard
psql -d fallguard -f db/schema.sql
psql -d fallguard -f db/seed.sql
cp deploy/web.env.example deploy/web.env    # 填写 DATABASE_URL / SECRET_KEY
uvicorn app.main:app --workers 1
```

单 worker 是硬约束：SSE 用进程内 asyncio.Queue 广播，多进程之间收不到彼此的事件。

训练与量化分别见 `training/README.md` 与 `quantization/README.md`。

## 取舍

判定的灵敏度与稳定：用"确认 / 撤回"两态而不是单次阈值，换来的是护士不被弯腰、下蹲这类
邻近动作频繁打扰——19 条语料上确认 28 条、撤回 25 条，撤回把大部分疑似压掉了。代价是
同一批动作里仍有 3 段慢速蹲跌没抓到，因为训练集里没有对应的样本。

目标域适应：在病区自己的语料上事件召回 14/17，代价是跨域下降明显，原因见上一节。

多人与单人：覆盖率门加人数闸门把单人场景的误报压得很低（7 条无跌倒录像里只误报 1 条）。代价是这套门在
4 人同框时概率会错位，目前按 3 人以内使用。

部署形态：单进程的板端和单 worker 的云端让实时推送、审计链路都很简单，没有额外中间件。
代价是不能水平扩展，接 HTTPS 或多实例要改推送这一层；云端目前只有 HTTP，域名备案完成后
才会上 TLS 与 Secure Cookie。

运行条件：板卡三处（NPU、DDR、CPU）全锁 performance，换来的是单人场景能稳定跑满相机
帧率（30 fps）。代价是功耗与发热按满频走，供电要 5V/3A 以上，供电不足会出现没有日志的
硬复位。

告警生命周期：事件没有超时自动关单，未处置的会一直留在队列里。这样不会漏掉待办，代价是
需要人工清理。

## 仓库结构

```
board/           RK3576 板端：采集、推理、跟踪、判定、推流、生产入口
gateway/         C 网关：共享内存消费、SQLite 落库、HTTP 上报与快照服务
deploy/          板端部署模块：共享内存适配、RTSP 推流、RKNN 运行库安装
web/             云端 Web：FastAPI、PostgreSQL、Jinja2、SSE
training/        YOLO26n-pose 与 ST-GCN++ 训练代码与配置
quantization/    ONNX 导出、RKNN 转换、PC 侧管线复刻与验收
models/          交付的两个 FP16 RKNN 模型
reference/       ground truth 区间、基线数字、视频清单
tools/           维护这个仓库自己的脚本
docs/            五篇正文记录，索引见 docs/README.md
```

数据集与训练权重不在此仓库内。

## 文档

| 文档 | 内容 |
|---|---|
| `docs/README.md` | 全部文档的索引，每篇一句话 |
| `gateway/README.md` | 共享内存契约、五个线程、幂等与推送闸门、构建与验收 |
| `docs/BOARD_DEPLOYMENT.md` | 模型输入输出契约、版本清单、部署步骤 |
| `docs/判定逻辑与实测.md` | 窗口与状态机、四项改动的来龙去脉与实测数字 |
| `docs/性能与资源实测.md` | 推理耗时、零拷贝、锁频与核分配、RTSP 链路、后台线程打分 |
| `docs/摄像头与推流.md` | ISP 双节点共享、采集参数、标注画布与云 HLS |
| `docs/云端与服务.md` | 云端验收、三方对账、切流与暴露的问题 |

## 许可证

GNU Affero General Public License v3.0，全文见 `LICENSE`。
第三方组件与模型来源见 `NOTICE`。
