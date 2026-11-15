# 板端部署

板端跑两个模型：YOLO26n-pose 出人和关键点，ST-GCN++ 吃一段骨架序列判跌倒。
两个 `.rknn` 各自独立加载、独立推理，中间靠管线的窗口逻辑串起来。交付路径上
没有任何 INT8 产物。

## 一、交付文件

| 文件 | 字节 | sha256 | 用途 |
|---|---:|---|---|
| `models/yolo26n_pose_480x640_fp16.rknn` | 12,123,446 | `81ca4bb8c204370724e902b908d2f0d104fa4ef25fb219946b9b10abc2ad4db4` | 人体检测 + 17 关键点 |
| `models/stgcnpp_v2_fp16.rknn` | 13,386,094 | `3aa0d39afcdfbdc679bbf525aae5b63994e36f95bb070cd1858172ff4ab47ede` | 跌倒 / 非跌倒二分类 |

上板前先核对哈希，对不上说明拿的不是本项目的产物，后面的数字都不能用：

```bash
sha256sum yolo26n_pose_480x640_fp16.rknn stgcnpp_v2_fp16.rknn
```

`.rknn` 的哈希不是可复现量，toolkit 会把时间戳写进文件头，重转一次就变。
这两个值是用来回答"报告里的数字出自哪个文件"的。可复现的是评分数值，不是哈希。

这两个文件由 rknn-toolkit2 2.3.0 从 `quantization/` 导出的 ONNX 转换而来，
目标平台 `rk3576`。板端的运行库必须和转换版本兼容，版本对不上会报
`Invalid RKNN model version`，而 `RKNNLite` 加载的是硬编码路径
`/usr/lib/librknnrt.so`，`LD_LIBRARY_PATH` 覆盖不了，只能直接替换那个文件，
换之前先备份。板子上验证过的组合是 rknn-toolkit-lite2 2.3.2 加 librknnrt 2.3.2
加 driver 0.9.7。上板先查：

```bash
strings /usr/lib/librknnrt.so | grep -i "librknnrt version"
```

## 二、检测器的输入输出契约

输入 `(1, 3, 480, 640)` float32、NCHW。输出 `(1, 56, 6300)` float32，
56 是 4 个框加 1 个置信度加 17×3 个关键点，6300 是 60×80 + 30×40 + 15×20
三个 stride 的锚点总数。输出没有 NMS，因为用的是端到端的头。

预处理必须逐位照做：

```python
rgb = frame[:, :, ::-1].astype(np.float32) / 255.0
chw = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])
```

640×480 的输入是恒等变换，`r = min(640/480, 640/640) = 1.0`，不缩放也不补边。
所以不要做 letterbox，也不要补到 640×640。送 640×640 加灰边的话关键点最大会偏
485 像素。转换时 `mean_values=[[0,0,0]]`、`std_values=[[1,1,1]]`，归一化已经在
上面的 `/255` 里做完了，板端不要再减均值或除方差。

推理调用要显式写 `data_format="nchw"`。省略时默认按 `nhwc` 解释，
`(1,3,480,640)` 会被拒；也不要写 `"numpy"`，那个值不被接受，而且返回 `None`
而不是抛异常，错误会拖到下一帧才以 `'NoneType' object is not subscriptable`
的形式爆出来。

后处理：

```python
pred = pred[0].T                     # (6300, 56)
conf = pred[:, 4]
keep = np.nonzero(conf > 0.25)[0]                       # 严格大于
keep = keep[np.argsort(-conf[keep], kind="stable")][:4] # 按置信度降序取前 4
boxes = pred[keep, 0:4]                                 # xyxy，输入图像像素
kpts  = pred[keep, 5:].reshape(-1, 17, 3)
```

阈值 0.25 且是严格大于（用大于等于会让恰好等于阈值的框在一边存活，检出数就对不上）。
最多 4 人，框是 xyxy，框和关键点都要裁到画面内（不裁的话关节会跑到画面外约 23 像素）。
裁剪要在反 letterbox 之后做。

FP16 相对 fp32 的精度基线是 228 帧：IoU 中位数 0.99829、p5 0.99604、
1 像素内的关键点占 0.99224、关键点 L2 最大 2.823 像素、置信度绝对误差最大 0.02859、
检出不一致 1/228。这是模拟器上的六条判据同时满足，不是加权平均。

## 三、时序模型的契约

输入 `(1, 1, 64, 17, 3)` float32，维度顺序是 `(N, M, T, V, C)`。输出 `(1, 2)`
logits，第二列是跌倒。转换时 `mean_values=[[0]]`、`std_values=[[1]]`，
5 维输入只接受一个值，给三个会报长度错。

维度顺序是最容易错的地方。PYSKL 的 `FormatGCNInput` 产出 `(M, T, V, C)`，
按字面写成 `(N,M,C,T,V)` 导出会得到一个能跑但结果全错的图。

输入不是原始像素坐标，要过三步：

```python
kp_unit = kp_px.copy()
kp_unit[..., 0] /= width          # 按帧尺寸归一化，不是按框
kp_unit[..., 1] /= height
idx = np.rint(np.linspace(0, len(kp_unit) - 1, 64)).astype(np.int64)
kp_unit, scores = kp_unit[idx], scores[idx]        # 抽帧，不是插值
kp = kp_unit * 2.0 - 1.0                           # 映射到 [-1,1]
low = scores <= 0.01
kp[..., 0][low] = 0.0                              # 置零在映射之后
kp[..., 1][low] = 0.0
model_input = np.concatenate([kp, scores[..., None]], axis=-1)[None, None]
```

按帧尺寸归一化而不是按框，是因为按框归一会把远处和近处的人尺度抹平，与训练分布
不符。通道顺序是 `(x, y, score)`。低分关节置零发生在乘二减一之后，顺序不能反。

推理同样要显式 `data_format="nchw"`。如果用了带 batch 的产物，喂单个样本不会报错，
只会警告，然后把这一份广播到所有槽位返回多份相同结果；交付的这份是 batch-1。

softmax 在 CPU 上用 float64 做，用 float32 时第二位会在第 7 位有效数字上抖动，
与基线对不上。判定阈值是 0.5。

FP16 相对 ONNX 的基线是 1189 个样本（全体不是抽样）：决策翻转 0 个，
logit 最大绝对差 0.02033，概率最大差 0.00298。最靠近边界的那个样本，
ONNX 给 0.49787、板端给 0.49665，误差是翻转所需距离的 0.57 倍，有风险但没翻。

## 四、管线常量

这些值在 `board/src/spec.py` 里，板端全部照抄：

| 常量 | 值 | 含义 |
|---|---:|---|
| `CONF` | 0.25 | 检测阈值，严格大于 |
| `MAX_PERSONS` | 4 | 每帧最多 4 人，按置信度降序 |
| `TRACK_IOU` | 0.30 | 贪心 IoU 关联阈值 |
| `MAX_AGE` | 30 | 轨迹消失多少帧后丢弃 |
| `DURATION_S` / `STRIDE_S` | 2.0 / 0.5 | 窗口长度 / 步长（按秒定义时用） |
| `GUARD_S` | 0.5 | 标注保护带，只影响训练标签 |
| `RESAMPLE_FRAMES` | 64 | 窗口重采样后的帧数 |
| `MIN_COVERAGE` | 0.70 | 窗口内该人出现的帧占比下限 |
| `MAX_INTERP_GAP_FRAMES` | 10 | 只对内部间隔不超过 10 帧的空洞插值 |
| `COORDINATE_NORM` | `image_size` | 按帧尺寸归一化 |
| `SCORE_THRESHOLD` | 0.01 | 低分关节置零的阈值 |
| `THRESHOLD` | 0.5 | 窗口分数的判定阈值 |
| `STITCH_GAP_MAX` / `STITCH_OVERLAP_MAX` | 5 / 1 | 换号拼接的间隔 / 重叠帧上限 |
| `STITCH_DIST_FACTOR` | 0.25 | 拼接的中心距上限，乘两框对角线之和 |

几个容易做错的点：跟踪关联用的是大于等于而不是大于；配对排序是
`(-IoU, 框序号, 轨迹号)`，这个排序是轨迹编号确定性的来源，换排序下游全变；
新号按检测顺序发，所以送进跟踪器的框必须已经按置信度降序排好；插值只填空洞不填
首尾；覆盖率在插值之前算；一个窗口里有多人时取最大概率，不取平均。

## 五、部署步骤

板端依赖装完之后，从瑞芯微 SDK 装 librknnrt 2.3.x 并核对版本。

```bash
cd board
python3 -m pip install -r requirements.txt
export FG_DEPLOY_DIR=$(cd ../deploy && pwd)   # 共享内存适配与网页直播从这里导入
python3 scripts/run_sensor_multi.py --web-port <端口> --yolo-core 3 --stgcn-core 3 --zerocopy full
```

`FG_DEPLOY_DIR` 指到 `deploy/`，那里放共享内存读写和 MJPEG 直播页。
`--yolo-core 3` 是 `NPU_CORE_0_1`；只给 `AUTO` 的话在 RK3576 上只会用一个核。

生产由 systemd 托管，unit 在 `board/p8/` 下：感知是
`fallguard-sensor-multi.service`，另有网关、护士端、两路推流、快照上传、
配置下发代理和每日重启。感知 unit 里要注意三处：

- 判定开关写在 unit 的 `Environment=` 里，不要写进 `runtime/multi.env`，
  那个文件是云端下发、下次拉取整份覆盖的；也不要加 drop-in，
  drop-in 的 `Environment=` 会盖掉 `EnvironmentFile=`。
- 锁频三条命令要加 `+` 前缀。unit 以普通用户跑，`ExecStartPre` 默认也是普通用户，
  写 `/sys` 会 Permission denied，加 `-` 前缀又把错误吞掉，结果三条全是静默无效。
  曾经因为这一点 NPU 和 DDR 一直是 ondemand，帧率只有 13.2。三处都要锁：
  NPU、DDR、CPU 全核。
- 不要写 `After=multi-user.target`，这个 unit 由那个 target `Wants`，写成排在它
  之后就成依赖环，systemd 解环时会丢掉排在后面的网关和护士端。

感知 unit 和旧的单路感知服务不能同时 enable，也不能同时 start：
两边各是两个 RKNN 会话，四个会话会让 NPU 挂死并触发整机重启。切换必须是
"停旧、起新"。

## 六、版本清单

| 项 | 值 |
|---|---|
| rknn-toolkit2（转换侧） | 2.3.0 |
| rknn-toolkit-lite2 / librknnrt（板端） | 2.3.2 |
| NPU driver | 0.9.7 |
| GStreamer | 1.24.2 |
| 内核 | Linux 6.1.75 aarch64 |
| 摄像头节点 | `video42` mainpath 推流、`video43` selfpath 检测，脚本里用 by-path |
| 检测器 sha256 | `81ca4bb8…ad4db4` |
| 时序模型 sha256 | `3aa0d39a…47ede` |

GStreamer 元素可用性：`v4l2src`、`mpph264enc`、`rtph264pay`、`appsink`、
`appsrc`、`shmsink`、`shmsrc`、`queue`、`tee`、`videoconvert` 都在；
`interpipe_src` 缺失。板上没有硬件缩放元素（`rkvideoscale`、`rgaconvert`、`rga`），
要缩只能让 ISP 出目标尺寸，软件 `videoscale` 是 6.3 毫秒一帧。

## 七、上板检查顺序

按顺序查，前一条不过不要往下走：

1. 先做只加载加空跑：`load_rknn`、`init_runtime`、一次全零输入推理，不做后处理。
   这一步只回答"交付的字节能不能被板端 runtime 接受"。
2. 哈希与上面一致。
3. `init_runtime` 返回 0；失败先查 librknnrt 版本。
4. 输入形状对：YOLO 喂 `(1,3,480,640)`、时序模型喂 `(1,1,64,17,3)`，都带
   `data_format="nchw"`。
5. `inference()` 的返回值不是 `None`。它不抛异常，必须显式检查。
6. 一帧有明显行人的画面，检测置信度最大值应该超过 0.5；全接近 0 就查预处理，
   多半是做了两次归一化或者送成了 NHWC。
7. 同一段视频的检出数应该接近 PC 上 fp32 的检出数（228 帧基线是 1 帧不一致）。
8. 已知跌倒窗口的概率应该大于 0.5，非跌倒窗口小于 0.5；与 PC 比逐样本翻转应为 0。
9. 全链路回放：跑完 19 条录像，看段召回和窗口数。窗口数变了说明是跟踪、
   覆盖率门或拼接的问题，与量化无关。
10. 延迟：分别测检测单帧、时序单窗和端到端帧率，记录 NPU 频率档、核掩码、
    是否零拷贝、散热状态。

## 八、冷启动验收

整机重启之后三个服务（感知、网关、护士端）应该全部自动起来，三处 governor
都是 performance，网页、网关接口、护士端页面都返回 200，感知帧率在
21 fps 门槛以上。回放一条录像，事件应该能一路走到护士端，撤回时同一行改状态
而不是新插一行。

## 九、回滚

切流时把旧的部署目录整份备份过（`cp -a`），回滚只做三件事：停掉并 disable 新
感知 unit，恢复旧部署目录和网关二进制，enable 起旧感知、网关、护士端。
注意旧感知 unit 与新 unit 不能同时 enable，否则开机互相抢摄像头、各自重启。

旧部署目录里有一处向后兼容的改动：共享内存适配的 `push_fall_event` 多了一个
默认 False 的关键字参数，旧入口不传、行为不变，新入口传 True（新管线已经按人
冷却，不能再叠一层全局冷却）。回滚到旧感知不受影响；但如果整份恢复备份之后又去
跑新感知，会因为缺这个参数报 TypeError，把这一处补回即可。

## 十、INT8 为什么不在交付里

8 位 INT8 不可用：PTQ 的各种变体都失败（最好的 31/228，门槛是 2/228），
QAT 跑完是 26/228，机制仍是排序置换，还引入了幽灵检出（某个锚点从 0 变成 0.617）。
16 位的 `w16a16i` 能在模拟器上过判据（1/228），但它不是 INT8，而且 2026-10-27
板上实测这个降级风险成真了：两个变体在单进程单会话下 `load` 和 `init` 都成功，
每次推理却报 `failed to submit`，撞 6 秒 job timeout，帧率 6.4 秒一帧。
16 位激活与当前 driver 不兼容，模拟器上 1.28 像素的精度优势在板上兑现不了。
所以交付统一用 FP16。详细的门与数字见 `quantization/README.md`。

## 十一、取舍

两个模型各自独立加载、独立推理，换来的是换后端只改一处接缝，也能各自绑核。代价是
两个模型共用一块 NPU，同时最多两个会话，同一个会话也不能被两个线程共用。

模型契约固定在 640×480，换来的是 letterbox 恰好恒等、预处理能走查表加就地转换
（11.06 到 3.62 毫秒）、零拷贝后端也能用。代价是换分辨率必须重新导出模型，不是
调参数；零拷贝后端还额外要求原生布局与紧凑布局一致，`is_prealloc=1` 时 `size`
必须填。

预处理只覆盖了恒等快路径。接真实相机时如果分辨率不是模型尺寸，要补逐像素对拍，
因为 Python 缩小用 `INTER_AREA`，C 侧只有 `INTER_LINEAR`。

交付用 FP16 而不是 INT8，换来的是有板端实测支撑的精度。代价是模型体积和推理耗时
都比 INT8 大一档，所以必须靠零拷贝后端加三处锁频把帧率压在 30。

板端供电要 5 伏 3 安以上，供电不足会出现没有 panic 的硬复位。
