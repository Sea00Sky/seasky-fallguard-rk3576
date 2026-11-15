# 预处理规格

这份文件不是散文，是一份可被机器核对的清单：每一条都在 `src/deploy/spec.py` 里
有同名常量，`scripts/14_verify_spec.py` 会把它们逐条对回源项目的 `configs/*.yaml`
与函数默认值。文档、代码、源项目三者任何一处漂移，脚本就失败。

所有数字来自源项目实测，不是本项目重新设计的。

## 0. 三层结构与谁说了算

```
源项目 configs/*.yaml + src/window/build.py     ← 唯一权威（只读）
            │  照抄，不重新设计
            ▼
src/deploy/spec.py                              ← 本项目唯一出处（A14）
            │  import
            ├── src/runtime/     流式管线（板端形态）
            ├── src/reference/   离线参考的寻址层（指向源项目，不复制）
            └── scripts/         导出 / 对账 / 评估
```

规则是本文档、`spec.py`、源项目三者一致才算数。若 `spec.py` 的数与源项目不符，
以源项目为准，`spec.py` 错。`scripts/14_verify_spec.py` 就是这条规则的可执行形式，
产物是 `reports/reference/spec_audit.json`。

## 1. 前端（检测）

| 常量 | 值 | 出处 |
|---|---:|---|
| `IMGSZ` | 640 | `configs/pose.yaml:30` `inference.imgsz` |
| `CONF` | 0.25 | `configs/pose.yaml:34` `inference.conf` |
| `IOU` | 0.7 | `configs/pose.yaml:35` `inference.iou` |
| `MAX_PERSONS` | 4 | `configs/pose.yaml:39` `inference.max_persons` |
| `PERSON_CLASS` | 0 | `configs/pose.yaml:40` `inference.person_class` |

两个容易抄错的点：

1. `IMGSZ = 640` 不等于网络输入是 640×640。`LetterBox(auto=True)` 对 640×480 的
   板端帧是恒等变换（`r = 1.0`、`dw = dh = 0`），真正送进网络的是 480×640，
   导出也必须是 480×640。写论文时不要写"输入 640×640"。
2. 检测结果按框置信度降序排列（`argsort(-conf)`，`stable`）。这个顺序不是装饰，
   `IoUTracker` 按数组顺序发新号，排序不同编号就不同。

## 2. 跟踪

| 常量 | 值 | 出处 |
|---|---:|---|
| `TRACK_IOU` | 0.30 | `configs/pose.yaml:44` `tracking.iou_threshold` |
| `MAX_AGE` | 30 | `configs/pose.yaml:45` `tracking.max_age` |

贪心 IoU 关联，不用 ultralytics 自带的 tracker。`MAX_AGE = 30` 只约束 tracker
什么时候放弃一条轨迹，环形缓冲的过期时间是 `window_frames - 1`（60 帧窗就是 59），
两者不同，不要混用；混用 30 会漏掉 2161 条参考轨迹里的 323 条。

## 3. 窗口

| 常量 | 值 | 出处 |
|---|---:|---|
| `DURATION_S` | 2.0 | `configs/labels.yaml:20` |
| `STRIDE_S` | 0.5 | `configs/labels.yaml:21` |
| `GUARD_S` | 0.5 | `configs/labels.yaml:22` |
| `NEGATIVE_MIN_DISTANCE_S` | 0.5 | `configs/labels.yaml:23` |
| `RESAMPLE_FRAMES` | 64 | `configs/labels.yaml:24` |
| `MAX_INTERP_GAP_FRAMES` | 10 | `configs/labels.yaml:26` |

帧数由 fps 换算：`frames_for(fps, s) = max(1, int(round(fps * s)))`，
30 fps 下就是 60 帧窗口、15 帧步长、15 帧 guard。候选起点
`range(0, n - 60 + 1, 15)`，这一步不施加覆盖率门。

重采样：`np.linspace(0, n-1, 64)` 后 `np.rint`。

## 4. 选人与覆盖率

| 常量 | 值 | 出处 |
|---|---:|---|
| `MIN_COVERAGE` | 0.70 | `configs/labels.yaml:25`（D-043，不许改） |

三种选人口径（C6：任何召回数字都必须标明策略）：

| 策略 | 含义 | 本项目角色 |
|---|---|---|
| A | `max_presence`：窗内出现帧数最多的单条轨迹 | 仅作对照（7/17） |
| B | 窗内每条轨迹各算一个人，逐人取 max | G3 管线正确性判据用它（12/17） |
| C | B + D-042 换 ID 拼接链 | 验收口径（13/17） |

门的语义是：覆盖率低于 0.70 的人不送模型。它是打分之后的选人规则，不是模型输入。
覆盖率等于窗内"该轨迹被真实检测到"的帧数占比，插值补出来的帧不算。

## 5. 换 ID 拼接（D-042）

| 常量 | 值 | 出处 |
|---|---:|---|
| `STITCH_GAP_MAX` | 5 | `src/window/build.py:254` 默认值 |
| `STITCH_OVERLAP_MAX` | 1 | `src/window/build.py:255` 默认值 |
| `STITCH_DIST_FACTOR` | 0.25 | `src/window/build.py:244` 字面量 |

轨迹 `b` 可接在链尾 `a` 之后，当且仅当：

```
gap = b.first - a.last ∈ [-1, 5] 帧
a 与 b 同帧共存 ≤ 1 帧
每个共存帧的中心距 < 0.25 × (diag_a + diag_b)
```

只对链尾判定，因此两个人并排走不会被传递合并。

## 6. 归一化（最容易出错的一步）

| 常量 | 值 | 出处 |
|---|---:|---|
| `SCORE_THRESHOLD` | 0.01 | PYSKL `PreNormalize2D` 默认参数 |

顺序固定为：

```
1. 像素 → 单位坐标      x / width, y / height        （image_size 归一化）
2. 重采样到 64 帧        linspace + rint
3. 单位 → [-1, 1]       (c - w/2) / (w/2) = 2c - 1   （PreNormalize2D mode='fix'）
4. 低分关节点置零       score ≤ 0.01 的 x、y 置 0    （在映射之后）
5. 拼接 score 通道      (x, y, score)
```

第 1 步与第 3 步是两次不同的缩放，不是重复。G1b 第一次跑就抓到策略 A 把它们做了
两遍（`extract_window` 里已经归一化、重采样并转成 float16，外面又除了一次
width/height），913 个样本全错。

策略 A 的 float16 是源项目自己造成的：`extract_window` 返回前做了
`keypoints.astype(np.float16)`。所以 A 的最大概率差只能到 9.7e-3，而 B 和 C 是 1.5e-6。

## 7. 模型输入

| 常量 | 值 | 出处 |
|---|---:|---|
| `CLIP_LEN` | 64 | `weights/stgcnpp_v2_config_used.yaml` `data.clip_len` |
| `NUM_PERSON` | 1 | 同上 `data.num_person` |
| `IN_CHANNELS` | 3 | 通道序 (x, y, score) |
| `NUM_JOINTS` | 17 | COCO-17 |
| `MODEL_INPUT_SHAPE` | (1, 1, 64, 17, 3) | `(N, M, T, V, C)` |

布局是 `(N, M, T, V, C)`，不是计划里写的 `(N, M, C, T, V)`。PYSKL 的
`FormatGCNInput` 产出 `(M, T, V, C)`，`data_bn` 在 `(N*M, C, T*V)` 上做。
按字面顺序导出会得到一个能跑但结果全错的图。产物清单见
[`reference/MANIFEST_source.md`](../reference/MANIFEST_source.md)。

输出是 logits `(N, 2)`，softmax 留在 CPU 上做（只有两类，开销可忽略），
避开 `average_clip` 的 numpy 分支。

## 8. 判定

| 常量 | 值 | 出处 |
|---|---:|---|
| `THRESHOLD` | 0.5 | D-017 / D-023 |
| `BOARD_FPS` | 30.0 | `datasets/manifests/board/videos.csv` |

阈值 0.5 是源项目在验证集上定下的，不得在测试集或板端数据上重扫。

## 9. 结果表规模（对账基准）

| 量 | 值 | 为什么是这个数 |
|---|---:|---|
| 候选窗口 | 915 | 19 视频，`range(0, n-60+1, 15)`，不施门 |
| 样本 | 13,223 | A 913 + B 2,161 + C 10,149 |
| A 缺 2 个 | — | 915 窗里 2 个窗没有任何轨迹 → `extract_window` 返回 None |
| 施加门后 | 855 窗 / 1,189 样本 | `datasets/windows/board/windows.csv`，口径不同，别混用 |

915 与 855 不是矛盾：915 是不带门重建出来的，855 是事后施加 0.70 门之后的。
本项目的 runtime 一律按前者的口径实现。

## 10. 怎么复核这份文档

```bash
# 常量 x20、窗口表 x915、样本 x13223、拼接 x19 视频、本文档自身
python scripts/14_verify_spec.py

# 只要常量与拼接（约 10 秒，跳过 13k 样本重建）
python scripts/14_verify_spec.py --skip-samples
```

产物是 `reports/reference/spec_audit.json` 和 `spec_audit.md`。

这份文档有个局限：脚本只核对表格里的 `NAME = 数值` 与 `spec.py` 是否一致，
核对不了散文描述是否仍然正确。散文与代码的一致性靠评审，不靠脚本。
