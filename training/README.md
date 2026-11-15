# 训练侧

视频进 YOLO26n-pose 出 COCO-17 关键点，攒成骨架序列，再由 ST-GCN++ 判跌倒 / 非跌倒。
本目录只放训练相关的东西，板端实时推理在 `../board/`。

## 数据

三个公开数据集，原始文件只读：

| 数据集 | 内容 | 用法 |
|---|---|---|
| Le2i | 190 段视频，130 段有标注 | 跌倒区间加同场景负样本 |
| MCFD | 24 个 chute × 8 机位，120 fps | 多机位跌倒，按 chute 切分 |
| URFD | 30 段跌倒加 40 段日常，双视图 | 跨数据集测试，只用 RGB 半幅 |

数据是从下往上分层建的：原始只读，往上是规范化标注、视频与事件清单、关键点、
窗口，最后是按事件切分的训练集，每层都有 QC 判据门。

标注口径上有几条硬约束：主标签是二分类；切分按事件而不是按帧，同一个事件不能
跨 split；MCFD 同一个 chute 必须落在同一个 split；URFD 的左右视图算同一个事件。

窗口口径与板端一致：长度 2 秒（生产按帧数是 60 帧）、步长 0.5 秒、重采样到 64 帧，
关键点按帧尺寸归一化到 `[0,1]` 再映射到 `[-1,1]`。训练视频多是 24 到 30 fps，
所以一个窗口对应 48 到 60 个源帧。

## 阶段与脚本

`scripts/` 里的编号就是阶段顺序：00 到 07 建标注与人工区间，08 到 12 建清单与
关键点、窗口并做 QC，13 导出 PYSKL 格式，15 训练，16 到 32 是板端评测、集成、
模型选择、种子扫描与覆盖率门扫描。

已经走完的部分：数据层（原始数据 → 标注 → 清单 → 事件级切分 → QC）、关键点抽取、
窗口构建、PYSKL 导出、ST-GCN++ 训练与多组对照、板端测试台推理与真值评估。

```bash
python scripts/00_env_check.py
python scripts/01_copy_raw.py
python scripts/02_build_annotations.py
python scripts/03_build_manifest_splits.py
python scripts/04_qc_foundation.py
python scripts/09_extract_keypoints.py
python scripts/11_build_windows.py
python scripts/13_export_pyskl.py
python scripts/15_train_stgcn.py
```

## 配置

`configs/` 下除了数据与标签口径，还有一组训练配置：基线、数据增强、类别权重、
onset 标签、预训练初始化（三个学习率）、checkpoint 续训，以及 v3 版。跑过的对照
结果以 CSV 和 JSON 留在 `reports/` 里（种子扫描、预训练微调扫描、模型选择、
v2 与 v3 对比、集成评测、板端评测）。这些对照的结论文字没有随仓发布，
数字都在那些文件里。

## 权重

训练权重不进仓库。板端交付用的两个模型在 `../models/`，它们的来源、哈希和输入
形状见 `../reference/MANIFEST_source.md`。
