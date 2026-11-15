# 源权重与产物清单

所有权重和产物都用 sha256 冻结。任何一条对不上，说明它不是本项目数字的出处，
后面的对账都作废。`../quantization/src/deploy/paths.py` 里的 `verify_weights()`
会在启动时直接拒绝。

这份清单里的文件都留在原项目里，没有随仓发布；仓库只带两个交付的 FP16 `.rknn`。

## 源权重（输入，只读）

| 文件 | 字节 | sha256 | 出处 |
|---|---:|---|---|
| `weights/yolo26n-pose.pt` | 7,878,574 | `eb3bb826…c14522a9` | ultralytics 8.4.108 的 Pose26 头，3.68 M 参数 |
| `weights/stgcnpp_v2_best.pth` | 11,438,579 | `8cd5e74e…b7ab29d` | 时序模型基线，epoch 19，1.38 M 参数 |
| `weights/stgcnpp_v2_config_used.yaml` | 766 | `5ee701e6…2fe285c` | 训练时的配置快照，用来重建模型结构 |

时序模型的权重与源项目 `runs/stgcnpp_v2/best.pth` 是同一份字节。这是能用冻结的
参考概率做端到端对账的前提。

## 导出的 ONNX

| 文件 | 字节 | sha256 | 形状 | 门 |
|---|---:|---|---|---|
| `models/stgcnpp_v2.onnx` | 5,609,754 | `343ee0d6…9dbfed19` | `(1,1,64,17,3)` → `(1,2)` | G1 / G1b |
| `models/yolo26n_pose_480x640.onnx` | 12,001,226 | `4eb65ab9…f389deef` | `(1,3,480,640)` → `(1,56,6300)` | G2 |
| `models/stgcnpp_v2_b4.onnx` | 5,649,182 | `497f6dab…4b0ab0f1` | `(N,1,64,17,3)` → `(N,2)` | 批处理 |

批处理那份是 PC 侧的伴随产物，不是 RKNN 的转换输入。`stgcnpp_v2.onnx` 是静态
batch-1，没有改过，仍然是交付图和转换的唯一来源，G1/G4/G5 的数字都出自它的哈希。
板端批处理走 `rknn.build(rknn_batch_size=N)`，作用在同一张静态图上，三层实测
逐位等价。

形状为什么是这样：时序模型是 `(N, M, T, V, C)`，PYSKL 的 `FormatGCNInput` 产出
`(M, T, V, C)`，按计划里字面写的 `(N,M,C,T,V)` 导出会得到一个能跑但结果全错的图。
检测器是 `(1,3,480,640)`，因为 `LetterBox(auto=True)` 对 640×480 是恒等变换，
补灰边会让关键点最大偏 485 像素。输出 `(1,56,6300)` 里 56 是 4 个框加 1 个置信度
加 17×3 个关键点，6300 是三个 stride 的特征图加起来；`topk(300)` 被剥掉了，
NMS 也不做，因为端到端模式下 ultralytics 自己就不跑。

## 图结构核对

| 项 | 时序模型 | 检测器 |
|---|---:|---:|
| opset | 12 | 12 |
| 节点数 | 364 | 550 |
| Einsum | 0（已改写成 10 个 MatMul 加 10 个 ReduceSum） | — |
| TopK | — | 0 |
| NonMaxSuppression | — | 0 |
| Conv | 125 | 111 |

## RKNN 产物

| 文件 | 字节 | sha256 | 门 |
|---|---:|---|---|
| `models/stgcnpp_v2_fp16.rknn` | 13,386,094 | `3aa0d39a…ab47ede` | 通过 |
| `models/yolo26n_pose_480x640_fp16.rknn` | 12,123,446 | `81ca4bb8…ad4db4` | 通过 |
| `models/stgcnpp_v2_int8.rknn` | 8,085,192 | `2b65ce47…045ff02` | 0.98823，不到 0.99 |
| `models/yolo26n_pose_480x640_int8.rknn` | 9,402,128 | `01917178…e247ce2` | 置信度通道归零 |
| `models/stgcnpp_v2_int8_hybrid.rknn` | 8,091,195 | `f1edd923…0c9f93e5` | 与普通 INT8 逐位同精度 |
| `models/yolo26n_pose_480x640_int8_hybrid.rknn` | 9,408,131 | `8040aaa3…5d06e720` | 与普通 INT8 同精度 |

环境是 Ubuntu 24.04.1、Python 3.12.3、rknn-toolkit2 2.3.0、torch 2.4.0、
onnx 1.16.2、numpy 1.26.4，venv 在 `~/rknn-env`。目标平台 `rk3576`。

`.rknn` 的 sha256 不是可复现量，toolkit 会把时间戳和构建信息写进文件头，
同配置重转一次哈希就变。这里冻结哈希是为了回答"报告里的数字出自哪个文件"，
不是要求重转后对得上。字节级可复现的是 ONNX 那一层。

FP16 的验证对象其实是"这套转换配方"而不是"这个文件"：toolkit 2.3.0 不允许对
`load_rknn` 载入的模型做模拟器推理，所以验证脚本是用同一份配置从 ONNX 重建之后
再比。交付文件真正被加载是上板那一次。

## INT8 的两个结论

时序模型不通过，而且这是结论不是缺陷。两个类别的输出层被量化到 0.078 的网格上，
决策要的是两个 logit 差值的符号，没有冗余可以平均，margin 被系统性压缩，
1189 个样本里 14 个翻转。QAT 重校准后能过，但需要重训。

检测器的失败更彻底：`output0` 的 56 个通道共用一个 scale 3.1526，而置信度量程
只有 0 到 0.0353，量化之后置信度和关键点分数被清零。损伤发生在拼接处，
比检测头更靠后。用 `--hybrid` 保护 `output0` 被 toolkit 接受、也确实把输出 dtype
改成了 fp16，但精度逐位不变。

`kl_divergence` 算法那条已经删掉：它的输出与默认算法逐位相同，留着只会让人以为
它是个独立配置。`--hybrid` 保留着，因为它是唯一能表达"保护这个张量"的接口，
记录下"试过、无效"本身也是结论，默认构建仍是普通 INT8。

## 实验性产物

2026-10-25 还做过一批九输出布局（t9）加 16 位的实验，用来找量化天花板。
结论有两条值得记：16 位权重加 16 位整数激活不是 INT8，它能把检测器的失败机制
从"排序置换"退化成"阈值抖动"（翻转锚点到阈值的距离中位数从 0.147 降到
0.000147），并且在密集校准下能让检出不一致降到 1/228；但这些产物在板上跑不起来
（16 位激活与当前驱动不兼容），而且 `kl_divergence` 换回来又会退回排序置换，
说明这个结果对校准算法极敏感。

这些文件都不进交付路径，交付仍是两个 FP16 图。

## 复现

```bash
# pyskl 环境：导出时序模型并做参考链对账
python scripts/11_export_stgcn.py
python scripts/13_verify_reference_chain.py

# base 环境：导出检测器，跑 runtime
python scripts/12_export_yolo.py
python scripts/30_runtime_eval.py
```

两次导出应当字节级可复现。如果 sha256 变了但门仍然通过，先查 torch 和 onnx 的
版本，再查是否有算子被常量折叠成了不同顺序，不要直接更新这份清单里的哈希。
