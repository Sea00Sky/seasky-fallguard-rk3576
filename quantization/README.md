# 量化与部署验收

把训练好的两个模型（YOLO26n-pose + ST-GCN++）导出成 ONNX、转成 RKNN，并在 PC 上
复刻板端的流式管线，用业务指标验收转换是否忠实。

产物本身（`.onnx`、`.rknn`）没有随仓发布，只有跨平台可用的两个 FP16 `.rknn`
放在 `../models/`。本目录留下的是脚本、判据与实测记录。

## 环境

两个 conda 环境，不能混用：

- `pyskl`：导出时序模型、跑参考。torch 2.0.1、pyskl、onnx、onnxruntime。
- base：导出检测器、跑 runtime。ultralytics 8.4.108、onnx、onnxruntime、opencv。

RKNN 转换要在 WSL2 里做，rknn-toolkit2 2.3.0，venv 在 `~/rknn-env`，
搭建脚本见 `scripts/91_setup_wsl_rknn.sh`。在 Git Bash 里调 WSL 要写
`wsl -- bash -lc '...'` 并给路径加引号，否则 Git Bash 会先把 `~` 展开成 Windows
主目录，报 "No such file" 但退出码是 0，看起来像成功。

## 六道门的结果

| 门 | 内容 | 结果 |
|---|---|---|
| G0 | 冻结参考、CPU 确定性 | 通过 |
| G1 / G1b | 时序模型导出、重建张量端到端对账 | 通过 |
| G2 | 检测器导出与解码对账 | 通过 |
| G3 | 流式管线对账 | 通过，判据后来改过 |
| G4 | RKNN FP16 转换 | 通过（延迟不在范围内） |
| G5 | RKNN INT8 转换 | 不通过，两个模型都是，且是结论不是空白 |
| G6 | 终验 | 通过 |

G3 的判据换过一次。原来要求逐窗口概率差不超过 1e-3、并且每窗选中的轨迹相同，
后来证明这两条不可达：参考数据是用 CUDA FP16 生成的，里面含有重复框。
替代证据是"管线正确性"——把检测器的输出固定成参考值，只跑管线，1133 对样本的
最大概率差是 1.479e-06。G6 的判据改成"逐段命中判定是否一致"，19 段对 17 段、
no_fall 7 个视频全对；概率只列值、不设门。

## 三件必须知道的事

一是计划里有三处与实测不符，都按实测执行了。检测器的 `end2end` 不能关，
关掉会换成另一组权重的 one2many 头，关键点与训练时序模型时用的不同源；
输入不是 640×640，`LetterBox(auto=True)` 对 640×480 是恒等变换，真实输入是
480×640，补灰边会让关键点最大偏 485 像素；时序模型的输入维度顺序实测是
`(N,M,T,V,C)`，不是计划里写的 `(N,M,C,T,V)`，按字面导出会得到一个能跑但结果
全错的图。

二是 G1b 不能跳过。G1 只证明"ONNX 与 PyTorch 在我喂的那批张量上一致"，
如果张量重建本身错了，两边一起错、G1 照样通过。G1b 拿冻结的参考概率做端到端
对账，第一次跑就抓出了 913 个样本全错（双重归一化）。以后任何转换或管线改动，
都要配一份端到端对账，不能只做 A/B 一致性检查。

三是参考数据本身也要先验。G3 的逐窗差异查了很久，最后发现不在自己的实现里：
参考数据里有一批重复框（无 NMS 加上 TopK，让重叠锚点双双入选）。剔除持续重叠
5 帧以上的 23 条重复轨迹之后，各策略的段召回不变，但这个结论依赖那个阈值——
阈值取 2 或 3 时有一段会掉到 9/17，因为重叠里有一多半只是正常的换号交接。
所以冻结参考之前要先验它的检出数分布、框两两 IoU、dtype 和生成设备。

## INT8 的负结果

两个模型都失败，而且是同一个原因。

时序模型的决策一致率是 0.98823，低于 0.99 的门槛（1189 个样本里 14 个翻转）。
做 QAT 并重校准之后能到 0.9933，说明差距是结构性的，不是调参不够；但需要重训，
按约定不作交付。检测器更直接：`output0` 的 56 个通道共用一个 scale 3.1526，
而置信度量程只有 0 到 0.0353，量化之后置信度通道被清零，228 帧里 221 帧检出数
对不上。

共同原因是 RKNN 对图的输出张量强制使用单一量化 scale，而且这个张量的 dtype
不允许修改。输出张量里各分量量级差得越大伤害越大：时序模型只损失一个量级，
检测器直接丢掉整个低幅值通道。

试过的四条修复路径都有明确结论，不是"效果不好"而是"明确无效"：改张量 dtype
被 toolkit 直接拒绝；用 `custom_quantize_layers` 保护图输出被接受，但精度逐位
不变；换 `kl_divergence` 算法输出与默认算法逐位相同；`mmse` 和 `w8a16` 没拿到
数据（超时或构建失败无产物），这一条要在报告里和上面三条区分开。

还有一条 16 位的路线：`w16a16i`（16 位权重加 16 位整数激活，不是 INT8）配九输出
布局和密集校准，在模拟器上能让检测器的检出不一致降到 1/228 并过门。但它在板上
跑不起来，见 `../docs/BOARD_DEPLOYMENT.md`。所以交付统一用 FP16，这不是妥协
而是有板端证据的决定。

## 目录

```
src/deploy/     预处理常量的唯一定义处、路径与权重校验、两个导出脚本
src/runtime/    流式管线（检测、跟踪、环形缓冲、拼接、覆盖率门）
src/reference/  参考的寻址层，调用源项目的真函数而不是复制一份
scripts/        10–16 冻结与导出、20–29 转换与验证、30–37 回放、40–53 量化实验
preprocess_spec.md  预处理规格（常量与归一化顺序）
```

跑出来的原始 JSON（各道门的对拍数据、参考数据体检、多人场景审计）默认写在
`reports/` 下，那是运行期产物，不随仓库发布。

## 怎么跑

Phase 0（冻结参考，只需一次）和 Phase 1（导出与对账）：

```bash
python scripts/10_freeze_reference.py
python scripts/14_verify_spec.py          # 必须用 pyskl 环境，约 14 秒
python scripts/11_export_stgcn.py
python scripts/13_verify_reference_chain.py
python scripts/12_export_yolo.py
```

Phase 3 的管线正确性证据（约 30 秒）：

```bash
python scripts/31_pipeline_correctness.py
```

流式 runtime 对账，全量 10 到 16 分钟：

```bash
python scripts/30_runtime_eval.py
python scripts/30_runtime_eval.py --no-stitch    # 另一套口径
python scripts/30_runtime_eval.py --limit 3      # 快速冒烟，产物带后缀不覆盖全量
```

Phase 2 的 RKNN 转换与验证在 WSL2 里跑：

```bash
wsl -- bash -lc 'cd "<项目路径>" && ~/rknn-env/bin/python scripts/20_convert_rknn.py --precision fp16'
wsl -- bash -lc 'cd "<项目路径>" && ~/rknn-env/bin/python scripts/21_verify_rknn.py --precision fp16'
```

检测器换成交付的 FP16 图、同进程配对回放的业务指标验证约 90 分钟：

```bash
wsl -- bash -lc 'cd "<项目路径>" && ~/rknn-env/bin/python scripts/37_runtime_eval_rknn.py --backend both --precision fp16'
```

模拟器回放只有 2.9 fps（ONNX 是 20.0 fps），这是 CPU 模拟 NPU 的代价，
与板端速度无关，不要当成性能数字引用。

## 约束

- 源项目与旧部署目录只读，不修改。
- 板端数据只用于评估，绝不进入任何训练集或 pkl。
- 覆盖率门 0.70 不许改，也不在板端数据上调参，板端只做验收。
- 预处理常量只在 `src/deploy/spec.py` 定义一次，别处不许硬编码。
- 同代码两次全量跑出的帧率是 28.13 对 15.25 fps，引用性能数字必须注明测量条件。

文中 `D-xxx` 是项目内部的决策编号，完整记录没有随仓发布。
