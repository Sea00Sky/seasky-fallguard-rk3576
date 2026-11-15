# Models

交付的两个 FP16 RKNN 图，就是在板端跑的那一份配置。

| 文件 | 大小 | MD5 | 输入 |
|---|---:|---|---|
| `yolo26n_pose_480x640_fp16.rknn` | 12,123,446 B | `7897533d61af4f1210db074420c68f4c` | `(1,3,480,640)` NCHW float32 |
| `stgcnpp_v2_fp16.rknn` | 13,386,094 B | `a319cb0456b92d9662de0c65edbcbec4` | `(1,1,64,17,3)` float32 |

检测器输出 COCO-17 关键点，分类器吃一段归一化骨架序列、返回跌倒/非跌倒两个
logit。两个图的转换配置里 mean 都是 0、std 都是 1，归一化必须在送进 NPU 之前
做完，板端不要再减均值或除方差。推理时检测器要显式指定 `data_format="nchw"`，
时序模型是 5 维输入，`mean_values` 只能给一个值。

## 来源

两个文件都由 rknn-toolkit2 2.3.0 从 `quantization/` 导出的 ONNX 转换而来，目标
平台 `rk3576`。检测器源自 Ultralytics YOLO26n-pose，分类器源自 MMAction2
ST-GCN++。许可证条款见 `../NOTICE`。

没有附带 INT8 版本。按交付验收标准，INT8 检测器解码出的框数对不上（32/228），
INT8 分类器的一致率 0.98823 低于 0.99 的门槛，FP16 才是交付配置。

## 运行环境

需要 librknnrt 2.3.x。仓库里不附带这个运行库，从瑞芯微 SDK 装完后核对版本：

```bash
strings /usr/lib/librknnrt.so | grep -i "librknnrt version"
```

`RKNNLite` 加载的是硬编码路径 `/usr/lib/librknnrt.so`，用 `LD_LIBRARY_PATH`
覆盖不了它。

板卡上只有两个 NPU 核。三个 RKNN 会话同时推理会把 NPU 挂死（job timeout 之后
无限 soft reset），同时最多两个；同一个会话也不能被两个线程共用，每个线程要
各建一个实例。
