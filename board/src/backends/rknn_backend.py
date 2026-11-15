"""RKNN 后端的两个适配器：把 `rknnlite` 包成管线期望的接口。"""
from __future__ import annotations

import numpy as np

from ..yolo_decode import INPUT_SHAPE, decode, preprocess

# 与源项目 `src/deploy/spec.py` 保持一致。
CONF = 0.25
MAX_PERSONS = 4
IMGSZ = 640

LUT_F16 = (np.arange(256, dtype=np.float16) / np.float16(255.0))


class Detection:
    """一帧里的一个人，坐标已是原图像素。"""

    __slots__ = ("bbox", "keypoints", "scores", "track_id")

    def __init__(self, bbox: np.ndarray, keypoints: np.ndarray,
                 scores: np.ndarray, track_id: int = -1) -> None:
        self.bbox = bbox
        self.keypoints = keypoints
        self.scores = scores
        self.track_id = track_id


class YoloBackend:
    """RKNNLite 上的 YOLO26n-pose。"""

    def __init__(self, rknn, conf: float = CONF,
                 max_persons: int = MAX_PERSONS,
                 fast_f16: bool = False) -> None:
        self.rknn = rknn
        self.conf = float(conf)
        self.max_persons = int(max_persons)
        self.n_calls = 0
        self.fast_f16 = bool(fast_f16)
        self._rgb = None
        self._f16 = None

    def _prep_f16_nhwc(self, frame_bgr: np.ndarray) -> np.ndarray:
        """BGR uint8 → f16 NHWC，预分配缓冲 + cv2.LUT(dst=)）。"""
        import cv2
        h, w = frame_bgr.shape[:2]
        if (h, w) != (INPUT_SHAPE[2], INPUT_SHAPE[3]):
            raise ValueError(
                f"fast_f16 只支持 {INPUT_SHAPE[3]}x{INPUT_SHAPE[2]}，"
                f"收到 {w}x{h}；换分辨率必须重新导出模型。")
        if self._rgb is None or self._rgb.shape != frame_bgr.shape:
            self._rgb = np.empty(frame_bgr.shape, dtype=np.uint8)
            self._f16 = np.empty(frame_bgr.shape, dtype=np.float16)
        cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB, dst=self._rgb)
        # 换 cv2.LUT：板上实测 4.36 ms → 0.88 ms（逐位相同，用生产
        # 视频真实帧对拍 uint16 位型）。np.take 是纯 C 但按元素 gather，OpenCV 的
        # LUT 有 SIMD 化的 256 项表查表。dst= 是预分配缓冲 ⇒ 每帧不 malloc 1.84 MB。
        cv2.LUT(self._rgb, LUT_F16, dst=self._f16)
        return self._f16[None]

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        """一帧 BGR uint8 -> 按置信度降序的检出列表。"""
        if self.fast_f16:
            tensor = self._prep_f16_nhwc(frame_bgr)
            h, w = frame_bgr.shape[:2]
            # 640×480 下 LetterBox 是恒等变换，参数写死即可。
            params = {"ratio": 1.0, "pad": (0, 0), "src": (w, h),
                      "shape": (h, w), "auto": True}
            outs = self.rknn.inference(
                inputs=[tensor], data_type="float16", data_format="nhwc",
                inputs_pass_through=[1])
        else:
            chw, params = preprocess(frame_bgr, imgsz=IMGSZ)
            if chw.shape != INPUT_SHAPE:
                raise ValueError(
                    f"预处理产出 {chw.shape}，图要的是 {INPUT_SHAPE}。"
                    f"导出的图静态固定在 480x640；所有板端视频都是 640x480，"
                    f"此时 LetterBox(auto=True) 是恒等变换。"
                    f"别的分辨率需要单独导出。")

            # `data_format="nchw"` 必须显式写：省略时 4-D 输入默认 nhwc，
            # (1,3,480,640) 会被拒。写 "numpy" 也不接受，且返回 None 而非抛异常。
            outs = self.rknn.inference(inputs=[chw], data_format="nchw")
        if outs is None:
            raise RuntimeError(
                "RKNN YOLO 推理返回 None——真正的错误在 toolkit/rknnlite 的"
                "日志里，它不抛异常。检查 data_format、输入形状、librknnrt 版本。")

        raw = np.asarray(outs[0])
        self.n_calls += 1
        boxes, kpts, scores = decode(raw, params, conf=self.conf,
                                     max_persons=self.max_persons)
        return [Detection(bbox=boxes[i], keypoints=kpts[i], scores=scores[i])
                for i in range(len(boxes))]


class _Input:
    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name


class StgcnBackend:
    """RKNNLite 上的 ST-GCN++，伪装成 onnxruntime 的 `InferenceSession`。"""

    def __init__(self, rknn, input_name: str = "input") -> None:
        self.rknn = rknn
        self._inputs = [_Input(input_name)]
        self.n_calls = 0

    def get_inputs(self):
        return self._inputs

    def run(self, _output_names, feeds):
        x = next(iter(feeds.values()))
        # 图是 5-D，默认布局恰好能用，但仍然显式写 nchw，
        # 免得将来与 YOLO 那条路径的行为漂移。
        outs = self.rknn.inference(inputs=[x], data_format="nchw")
        if outs is None:
            raise RuntimeError(
                "RKNN ST-GCN 推理返回 None——错误在 rknnlite 日志里，它不抛异常")
        self.n_calls += 1
        logits = np.asarray(outs[0])
        if logits.shape[0] != x.shape[0]:
            raise RuntimeError(
                f"ST-GCN 返回 {logits.shape[0]} 行，喂了 {x.shape[0]} 个样本。"
                f"batch-N 图喂不满会广播而不是报错，所以这里显式检查。")
        return [logits]
