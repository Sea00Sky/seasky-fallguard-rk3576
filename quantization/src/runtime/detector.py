"""Frame -> detections, on top of the exported YOLO ONNX graph."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..deploy import spec, yolo_export


@dataclass
class Detection:
    """One person in one frame, in original-image pixels."""

    bbox: np.ndarray
    keypoints: np.ndarray
    scores: np.ndarray
    track_id: int = -1


class Detector:
    """Wraps the YOLO26n-pose ONNX graph."""

    def __init__(self, model_path: Path, conf: float = spec.CONF,
                 max_persons: int = spec.MAX_PERSONS) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.session = ort.InferenceSession(
            str(model_path), opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.conf = float(conf)
        self.max_persons = int(max_persons)
        shape = self.session.get_inputs()[0].shape
        self.input_shape = tuple(shape)
        if tuple(shape) != yolo_export.INPUT_SHAPE:
            raise ValueError(
                f"{model_path.name} has input {tuple(shape)}, expected "
                f"{yolo_export.INPUT_SHAPE}; re-export with scripts/12")

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        """One BGR uint8 frame -> detections sorted by confidence descending."""
        chw, params = yolo_export.preprocess(frame_bgr, imgsz=spec.IMGSZ)
        if chw.shape != self.input_shape:
            raise ValueError(
                f"preprocess produced {chw.shape} but the graph wants "
                f"{self.input_shape}. The exported graph is static at "
                f"{yolo_export.INPUT_SHAPE}; every board clip is 640x480 so "
                f"LetterBox(auto=True) is the identity. A different frame size "
                f"needs its own export.")
        raw = self.session.run(None, {self.input_name: chw})[0]
        boxes, kpts, scores = yolo_export.decode(
            raw, params, conf=self.conf, max_persons=self.max_persons)
        return [Detection(bbox=boxes[i], keypoints=kpts[i], scores=scores[i])
                for i in range(len(boxes))]
