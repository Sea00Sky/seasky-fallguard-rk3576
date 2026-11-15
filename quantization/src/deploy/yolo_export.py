"""YOLO26n-pose export wrapper and the CPU-side decode."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

# Output layout of the exported graph.
N_BOX = 4
N_CLS = 1
N_KPT = 17
KPT_DIM = 3
OUT_CHANNELS = N_BOX + N_CLS + N_KPT * KPT_DIM      # 56
# 480x640 -> stride 8/16/32 feature maps 60x80 + 30x40 + 15x20.
INPUT_SHAPE = (1, 3, 480, 640)
N_ANCHORS = (480 // 8) * (640 // 8) + (480 // 16) * (640 // 16) + (480 // 32) * (640 // 32)


class YOLOPoseRaw(nn.Module):
    """The one2one detection head with TopK/Gather stripped off."""

    def __init__(self, model: nn.Module, end2end_branch: bool = True) -> None:
        super().__init__()
        self.net = model
        self.branch = "one2one" if end2end_branch else "one2many"
        # `export = True` switches the head's decode to the static, tensor-only
        # path (`kpts_decode` export branch), which is what we want in a graph.
        self.net.model[-1].export = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y: list = []
        for m in self.net.model[:-1]:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.net.save else None)
        head = self.net.model[-1]
        preds = head.forward_head([y[i] for i in head.f], **getattr(head, self.branch))
        return head._inference(preds)


# ------------------------------------------------------------------ CPU decode


def preprocess(frame_bgr: np.ndarray, imgsz: int = 640, stride: int = 32,
               auto: bool = True):
    """BGR uint8 frame -> (1, 3, H, W) float32 RGB in 0..1, plus params."""
    import cv2

    h, w = frame_bgr.shape[:2]
    r = min(imgsz / h, imgsz / w)
    new_w, new_h = round(w * r), round(h * r)          # `new_unpad`
    dw, dh = imgsz - new_w, imgsz - new_h
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    dw, dh = dw / 2, dh / 2                            # center=True
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)

    out_h, out_w = new_h + top + bottom, new_w + left + right
    if (new_w, new_h) != (w, h):
        interp = cv2.INTER_LINEAR if r > 1 else cv2.INTER_AREA
        resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=interp)
    else:
        resized = frame_bgr

    canvas = np.full((out_h, out_w, 3), 114, dtype=np.uint8)
    canvas[top:top + new_h, left:left + new_w] = resized
    rgb = canvas[:, :, ::-1].astype(np.float32) / 255.0
    chw = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])
    return chw, {"ratio": r, "pad": (left, top), "src": (w, h),
                 "shape": (out_h, out_w), "auto": auto}


def unscale_boxes(boxes: np.ndarray, params: dict) -> np.ndarray:
    """Letterboxed pixels -> original-image pixels, without clipping."""
    r, (left, top) = params["ratio"], params["pad"]
    out = np.array(boxes, dtype=np.float32, copy=True)
    out[:, [0, 2]] -= left
    out[:, [1, 3]] -= top
    out /= r
    return out


def clip_boxes(boxes: np.ndarray, params: dict) -> np.ndarray:
    """Clamp to the source frame, as ultralytics does after NMS."""
    w, h = params["src"]
    out = np.array(boxes, dtype=np.float32, copy=True)
    out[:, [0, 2]] = out[:, [0, 2]].clip(0, w)
    out[:, [1, 3]] = out[:, [1, 3]].clip(0, h)
    return out


def scale_boxes(boxes: np.ndarray, params: dict) -> np.ndarray:
    """Unscale then clip. Kept as the one-call form for callers that want both."""
    return clip_boxes(unscale_boxes(boxes, params), params)


def decode(
    out: np.ndarray,
    params: dict,
    conf: float = 0.25,
    max_persons: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(1, 56, 6720) -> boxes [n,4], keypoints [n,17,2], scores [n,17]."""
    pred = np.asarray(out, dtype=np.float32)[0].T          # (N_ANCHORS, 56)
    cls_conf = pred[:, 4]

    # ultralytics uses a strict `>` here (`conf.view(-1) > conf_thres`); with
    # `>=` a box sitting exactly on 0.25 survives on one side only and the
    # per-frame detection count diverges.
    keep = np.nonzero(cls_conf > conf)[0]
    if keep.size == 0:
        return (np.zeros((0, 4), np.float32), np.zeros((0, N_KPT, 2), np.float32),
                np.zeros((0, N_KPT), np.float32))

    boxes = scale_boxes(pred[keep, 0:4], params)
    kpts = pred[keep, 5:].reshape(-1, N_KPT, KPT_DIM)

    r, (left, top) = params["ratio"], params["pad"]
    kpts = np.array(kpts, dtype=np.float32, copy=True)
    kpts[:, :, 0] = (kpts[:, :, 0] - left) / r
    kpts[:, :, 1] = (kpts[:, :, 1] - top) / r
    # ultralytics' `scale_coords` ends in `clip_coords`, so out-of-frame joints
    # are clamped to the image border. Skipping this leaves joints up to
    # ~23 px outside the frame and desynchronises G2.
    w, h = params["src"]
    kpts[:, :, 0] = kpts[:, :, 0].clip(0, w)
    kpts[:, :, 1] = kpts[:, :, 1].clip(0, h)

    order = np.argsort(-cls_conf[keep], kind="stable")[:max_persons]
    # (x, y) and the per-joint confidence are returned separately: the source
    # pipeline stores `keypoints [T,P,17,2]` and `scores [T,P,17]`.
    return (boxes[order].astype(np.float32),
            kpts[order, :, :2].astype(np.float32),
            kpts[order, :, 2].astype(np.float32))
