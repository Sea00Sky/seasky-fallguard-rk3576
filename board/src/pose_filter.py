"""检出净化：把「一个人被检出好几个框」按下去。"""
from __future__ import annotations

import numpy as np


def _area(b) -> float:
    return max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))


def _inter(a, b) -> float:
    iw = max(0.0, min(float(a[2]), float(b[2])) - max(float(a[0]), float(b[0])))
    ih = max(0.0, min(float(a[3]), float(b[3])) - max(float(a[1]), float(b[1])))
    return iw * ih


def iou(a, b) -> float:
    inter = _inter(a, b)
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0


def containment(outer, inner) -> float:
    """`inner` 有多大比例落在 `outer` 里（0–1）。"""
    ai = _area(inner)
    return _inter(outer, inner) / ai if ai > 0 else 0.0


def ios(a, b) -> float:
    """交集 / 较小的那个面积（"谁被谁包住"的对称版）。"""
    inter = _inter(a, b)
    m = min(_area(a), _area(b))
    return inter / m if m > 0 else 0.0


def count_kp(scores, kp_conf: float) -> int:
    return int(np.count_nonzero(np.asarray(scores) > float(kp_conf)))


def continuity(box, prev_boxes) -> float:
    """这个框和上一帧已有轨迹的最大 IoU —— 身份连续性的代理量。"""
    if prev_boxes is None or len(prev_boxes) == 0:
        return 0.0
    best = 0.0
    for pb in prev_boxes:
        v = iou(box, pb)
        if v > best:
            best = v
    return best


def filter_detections(detections, min_kp: int = 4, kp_conf: float = 0.3,
                      contain: float = 0.55, iou_thr: float = 0.70,
                      prev_boxes=None):
    """返回净化后的检出列表（顺序可能变）。参数为 0/负 时对应那一闸关闭。"""
    if not detections:
        return detections

    kept = list(detections)

    # --- 闸 1：姿态有效性 ------------------------------------------------
    if min_kp and min_kp > 0:
        kept = [d for d in kept if count_kp(d.scores, kp_conf) >= min_kp]
        if not kept:
            return []

    # --- 闸 2：嵌套抑制（谁能接上上一帧的身份，谁赢）--------------------
    if (contain and contain > 0) or (iou_thr and iou_thr > 0):
        def _rank(d):
            sc = np.asarray(d.scores, dtype=np.float32)
            return (-continuity(d.bbox, prev_boxes), -_area(d.bbox),
                    -float(sc.mean()) if sc.size else 0.0)

        ordered = sorted(kept, key=_rank)
        out = []
        for det in ordered:
            drop = False
            for other in out:
                if iou_thr and iou_thr > 0 and \
                        iou(det.bbox, other.bbox) >= iou_thr:
                    drop = True
                    break
                if contain and contain > 0 and \
                        ios(det.bbox, other.bbox) >= contain:
                    drop = True
                    break
            if not drop:
                out.append(det)
        kept = out

    return kept


def apply_to(detections, cfg, prev_boxes=None) -> tuple[list, int, int]:
    """按 `PipelineConfig` 的字段净化，返回 `(detections, 丢了几个姿态无效的, 丢了几个重复的)`。"""
    n0 = len(detections)
    got = filter_detections(
        detections,
        min_kp=getattr(cfg, "pose_min_kp", 4),
        kp_conf=getattr(cfg, "pose_kp_conf", 0.3),
        contain=getattr(cfg, "dedupe_contain", 0.55),
        iou_thr=getattr(cfg, "dedupe_iou", 0.70),
        prev_boxes=prev_boxes,
    )
    n1 = len(got)
    # 两道闸的贡献分开报：先跑只有闸 1 的结果，差值就是闸 2 丢的。
    if n0 and (getattr(cfg, "pose_min_kp", 0) or 0) > 0:
        only_kp = sum(1 for d in detections
                      if count_kp(d.scores, getattr(cfg, "pose_kp_conf", 0.3))
                      >= int(cfg.pose_min_kp))
    else:
        only_kp = n0
    return got, n0 - only_kp, only_kp - n1
