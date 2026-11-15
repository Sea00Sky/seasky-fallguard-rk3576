"""8888 标注画布自检（PC，不需要 NPU / 摄像头 / 视频）。"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.multi import MultiConfig, MultiPipeline, overlay        # noqa: E402
from src.pipeline import PipelineConfig                          # noqa: E402
from src.ring import RingBuffer                                  # noqa: E402

OK = True


def check(name, got, want):
    global OK
    good = got == want
    OK = OK and good
    print(f"  {'PASS' if good else 'FAIL'}  {name}  got={got!r} want={want!r}")


# ---------------------------------------------------------------- 假后端

def _fake_backends(n_people_at):
    """确定性假后端：不需要 NPU，只为验证本仓库的代码。"""
    from src.backends import Detection

    class FakeDetector:
        def __init__(self):
            self.n_calls = 0

        def detect(self, frame):
            n = n_people_at(self.n_calls)
            self.n_calls += 1
            out = []
            for i in range(n):
                x = 10.0 + 120.0 * i          # 固定不动 ⇒ IoU=1 ⇒ tracker 稳定续号
                kp = np.tile(np.array([[x + 50.0, 100.0]], np.float32), (17, 1))
                out.append(Detection(
                    bbox=np.array([x, 10.0, x + 100.0, 210.0], np.float32),
                    keypoints=kp, scores=np.full((17,), 0.9, np.float32)))
            return out

    class FakeStgcn:
        def __init__(self):
            self._inputs = [type("I", (), {"name": "input"})()]

        def get_inputs(self):
            return self._inputs

        def run(self, _outs, feeds):
            p = 0.9                            # 恒定高概率 ⇒ 出 confirmed 事件
            return [np.array([[np.log(1 - p), np.log(p)]], np.float32)]

    return FakeDetector, FakeStgcn


def _spread_kp(x, y, w, h, conf=0.9):
    """17 个点铺开成人形。"""
    kp = np.zeros((17, 3), np.float32)
    for i in range(17):
        kp[i, 0] = x + w * (0.25 + 0.5 * (i % 3) / 2.0)
        kp[i, 1] = y + h * (i / 16.0)
        kp[i, 2] = conf
    return kp


def _build(n_people=1):
    FakeDetector, FakeStgcn = _fake_backends(lambda f: n_people)
    return MultiPipeline(
        PipelineConfig(fps=30.0),
        detector=FakeDetector(), stgcn_session=FakeStgcn(),
        mcfg=MultiConfig(confirm_windows=1, retract_windows=1,
                         cooldown_s=0.0, max_persons_stgcn=3))


# ---------------------------------------------------------------- V1

def v1_ring_latest():
    print("[V1] RingBuffer.latest()")
    rb = RingBuffer(window_frames=60)
    kp = np.arange(34, dtype=np.float32).reshape(17, 2)
    sc = np.full((17,), 0.8, np.float32)

    rb.update(7, 0, kp, sc)
    got_kp, got_sc = rb.latest(7)
    check("刚写入就能取到", bool(np.array_equal(got_kp, kp)), True)

    # 第 1 帧检测器没看到这个人 ⇒ 洞
    rb.mark_missing(1)
    got_kp2, _ = rb.latest(7)
    check("洞不冒充（仍是第 0 帧的数据）", bool(np.array_equal(got_kp2, kp)), True)
    check("last_frame 不因洞推进", rb.last_frame(7), 0)

    # 第 2 帧又看到，但换了位置
    kp2 = kp + 5.0
    rb.update(7, 2, kp2, sc)
    got_kp3, _ = rb.latest(7)
    check("再看到时更新到新位置", bool(np.array_equal(got_kp3, kp2)), True)

    check("不存在的 track", rb.latest(99), None)

    rb.expire(2 + 60)                     # max_age = window_frames-1 = 59
    check("expire 后返回 None", rb.latest(7), None)


# ---------------------------------------------------------------- V2

def v2_display_persons():
    print("[V2] MultiPipeline.display_persons()")
    pipe = _build(n_people=1)
    frame = np.zeros((240, 400, 3), np.uint8)
    for _ in range(80):
        pipe.push(frame)

    before = (len(pipe._per_person), len(pipe._events),
              len(pipe.ring.track_ids()), pipe._frame)
    persons = pipe.display_persons()
    after = (len(pipe._per_person), len(pipe._events),
             len(pipe.ring.track_ids()), pipe._frame)

    check("只读：四个状态量不变", after, before)
    check("有 1 个人", len(persons), 1)
    p = persons[0]
    check("字段齐全",
          sorted(p.keys()),
          ["bbox", "coverage", "keypoints", "person_id", "prob", "status",
           "track_id"])
    check("keypoints 形状", p["keypoints"].shape, (17, 3))
    check("bbox 是 4 个数", len(p["bbox"]), 4)
    check("bbox 与 tracker 一致",
          [round(v, 3) for v in p["bbox"]],
          [round(float(v), 3) for v in pipe.tracker.bbox_of(p["track_id"])])
    check("出过窗口后 prob 有值", p["prob"] is not None, True)
    check("prob 在 [0,1]", 0.0 <= float(p["prob"]) <= 1.0, True)

    # 人消失 ⇒ 2 帧后不再画
    for _ in range(5):
        pipe.push(frame)
    FakeDetector, _ = _fake_backends(lambda f: 0)
    pipe.detector = FakeDetector()
    for _ in range(6):
        pipe.push(frame)
    check("消失的人不再出现", len(pipe.display_persons()), 0)


# ---------------------------------------------------------------- V3

def v3_render():
    print("[V3] overlay.render()")
    frame = np.full((240, 400, 3), 40, np.uint8)
    persons = [{
        "track_id": 3, "person_id": 3,
        "bbox": [10.0, 10.0, 110.0, 210.0],
        "keypoints": _spread_kp(10.0, 10.0, 100.0, 200.0),
        "prob": 0.87, "coverage": 0.92, "status": "scored",
    }]
    before = frame.copy()

    canvas = overlay.render(frame, persons, falling=True, fps=22.4)
    check("原帧一个字节没改", bool(np.array_equal(frame, before)), True)
    check("画布是新数组（不是原帧）", canvas is frame, False)
    check("形状不变", canvas.shape, frame.shape)
    check("确实画上了东西", bool(np.any(canvas != frame)), True)

    canvas2 = overlay.render(frame, persons, falling=False, fps=22.4,
                             canvas=canvas)
    check("画布跨帧复用（同一对象）", canvas2 is canvas, True)

    # 关键点置信度低于阈值 ⇒ 一根绿线都不该有（框/标签仍在，所以不能数总差异）
    def n_limb(im):
        # 不能用「精确等于 COLOR_LIMB」：cv2.LINE_AA 会抗锯齿混色，
        # 线芯才接近纯绿。按色相判：G 高、R/B 低。
        b = im[:, :, 0].astype(np.int16)
        g = im[:, :, 1].astype(np.int16)
        r = im[:, :, 2].astype(np.int16)
        return int(np.count_nonzero((g > 150) & (r < 100) & (b < 100)))

    check("高置信度画了绿线", n_limb(canvas) > 0, True)
    low = [dict(persons[0], keypoints=persons[0]["keypoints"].copy())]
    low[0]["keypoints"][:, 2] = 0.1
    bare = overlay.render(frame, low, falling=False, canvas=None)
    check("低置信度一根绿线都没有", n_limb(bare), 0)

    # 无人 + 无横幅 ⇒ 与原帧逐字节相同
    empty = overlay.render(frame, [], falling=False, canvas=None)
    check("空场景不画任何东西", bool(np.array_equal(empty, frame)), True)


# ---------------------------------------------------------------- V4

def v4_cost():
    print("[V4] 渲染成本（PC 参考值，板端另测）")
    frame = np.random.randint(0, 255, (480, 640, 3), np.uint8)

    def mk(n):
        out = []
        for i in range(n):
            x = 20.0 + 150.0 * i
            out.append({
                "track_id": i, "person_id": i,
                "bbox": [x, 40.0, x + 130.0, 400.0],
                "keypoints": _spread_kp(x, 40.0, 130.0, 360.0),
                "prob": 0.8, "coverage": 0.9, "status": "scored"})
        return out

    for n in (1, 3):
        ps = mk(n)
        cv = None
        for _ in range(20):                # 预热
            cv = overlay.render(frame, ps, falling=False, canvas=cv)
        t0 = time.perf_counter()
        for _ in range(200):
            cv = overlay.render(frame, ps, falling=False, canvas=cv)
        ms = (time.perf_counter() - t0) / 200 * 1000
        print(f"  {n} 人  {ms:.2f} ms/帧（PC 参考）")
    print("   板端数字必须重新测，这里只证明没有 O(人数) 的整帧拷贝")


def main() -> int:
    v1_ring_latest()
    v2_display_persons()
    v3_render()
    v4_cost()
    print()
    print("全部 PASS" if OK else " 有 FAIL")
    return 0 if OK else 1


if __name__ == "__main__":
    sys.exit(main())
