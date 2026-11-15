"""零拷贝 YOLO 后端：`ctypes` 直调 `librknnrt` + `cv2` 直写 NPU 内存。"""
from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

from ..yolo_decode import INPUT_SHAPE, decode
from .rknn_backend import CONF, MAX_PERSONS, Detection, LUT_F16

# ---------------------------------------------------------------- RKNN C API
# 常量逐条照 `rknn_api.h` 2.3.x 抄（板上 `strings librknnrt.so` 报 2.3.2）。

Q_IN_OUT_NUM, Q_OUTPUT_ATTR = 0, 2
Q_NATIVE_INPUT_ATTR, Q_NATIVE_OUTPUT_ATTR = 8, 9
T_FLOAT32, T_FLOAT16, T_INT8 = 0, 1, 2
FMT_NHWC = 1
SYNC_FROM_DEV = 0x2
CORE_0_1 = 3
INIT_FLAG = 0 # 见模块 docstring：不要动这个 0
_MAXD, _MAXNAME = 16, 256
LIB_CANDIDATES = ("/usr/lib/librknnrt.so", "librknnrt.so")


class _Attr(ctypes.Structure):
    """`rknn_tensor_attr`（rknn_api.h:278）。sizeof 必须 = 376。"""
    _fields_ = [
        ("index", ctypes.c_uint32),
        ("n_dims", ctypes.c_uint32),
        ("dims", ctypes.c_uint32 * _MAXD),
        ("name", ctypes.c_char * _MAXNAME),
        ("n_elems", ctypes.c_uint32),
        ("size", ctypes.c_uint32),
        ("fmt", ctypes.c_int),
        ("type", ctypes.c_int),
        ("qnt_type", ctypes.c_int),
        ("fl", ctypes.c_int8),
        ("zp", ctypes.c_int32),
        ("scale", ctypes.c_float),
        ("w_stride", ctypes.c_uint32),
        ("size_with_stride", ctypes.c_uint32),
        ("pass_through", ctypes.c_uint8),
        ("h_stride", ctypes.c_uint32),
    ]


class _TensorMem(ctypes.Structure):
    """`rknn_tensor_mem`（rknn_api.h:399）。sizeof 必须 = 40。"""
    _fields_ = [
        ("virt_addr", ctypes.c_void_p),
        ("phys_addr", ctypes.c_uint64),
        ("fd", ctypes.c_int32),
        ("offset", ctypes.c_int32),
        ("size", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("priv_data", ctypes.c_void_p),
    ]


class _Input(ctypes.Structure):
    """`rknn_input`（rknn_api.h:412）。sizeof 必须 = 32。"""
    _fields_ = [
        ("index", ctypes.c_uint32),
        ("buf", ctypes.c_void_p),
        ("size", ctypes.c_uint32),
        ("pass_through", ctypes.c_uint8),
        ("type", ctypes.c_int),
        ("fmt", ctypes.c_int),
    ]


class _Output(ctypes.Structure):
    """`rknn_output`（rknn_api.h:431）。sizeof 必须 = 24。"""
    _fields_ = [
        ("want_float", ctypes.c_uint8),
        ("is_prealloc", ctypes.c_uint8),
        ("index", ctypes.c_uint32),
        ("buf", ctypes.c_void_p),
        ("size", ctypes.c_uint32),
    ]


class _IONum(ctypes.Structure):
    _fields_ = [("n_input", ctypes.c_uint32), ("n_output", ctypes.c_uint32)]


_EXPECT_SIZEOF = {_Attr: 376, _TensorMem: 40, _Input: 32, _Output: 24, _IONum: 8}


def _load_lib():
    last = None
    for cand in LIB_CANDIDATES:
        try:
            lib = ctypes.CDLL(cand)
            break
        except OSError as e:                      # pragma: no cover
            last = e
    else:                                          # pragma: no cover
        raise RuntimeError(
            f"加载 librknnrt 失败（试过 {LIB_CANDIDATES}）。"
            f"板端应装在 /usr/lib/，PC 上没有是正常的——本后端只在板上用。") from last
    checks = [
        ("rknn_init", [ctypes.POINTER(ctypes.c_uint64), ctypes.c_void_p,
                       ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]),
        ("rknn_query", [ctypes.c_uint64, ctypes.c_int, ctypes.c_void_p,
                        ctypes.c_uint32]),
        ("rknn_set_core_mask", [ctypes.c_uint64, ctypes.c_int]),
        ("rknn_create_mem", [ctypes.c_uint64, ctypes.c_uint32]),
        ("rknn_destroy_mem", [ctypes.c_uint64, ctypes.POINTER(_TensorMem)]),
        ("rknn_set_io_mem", [ctypes.c_uint64, ctypes.POINTER(_TensorMem),
                             ctypes.POINTER(_Attr)]),
        ("rknn_inputs_set", [ctypes.c_uint64, ctypes.c_uint32,
                             ctypes.POINTER(_Input)]),
        ("rknn_run", [ctypes.c_uint64, ctypes.c_void_p]),
        ("rknn_outputs_get", [ctypes.c_uint64, ctypes.c_uint32,
                              ctypes.POINTER(_Output), ctypes.c_void_p]),
        ("rknn_outputs_release", [ctypes.c_uint64, ctypes.c_uint32,
                                  ctypes.POINTER(_Output)]),
        ("rknn_destroy", [ctypes.c_uint64]),
    ]
    for name, argtypes in checks:
        fn = getattr(lib, name)
        fn.argtypes = argtypes
        fn.restype = ctypes.c_int
    lib.rknn_create_mem.restype = ctypes.POINTER(_TensorMem)
    return lib


def _dtype_of(attr: _Attr) -> np.dtype:
    if attr.type == T_FLOAT16:
        return np.dtype(np.float16)
    if attr.type == T_FLOAT32:
        return np.dtype(np.float32)
    if attr.type == T_INT8:
        return np.dtype(np.int8)
    raise RuntimeError(f"没见过的张量类型 type={attr.type}")


def _np_at(addr: int, attr: _Attr) -> np.ndarray:
    """把裸地址包成一维 numpy 视图（零拷贝）。"""
    dt = _dtype_of(attr)
    n = int(attr.n_elems)
    raw = (ctypes.c_uint8 * (n * dt.itemsize)).from_address(addr)
    return np.ctypeslib.as_array(raw).view(dt)


class _Session:
    """一个裸 librknnrt 会话。绑定与否在构造时定死（`set_io_mem` 不能撤销）。"""

    def __init__(self, model_path: Path, core_mask: int = CORE_0_1,
                 bind_input: bool = True, bind_output: bool = True) -> None:
        for st, exp in _EXPECT_SIZEOF.items():
            got = ctypes.sizeof(st)
            if got != exp:                          # pragma: no cover
                raise RuntimeError(
                    f"{st.__name__} 布局不符：ctypes {got} ≠ C {exp}。"
                    f"头文件版本与 librknnrt 可能不匹配，继续跑会静默写坏内存。")
        self.lib = _load_lib()
        self.model = Path(model_path).read_bytes()
        # 这块缓冲必须活到 rknn_init 之后（model 是借用指针）
        self._mbuf = ctypes.create_string_buffer(self.model, len(self.model))
        self.ctx = ctypes.c_uint64(0)
        rc = self.lib.rknn_init(ctypes.byref(self.ctx), self._mbuf,
                                len(self.model), INIT_FLAG, None)
        if rc != 0:
            raise RuntimeError(f"rknn_init 失败 rc={rc}（{model_path}）")
        rc = self.lib.rknn_set_core_mask(self.ctx, int(core_mask))
        if rc != 0:
            raise RuntimeError(f"rknn_set_core_mask({core_mask}) 失败 rc={rc}")

        io = _IONum()
        self._q(Q_IN_OUT_NUM, io)
        self.n_out = int(io.n_output)

        self.in_attr = _Attr()
        self.in_attr.index = 0
        self._q(Q_NATIVE_INPUT_ATTR, self.in_attr)
        # 绑输入要求「原生布局 == 紧凑布局」，否则直写会写错位置。
        if self.in_attr.size_with_stride != self.in_attr.size:
            raise RuntimeError(
                f"输入有 padding（size={self.in_attr.size} "
                f"stride={self.in_attr.size_with_stride}）⇒ 不能直写，"
                f"请改用 bind_input=False")

        self.out_attr: list[_Attr] = []
        self.out_nat: list[_Attr] = []
        for i in range(self.n_out):
            a = _Attr()
            a.index = i
            self._q(Q_OUTPUT_ATTR, a)
            n = _Attr()
            n.index = i
            self._q(Q_NATIVE_OUTPUT_ATTR, n)
            if (n.type, n.n_elems, n.size_with_stride) != (
                    a.type, a.n_elems, a.size_with_stride):
                raise RuntimeError(
                    f"输出 {i} 原生/逻辑 attr 不一致：native(type={n.type}, "
                    f"n_elems={n.n_elems}, stride={n.size_with_stride}) vs "
                    f"logical(type={a.type}, n_elems={a.n_elems}, "
                    f"stride={a.size_with_stride}) ⇒ 按逻辑 attr 绑输出会读错。"
                    f"换模型后要改成用 NATIVE_OUTPUT_ATTR 建缓冲并重新对拍。")
            self.out_attr.append(a)
            self.out_nat.append(n)

        self.in_mem = None
        self.in_view = None
        if bind_input:
            self.in_mem = self.lib.rknn_create_mem(
                self.ctx, self.in_attr.size_with_stride)
            if not self.in_mem:
                raise RuntimeError("rknn_create_mem(input) 失败")
            a = _Attr.from_buffer_copy(self.in_attr)
            a.pass_through = 1          # 绑输入必须直通，否则语义不同
            rc = self.lib.rknn_set_io_mem(self.ctx, self.in_mem, ctypes.byref(a))
            if rc != 0:
                raise RuntimeError(f"rknn_set_io_mem(input) 失败 rc={rc}")
            if self.in_attr.fmt != FMT_NHWC:        # pragma: no cover
                raise RuntimeError(
                    f"直写要求原生输入是 NHWC，实际 fmt={self.in_attr.fmt}；"
                    f"别的布局写进去会静默错位，请改用 bind_input=False")
            flat = _np_at(self.in_mem.contents.virt_addr, self.in_attr)
            d = tuple(int(self.in_attr.dims[k])
                      for k in range(self.in_attr.n_dims))
            # (1,H,W,C) → (H,W,C)：`cv2.LUT(dst=)` 要求与源图同形状，
            # 给一维会报形状不符（原型第二轮之前就是这么埋雷的）。
            self.in_view = flat.reshape(d[1:]) if len(d) == 4 else flat
            if not self.in_view.flags.writeable:    # pragma: no cover
                raise RuntimeError("绑定的输入视图不可写")

        self.out_mem = []
        if bind_output:
            for i in range(self.n_out):
                m = self.lib.rknn_create_mem(
                    self.ctx, self.out_attr[i].size_with_stride)
                if not m:
                    raise RuntimeError(f"rknn_create_mem(out {i}) 失败")
                a = _Attr.from_buffer_copy(self.out_attr[i])
                a.pass_through = 0
                rc = self.lib.rknn_set_io_mem(self.ctx, m, ctypes.byref(a))
                if rc != 0:
                    raise RuntimeError(f"rknn_set_io_mem(out {i}) 失败 rc={rc}")
                self.out_mem.append(m)

        self.bind_input = bind_input
        self.bind_output = bind_output
        self._closed = False

    def _q(self, cmd: int, struct) -> None:
        rc = self.lib.rknn_query(self.ctx, cmd, ctypes.byref(struct),
                                 ctypes.sizeof(struct))
        if rc != 0:
            raise RuntimeError(f"rknn_query(cmd={cmd}) 失败 rc={rc}")

    # ------------------------------------------------------------------ 推理
    def feed(self, arr: np.ndarray) -> None:
        """不绑输入时用：`rknn_inputs_set(pass_through=1)`。"""
        inp = _Input()
        inp.index = 0
        inp.buf = arr.ctypes.data
        inp.size = self.in_attr.size_with_stride
        inp.pass_through = 1
        inp.type = T_FLOAT16
        inp.fmt = FMT_NHWC
        rc = self.lib.rknn_inputs_set(self.ctx, 1, ctypes.byref(inp))
        if rc != 0:
            raise RuntimeError(f"rknn_inputs_set 失败 rc={rc}")

    def run(self) -> list[np.ndarray]:
        """跑一次并返回 float32 副本（每个输出一个）。"""
        if self.lib.rknn_run(self.ctx, None) != 0:
            raise RuntimeError("rknn_run 失败")
        outs = (_Output * self.n_out)()
        for i in range(self.n_out):
            outs[i].index = i
            if self.bind_output:
                # is_prealloc=1 时 size 也必须设，漏设 ⇒ 输出恒零
                outs[i].is_prealloc = 1
                outs[i].buf = self.out_mem[i].contents.virt_addr
                outs[i].size = self.out_attr[i].size
        if self.lib.rknn_outputs_get(self.ctx, self.n_out, outs, None) != 0:
            raise RuntimeError("rknn_outputs_get 失败")
        try:
            res = []
            for i in range(self.n_out):
                a = self.out_attr[i]
                src = (self.out_mem[i].contents.virt_addr
                       if self.bind_output else outs[i].buf)
                dims = tuple(int(a.dims[k]) for k in range(a.n_dims))
                # 原生缓冲是一维的；不 reshape 喂 decode 会在 pred[:,4] 抛
                # IndexError（原型第一轮就是这么崩的）。
                res.append(_np_at(src, a).reshape(dims).astype(np.float32))
            return res
        finally:
            self.lib.rknn_outputs_release(self.ctx, self.n_out, outs)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for m in self.out_mem:
            self.lib.rknn_destroy_mem(self.ctx, m)
        self.out_mem = []
        if self.in_mem is not None:
            self.lib.rknn_destroy_mem(self.ctx, self.in_mem)
            self.in_mem = None
        self.in_view = None
        self.lib.rknn_destroy(self.ctx)


class YoloBackendCtypes:
    """零拷贝 YOLO 后端，接口与 `YoloBackend` 同构 ⇒ 可直接替换。"""

    def __init__(self, model_path, conf: float = CONF,
                 max_persons: int = MAX_PERSONS, core_mask: int | None = CORE_0_1,
                 bind_output: bool = True, log=print) -> None:
        h, w = INPUT_SHAPE[2], INPUT_SHAPE[3]
        self._h, self._w = int(h), int(w)
        self.conf = float(conf)
        self.max_persons = int(max_persons)
        self.n_calls = 0
        cm = CORE_0_1 if core_mask is None else int(core_mask)
        self._s = _Session(Path(model_path), core_mask=cm,
                           bind_input=True, bind_output=bool(bind_output))
        self._rgb = np.empty((self._h, self._w, 3), np.uint8)
        # 640×480 下 LetterBox 是恒等变换（模型静态输入 480×640），参数写死。
        self._params = {"ratio": 1.0, "pad": (0, 0), "src": (self._w, self._h),
                        "shape": (self._h, self._w), "auto": True}
        log(f"[config] YOLO 零拷贝后端：ctypes 直调 librknnrt，"
            f"核掩码={cm} 绑输入=是 绑输出={'是' if bind_output else '否'} "
            f"（预期 detect ≈ 24 ms，替代 rknnlite ≈ 33 ms）")

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        h, w = frame_bgr.shape[:2]
        if (h, w) != (self._h, self._w):
            raise ValueError(
                f"零拷贝后端只支持 {self._w}x{self._h}，收到 {w}x{h}；"
                f"换分辨率必须重新导出模型。")
        if frame_bgr.dtype != np.uint8 or not frame_bgr.flags.c_contiguous:
            raise ValueError("零拷贝后端要求 uint8 且 C 连续的 BGR 帧")

        import cv2
        cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB, dst=self._rgb)
        # 落点就是 NPU 的 DMA 缓冲：这一句同时完成了"预处理"和"喂数据"，
        # `rknn_inputs_set` 那一段因此整个消失。
        cv2.LUT(self._rgb, LUT_F16, dst=self._s.in_view)

        raw = self._s.run()[0]
        self.n_calls += 1
        boxes, kpts, scores = decode(raw, self._params, conf=self.conf,
                                     max_persons=self.max_persons)
        return [Detection(bbox=boxes[i], keypoints=kpts[i], scores=scores[i])
                for i in range(len(boxes))]

    def close(self) -> None:
        self._s.close()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
