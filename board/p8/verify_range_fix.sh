#!/bin/bash
# 端到端验证"压到 limited 再编码"能否往返还原
# 全程用落盘的真实 raw，不碰摄像头、不停服务。
#
# 现状链路： raw(full) --mpph264enc--> h264 --解码--> NV12 期望：被拉成 1.164*(Y-16)
# 修复链路： raw(full) --videobalance(0.8588/+0.0627/0.8784)--> mpph264enc --> h264 --解码--> NV12
# 期望：往返接近恒等（斜率≈1.0）
OUT=/tmp/fg_diag
SRC=$OUT/raw_1280x720.nv12
SRC1080=$OUT/raw_1920x1080.nv12

echo "### videobalance 属性范围"
python3 -c "
from gi.repository import Gst; Gst.init(None)
e=Gst.ElementFactory.make('videobalance',None)
for p in ('contrast','brightness','saturation','hue'):
    pr=e.find_property(p); print(f'  {p:12} default={pr.default_value:+.4f} min={pr.minimum:+.4f} max={pr.maximum:+.4f}')
" 2>&1 | grep -v PyGI

encdec() {  # $1=前置元素串 $2=输入文件 $3=tag  $4=宽 $5=高
  rm -f "$OUT/$3.h264" "$OUT/$3.nv12"
  timeout 60 gst-launch-1.0 -q -e filesrc location="$2" num-buffers=6 \
    ! rawvideoparse format=nv12 width=$4 height=$5 framerate=30/1 \
    $1 \
    ! mpph264enc rc-mode=cbr bps=4000000 gop=30 ! h264parse config-interval=-1 \
    ! filesink location="$OUT/$3.h264" 2>&1 | tail -2
  timeout 60 gst-launch-1.0 -q -e filesrc location="$OUT/$3.h264" \
    ! h264parse ! avdec_h264 ! videoconvert ! video/x-raw,format=NV12 \
    ! filesink location="$OUT/$3.nv12" 2>&1 | tail -2
  echo "  $3: h264=$(stat -c%s "$OUT/$3.h264" 2>/dev/null)B 解码=$(stat -c%s "$OUT/$3.nv12" 2>/dev/null)B"
}

echo
echo "### A. 现状链路（不加任何处理）"
encdec "" "$SRC" cur 1280 720
[ -s "$OUT/cur.nv12" ] && python3 /tmp/_chk.py "$OUT/cur.nv12" "$SRC" "现状"

echo
echo "### B. 修复链路（videobalance 压到 limited 再编码）"
encdec "! videobalance contrast=0.8588 brightness=0.0627 saturation=0.8784 hue=0.0" "$SRC" fix 1280 720
[ -s "$OUT/fix.nv12" ] && python3 /tmp/_chk.py "$OUT/fix.nv12" "$SRC" "修复"

echo
echo "### C. 修复链路的色度是否也还原（只看 U/V，用原始 raw 的 UV 平面对比）"
python3 - <<'PY'
import numpy as np
W,H=1280,720
def uv(p):
    d=np.frombuffer(open(p,'rb').read(W*H*3//2),np.uint8)[W*H:]
    return d.reshape(H//2,W).astype(np.float32)
s=uv('/tmp/fg_diag/raw_1280x720.nv12')
for t in ('cur','fix'):
    try: o=uv(f'/tmp/fg_diag/{t}.nv12')
    except Exception as e: print(f'  {t}: {e}'); continue
    a=np.polyfit(s[::3,::3].ravel(),o[::3,::3].ravel(),1)
    print(f"  {t}: U/V 映射斜率={a[0]:.4f} 截距={a[1]:+.2f}   (期望修复后≈1.0/+0.0，现状≈1.138/-16.5)")
PY

echo
echo "### D. videobalance 的耗时（离线全速跑 300 帧的墙钟）"
for extra in "" "! videobalance contrast=0.8588 brightness=0.0627 saturation=0.8784"; do
  st=$(date +%s.%N)
  timeout 120 gst-launch-1.0 -q filesrc location="$SRC" num-buffers=6 \
    ! rawvideoparse format=nv12 width=1280 height=720 framerate=30/1 $extra \
    ! fakesink sync=false >/dev/null 2>&1
  en=$(date +%s.%N)
  echo "  额外元素='${extra:-无}'  6 帧耗时=$(echo "$en-$st"|bc)s"
done
