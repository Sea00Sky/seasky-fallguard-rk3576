#!/bin/sh

BASE=${BASE:-/opt/fallguard/board}
PORT=${1:-9103}
PASS=${2:-TestPassphrase123}

echo "=== 0. 清场 ==="
for pid in $(ss -lunp 2>/dev/null | grep ":${PORT} " | grep -o 'pid=[0-9]*' | cut -d= -f2); do
    kill "$pid" 2>/dev/null
done
for pid in $(ss -ltnp 2>/dev/null | grep ":8554 " | grep -o 'pid=[0-9]*' | cut -d= -f2); do
    kill "$pid" 2>/dev/null
done
sleep 1
rm -f /tmp/fg_srt.ts /tmp/fg_stream.log /tmp/fg_cloud.log /tmp/fg_recv.log

echo "=== 1. 起本地 RTSP（mainpath 1080p）==="
cd "$BASE" || exit 1
setsid nohup python3 scripts/run_stream.py > /tmp/fg_stream.log 2>&1 &
sleep 4
ss -ltnp 2>/dev/null | grep ':8554 ' || { echo " 8554 没起来"; tail -10 /tmp/fg_stream.log; exit 1; }

echo "=== 2. 起 SRT listener ==="
# listener 必须先绑定端口，caller 才能连上
setsid nohup gst-launch-1.0 -q \
    srtsrc uri="srt://127.0.0.1:${PORT}?mode=listener" passphrase="${PASS}" \
    ! filesink location=/tmp/fg_srt.ts > /tmp/fg_recv.log 2>&1 &
sleep 2
ss -lunp 2>/dev/null | grep ":${PORT} " || { echo " ${PORT} 没在听"; tail -10 /tmp/fg_recv.log; exit 1; }

echo "=== 3. 跑云出口 12 秒 ==="
# timeout 发 SIGTERM ⇒ run_cloud 的信号处理器会干净退出，rc=124 是 timeout 的正常结果
timeout 12 python3 scripts/run_cloud.py \
    --uri "srt://127.0.0.1:${PORT}" \
    --streamid "publish:live" \
    --passphrase "${PASS}" > /tmp/fg_cloud.log 2>&1
echo "run_cloud rc=$?"
cat /tmp/fg_cloud.log
sleep 2

echo "=== 4. 判据 ==="
SIZE=$(stat -c %s /tmp/fg_srt.ts 2>/dev/null || echo 0)
echo "收到字节: $SIZE"
RC=1
if [ "$SIZE" -gt 100000 ]; then
    echo " PASS：SRT 链路通（真实代码路径）"
    RC=0
else
    echo " FAIL：字节不足"
fi

echo "--- stream 日志（尾 5）---"; tail -5 /tmp/fg_stream.log
echo "--- cloud 日志 ---";         cat /tmp/fg_cloud.log
echo "--- recv 日志（尾 5）---";   tail -5 /tmp/fg_recv.log

echo "=== 5. 收尾 ==="
for pid in $(ss -lunp 2>/dev/null | grep ":${PORT} " | grep -o 'pid=[0-9]*' | cut -d= -f2); do kill "$pid" 2>/dev/null; done
for pid in $(ss -ltnp 2>/dev/null | grep ":8554 " | grep -o 'pid=[0-9]*' | cut -d= -f2); do kill "$pid" 2>/dev/null; done
sleep 1
echo "done"
exit $RC
