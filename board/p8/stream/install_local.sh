#!/bin/sh
# 板端安装本地 RTSP 推流服务
#
# 用法（在板子上，以 root）：
# sudo sh p8/stream/install_local.sh
#
# 做四件事：停手工进程 → 装 env → 装 unit → 起服务并自检。
#
# 装之前必须停掉手工跑的 `python3 scripts/run_stream.py`：
# mainpath 是 V4L2 独占的，两个进程抢同一个节点，第二个 EBUSY。
#
# 改了本 unit 或 stream.env 之后，云出口 unit 也要重装 + 重启：
# 它读同一份 stream.env 来拼本地拉流地址。
set -eu

DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

[ "$(id -u)" -eq 0 ] || { echo "必须以 root 运行" >&2; exit 2; }

echo "=== 0/4 停手工进程 ==="
if pgrep -f 'python3 scripts/run_stream[.]py' >/dev/null 2>&1; then
    echo "  发现手工跑的 run_stream.py，停掉（否则抢 mainpath）"
    pkill -f 'python3 scripts/run_stream[.]py' || true
    sleep 1
fi

echo "=== 1/4 装 env ==="
mkdir -p /etc/fallguard
install -m 0600 "$DIR/stream.env" /etc/fallguard/stream.env

echo "=== 2/4 装 unit ==="
install -m 0644 "$DIR/fallguard-streamer-local.service" \
    /etc/systemd/system/fallguard-streamer-local.service

echo "=== 3/4 起服务 ==="
systemctl daemon-reload
systemctl enable --now fallguard-streamer-local
sleep 4

echo "=== 4/4 自检 ==="
echo "  is-active: $(systemctl is-active fallguard-streamer-local || true)"
systemctl show -p NRestarts -p SubState -p ExecMainPID \
    fallguard-streamer-local | sed 's/^/  /'
if ss -ltn 2>/dev/null | grep -q ':8554 '; then
    echo "   8554 在听"
else
    echo "   8554 没在听 —— 看 journalctl -u fallguard-streamer-local"
fi

cat <<'EOF'

=== 装完了，还剩一步 ===
云出口读同一份 stream.env，所以要重装并重启它：
    sudo install -m 644 p8/cloud/fallguard-streamer-cloud.service /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl restart fallguard-streamer-cloud
EOF
