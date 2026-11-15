#!/bin/sh
# 板端安装「每日低峰重启感知进程」
#
# 用法（在板子上，以 root）：
# sudo sh p8/install_daily_restart.sh
#
# 做四件事：前置检查 → 装两个 unit → enable --now timer → 打自检。
# 样式与 p8/stream/install_local.sh 保持一致。
#
# 装之前先确认 `fallguard-sensor-multi` 是 enable 的，否则定时器踢的是一具空壳。
# 本脚本只碰感知进程，不碰网关、不碰推流、不做整机重启。理由见
# p8/fallguard-daily-restart.service 顶部的注释。
set -eu

DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

[ "$(id -u)" -eq 0 ] || { echo "必须以 root 运行" >&2; exit 2; }

echo "=== 0/4 前置检查 ==="
if systemctl is-enabled fallguard-sensor-multi >/dev/null 2>&1; then
    echo "   fallguard-sensor-multi: $(systemctl is-enabled fallguard-sensor-multi) / $(systemctl is-active fallguard-sensor-multi || true)"
else
    echo "   fallguard-sensor-multi 没 enable —— 定时器会去重启一个不会自启的服务"
fi
# 互斥的另一半：两个感知 unit 不能同时 enable（4 个 RKNN 会话会 NPU 挂死）
for u in fallguard-sensor-assembly fallguard-sensor-multi; do
    st=$(systemctl is-enabled "$u" 2>/dev/null || echo "not-found")
    echo "  $u: $st"
done
if systemctl is-enabled fallguard-sensor-assembly >/dev/null 2>&1; then
    echo "   旧 sensor 还 enable 着 —— 按切流纪律应先停旧再起新，别在这里顺手改。"
fi

echo "=== 1/4 装 unit ==="
install -m 0644 "$DIR/fallguard-daily-restart.service" /etc/systemd/system/fallguard-daily-restart.service
install -m 0644 "$DIR/fallguard-daily-restart.timer"   /etc/systemd/system/fallguard-daily-restart.timer

echo "=== 2/4 起定时器 ==="
systemctl daemon-reload
systemctl enable --now fallguard-daily-restart.timer

echo "=== 3/4 自检 ==="
systemctl list-timers fallguard-daily-restart.timer --no-pager | sed 's/^/  /'
echo "  is-enabled: $(systemctl is-enabled fallguard-daily-restart.timer || true)"
echo "  is-active : $(systemctl is-active fallguard-daily-restart.timer || true)"

echo "=== 4/4 记下「重启前」的基线（明天这个时候好对比）==="
PID=$(systemctl show fallguard-sensor-multi -p MainPID --value)
echo "  MainPID=$PID  NRestarts=$(systemctl show fallguard-sensor-multi -p NRestarts --value)"
echo "  启动时刻: $(systemctl show fallguard-sensor-multi -p ActiveEnterTimestamp --value)"
if [ -n "$PID" ] && [ "$PID" != "0" ]; then
    ps -o pid,etime,rss,comm -p "$PID" | sed 's/^/  /'
fi
echo "  系统内存: $(free -m | awk '/^Mem:/{print $3" MB used / "$2" MB total"}')"

cat <<'EOF'

=== 装完了 ===
明早自检（对比 RSS 与 etime 是否被重置）：

    systemctl show fallguard-sensor-multi -p ActiveEnterTimestamp -p NRestarts -p MainPID
    ps -o pid,etime,rss -p "$(systemctl show fallguard-sensor-multi -p MainPID --value)"
    journalctl -u fallguard-daily-restart.service -n 20 --no-pager

手工立刻验一次（会真的重启感知进程，数秒盲区）：

    sudo systemctl start fallguard-daily-restart.service

 想停掉：

    sudo systemctl disable --now fallguard-daily-restart.timer

 改「多久一次」之前先读 p8/verify_retract_window.py —— 间隔必须远大于云端
   撤回匹配的 2 分钟窗，否则重启会开始静默改历史告警。
EOF
