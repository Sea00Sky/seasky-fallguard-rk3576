#!/bin/sh
set -u

CLIP="${1:-}"
if [ -z "$CLIP" ]; then
    echo "用法: $0 <clip.mp4> [--plain] [--seconds N]"
    exit 2
fi
shift

MODE=bridge
LIMIT=0
while [ $# -gt 0 ]; do
    case "$1" in
        --plain)   MODE=plain; shift ;;
        --seconds) LIMIT="$2"; shift 2 ;;
        *) echo "[demo] 未知参数 $1"; exit 2 ;;
    esac
done

DIR=${FG_BOARD_DIR:-/opt/fallguard/board}
DEPLOY_DIR=${FG_DEPLOY_DIR:-/opt/fallguard/deploy}
PW=${FG_SUDO_PASS:-}
LOG=/tmp/fall_demo.log
BLOG=/tmp/ann_bridge.log
BRIDGE_PID=""

sudo_do() {
    if [ -n "$PW" ]; then
        echo "$PW" | sudo -S "$@"
    else
        sudo "$@"
    fi
}

cd "$DIR" || { echo "[demo] 进不去 $DIR"; exit 1; }

# 与 systemd 单元读同一份配置
if [ -r /etc/fallguard/stream.env ]; then
    set -a
    . /etc/fallguard/stream.env
    set +a
fi

GEO_CONF=/etc/systemd/system/fallguard-sensor-multi.service.d/geo-veto.conf
if [ -r "$GEO_CONF" ]; then
    set -a
    eval "$(sed -n 's/^Environment=//p' "$GEO_CONF")"
    set +a
    echo "[demo] geo_veto 来自 $GEO_CONF: ${FG_MULTI_GEO_VETO_AR_MAX:-?}/${FG_MULTI_GEO_VETO_BOTTOM_SHIFT:-?}/${FG_MULTI_GEO_VETO_R_END:-?}"
else
    echo "[demo]  没找到 $GEO_CONF，回放将不带几何否决"
fi

restore() {
    echo "[demo] ==== 回滚 ===="
    if [ -n "$BRIDGE_PID" ]; then
        kill "$BRIDGE_PID" 2>/dev/null
        wait "$BRIDGE_PID" 2>/dev/null
        BRIDGE_PID=""
    fi
    sleep 1
    sudo_do systemctl start fallguard-streamer-local 2>/dev/null
    sleep 2
    sudo_do systemctl start fallguard-sensor-multi 2>/dev/null
    sleep 3
    echo "[demo] sensor=$(systemctl is-active fallguard-sensor-multi)" \
         "streamer-local=$(systemctl is-active fallguard-streamer-local)" \
         "streamer-cloud=$(systemctl is-active fallguard-streamer-cloud)" \
         "gateway=$(systemctl is-active fallguard-gateway)"
}
trap restore EXIT INT TERM

[ -f "$CLIP" ] || { echo "[demo] 找不到视频: $CLIP"; exit 2; }

echo "[demo] $(date) 模式=$MODE clip=$(basename "$CLIP")"

# ① 停检测：回放要独占 NPU
sudo_do systemctl stop fallguard-sensor-multi 2>/dev/null
sleep 2
echo "[demo] sensor-multi = $(systemctl is-active fallguard-sensor-multi)"

# ② 停本地推流：bridge 要占 8554
if [ "$MODE" = bridge ]; then
    sudo_do systemctl stop fallguard-streamer-local 2>/dev/null
    sleep 2
    echo "[demo] streamer-local = $(systemctl is-active fallguard-streamer-local)"
    WEBARGS="--web-port 8888 --web-fps 15"
else
    WEBARGS="--web-port 0"
fi

# ③ 起回放检测
# 不能加 sudo：numpy 装在普通用户目录，root 的 sys.path 里没有
PYTHONUNBUFFERED=1 FG_DEPLOY_DIR="$DEPLOY_DIR" \
    /usr/bin/python3 scripts/run_sensor_multi.py \
    --video "$CLIP" --loop --cooldown-s 5 \
    --snap-w 640 --snap-h 480 $WEBARGS \
    > "$LOG" 2>&1 &
SENSOR_PID=$!
echo "[demo] 回放进程 pid=$SENSOR_PID 日志=$LOG"

# ④ 起标注桥（等 8888 先就绪）
if [ "$MODE" = bridge ]; then
    i=0
    while [ $i -lt 40 ]; do
        if curl -sf -m 1 -o /dev/null http://127.0.0.1:8888/ 2>/dev/null; then break; fi
        sleep 1
        i=$((i + 1))
    done
    echo "[demo] 8888 就绪（等了 ${i}s）"

    if ! /usr/bin/python3 p8/ann_bridge.py --probe >> "$BLOG" 2>&1; then
        echo "[demo]  标注桥自检失败，详见 $BLOG（演示会没有骨架）"
    fi
    /usr/bin/python3 p8/ann_bridge.py --port 8554 --mount /live >> "$BLOG" 2>&1 &
    BRIDGE_PID=$!
    sleep 3
    echo "[demo] 标注桥 pid=$BRIDGE_PID 日志=$BLOG"
    tail -2 "$BLOG"
fi

# ⑤ 等
if [ "$LIMIT" -gt 0 ] 2>/dev/null; then
    echo "[demo] 跑 ${LIMIT}s 后自动停止"
    i=0
    while [ $i -lt "$LIMIT" ] && kill -0 "$SENSOR_PID" 2>/dev/null; do
        sleep 1
        i=$((i + 1))
    done
    kill "$SENSOR_PID" 2>/dev/null
else
    wait "$SENSOR_PID"
fi

echo "[demo] ---- 回放日志（尾部）----"
tail -25 "$LOG"
echo "[demo] ---- 结束 ----"
echo "[demo] DONE"
