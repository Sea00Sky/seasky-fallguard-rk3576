#!/bin/sh
set -u

DIR=${FG_BOARD_DIR:-/opt/fallguard/board}
CLIP_SECONDS="${1:-75}"
LOG=/tmp/replay_all.log
shift || true
if [ $# -gt 0 ]; then
    SETS="$*"
else
    SETS="$DIR/videos/整体测试视频 $DIR/videos/蹲姿实录视频"
fi

echo "[all] $(date) 开始，每条 ${CLIP_SECONDS}s，目录: $SETS"
n=0
for d in $SETS; do
    for clip in "$d"/*.mp4; do
        [ -f "$clip" ] || continue
        n=$((n + 1))
        echo "[all] $(date) ==== [$n] $(basename "$clip") ===="
        sh "$DIR/demo/fall_demo.sh" "$clip" --seconds "$CLIP_SECONDS"
        echo "[all] $(date) ---- [$n] 完成 ----"
        sleep 5
    done
done
echo "[all] $(date) 全部完成，共 $n 条"
echo "[all] DONE"
