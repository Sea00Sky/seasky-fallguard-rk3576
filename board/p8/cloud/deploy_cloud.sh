#!/bin/sh
set -eu

TARBALL="${1:-}"
if [ -z "$TARBALL" ]; then
    echo "用法: sh deploy_cloud.sh <mediamtx_vX_linux_amd64.tar.gz>" >&2
    exit 2
fi
if [ "$(id -u)" -ne 0 ]; then
    echo "必须以 root 运行" >&2
    exit 2
fi
if [ ! -f "$TARBALL" ]; then
    echo "找不到 $TARBALL" >&2
    exit 2
fi

DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

ARCH=$(uname -m)
case "$ARCH" in
    x86_64)  ;;
    aarch64) echo " 本脚本按 amd64 写的；aarch64 要换 arm64 包，并确认二进制名一致" ;;
    *) echo "不支持的架构: $ARCH" >&2; exit 2 ;;
esac

echo "=== 1/4 解包 ==="
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
tar -xzf "$TARBALL" -C "$TMP"
[ -f "$TMP/mediamtx" ] || { echo "包里没有 mediamtx 可执行文件" >&2; exit 2; }

echo "=== 2/4 装二进制 ==="
install -m 0755 "$TMP/mediamtx" /usr/local/bin/mediamtx
/usr/local/bin/mediamtx --version 2>/dev/null || true

echo "=== 3/4 装配置与 unit ==="
mkdir -p /etc/mediamtx
if [ -f /etc/mediamtx/mediamtx.yml ]; then
    BAK="/etc/mediamtx/mediamtx.yml.bak_$(date +%Y%m%d_%H%M%S)"
    cp -p /etc/mediamtx/mediamtx.yml "$BAK"
    echo "  已备份旧配置 -> $BAK"
fi
install -m 0600 "$DIR/mediamtx.yml" /etc/mediamtx/mediamtx.yml
install -m 0644 "$DIR/mediamtx.service" /etc/systemd/system/mediamtx.service

echo "=== 4/4 启动并自检 ==="
systemctl daemon-reload
systemctl enable --now mediamtx
sleep 2

if ! systemctl is-active --quiet mediamtx; then
    echo " mediamtx 没起来，最近日志：" >&2
    journalctl -u mediamtx -n 30 --no-pager >&2
    exit 1
fi

echo "--- 监听端口 ---"
ss -lunp 2>/dev/null | grep ':8890 ' || echo " UDP 8890 没在听"
ss -ltnp 2>/dev/null | grep ':8554 ' || echo " TCP 8554 没在听"

echo "--- 生效配置（从日志读）---"
journalctl -u mediamtx -n 12 --no-pager | sed 's/^/  /'

cat <<'EOF'

=== 部署完成，还剩两件事 ===
1. 控制台防火墙放行：UDP 8890（SRT 收流）、TCP 8554（RTSP 出流）
2. 本机自检：
     curl -s 127.0.0.1:9997/v3/paths/list
   板子推上来之前 paths 是空的，推上来后能看到 live
EOF
