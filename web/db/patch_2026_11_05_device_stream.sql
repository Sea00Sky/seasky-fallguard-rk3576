-- ============================================================================
-- 设备 ↔ 推流路径（2026-11-05 第92轮五改）
--
-- 执行：
--   psql -U postgres -d fallguard -v ON_ERROR_STOP=1 -f db/patch_2026_11_05_device_stream.sql
--
-- ⚠️ **必须先跑本补丁，再部署代码**。新代码会 SELECT d.stream_path，
--    补丁没跑就是 /admin/devices、/admin/overview 与所有 ingest 接口全部 500。
--
-- 为什么设备也要一条流名（与 patch_2026_11_02_point_stream.sql 的
-- 「stream_path 是点位的属性」**不矛盾**，是两个不同的事实）：
--   · fg.point.stream_path  = 「这个点位**显示**哪条流」——是给人看的绑定位；
--   · fg.device.stream_path = 「这台板子**推**的是哪条流」——是设备的事实。
--   同一台板子的画面可以被管理员绑到任意点位（演示时跨病区绑过），
--   这时画面会跟着走，但骨架帧是板端带着**设备自己**的病区/区域报上来的，
--   于是出现「有画面、没骨架」。
--
--   板端新固件会在 /frames 的 payload 里带 stream（优先用上报值）；
--   老固件没有这个字段，就用这里的登记值兜底 —— 不用碰板子也能把口径对齐。
--
-- 可重复执行：ADD COLUMN IF NOT EXISTS / 约束先 DROP 再建。
-- ============================================================================

BEGIN;
SET search_path = fg, public;
SET client_encoding = 'UTF8';

-- ── 1. 加列 ────────────────────────────────────────────────────────────────
ALTER TABLE fg.device
  ADD COLUMN IF NOT EXISTS stream_path TEXT;

COMMENT ON COLUMN fg.device.stream_path IS
  '这台板子推的 mediamtx 路径名。NULL = 未知（骨架退回按设备病区+区域分发）。'
  '板端 /frames 上报的 stream 优先于本列。';

-- ── 2. 约束 ────────────────────────────────────────────────────────────────
-- 与 fg.point.stream_path 用**逐字相同**的字符集：同一个名字两边都要能填，
-- 且它会被拿去和 mediamtx 的 path 精确比较（区分大小写，不能有 '/' ':' '?'）。
ALTER TABLE fg.device DROP CONSTRAINT IF EXISTS ck_device_stream_path;
ALTER TABLE fg.device
  ADD CONSTRAINT ck_device_stream_path
  CHECK (stream_path IS NULL OR stream_path ~ '^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$');

-- ── 3. **故意不做自动回填** ────────────────────────────────────────────────
-- 第一版这里写过一条 UPDATE：把点位的 stream_path 抄给挂在该点位的设备。
-- 现网第一次跑就抄错了 —— 它把 live 安到了 4F-B 的演示设备 K7C-4FB-01 上，
-- 而真正推流的是 4F-A 的 K7C-4FA-01（演示时把 live 改绑到了别的点位）。
-- 就算加「同病区才回填」也挡不住：那次错的设备与点位恰好同病区。
--
-- 结论：「点位显示的流」和「板子推的流」是两个独立事实，**猜不得**。
-- 后者的唯一可靠来源是板端上报（新固件）；老固件只能靠管理员在
-- /admin/devices 的「板端推流路径」里显式填一次。
-- 现网那次误填的订正记录在 _vendor/cloud_ops/_fix_device_stream_data_srv.sql。

-- ── 4. 授权 ────────────────────────────────────────────────────────────────
GRANT SELECT, INSERT, UPDATE ON fg.device TO fg_app;

COMMIT;
