-- ============================================================================
-- 七改：设备行上不再有「安装位置」副本（2026-11-05，用户拍板「点位选设备」）
--
-- 用户原话：「不能设备页面不能选安装点位，设备页面只有设备号，这些真实的设备信息，
--            不然信息重复了，我现在的逻辑是点位选设备，不是设备选点位」
--
-- 模型：位置只存在点位（point.ward_id / zone_id / light_key），
--       设备与点位的唯一关系是 point.device_id。
--       旧模型里 device.point_id / zone_id / location 是同一事实的第二份副本。
--
-- ⚠️⚠️ 执行顺序**不可反**（和 point.stream_path 那次一样）：
--   ① 先发「停读写」代码（admin.create/patch、main._devices、ingest、metrics、events）
--   ② 跑验收（_verify_point_device_srv.py / _verify_point_drawer_srv.py / _verify_pages_srv.py）
--   ③ 最后才跑本补丁
--   反了 = /admin/devices、/head/devices、/alerts、/nurse/today、/head/zones 全 500。
--
-- 保留 device.ward_id：那是「这台机器归哪个病区管」，是 P12 权限与列表过滤的依据，
-- 不是位置副本（一台 4F-B 的机器可以被 4F-A 的点位绑，归属按点位算）。
--
-- 回滚：fg.device_place_bak_20261105 存了删列前的原值，按 device_id 可逐行恢复。
-- ============================================================================
BEGIN;
SET search_path = fg, public;
SET client_encoding = 'UTF8';

-- ── 0. 先留底（幂等：表已存在就不动）──────────────────────────────────────
CREATE TABLE IF NOT EXISTS fg.device_place_bak_20261105 AS
SELECT device_id, point_id, zone_id, location
  FROM fg.device;

COMMENT ON TABLE fg.device_place_bak_20261105 IS
  '七改删列前的设备位置副本（2026-11-05）：device.point_id / zone_id / location';

-- 留底表要给应用角色只读（本补丁由 postgres 跑，表属主是 postgres；
-- 不 GRANT 的话验收脚本读它直接 InsufficientPrivilege —— 2026-11-05 踩过）
GRANT SELECT ON fg.device_place_bak_20261105 TO fg_app;

-- ── 1. 删列（依赖的索引 ix_device_point 与 CHECK ck_device_location_nonempty
--       会随列自动删除，不必单写）─────────────────────────────────────────
ALTER TABLE fg.device DROP COLUMN IF EXISTS point_id;
ALTER TABLE fg.device DROP COLUMN IF EXISTS zone_id;
ALTER TABLE fg.device DROP COLUMN IF EXISTS location;

COMMIT;

-- ── 2. 自检（应返回 0 行）─────────────────────────────────────────────────
-- SELECT column_name FROM information_schema.columns
--  WHERE table_schema='fg' AND table_name='device'
--    AND column_name IN ('point_id','zone_id','location');