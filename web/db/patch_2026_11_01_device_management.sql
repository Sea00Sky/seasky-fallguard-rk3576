-- ============================================================================
-- 设备管理 P0：区域 + 设备台账 + 事件位置快照
-- 依据：2026-11-01 设备管理方案（P0 阶段）
--
-- 执行：
--   psql -U postgres -d fallguard -v ON_ERROR_STOP=1 -f db/patch_2026_11_01_device_management.sql
--
-- 可重复执行：表/索引用 IF NOT EXISTS，约束先 DROP IF EXISTS 再建，回填带条件。
--
-- 三条设计决定（改之前先读）：
--   1. 一台板子 = 一路摄像头 = 一行 fg.device。多摄像头 = 多行，不建 camera 表。
--   2. 灵敏度不在这张表里。P1 会加 device_config（期望）+ device_config_rev（历史）；
--      本补丁只加「板端上报的镜像」列（last_policy_rev / last_config_hash / last_status）。
--   3. fg.event 加 location_text / zone_id 快照：设备位置可编辑，但历史告警
--      必须留在当时的位置，否则门灯和按位置统计会整体漂移。
-- ============================================================================

BEGIN;
SET search_path = fg, public;
SET client_encoding = 'UTF8';

-- ── 1. 区域 ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS fg.zone (
  zone_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  ward_id     TEXT NOT NULL REFERENCES fg.ward(ward_id),
  name        TEXT NOT NULL,                      -- '东走廊' / '3 号病房'
  zone_type   TEXT NOT NULL DEFAULT 'other',
  floor       TEXT,
  is_active   BOOLEAN NOT NULL DEFAULT TRUE,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
  CONSTRAINT ck_zone_name CHECK (btrim(name) <> '' AND char_length(name) <= 40),
  CONSTRAINT ck_zone_type CHECK (zone_type IN
    ('corridor','ward_room','bathroom','dayroom','stairwell','nurse_station','other')),
  CONSTRAINT ck_zone_floor CHECK (floor IS NULL OR char_length(floor) <= 16),
  CONSTRAINT uq_zone_ward_name UNIQUE (ward_id, name)
);
CREATE INDEX IF NOT EXISTS ix_zone_ward ON fg.zone(ward_id) WHERE is_active;

-- ── 2. device 扩列 ─────────────────────────────────────────────────────────
ALTER TABLE fg.device
  ADD COLUMN IF NOT EXISTS zone_id            UUID REFERENCES fg.zone(zone_id),
  ADD COLUMN IF NOT EXISTS device_name        TEXT,
  ADD COLUMN IF NOT EXISTS location_detail    TEXT,
  ADD COLUMN IF NOT EXISTS mount_type         TEXT,
  ADD COLUMN IF NOT EXISTS mount_height_cm    SMALLINT,
  ADD COLUMN IF NOT EXISTS view_direction     TEXT,
  ADD COLUMN IF NOT EXISTS lens_fov_deg       SMALLINT,
  ADD COLUMN IF NOT EXISTS device_model       TEXT,
  ADD COLUMN IF NOT EXISTS hardware_id        TEXT,
  ADD COLUMN IF NOT EXISTS firmware_ver       TEXT,
  ADD COLUMN IF NOT EXISTS camera_model       TEXT,
  ADD COLUMN IF NOT EXISTS camera_serial      TEXT,
  ADD COLUMN IF NOT EXISTS notes              TEXT,
  ADD COLUMN IF NOT EXISTS commissioned_at    DATE,
  ADD COLUMN IF NOT EXISTS profile_updated_at TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS profile_updated_by BIGINT REFERENCES fg.app_user(user_id),
  ADD COLUMN IF NOT EXISTS last_seen_at       TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS last_seen_ip       INET,
  ADD COLUMN IF NOT EXISTS last_policy_rev    INT,
  ADD COLUMN IF NOT EXISTS last_config_hash   TEXT,
  ADD COLUMN IF NOT EXISTS last_status        JSONB;

-- 回填必须在加 CHECK 之前
UPDATE fg.device
   SET device_name = COALESCE(NULLIF(btrim(device_name), ''), NULLIF(btrim(location), ''), device_code)
 WHERE device_name IS NULL OR btrim(device_name) = '';
UPDATE fg.device
   SET location = device_code
 WHERE location IS NULL OR btrim(location) = '';
UPDATE fg.device
   SET profile_updated_at = created_at
 WHERE profile_updated_at IS NULL;

-- ── 3. device 约束 ─────────────────────────────────────────────────────────
ALTER TABLE fg.device DROP CONSTRAINT IF EXISTS ck_device_code_format;
ALTER TABLE fg.device ADD CONSTRAINT ck_device_code_format
  CHECK (device_code = upper(device_code) AND device_code ~ '^[A-Z0-9][A-Z0-9_-]{2,31}$');

ALTER TABLE fg.device DROP CONSTRAINT IF EXISTS ck_device_location_nonempty;
ALTER TABLE fg.device ADD CONSTRAINT ck_device_location_nonempty
  CHECK (location IS NULL OR (btrim(location) <> '' AND char_length(location) <= 64));

ALTER TABLE fg.device DROP CONSTRAINT IF EXISTS ck_device_name_nonempty;
ALTER TABLE fg.device ADD CONSTRAINT ck_device_name_nonempty
  CHECK (device_name IS NULL OR (btrim(device_name) <> '' AND char_length(device_name) <= 64));

ALTER TABLE fg.device DROP CONSTRAINT IF EXISTS ck_device_mount_type;
ALTER TABLE fg.device ADD CONSTRAINT ck_device_mount_type
  CHECK (mount_type IS NULL OR mount_type IN ('ceiling','wall','pole','bedside','other'));

ALTER TABLE fg.device DROP CONSTRAINT IF EXISTS ck_device_mount_height;
ALTER TABLE fg.device ADD CONSTRAINT ck_device_mount_height
  CHECK (mount_height_cm IS NULL OR mount_height_cm BETWEEN 30 AND 600);

ALTER TABLE fg.device DROP CONSTRAINT IF EXISTS ck_device_fov;
ALTER TABLE fg.device ADD CONSTRAINT ck_device_fov
  CHECK (lens_fov_deg IS NULL OR lens_fov_deg BETWEEN 10 AND 180);

ALTER TABLE fg.device DROP CONSTRAINT IF EXISTS ck_device_config_hash;
ALTER TABLE fg.device ADD CONSTRAINT ck_device_config_hash
  CHECK (last_config_hash IS NULL OR last_config_hash ~ '^[0-9a-f]{64}$');

-- ── 4. event 位置快照 ──────────────────────────────────────────────────────
ALTER TABLE fg.event
  ADD COLUMN IF NOT EXISTS location_text TEXT,
  ADD COLUMN IF NOT EXISTS zone_id       UUID;

-- 老事件：用设备当前值回填（一次性；之后入库时写当时值）
UPDATE fg.event e
   SET location_text = COALESCE(e.location_text, d.location),
       zone_id       = COALESCE(e.zone_id, d.zone_id)
  FROM fg.device d
 WHERE d.device_id = e.device_id
   AND (e.location_text IS NULL OR e.zone_id IS NULL);

CREATE INDEX IF NOT EXISTS ix_event_zone
  ON fg.event(zone_id, received_at DESC);

-- ── 5. 权限 ────────────────────────────────────────────────────────────────
-- 新表必须单独 GRANT（P9 漏 GRANT 导致运行期 500 的教训）。
-- device / event 的新列继承表级权限，不用重复授权。
GRANT SELECT, INSERT, UPDATE ON fg.zone TO fg_app;

COMMIT;
