-- ============================================================================
-- 点位（Point）：设备只能挂到已有点位上，点位名/门灯键不再手输
--
-- 执行：
--   psql -U postgres -d fallguard -v ON_ERROR_STOP=1 -f db/patch_2026_11_01_point.sql
--
-- 为什么要有这一层：
--   点位名（这台摄像头叫什么）和门灯键（这盏灯管哪一片）是两件事。
--   一个走廊装两台摄像头时，点位名不同、门灯键相同 —— 走廊口只亮一盏灯。
--   把两者合成一个字段，多摄像头场景下门灯会重复。
--
-- 设备上的 device_name / location 仍然保留，作为点位的**冗余副本**：
--   查询和事件快照都不用再 JOIN；改点位时由应用层同步回设备。
-- ============================================================================

BEGIN;
SET search_path = fg, public;
SET client_encoding = 'UTF8';

CREATE TABLE IF NOT EXISTS fg.point (
  point_id    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  ward_id     TEXT NOT NULL REFERENCES fg.ward(ward_id),
  zone_id     UUID REFERENCES fg.zone(zone_id),
  name        TEXT NOT NULL,                      -- 点位名：'东走廊 01 号机'
  light_key   TEXT NOT NULL,                      -- 门灯键：'4F-A 东走廊'（多台可共用）
  is_active   BOOLEAN NOT NULL DEFAULT TRUE,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
  CONSTRAINT ck_point_name CHECK (btrim(name) <> '' AND char_length(name) <= 64),
  CONSTRAINT ck_point_light_key CHECK (btrim(light_key) <> '' AND char_length(light_key) <= 64),
  CONSTRAINT uq_point_ward_name UNIQUE (ward_id, name)
);
CREATE INDEX IF NOT EXISTS ix_point_ward ON fg.point(ward_id) WHERE is_active;

ALTER TABLE fg.device
  ADD COLUMN IF NOT EXISTS point_id UUID REFERENCES fg.point(point_id);

-- 回填：每台现有设备生成一个点位（名称/门灯键沿用设备上的值）
INSERT INTO fg.point (ward_id, zone_id, name, light_key)
SELECT d.ward_id, d.zone_id,
       COALESCE(NULLIF(btrim(d.device_name), ''), d.device_code),
       COALESCE(NULLIF(btrim(d.location), ''), NULLIF(btrim(d.device_name), ''), d.device_code)
  FROM fg.device d
 WHERE d.point_id IS NULL
ON CONFLICT (ward_id, name) DO NOTHING;

UPDATE fg.device d
   SET point_id = p.point_id
  FROM fg.point p
 WHERE d.point_id IS NULL
   AND p.ward_id = d.ward_id
   AND p.name = COALESCE(NULLIF(btrim(d.device_name), ''), d.device_code);

GRANT SELECT, INSERT, UPDATE ON fg.point TO fg_app;

COMMIT;
