-- ============================================================================
-- 2026-11-05 六改：点位**可选绑定设备**，推流路径回归设备属性
--
-- 模型（用户拍板）：
--   ① 设备：身份 + 安装位置 + **推哪条流**（device.stream_path）
--   ② 点位：可选绑定一台设备（point.device_id），不再自己存 stream_path
--   ③ 画面/骨架/告警归属 = 绑定了这台设备的那个点位；没绑定 = 按安装位置
--
-- 为什么要改：五改-2 让「保存点位流」把设备登记一起搬走，等于用归属覆盖安装 ——
-- 409病房 被搬空、device_name 被改写成 512病房、4F-A 护士长看不见自己的机器。
-- 那是一个字段装两个事实的必然结果，这次把两个事实拆到两张表。
--
-- 执行顺序：本补丁 → 推代码 →（验收后）patch_2026_11_05c_drop_point_stream.sql
-- 本补丁只增列 + 回填，不删任何东西，旧代码在补丁后仍可运行（读不到 device_id 不影响）。
-- ============================================================================
BEGIN;

-- 1) 点位绑定设备（可选：NULL = 这个点位现在没接设备）
ALTER TABLE fg.point ADD COLUMN IF NOT EXISTS device_id uuid;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'fk_point_device') THEN
        ALTER TABLE fg.point
            ADD CONSTRAINT fk_point_device FOREIGN KEY (device_id)
            REFERENCES fg.device (device_id) ON DELETE SET NULL;
    END IF;
END $$;

-- 一台设备最多被一个点位显示：两个点位绑同一台 = 两处画面/告警归属打架
CREATE UNIQUE INDEX IF NOT EXISTS uq_point_device_id
    ON fg.point (device_id) WHERE device_id IS NOT NULL;

COMMENT ON COLUMN fg.point.device_id IS
    '这个点位显示哪台设备（可选）。画面/骨架/告警归属都看它；流本身是 device.stream_path';

GRANT SELECT, INSERT, UPDATE ON fg.point TO fg_app;

-- 2) 回填
-- ① 先按「流」等价迁移：原来 p.stream_path = d.stream_path 的，就是同一个绑定
UPDATE fg.point p
   SET device_id = d.device_id
  FROM fg.device d
 WHERE p.device_id IS NULL
   AND p.stream_path IS NOT NULL
   AND d.stream_path = p.stream_path;

-- ② 其余点位按**安装关系**补齐一台（装在这儿的设备）
--    DISTINCT ON：一个点位装过多台时只取最近有心跳的那台，别让回填炸唯一索引
UPDATE fg.point p
   SET device_id = x.device_id
  FROM (
        SELECT DISTINCT ON (d.point_id) d.point_id, d.device_id
          FROM fg.device d
         WHERE d.point_id IS NOT NULL
         ORDER BY d.point_id, d.last_seen_at DESC NULLS LAST, d.device_code
       ) x
 WHERE p.point_id = x.point_id
   AND p.device_id IS NULL;

-- 3) 修五改-2 造成的数据损伤：K7C-4FA-01 的**安装登记**回 409病房
--    （归属仍在 512病房 —— 那是 point.device_id 的事，两回事）
--    审计里记着 from_point=409病房 / from_ward=4F-A，这里按点位名精确还原。
UPDATE fg.device d
   SET point_id = p.point_id,
       zone_id = p.zone_id,
       ward_id = p.ward_id,
       device_name = p.name,
       location = p.light_key,
       profile_updated_at = now()
  FROM fg.point p
 WHERE d.device_code = 'K7C-4FA-01'
   AND p.name = '409病房';

COMMIT;
