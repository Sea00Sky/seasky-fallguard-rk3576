-- ============================================================================
-- 2026-11-05 六改·收尾：删掉 fg.point.stream_path
--
-- ⚠️ **必须在「六改代码」上线并验收之后再跑**。
--    顺序反了（先删列、后推代码）会让所有还在 `SELECT p.stream_path` 的
--    页面整页 500 —— /nurse/today、/alerts、/head/events、/head/points 全中。
--
-- 为什么删而不是留着不用：留一列没人读、但还在界面上有输入框的字段，
-- 下一次「改一下这个流」就会又变成第二个真源 —— 六改就是为了消灭它。
-- 绑定关系现在只有两处：
--     point.device_id  = 这个点位显示哪台设备
--     device.stream_path = 那台设备推哪条流
-- ============================================================================
BEGIN;

DROP INDEX IF EXISTS uq_point_stream_path;
ALTER TABLE fg.point DROP CONSTRAINT IF EXISTS ck_point_stream_path;
ALTER TABLE fg.point DROP COLUMN IF EXISTS stream_path;

COMMIT;