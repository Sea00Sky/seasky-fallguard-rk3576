-- ============================================================================
-- 点位 ↔ 摄像头流（2026-11-02）
--
-- 执行：
--   psql -U postgres -d fallguard -v ON_ERROR_STOP=1 -f db/patch_2026_11_02_point_stream.sql
--
-- ⚠️ **必须先跑本补丁，再部署代码**。新代码会 SELECT p.stream_path，
--    补丁没跑就是 /nurse/today、/alerts、/head/events、/head/devices 全部 500。
--    （与 patch_2026_10_29_ever_confirmed.sql 是同一类坑，/healthz 会先暴露。）
--
-- 为什么是「每路流各自独立」的模型：
--   摄像头自己把画面推到云端（板端 SRT → mediamtx 的某条 path），
--   管理员把 path 名字填到点位上，这个点位就出画面；填别的点位，画面就跟着走。
--   所以 stream_path 是**点位**的属性，不是设备的属性：
--     · 点位可以没有设备（实验点位 / 尚未装机的点位），不该逼着伪造 fg.device 行，
--       fg.device 的 ingest_key_hash 是 NOT NULL，伪造出来还会生成多余的推送密钥；
--     · device_code 的字符集是 ^[A-Z0-9][A-Z0-9_-]{2,31}$（全大写），
--       而 mediamtx 的 path 区分大小写，直接复用会留下"看着一样其实不是一条"的雷。
--
-- 已知限制（本轮不解决）：
--   fg.point → fg.device 目前是 1:N。一个点位挂多台摄像头时只能有一条 stream_path，
--   多台同时往同一条 path 推会被 mediamtx 的 overridePublisher 互顶（画面在源之间跳）。
--   真要多摄像头，下一步迁到 camera / point_stream 表，stream_path 是平滑起点。
--
-- 可重复执行：ADD COLUMN IF NOT EXISTS / 约束先 DROP 再建 / 索引 IF NOT EXISTS。
-- ============================================================================

BEGIN;
SET search_path = fg, public;
SET client_encoding = 'UTF8';

-- ── 1. 加列 ────────────────────────────────────────────────────────────────
ALTER TABLE fg.point
  ADD COLUMN IF NOT EXISTS stream_path TEXT,
  ADD COLUMN IF NOT EXISTS source_type TEXT NOT NULL DEFAULT 'camera';

COMMENT ON COLUMN fg.point.stream_path IS
  'mediamtx 的路径名。NULL = 该点位还没有配置摄像头（护士端显示「未配置摄像头」）。';
COMMENT ON COLUMN fg.point.source_type IS
  'camera=真实摄像头 | video_file=演示用文件循环推流 | none=未接入';

-- ── 2. 约束 ────────────────────────────────────────────────────────────────
-- 路径名会被拼进上游 URL（live.py），字符集必须比 mediamtx 更严：
--   单段、不以分隔符开头、最长 32、只允许字母数字下划线连字符。
-- 显式排除 '/'（HLS 代理会拒绝，填了也是白填）、':'（SRT streamid 的分隔符）、
-- 以及会改写 URL 的 '?' '#' '%' '\'。
ALTER TABLE fg.point DROP CONSTRAINT IF EXISTS ck_point_stream_path;
ALTER TABLE fg.point
  ADD CONSTRAINT ck_point_stream_path
  CHECK (stream_path IS NULL OR stream_path ~ '^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$');

ALTER TABLE fg.point DROP CONSTRAINT IF EXISTS ck_point_source_type;
ALTER TABLE fg.point
  ADD CONSTRAINT ck_point_source_type
  CHECK (source_type IN ('camera', 'video_file', 'none'));

-- ── 3. 全局唯一（部分索引：NULL 不参与）────────────────────────────────────
-- mediamtx 的 path 是全机共用的，不是按病区划分的。两条点位填同一个 path，
-- 表现是"两个点位看到的其实是同一幅画面"，排查起来极费劲——所以在库层挡住。
CREATE UNIQUE INDEX IF NOT EXISTS uq_point_stream_path
  ON fg.point(stream_path)
  WHERE stream_path IS NOT NULL;

-- ── 4. 回填：把现役那台真实摄像头绑到板端正在推的 live ─────────────────────
-- 板端 FG_CLOUD_STREAMID=publish:live:board:<pass>，落点就是 mediamtx 的 live。
--
-- 三条护栏，缺一条这个补丁就不可重跑：
--   ① 只在「全库还没有任何点位绑定 live」时才回填 —— 否则管理员把 live 改绑到
--      别的点位之后再跑一次补丁，会撞 uq_point_stream_path 让整个事务回滚；
--   ② 候选点在**一条语句里**只选一个（CTE + LIMIT 1）。不能写成
--      `WHERE p.point_id = (SELECT ... WHERE d.ward_id = p.ward_id)`：
--      那是按行求值，两个病区各有一台设备时会有两行同时命中，
--      两行都写 'live' → 撞唯一索引 → 整个补丁回滚。多病区部署必踩。
--   ③ 优先挑有启用设备的点位；再按病区、点位排序，结果稳定可复现。
--
-- 挑中的是「挂着启用设备」的那个点位——现网就是 4F-A / 409病房 / K7C-4FA-01。
WITH target AS (
    SELECT p.point_id
      FROM fg.point p
     WHERE p.stream_path IS NULL
       AND NOT EXISTS (SELECT 1 FROM fg.point q WHERE q.stream_path = 'live')
       AND EXISTS (
             SELECT 1 FROM fg.device d
              WHERE d.point_id = p.point_id AND d.is_active
           )
     ORDER BY p.ward_id, p.point_id
     LIMIT 1
)
UPDATE fg.point p
   SET stream_path = 'live',
       source_type = 'camera',
       updated_at  = now()
  FROM target t
 WHERE p.point_id = t.point_id;

-- ── 5. 授权 ────────────────────────────────────────────────────────────────
-- 加列不需要新的列级授权（表级 GRANT 已覆盖），这里显式再给一次，
-- 防止有人从零重建时漏掉 patch_2026_11_01_point.sql 里的那行。
GRANT SELECT, INSERT, UPDATE ON fg.point TO fg_app;

COMMIT;
