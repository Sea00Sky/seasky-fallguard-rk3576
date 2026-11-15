-- ============================================================================
-- 补丁：告警视频片段（P9 ③）—— 新增 fg.event_evidence 表
-- 日期：2026-10-29
-- 执行：sudo -u postgres psql -d fallguard -v ON_ERROR_STOP=1 \
--         -f /tmp/patch_2026_10_29_event_evidence.sql
--
-- 幂等：全部 IF NOT EXISTS，可重复执行。
-- 回滚：本补丁只新增表，不碰任何既有表 ⇒ 回滚 = DROP TABLE fg.event_evidence;
--       （先确认没有代码在读它，否则 /alerts 详情页会 500）
--
-- ⚠️ 执行顺序：先本补丁，再部署新代码。反过来会让 clips.py 找不到表、
--    每次 ingest 都抛异常（虽然被 catch 住，但日志会刷屏）。
-- ============================================================================

SET search_path = fg, public;
SET client_encoding = 'UTF8';

CREATE TABLE IF NOT EXISTS event_evidence (
  evidence_id   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  event_uid     UUID NOT NULL REFERENCES event(event_uid) ON DELETE CASCADE,
  kind          TEXT NOT NULL DEFAULT 'clip' CHECK (kind IN ('clip')),
  status        TEXT NOT NULL DEFAULT 'PENDING'
     CHECK (status IN ('PENDING','READY','FAILED')),
  clip_start_at TIMESTAMPTZ NOT NULL,
  storage_path  TEXT,
  mime          TEXT NOT NULL DEFAULT 'video/mp4',
  width         INT,
  height        INT,
  duration_ms   INT,
  bytes         BIGINT,
  sha256        CHAR(64),
  attempts      INT NOT NULL DEFAULT 0,
  fail_reason   TEXT,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  ready_at      TIMESTAMPTZ,
  CONSTRAINT uq_evidence_event_kind UNIQUE (event_uid, kind)
);

CREATE INDEX IF NOT EXISTS ix_evidence_pending
  ON event_evidence(clip_start_at) WHERE status = 'PENDING';

-- ⚠️ 新表必须单独授权：CREATE TABLE 不会自动继承 schema 里已有的 GRANT。
--    漏了这一步的表现是 fg-web 日志每 5 秒刷一行
--    「InsufficientPrivilege: permission denied for table event_evidence」，
--    页面本身不报错（只是没有片段）—— 2026-10-29 首次部署就是这么踩的。
GRANT SELECT, INSERT, UPDATE ON event_evidence TO fg_app;

-- 自检：表存在且列齐
SELECT count(*) AS evidence_cols
  FROM information_schema.columns
 WHERE table_schema = 'fg' AND table_name = 'event_evidence';
