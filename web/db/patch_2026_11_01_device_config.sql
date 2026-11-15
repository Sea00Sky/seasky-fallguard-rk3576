-- ============================================================================
-- 判定策略下发（P1）：期望配置 + 不可变历史
--
-- 执行：
--   psql -U postgres -d fallguard -v ON_ERROR_STOP=1 -f db/patch_2026_11_01_device_config.sql
--
-- 模型：
--   device_config      当前期望（每台一行，rev 单调递增）
--   device_config_rev  历史（(device_id, rev) 主键，只追加）
--   实际是否生效看 device.last_policy_rev / last_config_hash（板端 ack 回写）
--
-- 为什么不做 desired/applied 两张表：applied 只是 device 上的两个标量，
-- 单独建表会多一次 JOIN 和一类"行不存在"的失败态。
-- ============================================================================

BEGIN;
SET search_path = fg, public;
SET client_encoding = 'UTF8';

CREATE TABLE IF NOT EXISTS fg.device_config (
  device_id   UUID PRIMARY KEY REFERENCES fg.device(device_id) ON DELETE CASCADE,
  rev         INT NOT NULL,
  params      JSONB NOT NULL,
  note        TEXT,
  updated_by  BIGINT REFERENCES fg.app_user(user_id),
  updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS fg.device_config_rev (
  device_id   UUID NOT NULL REFERENCES fg.device(device_id) ON DELETE CASCADE,
  rev         INT NOT NULL,
  params      JSONB NOT NULL,
  note        TEXT,
  created_by  BIGINT REFERENCES fg.app_user(user_id),
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (device_id, rev)
);

CREATE INDEX IF NOT EXISTS ix_config_rev_device
  ON fg.device_config_rev(device_id, rev DESC);

-- 只追加：历史行不允许改
CREATE OR REPLACE FUNCTION fg.deny_config_rev_mutation() RETURNS trigger AS $$
BEGIN
  RAISE EXCEPTION 'device_config_rev 只允许追加，禁止 %', TG_OP;
END $$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_config_rev_no_update ON fg.device_config_rev;
CREATE TRIGGER trg_config_rev_no_update BEFORE UPDATE ON fg.device_config_rev
  FOR EACH ROW EXECUTE FUNCTION fg.deny_config_rev_mutation();

DROP TRIGGER IF EXISTS trg_config_rev_no_delete ON fg.device_config_rev;
CREATE TRIGGER trg_config_rev_no_delete BEFORE DELETE ON fg.device_config_rev
  FOR EACH ROW EXECUTE FUNCTION fg.deny_config_rev_mutation();

GRANT SELECT, INSERT, UPDATE ON fg.device_config TO fg_app;
GRANT SELECT, INSERT ON fg.device_config_rev TO fg_app;

COMMIT;
