-- ============================================================================
-- P11 · P0：三角色底座（2026-11-01）
--
-- 执行：
--   psql -U postgres -d fallguard -v ON_ERROR_STOP=1 -f db/patch_2026_11_01_roles.sql
--
-- 改三件事：
--   1. role 允许 'admin'
--   2. ward_id 允许 NULL（管理员是全院角色，不绑病区）
--   3. 强约束：admin ⇔ ward_id IS NULL（两者必须同时成立）
--
-- ⚠️ 现有账号全部是 nurse/head_nurse 且有病区 ⇒ 新约束天然满足，不需要回填。
-- ⚠️ ward_id 的外键保留（REFERENCES ward），只是允许 NULL。
-- ============================================================================

BEGIN;
SET search_path = fg, public;

-- 1. 角色枚举
ALTER TABLE fg.app_user DROP CONSTRAINT IF EXISTS app_user_role_check;
ALTER TABLE fg.app_user ADD CONSTRAINT app_user_role_check
  CHECK (role IN ('nurse', 'head_nurse', 'admin'));

-- 2. 管理员不绑病区
ALTER TABLE fg.app_user ALTER COLUMN ward_id DROP NOT NULL;

-- 3. 角色与病区必须一致
ALTER TABLE fg.app_user DROP CONSTRAINT IF EXISTS ck_user_ward_by_role;
ALTER TABLE fg.app_user ADD CONSTRAINT ck_user_ward_by_role
  CHECK ((role = 'admin') = (ward_id IS NULL));

COMMIT;
