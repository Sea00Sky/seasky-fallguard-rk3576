-- ============================================================================
-- 衡护 HENGUARD 云端 Web · 种子数据（V1）
-- 内容：1 病区（4F-A 心血管内科）+ 1 护士长 + 2 护士 + 1 设备
-- 执行：
--   psql -U postgres -d fallguard -v ON_ERROR_STOP=1 -f db/seed.sql
-- 可重复执行：
--   全部 INSERT ... ON CONFLICT DO NOTHING；重复执行不会覆盖已改过的数据。
-- ============================================================================

SET search_path = fg, public;
-- 文件是 UTF-8；显式声明客户端编码，保证中文病区名/姓名正确入库
SET client_encoding = 'UTF8';

-- ── 1. 病区 ────────────────────────────────────────────────────────────────
INSERT INTO fg.ward (ward_id, ward_name, floor, created_at)
VALUES ('4F-A', '心血管内科', '4F', now())
ON CONFLICT (ward_id) DO NOTHING;

-- ── 2. 账号 ────────────────────────────────────────────────────────────────
-- ⚠️ __PASSWORD_HASH__ 只是占位符，不是合法的 argon2id 哈希，直接登录会失败。
--    首次部署用 scripts/create_user.py 生成哈希（脚本会直接写库）：
--      python scripts/create_user.py create \
--          --login-name limin --display-name 李敏 --role nurse --ward 4F-A
--    三个账号各执行一次；已存在的账号用 passwd / reset-password 子命令改密。
--    也可以手工把脚本输出的 argon2id 哈希填到下面，再重新执行本文件
--    （ON CONFLICT DO NOTHING 不会覆盖已有行，需用 UPDATE 才生效）。
--
--    三个账号的初始密码由部署者设定，且 must_change_password 默认为 TRUE，
--    首次登录后必须改密码（设计方案 §8.2）。
INSERT INTO fg.app_user (login_name, display_name, role, ward_id, password_hash)
VALUES
  ('wangf',  '王芳', 'head_nurse', '4F-A', '__PASSWORD_HASH__'),
  ('limin',  '李敏', 'nurse',      '4F-A', '__PASSWORD_HASH__'),
  ('zhangw', '张伟', 'nurse',      '4F-A', '__PASSWORD_HASH__')
ON CONFLICT (login_name) DO NOTHING;

-- ── 3. 设备 ────────────────────────────────────────────────────────────────
-- 设备接入密钥：明文不放仓库，首次部署时现场生成并保管。
--   device_code = K7C-4FA-01
--   ingest_key  = 现场生成，形如 fg-<device_code>-<32 位十六进制>
--
-- 生成 ingest_key_hash（sha256 小写十六进制）：
--   python -c "import hashlib,sys;print(hashlib.sha256(sys.argv[1].encode()).hexdigest())" <ingest_key>
--
-- ⚠️ 下面的 __INGEST_KEY_HASH__ 是占位符，必须替换成上面算出的值，
--    否则板端 push 会一直 401。
--    替换示例：
--      UPDATE fg.device
--         SET ingest_key_hash = '<上面算出的 sha256>'
--       WHERE device_code = 'K7C-4FA-01';
INSERT INTO fg.device (device_code, ward_id, location, ingest_key_hash)
VALUES ('K7C-4FA-01', '4F-A', '4F-A 东走廊', '__INGEST_KEY_HASH__')
ON CONFLICT (device_code) DO NOTHING;

-- ── 部署后自检（手工执行，确认种子已落库） ────────────────────────────────
--   SELECT ward_id, ward_name, floor FROM fg.ward;
--   SELECT login_name, display_name, role, ward_id, is_active FROM fg.app_user ORDER BY user_id;
--   SELECT device_code, location, is_active FROM fg.device;
