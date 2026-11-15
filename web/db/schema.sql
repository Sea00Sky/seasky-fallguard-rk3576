-- ============================================================================
-- 衡护 HENGUARD 云端 Web · 数据库结构（V1）
--   §7.3 核心表 DDL（7 张表 + 索引 + 只追加审计）
--   §7.4 事件身份与幂等键（uq_event_board / ix_event_retract）
--   §11.2 事实视图 fg.v_event_fact
-- 执行：
--   psql -U postgres -d fallguard -v ON_ERROR_STOP=1 -f db/schema.sql
-- 可重复执行：
--   表、索引用 IF NOT EXISTS；触发器先 DROP IF EXISTS 再建；视图用 OR REPLACE。
-- 红线：
--   表名、列名必须与设计方案逐字一致，不得改名、不得加列。
-- ============================================================================

-- ⚠️ 2026-11-05：本文件是 **V1 基线，不再回溯改**。后续变更一律走同目录 `patch_*.sql`。
--    已删列：`point.stream_path`（patch_2026_11_05c）、
--    `device.location / zone_id / point_id`（patch_2026_11_05_drop_device_place）。
--    新环境 = 先跑本文件，再按文件名顺序跑全部 patch；只跑本文件会得到一版带死列的旧结构。

CREATE SCHEMA IF NOT EXISTS fg;
SET search_path = fg, public;
-- 文件是 UTF-8；显式声明客户端编码，避免 Windows/GBK 终端把中文注释和种子数据读花
SET client_encoding = 'UTF8';

-- ── 1. 病区 ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS ward (
  ward_id     TEXT PRIMARY KEY,                    -- '4F-A'
  ward_name   TEXT NOT NULL,                       -- '心血管内科'
  floor       TEXT,                                -- '4F'
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ── 1b. 区域（2026-11-01 设备管理 P0） ────────────────────────────────────
-- 一台板子 = 一路摄像头 = 一行 device；zone 是门灯/统计的分组维度。
CREATE TABLE IF NOT EXISTS zone (
  zone_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  ward_id     TEXT NOT NULL REFERENCES ward(ward_id),
  name        TEXT NOT NULL,
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
CREATE INDEX IF NOT EXISTS ix_zone_ward ON zone(ward_id) WHERE is_active;

-- ── 1c. 点位（2026-11-01 设备管理 P0 补） ──────────────────────────────────
-- 点位名 = 这台摄像头叫什么；门灯键 = 这盏灯管哪一片（多台可共用）。
CREATE TABLE IF NOT EXISTS point (
  point_id    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  ward_id     TEXT NOT NULL REFERENCES ward(ward_id),
  zone_id     UUID REFERENCES zone(zone_id),
  name        TEXT NOT NULL,
  light_key   TEXT NOT NULL,
  is_active   BOOLEAN NOT NULL DEFAULT TRUE,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
  CONSTRAINT ck_point_name CHECK (btrim(name) <> '' AND char_length(name) <= 64),
  CONSTRAINT ck_point_light_key CHECK (btrim(light_key) <> '' AND char_length(light_key) <= 64),
  CONSTRAINT uq_point_ward_name UNIQUE (ward_id, name)
);
CREATE INDEX IF NOT EXISTS ix_point_ward ON point(ward_id) WHERE is_active;

-- ── 2. 账号 ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS app_user (
  user_id        BIGSERIAL PRIMARY KEY,
  login_name     TEXT NOT NULL UNIQUE,
  display_name   TEXT NOT NULL,
  -- 2026-11-01 P11：加入信息科管理员；admin 是全院角色，不绑病区
  role           TEXT NOT NULL CHECK (role IN ('nurse','head_nurse','admin')),
  ward_id        TEXT REFERENCES ward(ward_id),
  CONSTRAINT ck_user_ward_by_role CHECK ((role = 'admin') = (ward_id IS NULL)),
  password_hash  TEXT NOT NULL,                    -- argon2id
  must_change_password BOOLEAN NOT NULL DEFAULT TRUE,
  is_active      BOOLEAN NOT NULL DEFAULT TRUE,
  failed_count   INT NOT NULL DEFAULT 0,
  locked_until   TIMESTAMPTZ,
  last_login_at  TIMESTAMPTZ,
  created_by     BIGINT REFERENCES app_user(user_id),
  created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_user_ward ON app_user(ward_id, role) WHERE is_active;

-- ── 2b. 护士 ↔ 负责区域（2026-11-04 P12）──────────────────────────────────
-- 数据范围 = 病区（老轴，app_user.ward_id）× 区域（新轴，本表）。两级**叠加**：
--   WHERE e.ward_id = %s AND e.zone_id = ANY(%s)
-- 只有 role='nurse' 参与；护士长/管理员不受本表限制（看全病区）。
-- ⚠️ 别把「区域 zone」和「病区 ward」混了：zone_type 里的 'ward_room' 是**病房**
--    （一个房间），属于区域层；「病区」永远只指 fg.ward。
-- 见 db/patch_2026_11_04_user_zone.sql。
CREATE TABLE IF NOT EXISTS app_user_zone (
  user_id    BIGINT NOT NULL REFERENCES app_user(user_id) ON DELETE CASCADE,
  zone_id    UUID   NOT NULL REFERENCES zone(zone_id)     ON DELETE CASCADE,
  created_by BIGINT REFERENCES app_user(user_id),
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (user_id, zone_id)
);
CREATE INDEX IF NOT EXISTS ix_auz_zone ON app_user_zone(zone_id);

-- 病区护栏：区域必须与账号同病区（跨两张表，CHECK 表达不了，用触发器）。
-- 应用层与查询层各还有一道（见补丁第 2 节），这是最后一道。
CREATE OR REPLACE FUNCTION fg.ck_user_zone_same_ward() RETURNS trigger AS $$
BEGIN
  IF NOT EXISTS (
    SELECT 1
      FROM fg.app_user u
      JOIN fg.zone z ON z.zone_id = NEW.zone_id
     WHERE u.user_id = NEW.user_id
       AND u.ward_id IS NOT NULL
       AND u.ward_id = z.ward_id
  ) THEN
    RAISE EXCEPTION
      'app_user_zone: 区域必须与账号同病区（user_id=%, zone_id=%）', NEW.user_id, NEW.zone_id
      USING ERRCODE = 'check_violation';
  END IF;
  RETURN NEW;
END
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_user_zone_same_ward ON app_user_zone;
CREATE TRIGGER trg_user_zone_same_ward
  BEFORE INSERT OR UPDATE ON app_user_zone
  FOR EACH ROW EXECUTE FUNCTION fg.ck_user_zone_same_ward();

-- ── 3. 设备 ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS device (
  device_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  device_code     TEXT NOT NULL UNIQUE,            -- 'K7C-4FA-01'
  ward_id         TEXT NOT NULL REFERENCES ward(ward_id),
  location        TEXT,                            -- '4F-A 东走廊'（自由文本，不是床号）
  ingest_key_hash TEXT NOT NULL,                   -- ingest_key 的 sha256
  model_ver       BIGINT,                          -- 最近上报的 model_ver
  is_active       BOOLEAN NOT NULL DEFAULT TRUE,
  last_event_at   TIMESTAMPTZ,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

  -- 设备台账（2026-11-01 设备管理 P0）：位置可编辑，历史靠 event 快照保稳
  zone_id            UUID REFERENCES zone(zone_id),
  point_id           UUID REFERENCES point(point_id),
  device_name        TEXT,                         -- '东走廊 01 号机'
  location_detail    TEXT,                         -- '3 号病房门口上方'
  mount_type         TEXT,                         -- ceiling/wall/pole/bedside/other
  mount_height_cm    SMALLINT,
  view_direction     TEXT,
  lens_fov_deg       SMALLINT,
  device_model       TEXT,                         -- 'Kickpi K7C'
  hardware_id        TEXT,
  firmware_ver       TEXT,
  camera_model       TEXT,                         -- 'IMX415'
  camera_serial      TEXT,
  notes              TEXT,
  commissioned_at    DATE,
  profile_updated_at TIMESTAMPTZ,
  profile_updated_by BIGINT REFERENCES app_user(user_id),

  -- 板端上报的镜像（P1 的 device_config 才是期望值）
  last_seen_at     TIMESTAMPTZ,
  last_seen_ip     INET,
  last_policy_rev  INT,
  last_config_hash TEXT,
  last_status      JSONB,

  CONSTRAINT ck_device_code_format CHECK (
    device_code = upper(device_code) AND device_code ~ '^[A-Z0-9][A-Z0-9_-]{2,31}$'),
  CONSTRAINT ck_device_location_nonempty CHECK (
    location IS NULL OR (btrim(location) <> '' AND char_length(location) <= 64)),
  CONSTRAINT ck_device_name_nonempty CHECK (
    device_name IS NULL OR (btrim(device_name) <> '' AND char_length(device_name) <= 64)),
  CONSTRAINT ck_device_mount_type CHECK (
    mount_type IS NULL OR mount_type IN ('ceiling','wall','pole','bedside','other')),
  CONSTRAINT ck_device_mount_height CHECK (
    mount_height_cm IS NULL OR mount_height_cm BETWEEN 30 AND 600),
  CONSTRAINT ck_device_fov CHECK (
    lens_fov_deg IS NULL OR lens_fov_deg BETWEEN 10 AND 180),
  CONSTRAINT ck_device_config_hash CHECK (
    last_config_hash IS NULL OR last_config_hash ~ '^[0-9a-f]{64}$')
);
CREATE INDEX IF NOT EXISTS ix_device_ward ON device(ward_id);
-- 2026-11-04 P12：区域对账与设备台账都按 point_id 找设备
CREATE INDEX IF NOT EXISTS ix_device_point ON device(point_id);

-- ── 3b. 判定策略下发（P1，2026-11-01） ─────────────────────────────────────
-- device_config = 当前期望；device_config_rev = 不可变历史；
-- 实际是否生效看 device.last_policy_rev / last_config_hash（板端 ack 回写）。
CREATE TABLE IF NOT EXISTS device_config (
  device_id   UUID PRIMARY KEY REFERENCES device(device_id) ON DELETE CASCADE,
  rev         INT NOT NULL,
  params      JSONB NOT NULL,
  note        TEXT,
  updated_by  BIGINT REFERENCES app_user(user_id),
  updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS device_config_rev (
  device_id   UUID NOT NULL REFERENCES device(device_id) ON DELETE CASCADE,
  rev         INT NOT NULL,
  params      JSONB NOT NULL,
  note        TEXT,
  created_by  BIGINT REFERENCES app_user(user_id),
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (device_id, rev)
);
CREATE INDEX IF NOT EXISTS ix_config_rev_device
  ON device_config_rev(device_id, rev DESC);

-- 只追加：历史行不允许改
CREATE OR REPLACE FUNCTION fg.deny_config_rev_mutation() RETURNS trigger AS $$
BEGIN
  RAISE EXCEPTION 'device_config_rev 只允许追加，禁止 %', TG_OP;
END $$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_config_rev_no_update ON device_config_rev;
CREATE TRIGGER trg_config_rev_no_update BEFORE UPDATE ON device_config_rev
  FOR EACH ROW EXECUTE FUNCTION fg.deny_config_rev_mutation();

DROP TRIGGER IF EXISTS trg_config_rev_no_delete ON device_config_rev;
CREATE TRIGGER trg_config_rev_no_delete BEFORE DELETE ON device_config_rev
  FOR EACH ROW EXECUTE FUNCTION fg.deny_config_rev_mutation();

-- ── 4. 事件 ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS event (
  event_uid      UUID PRIMARY KEY,                 -- UUIDv7，云端生成，永不变
  device_id      UUID NOT NULL REFERENCES device(device_id),
  ward_id        TEXT NOT NULL REFERENCES ward(ward_id),

  -- 板端来源（幂等键）
  board_seq      BIGINT NOT NULL,                  -- 网关 SQLite row_seq（撤回与 confirmed 共用）
  board_ts       TIMESTAMPTZ NOT NULL,             -- 板端墙钟（仅参考）
  received_at    TIMESTAMPTZ NOT NULL DEFAULT now(),

  -- 模型侧
  person_id      INT NOT NULL,                     -- 跟踪号，会复用
  window_start   BIGINT NOT NULL,                  -- 感知进程内帧号，进程重启归零
  window_end     BIGINT,
  model_ver      BIGINT NOT NULL,
  prob_fall      REAL NOT NULL,
  coverage       REAL NOT NULL,
  hybrid_score   REAL,
  bbox           REAL[4],                          -- 640×480 原图像素；V1 不渲染

  -- 状态（双状态模型）
  severity          TEXT NOT NULL CHECK (severity IN ('S1','S2','S3')),
  detection_state   TEXT NOT NULL CHECK (detection_state IN ('CONFIRMED','RETRACTED')),
  disposition_state TEXT NOT NULL DEFAULT 'PENDING'
     CHECK (disposition_state IN
       ('PENDING','ACKED','ON_SCENE','RESOLVED','FALSE_ALARM','CLOSED')),

  -- 人工侧
  outcome            TEXT CHECK (outcome IN
     ('ASSISTED','NO_ABNORMALITY','MEDICAL','HOSPITALIZED','OTHER')),
  false_alarm_reason TEXT CHECK (false_alarm_reason IN
     ('MODEL_MISJUDGE','OCCLUSION','NOT_PATIENT','DUPLICATE','OTHER')),
  assignee_id   BIGINT REFERENCES app_user(user_id),
  acked_at      TIMESTAMPTZ,
  arrived_at    TIMESTAMPTZ,
  resolved_at   TIMESTAMPTZ,
  note          TEXT,

  -- 运维
  orphan_retract BOOLEAN NOT NULL DEFAULT FALSE,
  is_test        BOOLEAN NOT NULL DEFAULT FALSE,
  revision       INT NOT NULL DEFAULT 0,

  -- 位置快照（2026-11-01）：设备位置可编辑，历史告警留在当时的位置
  location_text  TEXT,
  zone_id        UUID,

  -- 幂等键：confirmed 走它；撤回不走（见 §7.4，用唯一键会插成两条）
  CONSTRAINT uq_event_board UNIQUE (device_id, board_seq, board_ts),

  -- 4 个 CHECK 约束
  CONSTRAINT ck_event_acked   CHECK (acked_at   IS NULL OR acked_at   >= received_at),
  CONSTRAINT ck_event_arrived CHECK (arrived_at IS NULL OR arrived_at >= acked_at),
  -- 2026-11-04：等价 → 蕴含。CLOSE 只改状态、保留处置结果，等价约束会让
  -- 「归档」必炸（RESOLVED→CLOSE / FALSE_ALARM→CLOSE 都 CheckViolation ⇒ 500）。
  -- 见 patch_2026_11_04_close_state.sql 与 P12 文档 §11。
  CONSTRAINT ck_event_fa      CHECK (
      (disposition_state IN ('FALSE_ALARM', 'CLOSED') OR false_alarm_reason IS NULL)
  AND (disposition_state <> 'FALSE_ALARM' OR false_alarm_reason IS NOT NULL)),
  CONSTRAINT ck_event_outcome CHECK (
      (disposition_state IN ('RESOLVED', 'CLOSED') OR outcome IS NULL)
  AND (disposition_state <> 'RESOLVED' OR outcome IS NOT NULL))
);

CREATE INDEX IF NOT EXISTS ix_event_ward_time
  ON event(ward_id, received_at DESC);
CREATE INDEX IF NOT EXISTS ix_event_ward_disp
  ON event(ward_id, disposition_state, received_at DESC);
CREATE INDEX IF NOT EXISTS ix_event_assignee
  ON event(assignee_id, disposition_state);
-- 撤回匹配专用：只索引 CONFIRMED 行
CREATE INDEX IF NOT EXISTS ix_event_retract
  ON event(device_id, person_id, window_start, model_ver)
  WHERE detection_state = 'CONFIRMED';
CREATE INDEX IF NOT EXISTS ix_event_zone
  ON event(zone_id, received_at DESC);

-- ── 5. 动作流水 ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS event_action (
  action_id    BIGSERIAL PRIMARY KEY,
  event_uid    UUID NOT NULL REFERENCES event(event_uid) ON DELETE RESTRICT,
  actor_id     BIGINT NOT NULL REFERENCES app_user(user_id),
  action       TEXT NOT NULL CHECK (action IN
     ('ACK','ARRIVE','RESOLVE','FALSE_ALARM','CLOSE','REOPEN','NOTE','RETRACT')),
  from_state   TEXT,
  to_state     TEXT,
  outcome      TEXT,
  reason       TEXT,
  client_request_id TEXT UNIQUE,                   -- 人工动作幂等
  created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_action_event ON event_action(event_uid, created_at);

-- ── 6. 快照 ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS snapshot (
  snapshot_id  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  event_uid    UUID NOT NULL REFERENCES event(event_uid) ON DELETE CASCADE,
  storage_path TEXT NOT NULL,                      -- 不在 webroot 下（见 §8.7）
  width        INT NOT NULL DEFAULT 320,
  height       INT NOT NULL DEFAULT 240,
  bytes        INT NOT NULL,
  sha256       CHAR(64) NOT NULL,
  source       TEXT NOT NULL DEFAULT 'board_upload',
  uploaded_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
  CONSTRAINT uq_snapshot_event UNIQUE (event_uid)
);

-- ── 6b. 事件证据：告警视频片段（P9 ③，2026-10-29） ────────────────────────
-- 为什么单独建表而不是复用 snapshot：
--   ① snapshot 有 UNIQUE(event_uid)，一个事件只能一张图；片段是同事件的**另一种**证据
--   ② 片段有生命周期（PENDING → READY / FAILED），快照没有
-- 表名沿用 P9 方案 §4.1 ER 里的 event_evidence，P1 数据模型 V2 直接接着用，
--   届时只需把 kind 扩成 ('clip','bbox','kp')，不必迁移。
CREATE TABLE IF NOT EXISTS event_evidence (
  evidence_id   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  event_uid     UUID NOT NULL REFERENCES event(event_uid) ON DELETE CASCADE,
  kind          TEXT NOT NULL DEFAULT 'clip' CHECK (kind IN ('clip')),
  status        TEXT NOT NULL DEFAULT 'PENDING'
     CHECK (status IN ('PENDING','READY','FAILED')),

  -- 片段应覆盖的绝对起点（云端墙钟，已含 clip_lag_s 补偿）。
  -- 存绝对时刻而不是相对秒数：worker 重启后仍能算出该切哪一段。
  clip_start_at TIMESTAMPTZ NOT NULL,

  storage_path  TEXT,                              -- READY 才有；不在 webroot 下（§8.7）
  mime          TEXT NOT NULL DEFAULT 'video/mp4',
  width         INT,
  height        INT,
  duration_ms   INT,
  bytes         BIGINT,
  sha256        CHAR(64),

  attempts      INT NOT NULL DEFAULT 0,            -- 切失败重试计数，超限置 FAILED
  fail_reason   TEXT,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  ready_at      TIMESTAMPTZ,

  -- 一个事件一种证据只留一行：worker 重启后重复排队会被这条挡住
  CONSTRAINT uq_evidence_event_kind UNIQUE (event_uid, kind)
);
-- worker 的扫描路径：只找到期的 PENDING（部分索引，READY 行不进索引）
CREATE INDEX IF NOT EXISTS ix_evidence_pending
  ON event_evidence(clip_start_at) WHERE status = 'PENDING';

-- ── 7. 审计（只追加） ──────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS audit_log (
  audit_id     BIGSERIAL PRIMARY KEY,
  at           TIMESTAMPTZ NOT NULL DEFAULT now(),
  actor_id     BIGINT,                             -- 系统动作可为 NULL
  actor_role   TEXT,
  ward_id      TEXT,
  action       TEXT NOT NULL,                      -- LOGIN/ACK/SNAPSHOT_VIEW/EXPORT/...
  object_type  TEXT NOT NULL,                      -- event/user/snapshot/session
  object_id    TEXT,
  result       TEXT NOT NULL CHECK (result IN ('OK','DENIED','FAIL')),
  client_ip    INET,
  detail       JSONB
);
CREATE INDEX IF NOT EXISTS ix_audit_at     ON audit_log(at DESC);
CREATE INDEX IF NOT EXISTS ix_audit_ward   ON audit_log(ward_id, at DESC);
CREATE INDEX IF NOT EXISTS ix_audit_actor  ON audit_log(actor_id, at DESC);
CREATE INDEX IF NOT EXISTS ix_audit_object ON audit_log(object_type, object_id);

-- ── 只追加：禁 UPDATE / DELETE ─────────────────────────────────────────────
-- 函数名沿用设计方案 §7.3；CREATE OR REPLACE 保证可重复执行。
CREATE OR REPLACE FUNCTION fg.deny_mutation() RETURNS trigger AS $$
BEGIN
  RAISE EXCEPTION 'audit_log 只允许追加，禁止 %', TG_OP;
END $$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_audit_no_update ON fg.audit_log;
CREATE TRIGGER trg_audit_no_update BEFORE UPDATE ON fg.audit_log
  FOR EACH ROW EXECUTE FUNCTION fg.deny_mutation();

DROP TRIGGER IF EXISTS trg_audit_no_delete ON fg.audit_log;
CREATE TRIGGER trg_audit_no_delete BEFORE DELETE ON fg.audit_log
  FOR EACH ROW EXECUTE FUNCTION fg.deny_mutation();

-- ── 事实视图：所有 KPI 的唯一出发点（§11.2） ──────────────────────────────
-- 只含 is_test = FALSE 的事件；测试告警不进任何统计。
CREATE OR REPLACE VIEW fg.v_event_fact AS
SELECT
  e.event_uid, e.ward_id, e.device_id, e.received_at,
  e.detection_state, e.disposition_state, e.severity,
  e.outcome, e.false_alarm_reason, e.assignee_id,
  e.acked_at, e.arrived_at, e.resolved_at,
  EXTRACT(EPOCH FROM (e.acked_at - e.received_at))    AS response_s,
  EXTRACT(EPOCH FROM (e.arrived_at - e.received_at))  AS arrive_s,
  -- ⚠️ is_confirmed 是「此刻仍坚持」，**不是**告警数口径，KPI-1 请用下方 ever_confirmed
  (e.detection_state = 'CONFIRMED')                    AS is_confirmed,
  -- ⚠️ 2026-11-04：从 disposition_state 改成 false_alarm_reason。
  --    归档（CLOSED）会保留误报原因，按状态判会把「已归档的误报」从分子里漏掉。
  --    对旧数据逐行等价（ck_event_fa 保证只有 FALSE_ALARM 才带 reason）。
  (e.false_alarm_reason IS NOT NULL)                   AS is_false_alarm,
  (e.disposition_state IN ('RESOLVED','FALSE_ALARM','CLOSED')) AS is_closed,
  (e.outcome IN ('ASSISTED','MEDICAL','HOSPITALIZED'))  AS is_confirmed_fall,
  -- 「曾经确认过」：历史事实，与「此刻是否还在跌」解耦（P9 二期 ①）。
  --   确认后又被模型撤回的真实告警，当时确实响过，仍算发生过一次告警；
  --   只有 orphan_retract（撤回先到、从未匹配到确认行）才不算。
  (e.detection_state = 'CONFIRMED'
   OR (e.detection_state = 'RETRACTED' AND NOT e.orphan_retract)) AS ever_confirmed,
  -- P12（2026-11-04）：区域级权限的判据列。
  -- ⚠️ 必须在**最后一列** —— CREATE OR REPLACE VIEW 只允许在末尾加列，
  --    以后再加列也照此办理，别插在中间（会直接报错）。
  e.zone_id
FROM fg.event e
WHERE e.is_test = FALSE;

-- ── 应用账号 fg_app：最小权限（§7.3） ─────────────────────────────────────
-- 角色已存在则跳过，保证脚本可重复执行。
-- ⚠️ 下面的密码只是占位，部署时必须立即改掉：
--    ALTER ROLE fg_app PASSWORD '<强密码>';
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = 'fg_app') THEN
    CREATE ROLE fg_app LOGIN PASSWORD 'CHANGE_ME_BEFORE_DEPLOY';
    RAISE NOTICE '已创建角色 fg_app，请立即执行 ALTER ROLE fg_app PASSWORD ...';
  END IF;
END
$$;

GRANT USAGE ON SCHEMA fg TO fg_app;

-- 业务表：可读、可写、可改；**不给 DELETE**（审计与事件都不许物理删除）
GRANT SELECT, INSERT, UPDATE
  ON fg.ward, fg.zone, fg.point, fg.app_user, fg.device, fg.event, fg.event_action,
     fg.snapshot, fg.event_evidence, fg.device_config
  TO fg_app;

-- ⚠️ 唯一的例外：app_user_zone 是**配置型关联表**（不是事实数据），
--    「取消某人的某个区域」在语义上就是删掉这条关联，所以给 DELETE。
--    不给 UPDATE：一条关联只有 (user_id, zone_id) 两个业务列，改等于删了重加。
GRANT SELECT, INSERT, DELETE ON fg.app_user_zone TO fg_app;

-- 配置历史只允许追加
GRANT SELECT, INSERT ON fg.device_config_rev TO fg_app;

-- 审计表：只允许追加（UPDATE/DELETE 连权限都没有，触发器再兜一层）
GRANT SELECT, INSERT ON fg.audit_log TO fg_app;

-- 序列（BIGSERIAL）：插入需要 nextval
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA fg TO fg_app;

-- 事实视图：KPI 查询入口
GRANT SELECT ON fg.v_event_fact TO fg_app;
