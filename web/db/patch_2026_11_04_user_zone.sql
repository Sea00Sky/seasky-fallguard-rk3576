-- ============================================================================
-- P12 · 区域级数据权限（2026-11-04）
--
-- 执行：
--   psql -U postgres -d fallguard -v ON_ERROR_STOP=1 -f db/patch_2026_11_04_user_zone.sql
--
-- 目标：让 nurse 的数据范围从「病区」收窄到「病区 ∩ 自己负责的区域」。
--   病区（fg.ward）＝ 老轴，来自 app_user.ward_id，**永远在**；
--   区域（fg.zone）＝ 新轴，来自本次新建的 app_user_zone。
--   两级是**叠加**：WHERE e.ward_id = %s AND e.zone_id = ANY(%s)。
--   护士长/管理员不参与新轴（看全病区），行为与改动前逐字相同。
--
-- ⚠️ **必须先跑本补丁，再部署代码**。新代码的 current_user 会去 JOIN
--    app_user_zone；补丁没跑就是**每一个已登录页面 500**（/healthz 的 schema
--    探针会先暴露）。与 patch_2026_11_02_point_stream.sql 是同一类坑。
--
-- ⚠️ 本补丁**只动数据与结构，不改行为**。存量护士会被绑上本病区全部启用区域
--    （见第 5 节），所以上线瞬间的范围与今天完全一致，护士长再按实际情况收窄。
--    「未分配＝什么都看不到」（默认拒绝）从第一个**新建**护士开始生效。
--
-- 可重复执行：CREATE TABLE IF NOT EXISTS / 索引 IF NOT EXISTS /
--   回填带 IS NULL 前置条件 / 灰度绑定带「表原本为空」护栏。
-- ============================================================================

BEGIN;
SET search_path = fg, public;
SET client_encoding = 'UTF8';

-- ── 1. 关联表 ──────────────────────────────────────────────────────────────
-- 多对多：一个护士可负责 N 个区域（一个区域也可有多名护士值班）。
-- 为什么不做成 app_user 上的一列：一个护士要看「东走廊 + 409病房」就得开两个账号。
CREATE TABLE IF NOT EXISTS app_user_zone (
  user_id    BIGINT NOT NULL REFERENCES app_user(user_id) ON DELETE CASCADE,
  zone_id    UUID   NOT NULL REFERENCES zone(zone_id)     ON DELETE CASCADE,
  created_by BIGINT REFERENCES app_user(user_id),
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (user_id, zone_id)
);

COMMENT ON TABLE app_user_zone IS
  '护士 ↔ 负责区域（P12）。只有 role=nurse 参与；护士长/管理员不受此表限制。';

-- 「区域值守」页要反查「这个区域现在谁负责」——PK 是 (user_id, zone_id)，
-- 按 zone_id 单查走不到索引，补一条。
CREATE INDEX IF NOT EXISTS ix_auz_zone ON app_user_zone(zone_id);

-- ── 2. 病区护栏（触发器）────────────────────────────────────────────────────
-- **区域绝不能用来跨病区**，否则 4F-A 的护士可以借 4F-B 的一个区域越界。
-- 这个不变式跨两张表（app_user.ward_id 与 zone.ward_id），DDL 的 CHECK 表达不了。
--   · 应用层：绑定接口写前校验（P4）；
--   · 查询层：current_user 的 JOIN 带 AND z.ward_id = u.ward_id（P1，纵深防御）；
--   · **库层：就是这个触发器** —— 前两层都可能被绕过（手工 SQL、新写的代码路径），
--     这一层不会。
-- 用触发器而不是「复合外键 + 冗余 ward_id 列」：后者要给 app_user 和 zone 各加
-- 一条 UNIQUE(user_id, ward_id) / UNIQUE(zone_id, ward_id)，为了一个不变式改动
-- 两张已上线的表，代价不成比例。本库已有触发器先例（audit_log 的禁改禁删）。
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

-- ── 3. 【前提 A】device.zone_id 与 point.zone_id 对账 ────────────────────────
-- ⚠️⚠️ 这是本次改动**最高优先**的一条，且它修的是一个**已存在的 bug**：
--
--   ingest 给事件打的 zone_id 快照取自**设备**（dev["zone_id"]，见 ingest.py:449/598），
--   而管理员改点位区域时，patch_point 只回写设备的 device_name/location，
--   **不回写 zone_id**（admin.py:1296-1304）。于是把点位从「东走廊」挪到「409病房」
--   之后，之后所有新事件仍被打上「东走廊」。
--
--   在病区级权限下这只是显示错。一旦区域变成授权单位，它就是**安全 bug**：
--   该看到的人看不到，不该看到的人反而看得到。
--
-- 应用层的同步（patch_point 里加 zone_id）在代码侧修；这里修已经产生的历史错配。
UPDATE fg.device d
   SET zone_id = p.zone_id
  FROM fg.point p
 WHERE d.point_id = p.point_id
   AND p.zone_id IS NOT NULL                              -- 点位没归区就不动设备（别抹掉已有值）
   AND d.zone_id IS DISTINCT FROM p.zone_id;

-- ── 4. 【决策 5】老事件回填 zone_id ─────────────────────────────────────────
-- 2026-11-01 加区域功能之前的事件完全没有 zone_id 快照。回填口径**刻意与 ingest
-- 逐字相同**（就是 device.zone_id），所以回填出来的值和「当时如果就有区域功能」
-- 会打的值是一致的。这也是为什么第 3 节必须先于本节。
--
-- 回填不到的（设备本身也没归区）保持 NULL。对护士天然不可见
-- （`NULL = ANY(...)` 结果为 NULL ⇒ 不匹配），对护士长天然可见 —— 正是决策 5 要的。
UPDATE fg.event e
   SET zone_id = d.zone_id
  FROM fg.device d
 WHERE e.device_id = d.device_id
   AND e.zone_id IS NULL
   AND d.zone_id IS NOT NULL;

-- ── 5. 灰度：存量护士绑上本病区全部启用区域 ─────────────────────────────────
-- ⚠️⚠️ **护栏：只在表原本为空时执行**。
--    没有这个护栏，补丁**第二次跑**会把护士长已经收窄掉的区域又全部塞回去
--    —— 回填类补丁不可重跑最危险的形态不是报错，是**静默把权限放大**。
--    （同 patch_2026_11_02 第 4 节的①号护栏。）
--
-- 效果：上线瞬间每个护士看到的还是全病区，与今天逐字相同。
--   护士长随后用 /head/staff 逐个收窄。
--   若希望「上线即默认拒绝」，把本节的 INSERT 连同 IF 一起删掉即可 ——
--   但那样存量护士当班就什么都看不到，必须先分配完再切。
DO $$
DECLARE
  n_users INT;
  n_links INT;
BEGIN
  IF NOT EXISTS (SELECT 1 FROM fg.app_user_zone) THEN
    SELECT count(*) INTO n_users
      FROM fg.app_user WHERE role = 'nurse' AND is_active;

    INSERT INTO fg.app_user_zone (user_id, zone_id, created_by)
    SELECT u.user_id, z.zone_id, NULL
      FROM fg.app_user u
      JOIN fg.zone z ON z.ward_id = u.ward_id AND z.is_active
     WHERE u.role = 'nurse' AND u.is_active
    ON CONFLICT (user_id, zone_id) DO NOTHING;

    GET DIAGNOSTICS n_links = ROW_COUNT;
    RAISE NOTICE 'P12 灰度：% 个启用护士 × 本病区启用区域 ⇒ 写入 % 条关联', n_users, n_links;
  ELSE
    RAISE NOTICE 'P12 灰度：app_user_zone 非空，跳过（补丁重跑不会放大权限）';
  END IF;
END
$$;

-- ── 6. 索引 ────────────────────────────────────────────────────────────────
-- 第 3/4 节的 UPDATE 与设备台账都要按 point_id 找设备；现在只有 ix_device_ward。
CREATE INDEX IF NOT EXISTS ix_device_point ON fg.device(point_id);

-- ix_event_zone 在 schema.sql:247 已存在，本节不重复建。

-- ── 7. 授权 ────────────────────────────────────────────────────────────────
-- ⚠️ **这是全库第一张需要 DELETE 的业务表**。现有 GRANT 一律是
--    「SELECT, INSERT, UPDATE，不给 DELETE」（schema.sql:389 的注释：审计与事件
--    都不许物理删除）。app_user_zone 是**配置型关联表**，不是事实数据：
--    护士长「取消某人的某个区域」在语义上就是删掉这条关联，用软删（加 is_active 列）
--    反而会出现「取消过的区域又出现在下拉里」的歧义。
--    所以单独给 DELETE，并把这条例外写在这里。
GRANT SELECT, INSERT, DELETE ON fg.app_user_zone TO fg_app;
-- UPDATE 不给：一条关联只有 (user_id, zone_id) 两个业务列，改就等于删了重加。

-- ── 8. 事实视图补 zone_id ──────────────────────────────────────────────────
-- 计划里漏了这一条：`kpi()` 的五个 KPI **全部**从 fg.v_event_fact 出发，
-- 而这个视图根本没暴露 zone_id ⇒ 区域条件无处可加。main.py 里还有 4 处也用它
-- （侧边栏红点 _pending_count、护士历史页汇总、/healthz 今日计数）。
--
-- 做法：**在列末尾追加** e.zone_id。
--   · PostgreSQL 的 CREATE OR REPLACE VIEW 只允许在**末尾**加列，
--     改名/删列/调序都会报错 —— 所以严格照抄原定义，只把 zone_id 放最后。
--   · 现有 20 列的名字与顺序逐字不变，所有按列名取值的调用方零影响。
--     （已确认全项目没有 `SELECT * FROM fg.v_event_fact`，也没有别的视图依赖它。）
--   · CREATE OR REPLACE 不会动属主和授权，第 7 节之外不用重发 GRANT。
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
  (e.disposition_state = 'FALSE_ALARM')                AS is_false_alarm,
  (e.disposition_state IN ('RESOLVED','FALSE_ALARM','CLOSED')) AS is_closed,
  (e.outcome IN ('ASSISTED','MEDICAL','HOSPITALIZED'))  AS is_confirmed_fall,
  -- 「曾经确认过」：历史事实，与「此刻是否还在跌」解耦（P9 二期 ①）。
  (e.detection_state = 'CONFIRMED'
   OR (e.detection_state = 'RETRACTED' AND NOT e.orphan_retract)) AS ever_confirmed,
  e.zone_id                                            -- ← P12 新增（必须在末尾）
FROM fg.event e
WHERE e.is_test = FALSE;

COMMIT;

-- ============================================================================
-- 跑完自检（手工执行，确认数字）：
--
--   -- 每个护士现在负责几个区域
--   SELECT u.login_name, u.ward_id, count(uz.zone_id) AS zones
--     FROM fg.app_user u
--     LEFT JOIN fg.app_user_zone uz ON uz.user_id = u.user_id
--    WHERE u.role = 'nurse' AND u.is_active
--    GROUP BY 1,2 ORDER BY 1;
--
--   -- 还没归区域的点位（这些人负责的区域里不存在它们 ⇒ 护士看不到）
--   SELECT count(*) FROM fg.point WHERE zone_id IS NULL AND is_active;
--
--   -- 回填后仍然没有区域的历史事件（对护士不可见、对护士长可见）
--   SELECT count(*) FROM fg.event WHERE zone_id IS NULL;
--
--   -- 病区护栏自检：下面这条**必须报错**
--   INSERT INTO fg.app_user_zone (user_id, zone_id)
--   SELECT u.user_id, z.zone_id FROM fg.app_user u, fg.zone z
--    WHERE u.ward_id <> z.ward_id LIMIT 1;
-- ============================================================================
