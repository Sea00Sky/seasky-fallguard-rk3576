-- P9 二期 ① 语义拆分：给事实视图补一列「曾经确认过」
-- 目的：把「历史事实（告警当时确实响过）」与「此刻状态（模型现在还坚持吗）」分开。
--   CONFIRMED                              -> 曾确认，且现在还坚持
--   RETRACTED 且 orphan_retract = FALSE    -> 曾确认，之后被模型撤回（真实告警，仍算告警）
--   RETRACTED 且 orphan_retract = TRUE     -> 从未确认过（孤立撤回，不算告警）
--
-- CREATE OR REPLACE VIEW 只允许在**末尾追加**列，所以 ever_confirmed 必须排在最后。
-- 幂等：可反复执行。

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
  (e.detection_state = 'CONFIRMED'
   OR (e.detection_state = 'RETRACTED' AND NOT e.orphan_retract)) AS ever_confirmed
FROM fg.event e
WHERE e.is_test = FALSE;

-- 视图权限：schema.sql 里 GRANT 过一次，重建视图不会丢 owner/ACL，
-- 但保留这一步以防视图被 DROP 过。
GRANT SELECT ON fg.v_event_fact TO fg_app;

-- ── 回滚（⚠️ 只能连代码一起回滚） ────────────────────────────────────────
-- CREATE OR REPLACE VIEW **删不掉列**，所以降级必须 DROP 后重建：
--   DROP VIEW fg.v_event_fact;   -- 改前已查 pg_depend：无依赖视图，安全
--   再执行 db/schema.sql 里 is_confirmed 那一版视图 + GRANT SELECT TO fg_app
-- 顺序：先回滚 app 代码（KPI 改回 is_confirmed），再 DROP VIEW。
-- 反过来会 500（代码在查一个不存在的列）。

