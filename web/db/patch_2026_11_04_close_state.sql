-- ============================================================================
-- 修复：`CLOSE`（归档）在每条可达路径上都 500（2026-11-04）
--
-- 执行：
--   psql -U postgres -d fallguard -v ON_ERROR_STOP=1 \
--        -f db/patch_2026_11_04_close_state.sql
--
-- 症状
-- ----
--   RESOLVED     → CLOSE  ⇒  CheckViolation: ck_event_outcome  ⇒ HTTP 500
--   FALSE_ALARM  → CLOSE  ⇒  CheckViolation: ck_event_fa       ⇒ HTTP 500
--   （两条都在 P12 灰度演练里实测过；现网 0 条 CLOSED 就是这个原因）
--
-- 根因
-- ----
--   两条约束都是**双向等价**：
--     ck_event_outcome = ((disposition_state = 'RESOLVED')    = (outcome IS NOT NULL))
--     ck_event_fa      = ((disposition_state = 'FALSE_ALARM') = (false_alarm_reason IS NOT NULL))
--   而 `_apply_update` 的 CLOSE 只写 disposition_state，不动那两列
--   （web/app/routes/events.py:818 一带）。于是「归档」把状态改成 CLOSED 的瞬间，
--   等式左边变假、右边还是真 ⇒ 约束必炸。
--
--   注意这不是「随便放宽」：代码里 CLOSE 的设计意图写得明明白白 ——
--   metrics.disposition_counts 的 docstring：「CLOSE 只能从 RESOLVED/FALSE_ALARM
--   进入并**保留** resolved_at」。**约束和设计打了架**，错的是约束。
--
-- 改法：等价 → 蕴含
-- ------------------
--   outcome 有值          ⇒ 状态必须是 RESOLVED 或 CLOSED（别的地方不许带处置结果）
--   状态是 RESOLVED       ⇒ outcome 必须有值
--   ⇒ CLOSED 可以带 outcome，也可以不带（取决于它是从 RESOLVED 还是 FALSE_ALARM 来的）
--   false_alarm_reason 同理。
--
--   真值表（改后）：
--     PENDING/ACKED/ON_SCENE + outcome=NULL      ✓（和以前一样）
--     RESOLVED + outcome≠NULL                    ✓（和以前一样）
--     FALSE_ALARM + reason≠NULL                  ✓（和以前一样）
--     CLOSED + outcome≠NULL, reason=NULL         ✓（新：从 RESOLVED 归档）
--     CLOSED + outcome=NULL, reason≠NULL         ✓（新：从 FALSE_ALARM 归档）
--     CLOSED + 两者都 NULL                       ✓（新：手工归档）
--     PENDING + outcome≠NULL                     ✗（和以前一样被挡）
--
-- ⚠️ 顺带修 `v_event_fact.is_false_alarm`：原来是 `disposition_state =
--    'FALSE_ALARM'`，归档后（CLOSED + reason≠NULL）就不算了，误报率分子会凭空少一条。
--    改成 `false_alarm_reason IS NOT NULL` —— 对旧数据**逐行等价**（约束保证
--    只有 FALSE_ALARM 才带 reason），对新增的 CLOSED 才正确。
--    `is_confirmed_fall` 不用改：它本来就只看 outcome。
--
-- 可重复执行：DROP CONSTRAINT IF EXISTS + ADD；视图 CREATE OR REPLACE。
-- 不含任何数据变更（现网 0 条 CLOSED，新旧约束对所有存量行等价）。
-- ============================================================================

BEGIN;
SET search_path = fg, public;
SET client_encoding = 'UTF8';

-- ── 1. 处置结果：等价 → 蕴含 ──────────────────────────────────────────────
ALTER TABLE event DROP CONSTRAINT IF EXISTS ck_event_outcome;
ALTER TABLE event ADD  CONSTRAINT ck_event_outcome CHECK (
    -- 带处置结果 ⇒ 必须是“已处置”或“已归档”
    (disposition_state IN ('RESOLVED', 'CLOSED') OR outcome IS NULL)
    -- 已处置 ⇒ 必须有处置结果（这一半不能松，否则 RESOLVED 能空着 outcome）
AND (disposition_state <> 'RESOLVED' OR outcome IS NOT NULL)
);

-- ── 2. 误报原因：等价 → 蕴含 ──────────────────────────────────────────────
ALTER TABLE event DROP CONSTRAINT IF EXISTS ck_event_fa;
ALTER TABLE event ADD  CONSTRAINT ck_event_fa CHECK (
    (disposition_state IN ('FALSE_ALARM', 'CLOSED') OR false_alarm_reason IS NULL)
AND (disposition_state <> 'FALSE_ALARM' OR false_alarm_reason IS NOT NULL)
);

-- ── 3. 事实视图：误报判据跟着改（见文件头 ⚠️） ────────────────────────────
-- ⚠️ CREATE OR REPLACE VIEW 只允许**沿用原来的列顺序与类型**，这里只改表达式，
--    不动列 → 安全。以后要加列只能加在最后（P12 的 zone_id 就是加在末列）。
CREATE OR REPLACE VIEW v_event_fact AS
SELECT
  e.event_uid, e.ward_id, e.device_id, e.received_at,
  e.detection_state, e.disposition_state, e.severity,
  e.outcome, e.false_alarm_reason, e.assignee_id,
  e.acked_at, e.arrived_at, e.resolved_at,
  EXTRACT(EPOCH FROM (e.acked_at - e.received_at))    AS response_s,
  EXTRACT(EPOCH FROM (e.arrived_at - e.received_at))  AS arrive_s,
  (e.detection_state = 'CONFIRMED')                    AS is_confirmed,
  -- ⚠️ 从 disposition_state 改成 false_alarm_reason：归档后状态变 CLOSED，
  --    按状态判会把「已归档的误报」漏掉（误报率分子凭空少一条）。
  --    对旧数据逐行等价（约束保证只有 FALSE_ALARM 才带 reason）。
  (e.false_alarm_reason IS NOT NULL)                   AS is_false_alarm,
  (e.disposition_state IN ('RESOLVED','FALSE_ALARM','CLOSED')) AS is_closed,
  (e.outcome IN ('ASSISTED','MEDICAL','HOSPITALIZED'))  AS is_confirmed_fall,
  (e.detection_state = 'CONFIRMED'
   OR (e.detection_state = 'RETRACTED' AND NOT e.orphan_retract)) AS ever_confirmed,
  e.zone_id
FROM fg.event e
WHERE e.is_test = FALSE;

COMMIT;

-- ── 跑完自检（应输出两行，且都有 CLOSED） ────────────────────────────────
\echo ''
\echo '── 约束现在的样子 ──────────────────────────────────────'
SELECT conname, pg_get_constraintdef(oid) AS def
  FROM pg_constraint
 WHERE conrelid = 'fg.event'::regclass
   AND conname IN ('ck_event_outcome', 'ck_event_fa')
 ORDER BY conname;

\echo ''
\echo '── 存量数据分布（CLOSED 应为 0，其余不变）────────────────'
SELECT disposition_state, count(*)
  FROM fg.event GROUP BY 1 ORDER BY 1;
