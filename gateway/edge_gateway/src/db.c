
#include "gateway.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <time.h>
#include <pthread.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sqlite3.h>

static sqlite3 *g_db = NULL;

/* 惰性重连：db_open 只在启动时执行一次，若当时数据目录不可用（如挂载未就绪）
 * 双检锁只保护「打开/指针置位」；连接操作本身由 FULLMUTEX 串行化。 */
static pthread_mutex_t g_db_open_lk = PTHREAD_MUTEX_INITIALIZER;
static long long g_last_try_ms = 0;

static const char *DB_SCHEMA =
    "CREATE TABLE IF NOT EXISTS events("
    " seq INTEGER PRIMARY KEY,"
    " ts_sec INTEGER, ts_ms INTEGER,"
    " person_id INTEGER, class_id INTEGER,"
    " confidence REAL,"
    " bbox_x1 REAL, bbox_y1 REAL, bbox_x2 REAL, bbox_y2 REAL,"
    " hybrid_score REAL,"
    " reason TEXT,"
    " snapshot_seq INTEGER,"
    " snap_file TEXT,"

    " state INTEGER DEFAULT 1,"            /* 1=confirmed 2=retracted*/
    " window_start INTEGER, window_end INTEGER,"
    " model_ver INTEGER,"
    " prob_fall REAL, coverage REAL,"
    " pushed INTEGER DEFAULT -1,"          /* 1=已推 0=推失败 -1=闸门未推*/
    " created_at TEXT DEFAULT (datetime('now','localtime'))"
    ");";

/* v3 幂等键索引 —— 部分索引 + 非唯一。
 *
 * ① 必须是【部分索引】`WHERE window_start > 0`：v2 遗留行与「上游没填窗口号」
 * 的事件 window_start 为 NULL/0，普通索引会把它们全部当成同一个键
 * ② 必须【非唯一】（原先是 UNIQUE）：`window_start` 是「进程启动以来的帧号」，
 * 感知进程重启后从 0 重新数 ⇒ 不同会话的两次真实跌倒可能落在同一个
 * (person, window, model) 上。唯一索引会把第二次插入直接拒掉，只能退回
 * 「更新旧行」⇒ 新告警顶掉旧卡 = 漏报（比多一条重复卡严重得多）。
 * 幂等现在由 `db_insert_event()` 显式判定（重放 / 撤回才更新），索引只加速查找。
 * 老库上那个 UNIQUE 索引必须先 DROP —— `CREATE INDEX IF NOT EXISTS` 不会改它。 */
static const char *DB_IDEM_INDEX =
    "CREATE INDEX IF NOT EXISTS idx_events_idem"
    " ON events(person_id, window_start, model_ver)"
    " WHERE window_start > 0;";

/* v2 库升级到 v3：`CREATE TABLE IF NOT EXISTS` 不会给老表加列，必须逐列 ALTER。
 * 重复执行时 sqlite 报 "duplicate column name" —— 那是「已迁过」，不是错误。 */
static void db_migrate_v3(void) {
    static const char *cols[] = {
        "ALTER TABLE events ADD COLUMN state INTEGER DEFAULT 1",
        "ALTER TABLE events ADD COLUMN window_start INTEGER",
        "ALTER TABLE events ADD COLUMN window_end INTEGER",
        "ALTER TABLE events ADD COLUMN model_ver INTEGER",
        "ALTER TABLE events ADD COLUMN prob_fall REAL",
        "ALTER TABLE events ADD COLUMN coverage REAL",
    };
    for (size_t i = 0; i < sizeof(cols) / sizeof(cols[0]); i++) {
        char *err = NULL;
        if (sqlite3_exec(g_db, cols[i], NULL, NULL, &err) != SQLITE_OK) {
            if (!err || !strstr(err, "duplicate column name"))
                log_status("db 迁移失败 [%s]: %s", cols[i], err ? err : "?");
        }
        sqlite3_free(err);
    }
    /* 先 DROP 再建：v3 早期版本建的是 UNIQUE 索引，`CREATE INDEX IF NOT EXISTS`
     * 见到同名索引会直接跳过 ⇒ 唯一约束还在，新会话重号事件仍会被拒。
     * 表只有千行级，每次启动重建的代价可以忽略。 */
    char *err = NULL;
    if (sqlite3_exec(g_db, "DROP INDEX IF EXISTS idx_events_idem;",
                     NULL, NULL, &err) != SQLITE_OK)
        log_status("db 幂等索引删除失败: %s", err ? err : "?");
    sqlite3_free(err);
    err = NULL;
    if (sqlite3_exec(g_db, DB_IDEM_INDEX, NULL, NULL, &err) != SQLITE_OK) {
        log_status("db 幂等索引创建失败: %s", err ? err : "?");
        sqlite3_free(err);
    }
}

int db_open(const char *data_dir) {
    /* 目录准备：data_dir 与 data_dir/snaps（两级足够，失败让 sqlite 报错兜底）*/
    if (mkdir(data_dir, 0755) != 0 && errno != EEXIST)
        log_status("mkdir %s 失败: %s", data_dir, strerror(errno));
    char snaps[300];
    snprintf(snaps, sizeof(snaps), "%s/snaps", data_dir);
    if (mkdir(snaps, 0755) != 0 && errno != EEXIST)
        log_status("mkdir %s 失败: %s", snaps, strerror(errno));

    char path[320];
    snprintf(path, sizeof(path), "%s/events.db", data_dir);

    int rc = sqlite3_open_v2(path, &g_db,
                             SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX,
                             NULL);
    if (rc != SQLITE_OK) {
        log_status("sqlite open %s 失败: %s", path, g_db ? sqlite3_errmsg(g_db) : "?");
        g_db = NULL;
        return -1;
    }
    char *err = NULL;
    if (sqlite3_exec(g_db, "PRAGMA journal_mode=TRUNCATE; PRAGMA synchronous=NORMAL;", NULL, NULL, &err)
        != SQLITE_OK) {
        log_status("sqlite pragma 失败: %s", err ? err : "?");
        sqlite3_free(err);
    }
    if (sqlite3_exec(g_db, DB_SCHEMA, NULL, NULL, &err) != SQLITE_OK) {
        log_status("sqlite 建表失败: %s", err ? err : "?");
        sqlite3_free(err);
        sqlite3_close(g_db);
        g_db = NULL;
        return -1;
    }
    db_migrate_v3();                 /* v2 老库补列 + 建幂等索引（幂等）*/
    log_status("db 已打开 %s", path);
    return 0;
}

void db_close(void) {
    if (g_db) {
        sqlite3_close(g_db);
        g_db = NULL;
    }
}

int db_ensure_open(const char *data_dir) {
    if (g_db)
        return 0;
    pthread_mutex_lock(&g_db_open_lk);
    if (g_db) {                       /* 别的线程刚抢到并打开成功*/
        pthread_mutex_unlock(&g_db_open_lk);
        return 0;
    }
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    long long now_ms = (long long)now.tv_sec * 1000LL + now.tv_nsec / 1000000LL;
    if (g_last_try_ms && now_ms - g_last_try_ms < 30000) {
        pthread_mutex_unlock(&g_db_open_lk);
        return -1;                    /* 30s 内已试过且失败，静默跳过*/
    }
    g_last_try_ms = now_ms;
    int rc = db_open(data_dir);       /* 成功置 g_db；失败打日志并保持 NULL*/
    pthread_mutex_unlock(&g_db_open_lk);
    return rc;
}

/* ---- 身份查找：这条事件在库里是不是已经有行了 ----
 *
 * ① `window_start > 0`（v3 新管线）：契约幂等键 = (person_id, window_start, model_ver)。
 * ② `window_start = 0`（v2 遗留 / 上游没填窗口号）：用 (person_id, ts_sec, ts_ms)
 * 这个自然键，只为兜住「网关重启重放事件环」；v2 老行的 window_start 是 NULL
 * 而不是 0，所以不能拿 window_start 去比。
 *
 * `ORDER BY seq DESC`：窗口号在感知进程重启后会从 0 重新数，同一个
 * (person, window, model) 可能既有上一轮的旧行、又有本轮新行；撤回要更新
 * 最新那一行（本轮确认的那条），不能随手拿最旧的。
 *
 * 返回已有行的 seq；没有返回 -1。 */
static long db_find_existing(const fg_event_t *e) {
    const char *sql;
    if (e->window_start > 0)
        sql = "SELECT seq FROM events WHERE person_id=?1 AND window_start=?2"
              " AND model_ver=?3 ORDER BY seq DESC LIMIT 1;";
    else
        sql = "SELECT seq FROM events WHERE person_id=?1 AND ts_sec=?2"
              " AND ts_ms=?3 ORDER BY seq DESC LIMIT 1;";
    sqlite3_stmt *st = NULL;
    if (sqlite3_prepare_v2(g_db, sql, -1, &st, NULL) != SQLITE_OK)
        return -1;
    sqlite3_bind_int(st, 1, e->person_id);
    if (e->window_start > 0) {
        sqlite3_bind_int64(st, 2, (sqlite3_int64)e->window_start);
        sqlite3_bind_int64(st, 3, (sqlite3_int64)e->model_ver);
    } else {
        sqlite3_bind_int64(st, 2, (sqlite3_int64)e->ts_sec);
        sqlite3_bind_int(st, 3, e->ts_ms);
    }
    long row_seq = -1;
    if (sqlite3_step(st) == SQLITE_ROW)
        row_seq = (long)sqlite3_column_int64(st, 0);
    sqlite3_finalize(st);
    return row_seq;
}

/* 更新已有行的状态类字段（confirmed → retracted 的第二次投递走这里）。
 *
 * 不动 seq / ts_sec / ts_ms / reason：云端的处置记录用
 * (seq, ts_sec, ts_ms) 三键匹配，改了它们会让「已确认」的卡片退回 pending。
 * 行身份保持首次那条。 */
static int db_update_existing(long row_seq, const fg_event_t *e,
                              const char *snap_rel) {
    static const char *SQL =
        "UPDATE events SET state=?1, confidence=?2, hybrid_score=?3,"
        " prob_fall=?4, coverage=?5, snapshot_seq=?6,"
        " snap_file=COALESCE(?7, snap_file)"
        " WHERE seq=?8;";
    sqlite3_stmt *st = NULL;
    if (sqlite3_prepare_v2(g_db, SQL, -1, &st, NULL) != SQLITE_OK)
        return -1;
    sqlite3_bind_int(st, 1, (int)e->state);
    sqlite3_bind_double(st, 2, (double)e->confidence);
    sqlite3_bind_double(st, 3, (double)e->hybrid_score);
    sqlite3_bind_double(st, 4, (double)e->prob_fall);
    sqlite3_bind_double(st, 5, (double)e->coverage);
    sqlite3_bind_int(st, 6, (int)e->snapshot_seq);
    sqlite3_bind_text(st, 7, snap_rel && snap_rel[0] ? snap_rel : NULL, -1, SQLITE_STATIC);
    sqlite3_bind_int64(st, 8, (sqlite3_int64)row_seq);
    int rc = sqlite3_step(st);
    sqlite3_finalize(st);
    return (rc == SQLITE_DONE) ? 0 : -1;
}

/* 用指定 seq 插一行新事件（不带任何 ON CONFLICT）。返回 sqlite3_step 的原始 rc。*/
static int db_insert_new(long seq, const fg_event_t *e, const char *snap_rel) {
    static const char *SQL =
        "INSERT INTO events"
        "(seq,ts_sec,ts_ms,person_id,class_id,confidence,"
        " bbox_x1,bbox_y1,bbox_x2,bbox_y2,hybrid_score,reason,snapshot_seq,snap_file,"
        " state,window_start,window_end,model_ver,prob_fall,coverage)"
        " VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16,?17,?18,?19,?20);";
    sqlite3_stmt *st = NULL;
    if (sqlite3_prepare_v2(g_db, SQL, -1, &st, NULL) != SQLITE_OK)
        return SQLITE_ERROR;
    sqlite3_bind_int64(st, 1, (sqlite3_int64)seq);
    sqlite3_bind_int64(st, 2, (sqlite3_int64)e->ts_sec);
    sqlite3_bind_int(st, 3, e->ts_ms);
    sqlite3_bind_int(st, 4, e->person_id);
    sqlite3_bind_int(st, 5, e->class_id);
    sqlite3_bind_double(st, 6, (double)e->confidence);
    sqlite3_bind_double(st, 7, (double)e->bbox[0]);
    sqlite3_bind_double(st, 8, (double)e->bbox[1]);
    sqlite3_bind_double(st, 9, (double)e->bbox[2]);
    sqlite3_bind_double(st, 10, (double)e->bbox[3]);
    sqlite3_bind_double(st, 11, (double)e->hybrid_score);
    sqlite3_bind_text(st, 12, e->reason, -1, SQLITE_STATIC);
    sqlite3_bind_int(st, 13, (int)e->snapshot_seq);
    sqlite3_bind_text(st, 14, snap_rel && snap_rel[0] ? snap_rel : NULL, -1, SQLITE_STATIC);
    sqlite3_bind_int(st, 15, (int)e->state);
    sqlite3_bind_int64(st, 16, (sqlite3_int64)e->window_start);
    sqlite3_bind_int64(st, 17, (sqlite3_int64)e->window_end);
    sqlite3_bind_int64(st, 18, (sqlite3_int64)e->model_ver);
    sqlite3_bind_double(st, 19, (double)e->prob_fall);
    sqlite3_bind_double(st, 20, (double)e->coverage);
    int rc = sqlite3_step(st);
    sqlite3_finalize(st);
    return rc;
}

static long db_next_seq(void) {
    sqlite3_stmt *st = NULL;
    long next = -1;
    if (sqlite3_prepare_v2(g_db, "SELECT COALESCE(MAX(seq),0)+1 FROM events;",
                           -1, &st, NULL) == SQLITE_OK) {
        if (sqlite3_step(st) == SQLITE_ROW)
            next = (long)sqlite3_column_int64(st, 0);
        sqlite3_finalize(st);
    }
    return next;
}

/* 落盘一条事件，返回这一行实际使用的 seq（失败 -1）。
 *
 * 流程：先按身份键找已有行 → 有则原地更新状态字段；没有则插入。
 * 插入时若 seq 主键被占，顺延为 `MAX(seq)+1` 再插。
 *
 * 不能再写成「无目标 `ON CONFLICT DO UPDATE`」：
 * shm 在 `/dev/shm`（tmpfs），整机重启后 last_evt_seq 归零 ⇒ 新事件的
 * seq 又从 1 开始；而库里早就有 seq=1..1267 的老行。无目标 UPSERT 会把
 * 主键相撞当成同一事件，把 2026-08 的老行原地改写 —— seq/ts/reason
 * 一个不动，只改 state/prob/coverage ⇒ count 和 max(seq) 完全没变、
 * 日志也没有「事件落盘失败」，表面上就像"根本没落盘"。
 * ⇒ 幂等只认身份键；seq 相撞一律视为「不同事件、序号重号」。
 *
 * 和推送体的 `seq`（撤回要更新同一张卡片，不是新建一张）。
 *
 * 「进程启动以来的帧号」，感知进程一重启就从 0 重新数 ⇒ 新一轮的 confirmed
 * 会撞上上一轮同 (person, window) 的旧行。若一律更新，新告警会顶掉旧卡，
 * 护士端看不到这次跌倒（漏报），比多一条重复卡严重得多。
 * ⇒ 命中后只有两种情况算「同一张卡」：
 * ① `e->seq == 行 seq` —— 网关重启重放事件环，同一条事件；
 * ② `e->state == retracted` —— 对同一窗口的更正（撤回必须灰掉原来那张卡）。
 * 其它情况（新 seq 的 confirmed）一律当新事件插入，seq 重号就顺延。 */
long db_insert_event(const fg_event_t *e, const char *snap_rel) {
    if (!g_db) return -1;

    long row_seq = db_find_existing(e);
    if (row_seq >= 0) {
        int is_replay = (row_seq == (long)e->seq);
        int is_retract = (e->state == FG_EVT_ST_RETRACTED);
        if (is_replay || is_retract)
            return (db_update_existing(row_seq, e, snap_rel) == 0) ? row_seq : -1;
        /* 新会话的窗口号重号：落到下面按新事件插入*/
    }

    long seq = (long)e->seq;
    for (int attempt = 0; attempt < 3; attempt++) {
        int rc = db_insert_new(seq, e, snap_rel);
        if (rc == SQLITE_DONE)
            return seq;
        if (rc != SQLITE_CONSTRAINT)
            return -1;
        /* 约束冲突只可能是 seq 主键重号（幂等索引已改成非唯一）。
         * 「新告警顶掉旧卡」；重号就顺延到 MAX(seq)+1。 */
        long next = db_next_seq();
        if (next <= 0)
            return -1;
        seq = next;
    }
    return -1;
}

/* 按 row_seq 更新 —— 传 db_insert_event() 的返回值，不是 evt->seq。
 * 幂等命中时两者不同：evt->seq 是这次撤回的新序号，row_seq 是首次告警的序号。 */
void db_mark_pushed(long row_seq, int pushed) {
    if (!g_db) return;
    static const char *SQL = "UPDATE events SET pushed=?1 WHERE seq=?2;";
    sqlite3_stmt *st = NULL;
    if (sqlite3_prepare_v2(g_db, SQL, -1, &st, NULL) != SQLITE_OK)
        return;
    sqlite3_bind_int(st, 1, pushed);
    sqlite3_bind_int64(st, 2, (sqlite3_int64)row_seq);
    sqlite3_step(st);
    sqlite3_finalize(st);
}

char *db_recent_json(int limit) {
    if (!g_db || limit <= 0 || limit > 512) limit = limit > 512 ? 512 : 20;
    static const char *SQL =
        "SELECT seq,ts_sec,ts_ms,person_id,class_id,confidence,"
        "bbox_x1,bbox_y1,bbox_x2,bbox_y2,hybrid_score,reason,snapshot_seq,"
        "COALESCE(snap_file,''),pushed,created_at,"
        "COALESCE(state,1),COALESCE(window_start,0),COALESCE(model_ver,0)"
        " FROM events ORDER BY seq DESC LIMIT ?1;";
    sqlite3_stmt *st = NULL;
    if (sqlite3_prepare_v2(g_db, SQL, -1, &st, NULL) != SQLITE_OK)
        return NULL;
    sqlite3_bind_int(st, 1, limit);

    size_t cap = 4096 + (size_t)limit * 384;
    char *buf = malloc(cap);
    if (!buf) { sqlite3_finalize(st); return NULL; }
    size_t len = 0;
    len += (size_t)snprintf(buf + len, cap - len, "[");

    int first = 1;
    while (sqlite3_step(st) == SQLITE_ROW) {
        char esc_reason[160];
        json_escape((const char *)sqlite3_column_text(st, 11), esc_reason, sizeof(esc_reason));
        int n = snprintf(buf + len, cap - len,
                         "%s{\"seq\":%lld,\"ts\":%lld.%03d,\"person_id\":%d,\"class_id\":%d,"
                         "\"confidence\":%.2f,\"bbox\":[%.1f,%.1f,%.1f,%.1f],"
                         "\"hybrid_score\":%.2f,\"reason\":\"%s\",\"snapshot_seq\":%d,"
                         "\"snap\":\"%s\",\"pushed\":%d,\"created_at\":\"%s\","
                         "\"state\":%d,\"window_start\":%lld,\"model_ver\":%lld}",
                         first ? "" : ",",
                         (long long)sqlite3_column_int64(st, 0),
                         (long long)sqlite3_column_int64(st, 1),
                         sqlite3_column_int(st, 2),
                         sqlite3_column_int(st, 3),
                         sqlite3_column_int(st, 4),
                         sqlite3_column_double(st, 5),
                         sqlite3_column_double(st, 6),
                         sqlite3_column_double(st, 7),
                         sqlite3_column_double(st, 8),
                         sqlite3_column_double(st, 9),
                         sqlite3_column_double(st, 10),
                         esc_reason,
                         sqlite3_column_int(st, 12),
                         (const char *)sqlite3_column_text(st, 13),
                         sqlite3_column_int(st, 14),
                         (const char *)sqlite3_column_text(st, 15),
                         sqlite3_column_int(st, 16),
                         (long long)sqlite3_column_int64(st, 17),
                         (long long)sqlite3_column_int64(st, 18));
        if (n < 0 || (size_t)n >= cap - len) break;   /* 缓冲不足则截断*/
        len += (size_t)n;
        first = 0;
    }
    sqlite3_finalize(st);
    snprintf(buf + len, cap - len, "]");
    return buf;
}
