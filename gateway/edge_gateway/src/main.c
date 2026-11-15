/* main.c — FallGuard 边缘网关：生命周期 / 信号处理 / 5 线程编排 / 推送与快照工具
 *
 * 事件推送：SQLite 落盘 + 告警快照存档 + 闸门过滤 + HTTP POST 护士站
 * HTTP 服务：每连接一线程（MJPEG 直播长连接不阻塞其他请求）
 * /api/status /api/events /api/snap /api/live.mjpg
 */
#include "gateway.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <errno.h>
#include <unistd.h>
#include <time.h>
#include <signal.h>
#include <poll.h>
#include <netdb.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <syslog.h>
#include <sys/un.h>     /* sd_notify：struct sockaddr_un*/
#include <stddef.h>     /* offsetof*/

volatile sig_atomic_t g_stop = 0;

char g_push_url[512] = "";
char g_data_dir[256] = "/mnt/sdcard/fallguard_data";

#define MAX_CONN         32
/* 栈大小不能按"256 KB 够用"拍脑袋定：
 * `handle_snap()` 在栈上放一个 `fg_frame_t`，实测 sizeof = 204816 B
 * （200 KB JPEG 缓冲）。256 KB 只剩 52 KB 余量，再加上 `char path[320]`、
 * 4 KB 页对齐与 red zone ⇒ 随时可能栈溢出。
 * 取 512 KB：相对默认 8 MB 省 16 倍，同时留 ~300 KB 余量。
 * （v3 删掉 `handle_live_mjpg` 后栈上仍只剩 `handle_snap` 这一处大对象，
 * 所以 512 KB 保持不变 —— 别因为少了一条路径就调小。） */
#define CONN_STACK_SIZE  (512 * 1024)
static pthread_mutex_t g_conn_mtx = PTHREAD_MUTEX_INITIALIZER;
static int g_conn_live = 0;             /* 当前活跃连接线程数*/
static unsigned long g_conn_rejected = 0;  /* 因超限被拒的累计数（metric）*/

static void on_signal(int sig) {
    (void)sig;
    g_stop = 1;
}

/* ================= systemd sd_notify（不依赖 systemd-dev） =================
 *
 * 板上没有 `sd-daemon.h`（`systemd-dev` 未装，且 apt 源 404、安装会连带
 * 升级 12 个 systemd 相关包 —— 风险不可接受）。
 * 但 `sd_notify` 的协议本身就只是「往 $NOTIFY_SOCKET 这个 unix dgram
 * socket 发一行 `KEY=VALUE\n`」⇒ 30 行自己实现即可，零依赖。
 *
 * 没有它就不能开 `WatchdogSec=` —— 开了但从不 `WATCHDOG=1`，
 * systemd 会在超时后杀掉网关，变成新的崩溃循环。 */

static int g_notify_fd = -1;
static struct sockaddr_un g_notify_addr;
static socklen_t g_notify_len = 0;

static void sd_notify_init(void) {
    const char *path = getenv("NOTIFY_SOCKET");
    if (!path || !path[0])
        return;                       /* 不是 systemd 起的（手工跑）⇒ 静默跳过*/
    /* 抽象命名空间：首字节是 '@'（systemd 约定），实际存的是 '\0'*/
    size_t n = strlen(path);
    if (n >= sizeof(g_notify_addr.sun_path))
        return;
    memset(&g_notify_addr, 0, sizeof(g_notify_addr));
    g_notify_addr.sun_family = AF_UNIX;
    memcpy(g_notify_addr.sun_path, path, n);
    if (path[0] == '@')
        g_notify_addr.sun_path[0] = '\0';
    g_notify_len = (socklen_t)(offsetof(struct sockaddr_un, sun_path) + n);
    g_notify_fd = socket(AF_UNIX, SOCK_DGRAM | SOCK_CLOEXEC, 0);
}

static void sd_notify_send(const char *state) {
    if (g_notify_fd < 0)
        return;
    /* 失败不重试、不记日志 —— 通知是 best-effort，不能反过来拖累主流程*/
    sendto(g_notify_fd, state, strlen(state), MSG_NOSIGNAL,
           (struct sockaddr *)&g_notify_addr, g_notify_len);
}

/* ================= metrics（JSON lines 到 journald） =================
 *
 * 必须限速：板上已经发生过「一次故障打满 259 MB journald」。 */

#define METRICS_INTERVAL_S 10
static unsigned long g_evt_seen = 0;        /* 累计从 shm 取到的事件数*/
static unsigned long g_push_ok = 0;
static unsigned long g_push_fail = 0;
static unsigned long g_push_gated = 0;      /* 被 enabled/state/cooldown/url 挡掉*/
static unsigned long g_metrics_lines = 0;

static void metrics_line(void) {
    unsigned long now_evt = g_evt_seen;
    char body[512];
    snprintf(body, sizeof(body),
             "{\"t\":%lld,\"evt_seen\":%lu,\"push_ok\":%lu,\"push_fail\":%lu,"
             "\"push_gated\":%lu,\"queue_depth\":%d,\"queue_dropped\":%lu,"
             "\"conn_live\":%d,\"conn_rejected\":%lu,"
             "\"ring_count\":%u,\"sensor_alive\":%d}",
             (long long)time(NULL), now_evt, g_push_ok, g_push_fail,
             g_push_gated, eq_depth(), eq_dropped_total(),
             g_conn_live, g_conn_rejected,
             g_shm ? g_shm->evt_count : 0,
             (g_shm && g_shm->sensor_pid > 0 && kill(g_shm->sensor_pid, 0) == 0) ? 1 : 0);
    /* 走 syslog(LOG_INFO) ⇒ journald 收；`journalctl -u fallguard-gateway | grep METRICS`*/
    syslog(LOG_DAEMON | LOG_INFO, "METRICS %s", body);
    g_metrics_lines++;
}

/* ================= 工具：JSON 转义 / 快照存档 / HTTP POST =================*/

void json_escape(const char *in, char *out, size_t n) {
    size_t o = 0;
    for (const char *p = in; p && *p && o + 6 < n; p++) {
        unsigned char c = (unsigned char)*p;
        if (c == '"' || c == '\\') {
            out[o++] = '\\';
            out[o++] = (char)c;
        } else if (c < 0x20) {
            o += (size_t)snprintf(out + o, n - o, "\\u%04x", c);
        } else {
            out[o++] = (char)c;   /* UTF-8 原样通过*/
        }
    }
    out[o] = '\0';
}

/* 存最新告警快照到 <data_dir>/snaps/snap_<frame_seq>.jpg（best-effort：
 * 若事件处理时已被更新的快照覆盖，则存当前帧，文件名仍用事件挂的 seq） */
int save_alarm_snap(uint32_t want_seq, char *rel_out, size_t n) {
    rel_out[0] = '\0';
    fg_frame_t fr;
    if (!g_data_dir[0] || fg_pick_latest_frame(1, &fr) != 0)
        return -1;
    char path[320];
    snprintf(path, sizeof(path), "%s/snaps/snap_%u.jpg", g_data_dir,
             want_seq ? want_seq : fr.frame_seq);
    FILE *fp = fopen(path, "wb");
    if (!fp) {
        log_status("快照写盘失败 %s: %s", path, strerror(errno));
        return -1;
    }
    fwrite(fr.jpeg, 1, fr.jpeg_len, fp);
    fclose(fp);
    snprintf(rel_out, n, "snaps/snap_%u.jpg", want_seq ? want_seq : fr.frame_seq);
    return 0;
}

/* 极简 HTTP POST JSON。非阻塞 connect + poll 超时；收发各设 SO_*TIMEO。
 * 返回 0=收到 2xx；-1=任何失败（调用方只记日志不重试）。 */
int http_post_json(const char *url, const char *body, int timeout_ms) {
    /* 只支持 http://host[:port]/path*/
    if (strncmp(url, "http://", 7) != 0)
        return -1;
    char host[256] = "", port[8] = "80", path[256] = "/";
    const char *p = url + 7, *slash = strchr(p, '/');
    const char *colon = strchr(p, ':');
    if (colon && (!slash || colon < slash)) {
        size_t hn = (size_t)(colon - p);
        if (hn >= sizeof(host)) return -1;
        memcpy(host, p, hn);
        size_t pn = (slash ? (size_t)(slash - colon) : strlen(colon));
        if (pn <= 1 || pn >= sizeof(port)) return -1;
        memcpy(port, colon + 1, pn - 1);
        port[pn - 1] = '\0';
        snprintf(path, sizeof(path), "%s", slash ? slash : "/");
    } else {
        size_t hn = slash ? (size_t)(slash - p) : strlen(p);
        if (hn == 0 || hn >= sizeof(host)) return -1;
        memcpy(host, p, hn);
        snprintf(path, sizeof(path), "%s", slash ? slash : "/");
    }

    struct addrinfo hints, *res = NULL;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(host, port, &hints, &res) != 0 || !res)
        return -1;

    int fd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (fd < 0) { freeaddrinfo(res); return -1; }

    /* 非阻塞 connect + poll 超时（护士站不可达时最多等 timeout_ms）*/
    int flags = fcntl(fd, F_GETFL, 0);
    fcntl(fd, F_SETFL, flags | O_NONBLOCK);
    int rc = connect(fd, res->ai_addr, res->ai_addrlen);
    freeaddrinfo(res);
    if (rc != 0 && errno == EINPROGRESS) {
        struct pollfd pf = { .fd = fd, .events = POLLOUT };
        if (poll(&pf, 1, timeout_ms) != 1) {
            close(fd);
            return -1;
        }
        int soerr = 0;
        socklen_t sl = sizeof(soerr);
        getsockopt(fd, SOL_SOCKET, SO_ERROR, &soerr, &sl);
        if (soerr != 0) {
            close(fd);
            return -1;
        }
    } else if (rc != 0) {
        close(fd);
        return -1;
    }
    fcntl(fd, F_SETFL, flags);   /* 恢复阻塞模式，靠 SO_*TIMEO 兜底*/

    struct timeval tv = { timeout_ms / 1000, (timeout_ms % 1000) * 1000 };
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    char hdr[512];
    int hn = snprintf(hdr, sizeof(hdr),
                      "POST %s HTTP/1.1\r\nHost: %s:%s\r\n"
                      "Content-Type: application/json\r\n"
                      "X-FallGuard-Gateway: 1\r\n"
                      "Content-Length: %zu\r\nConnection: close\r\n\r\n",
                      path, host, port, strlen(body));
    size_t total = 0;
    while (total < (size_t)hn) {
        ssize_t w = write(fd, hdr + total, (size_t)hn - total);
        if (w <= 0) { close(fd); return -1; }
        total += (size_t)w;
    }
    total = 0;
    size_t blen = strlen(body);
    while (total < blen) {
        ssize_t w = write(fd, body + total, blen - total);
        if (w <= 0) { close(fd); return -1; }
        total += (size_t)w;
    }
    /* 读状态行即可判定*/
    char resp[128] = "";
    ssize_t r = read(fd, resp, sizeof(resp) - 1);
    close(fd);
    if (r <= 0)
        return -1;
    resp[r] = '\0';
    return (strncmp(resp, "HTTP/1.", 7) == 0 &&
            strstr(resp, " 2")) ? 0 : -1;   /* "200"/"201"/...*/
}

void *t1_shm_listener(void *arg) {
    (void)arg;
    log_status("T1 shm_listener 启动");
    while (!g_stop) {
        struct timespec ts;
        /* ：`CLOCK_REALTIME` → `CLOCK_MONOTONIC`。
         * `sem_clockwait` 需要 `_GNU_SOURCE`（Makefile 已加）。 */
        if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0)
            break;
        ts.tv_nsec += 500000000L;                    /* 500ms 超时兜底*/
        if (ts.tv_nsec >= 1000000000L) {
            ts.tv_sec++;
            ts.tv_nsec -= 1000000000L;
        }
        if (sem_clockwait(g_sem_evt, CLOCK_MONOTONIC, &ts) != 0)
            continue;
        fg_event_t evt;
        while (!g_stop && fg_take_event(&evt) == 0) {  /* DRAN：唤醒≠一条*/
            g_evt_seen++;
            eq_push(&evt);
        }
    }
    log_status("T1 退出");
    return NULL;
}

/* row_seq = db_insert_event() 的返回值（幂等命中时是首次那条的 seq）。
 * 推送体的 `seq` 与 db_mark_pushed 都用它 —— 撤回要更新同一张卡片，不是新建一张。 */
static void push_one_event(const fg_event_t *evt, long row_seq) {
    const fg_shm_t *s = g_shm;

    static long long last_push_ms = 0;
    struct timespec now;
    /* ：`CLOCK_REALTIME` → `CLOCK_MONOTONIC`。
     * 冷却只算"两次推送之间隔了多久"这个时间差，用 REALTIME 时
     * NTP 跳变会让 `now_ms - last_push_ms` 变成负数（→ 30 秒冷却变 30 年，
     * 再也不推送）或突然巨大（→ 冷却形同失效）。这是真 bug，不是洁癖。
     * 事件里上报给护士端的 `ts_sec` 仍是 epoch 秒（契约要求），不受影响。 */
    clock_gettime(CLOCK_MONOTONIC, &now);
    long long now_ms = (long long)now.tv_sec * 1000LL + now.tv_nsec / 1000000LL;

    /* ---- 推送闸门：enabled → cooldown → url ----
     * 删除分数闸门 `hybrid_score < push_threshold`。
     * hybrid_score 与上游 decision 口径不一致，真机 5 次复现
     * 「上游已判 FALL、闸门按分数拒」⇒ 真摔 0 条送达。
     * 上游 `shm_adapter.py` 的 `GATE_REASON_WHITELIST`（= `_fusion_decision()`
     * 全部 7 个 FALL 分支）已经做过一次决策 ⇒ 决策只做一次，此处不再复查分数。
     * 不许改回「设阈值为 0」—— 那会留下一个语义错误的旋钮，下一个人会调回去。
     * `push_threshold` 字段仍保留在 shm 配置区（护士端/配置页还在读），
     * 但不再参与推送判定。 */
    int allow = 1;
    int is_retract = (evt->state == FG_EVT_ST_RETRACTED);
    if (!s || !s->push_enabled)
        allow = 0;                                   /* 总开关关：记库不推*/
    /* ---- 状态闸门----
     * 只放行 confirmed / retracted 两态；suspected 等其它值一律拒。
     * 不许写成 `state == confirmed` —— 那会漏掉 retracted，
     * 撤回绕过冷却：撤回是「更正」，不是新告警。若被 30s 冷却吞掉，
     * 「先红卡、后灰化」就永远发生不了 —— 冷却只约束 confirmed。 */
    if (allow && evt->state != FG_EVT_ST_CONFIRMED &&
                 evt->state != FG_EVT_ST_RETRACTED)
        allow = 0;
    if (allow && !is_retract && last_push_ms &&
        now_ms - last_push_ms < s->push_cooldown_ms)
        allow = 0;                                   /* 冷却期内（只约束 confirmed）*/
    if (allow && !g_push_url[0])
        allow = 0;                                   /* 未配置接收端*/
    if (!allow)
        g_push_gated++;                              /* metric：被闸门挡掉*/

    /* ---- 组装推送体（闸门全过才推） ----*/
    int pushed = -1;
    if (allow) {
        if (!is_retract)
            last_push_ms = now_ms;   /* 撤回不占用冷却窗口，否则会推迟下一条真告警*/
        char esc_reason[160];
        json_escape(evt->reason, esc_reason, sizeof(esc_reason));
        char body[1280];
        /* `seq` 用 row_seq（不是 evt->seq）：撤回要更新同一张卡片。
         * v3 新增字段是护士端「红卡 → 灰化」的判据，缺了就等于没撤回。 */
        snprintf(body, sizeof(body),
                 "{\"type\":\"fall_alert\",\"gateway\":{\"pid\":%d},"
                 "\"event\":{\"seq\":%ld,\"ts\":%lld.%03d,\"person_id\":%d,"
                 "\"class_id\":%d,\"confidence\":%.2f,"
                 "\"bbox\":[%.1f,%.1f,%.1f,%.1f],\"hybrid_score\":%.2f,"
                 "\"reason\":\"%s\",\"snapshot_seq\":%u,"
                 "\"state\":%d,\"window_start\":%u,\"window_end\":%u,"
                 "\"model_ver\":%u,\"prob_fall\":%.3f,\"coverage\":%.3f}}",
                 (int)getpid(),
                 row_seq, (long long)evt->ts_sec, evt->ts_ms,
                 evt->person_id, evt->class_id, (double)evt->confidence,
                 (double)evt->bbox[0], (double)evt->bbox[1],
                 (double)evt->bbox[2], (double)evt->bbox[3],
                 (double)evt->hybrid_score, esc_reason, evt->snapshot_seq,
                 (int)evt->state, evt->window_start, evt->window_end,
                 evt->model_ver, (double)evt->prob_fall, (double)evt->coverage);
        if (http_post_json(g_push_url, body, 2000) == 0) {
            pushed = 1;
            g_push_ok++;
            log_status("已推送护士站 seq=%ld state=%d(evt=%llu) hybrid=%.2f -> %s",
                       row_seq, (int)evt->state,
                       (unsigned long long)evt->seq, (double)evt->hybrid_score, g_push_url);
        } else {
            pushed = 0;
            g_push_fail++;
            syslog(LOG_DAEMON | LOG_WARNING, "推送失败 seq=%ld state=%d -> %s",
                   row_seq, (int)evt->state, g_push_url);
        }
    }
    db_mark_pushed(row_seq, pushed);
}

void *t2_event_pusher(void *arg) {
    (void)arg;
    log_status("T2 event_pusher 启动");
    fg_event_t evt;
    while (eq_pop(&evt) == 0) {
        log_event(&evt);
        /* 快照 best-effort 存档（有挂快照才存）*/
        char snap_rel[64] = "";
        if (evt.snapshot_seq > 0)
            save_alarm_snap(evt.snapshot_seq, snap_rel, sizeof(snap_rel));
        db_ensure_open(g_data_dir);
        /* row_seq = 这一行实际使用的 seq：幂等命中时是首次那条的 seq。
         * DB 不可用时退化为本次 seq —— 推送不能因为落盘失败而停掉。 */
        long row_seq = db_insert_event(&evt, snap_rel);
        if (row_seq < 0) {
            syslog(LOG_DAEMON | LOG_WARNING, "事件落盘失败 seq=%llu",
                   (unsigned long long)evt.seq);
            row_seq = (long)evt.seq;
        }
        push_one_event(&evt, row_seq);
    }
    log_status("T2 退出");
    return NULL;
}

#define HTTP_REQ_MAX 2048

/* 返回 0 = 全部写出；-1 = 对端已断开（EPIPE/ECONNRESET/超时）。
 * 永远等不到"客户端走了"这个信号 ⇒ 每被遗弃一个观看端就永久泄漏一个线程，
 * 以 ~12fps 对 200KB 的 fg_frame_t 做栈拷贝。短连接（其它几个路由）不看返回值即可。 */
static int send_all(int fd, const void *buf, size_t len) {
    size_t off = 0;
    while (off < len) {
        ssize_t w = write(fd, (const char *)buf + off, len - off);
        if (w < 0 && errno == EINTR)     /* 信号打断不是错误，重试*/
            continue;
        if (w <= 0)
            return -1;
        off += (size_t)w;
    }
    return 0;
}

static void http_respond(int fd, int code, const char *status,
                         const char *ctype, const void *body, size_t blen) {
    char hdr[512];
    int n = snprintf(hdr, sizeof(hdr),
                     "HTTP/1.1 %d %s\r\nContent-Type: %s\r\n"
                     "Content-Length: %zu\r\nConnection: close\r\n\r\n",
                     code, status, ctype, blen);
    if (n > 0)
        send_all(fd, hdr, (size_t)n);
    send_all(fd, body, blen);
}

static void handle_status(int fd) {
    char body[2048];   /* push_url(512)+data_dir(256) 全满时也够*/
    fg_shm_t *s = g_shm;
    snprintf(body, sizeof(body),
             "{\"gateway\":{\"pid\":%d,\"push_url\":\"%s\",\"data_dir\":\"%s\"},"
             "\"sensor\":{\"pid\":%d,\"uptime\":%llu},"
             "\"status\":{\"seq\":%llu,\"is_falling\":%d,\"people\":%d,\"fps\":%.2f,"
             "\"frame_count\":%llu,\"last_evt_seq\":%u},"
             "\"config\":{\"cfg_seq\":%u,\"sensitivity\":%d,\"threshold\":%.2f,"
             "\"cooldown_ms\":%d,\"enabled\":%d},"
             "\"ring\":{\"count\":%u},\"queue\":{\"depth\":%d}}\n",
             (int)getpid(), g_push_url, g_data_dir,
             (int)s->sensor_pid, (unsigned long long)s->sensor_uptime,
             (unsigned long long)s->status_seq, s->is_falling, s->people_count,
             (double)s->fps, (unsigned long long)s->frame_count, s->last_evt_seq,
             s->cfg_seq, s->sensitivity_level, (double)s->push_threshold,
             s->push_cooldown_ms, s->push_enabled,
             s->evt_count, eq_depth());
    http_respond(fd, 200, "OK", "application/json", body, strlen(body));
}

static void handle_events(int fd, const char *query) {
    int n = 20;
    if (query)
        sscanf(query, "n=%d", &n);
    db_ensure_open(g_data_dir);       /* 读路径同样补开：DB 恢复后 /api/events 不再 503*/
    char *json = db_recent_json(n);
    if (json) {
        http_respond(fd, 200, "OK", "application/json", json, strlen(json));
        free(json);
    } else {
        http_respond(fd, 503, "DB Unavailable", "text/plain", "db unavailable\n", 16);
    }
}

static void handle_snap(int fd) {
    fg_frame_t fr;
    if (fg_pick_latest_frame(1, &fr) != 0) {
        http_respond(fd, 404, "No Snapshot", "text/plain", "no snapshot yet\n", 17);
        return;
    }
    http_respond(fd, 200, "OK", "image/jpeg", fr.jpeg, fr.jpeg_len);
}

/* v3 起删除 `/api/live.mjpg`：
 * shm 里已无 `live_frame[2]`。直播改由感知进程自带的进程内 MJPEG（:8888）
 * 或 GStreamer/RTSP 承担，网关不再转发画面 —— 它只做事件/快照/配置/状态。
 * 这也顺带消掉了本项目最长寿的一类线程泄漏源（MJPEG 长连接）。 */

/* 连接处理线程：解析请求行 → 路由（detached，退出即自清理）*/
typedef struct { int fd; } conn_arg_t;

static void *conn_thread(void *arg) {
    conn_arg_t *ca = (conn_arg_t *)arg;
    int fd = ca->fd;
    free(ca);

    struct timeval tv = { 5, 0 };   /* 读请求最多等 5s，防止半开连接占线程*/
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    char req[HTTP_REQ_MAX];
    ssize_t n = recv(fd, req, sizeof(req) - 1, 0);
    if (n <= 0) {
        close(fd);
        /* 活跃连接计数必须在这里也递减 —— 否则半开连接会让 g_conn_live
         * 只增不减，几十次后所有新连接都被 MAX_CONN 拒掉。 */
        pthread_mutex_lock(&g_conn_mtx);
        g_conn_live--;
        pthread_mutex_unlock(&g_conn_mtx);
        return NULL;
    }
    req[n] = '\0';

    /* GET /path?query HTTP/x.x*/
    char method[8] = "", target[512] = "";
    sscanf(req, "%7s %511s", method, target);
    char *qm = strchr(target, '?');
    const char *query = NULL;
    if (qm) {
        *qm = '\0';
        query = qm + 1;
    }

    int is_get = strcmp(method, "GET") == 0;
    if (!is_get) {
        http_respond(fd, 405, "Method Not Allowed", "text/plain", "GET only\n", 9);
    } else if (strcmp(target, "/api/status") == 0) {
        handle_status(fd);
    } else if (strcmp(target, "/api/events") == 0) {
        handle_events(fd, query);
    } else if (strcmp(target, "/api/snap") == 0) {
        handle_snap(fd);
    } else if (strcmp(target, "/api/live.mjpg") == 0 ||
               strcmp(target, "/api/live") == 0) {
        /* v3：直播帧已不在 shm 里。410 而不是 404 —— 语义是「永久移除」，
         * 让仍指向这里的老客户端停止重试，而不是以为路径写错。 */
        http_respond(fd, 410, "Gone", "text/plain",
                     "live stream moved out of shm in v3; use :8888 or RTSP\n", 55);
    } else {
        http_respond(fd, 404, "Not Found", "text/plain", "not found\n", 10);
    }
    shutdown(fd, SHUT_WR);
    close(fd);
    pthread_mutex_lock(&g_conn_mtx);
    g_conn_live--;
    pthread_mutex_unlock(&g_conn_mtx);
    return NULL;
}

void *t3_http_server(void *arg) {
    (void)arg;
    int port = g_http_port;
    int lfd = socket(AF_INET, SOCK_STREAM, 0);
    if (lfd < 0) {
        log_status("socket 失败: %s", strerror(errno));
        return NULL;
    }
    int one = 1;
    setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons((uint16_t)port);
    if (bind(lfd, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
        log_status("bind :%d 失败: %s", port, strerror(errno));
        close(lfd);
        return NULL;
    }
    if (listen(lfd, 16) != 0) {
        log_status("listen 失败: %s", strerror(errno));
        close(lfd);
        return NULL;
    }
    log_status("T3 http_server 启动 :%d （每连接一线程）", port);

    struct pollfd pfd;
    pfd.fd = lfd;
    pfd.events = POLLIN;
    while (!g_stop) {
        int pr = poll(&pfd, 1, 500);
        if (pr <= 0) continue;
        int cfd = accept(lfd, NULL, NULL);
        if (cfd < 0) continue;

        /* 并发护栏：
         * 线程栈 8 MB → 512 KB（CONN_STACK_SIZE）。`conn_thread` 栈上有
         * `char req[2048]` 和 `fg_frame_t`（204816 B，在 handle_snap 里），
         * 余量约 300 KB —— 不要再往上加大局部数组。
         * 最多 MAX_CONN 个并发连接，超过直接 close（拒绝服务面收敛），
         * 超出计数进 `g_conn_rejected`，由 metrics 行上报。 */
        pthread_mutex_lock(&g_conn_mtx);
        int over = (g_conn_live >= MAX_CONN);
        if (!over)
            g_conn_live++;
        pthread_mutex_unlock(&g_conn_mtx);
        if (over) {
            g_conn_rejected++;
            close(cfd);
            continue;
        }

        pthread_attr_t attr;
        pthread_attr_init(&attr);
        pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
        pthread_attr_setstacksize(&attr, CONN_STACK_SIZE);
        conn_arg_t *ca = malloc(sizeof(conn_arg_t));
        if (!ca) {
            close(cfd);
            pthread_mutex_lock(&g_conn_mtx);
            g_conn_live--;
            pthread_mutex_unlock(&g_conn_mtx);
            pthread_attr_destroy(&attr);
            continue;
        }
        ca->fd = cfd;
        pthread_t tid;
        if (pthread_create(&tid, &attr, conn_thread, ca) != 0) {
            close(cfd);
            free(ca);
            pthread_mutex_lock(&g_conn_mtx);
            g_conn_live--;
            pthread_mutex_unlock(&g_conn_mtx);
        }
        pthread_attr_destroy(&attr);
    }
    close(lfd);
    log_status("T3 退出");
    return NULL;
}

void *t5_watchdog(void *arg) {
    (void)arg;
    log_status("T5 watchdog 启动");
    unsigned long hb = 0;
    int tick = 0;                       /* 1 秒粒度*/
    while (!g_stop) {
        for (int i = 0; i < 5 && !g_stop; i++) {
            sleep(1);
            tick++;
            /* ---- systemd 看门狗：每 5 s 一次（WatchdogSec=30，留足余量）----
             * 这正是 WatchdogSec 要抓的场景。 */
            if (tick % 5 == 0)
                sd_notify_send("WATCHDOG=1");
            /* ---- metrics：每 10 s 一行 ----*/
            if (tick % METRICS_INTERVAL_S == 0)
                metrics_line();
        }
        if (g_stop) break;
        hb++;
        pid_t spid = g_shm ? g_shm->sensor_pid : -1;
        int alive = (spid > 0) && (kill(spid, 0) == 0);
        syslog(LOG_DAEMON | LOG_INFO,
               "HB#%lu sensor_pid=%d alive=%d ring_count=%u queue_depth=%d status_seq=%llu",
               hb, (int)spid, alive ? 1 : 0,
               g_shm ? g_shm->evt_count : 0, eq_depth(),
               (unsigned long long)(g_shm ? g_shm->status_seq : 0));
        if (!alive && spid > 0)
            syslog(LOG_DAEMON | LOG_WARNING, "事件源(sensor_pid=%d)已消失，等待恢复", (int)spid);
    }
    log_status("T5 退出");
    return NULL;
}

/* ================= main =================*/

static void usage(const char *p) {
    fprintf(stderr, "用法: %s [--config <path>] [--help]\n", p);
}

int main(int argc, char **argv) {
    const char *config_path = "config.json";
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--config") == 0 && i + 1 < argc) {
            config_path = argv[++i];
        } else if (strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "未知参数: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }

    log_init();
    eq_init();
    sd_notify_init();          /* 必须在任何 sd_notify_send 之前*/

    if (fg_shm_connect(g_shm_name) != 0) {
        syslog(LOG_ERR, "连接共享内存失败，退出：先启动事件源(mock/感知侧)");
        log_close();
        return 1;
    }
    fg_mark_gateway_pid();

    /* 配置先加载（填 g_push_url/g_data_dir），DB 才知道往哪落*/
    fg_cfg_t cfg;
    fg_config_load(config_path, &cfg);
    fg_write_config(&cfg);
    if (db_open(g_data_dir) != 0)
        syslog(LOG_WARNING, "SQLite 初始打开失败（事件将只在日志中）");

    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = on_signal;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);
    signal(SIGPIPE, SIG_IGN);

    pthread_t t1, t2, t3, t4, t5;
    pthread_create(&t1, NULL, t1_shm_listener, NULL);
    pthread_create(&t2, NULL, t2_event_pusher, NULL);
    pthread_create(&t3, NULL, t3_http_server, NULL);
    pthread_create(&t4, NULL, t4_config_watch, (void *)config_path);
    pthread_create(&t5, NULL, t5_watchdog, NULL);

    log_status("fallguard-gateway 启动 pid=%d push_url=%s data_dir=%s",
               (int)getpid(),
               g_push_url[0] ? g_push_url : "(未配置)",
               g_data_dir);

    /* 告诉 systemd「已就绪」—— 只有 Type=notify 时才需要，但发了无害。
     * 必须在 5 个线程都起来之后再发，否则 systemd 会认为服务已可用。 */
    sd_notify_send("READY=1\nSTATUS=运行中");

    while (!g_stop)
        pause();

    log_status("收到退出信号，开始优雅关闭");
    eq_shutdown();
    pthread_join(t1, NULL);
    pthread_join(t2, NULL);
    pthread_join(t3, NULL);
    pthread_join(t4, NULL);
    pthread_join(t5, NULL);
    db_close();
    fg_shm_disconnect();
    log_close();
    return 0;
}
