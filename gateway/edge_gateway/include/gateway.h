/*
 * gateway.h — FallGuard 边缘网关内部接口
 *
 * 职责：消费共享内存事件环 → SQLite 落盘 + HTTP POST 推送护士站；
 * 暴露 /api/status /api/events /api/snap /api/live.mjpg；
 * 监听配置变更写回配置区。
 * 契约唯一事实来源：include/fallguard_shm.h（本文件不重复定义任何共享内存结构）。
 */
#ifndef FALLGUARD_GATEWAY_H
#define FALLGUARD_GATEWAY_H

#include "fallguard_shm.h"
#include <semaphore.h>
#include <pthread.h>
#include <signal.h>

/* 生效配置（映射到共享内存配置区；push_url/data_dir 是网关本地项不进契约）*/
typedef struct {
    int   sensitivity_level;    /* 0=low 1=medium 2=high*/
    float push_threshold;       /* v3 起不参与推送判定，仅保留字段兼容*/
    int   push_cooldown_ms;     /* 相邻两次推送最小间隔*/
    int   push_enabled;         /* 总开关*/
} fg_cfg_t;

/* ---- 全局状态（定义在各自模块，外部经此声明使用） ----*/
extern volatile sig_atomic_t g_stop;
extern fg_shm_t *g_shm;
extern sem_t *g_sem_mutex, *g_sem_evt, *g_sem_live;
extern int  g_http_port;
extern char g_shm_name[64];
extern char g_push_url[512];    /* 护士站接收 URL（空=不推）*/
extern char g_data_dir[256];    /* SQLite/快照落盘根目录*/

/* ---- shm_ipc.c ----*/
int  fg_shm_connect(const char *name);   /* 0=成功；失败返回-1*/
void fg_shm_disconnect(void);            /* 只 close/unmap，不 unlink（事件源可能在用）*/
void fg_mark_gateway_pid(void);
int  fg_take_event(fg_event_t *out);     /* 0=取到一条, -1=环空*/
int  fg_pick_latest_frame(int want_alarm, fg_frame_t *out); /* v3 无直播帧：want_alarm=1 取告警快照，0 直接 -1；返回 0=有帧*/
void fg_write_config(const fg_cfg_t *cfg);

/* ---- event_queue.c ----*/
void eq_init(void);
void eq_shutdown(void);                  /* 唤醒所有阻塞的 push/pop*/
void eq_push(const fg_event_t *evt);     /* 满时丢最旧，永不阻塞*/
int  eq_pop(fg_event_t *out);            /* 0=取到, -1=停止且空*/
int  eq_depth(void);
unsigned long eq_dropped_total(void);    /* 满时丢最旧的累计条数（metric）*/

/* ---- config.c ----*/
int  fg_config_load(const char *path, fg_cfg_t *cfg); /* 0=成功；失败用默认值*/
void *t4_config_watch(void *arg);

/* ---- db.c（SQLite 落盘） ----*/
int  db_open(const char *data_dir);      /* 建 <data_dir>/events.db + snaps/ 目录，0=成功*/
int  db_ensure_open(const char *data_dir); /* g_db 为空时惰性重开（30s 节流），已开直接成功*/
void db_close(void);
long db_insert_event(const fg_event_t *e, const char *snap_rel); /* 返回 seq，失败-1*/
void db_mark_pushed(long row_seq, int pushed);  /* 1=成功 0=失败 -1=闸门未推；
                                                 * 传 db_insert_event() 的返回值（幂等命中时≠evt->seq） */
char *db_recent_json(int limit);         /* 返回 malloc 的 JSON 数组文本，调用方 free()；失败 NULL*/

int  save_alarm_snap(uint32_t want_seq, char *rel_out, size_t n); /* 存最新告警快照 JPEG，返回相对名*/
int  http_post_json(const char *url, const char *body, int timeout_ms); /* 0=2xx 成功*/
void json_escape(const char *in, char *out, size_t n);

/* ---- log.c ----*/
void log_init(void);
void log_close(void);
void log_status(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
void log_event(const fg_event_t *evt);

/* ---- main.c 线程入口 ----*/
void *t1_shm_listener(void *arg);
void *t2_event_pusher(void *arg);
void *t3_http_server(void *arg);
void *t5_watchdog(void *arg);

#endif /* FALLGUARD_GATEWAY_H*/
