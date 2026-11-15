/* shm_ipc.c — 共享内存接入：连接 / 取事件 / 取帧 / 写配置区（与 Python ctypes 镜像语义一致）*/
#include "gateway.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>
#include <arpa/inet.h>   /* htonl*/

/* 全局定义（gateway.h extern）*/
fg_shm_t *g_shm = NULL;
sem_t *g_sem_mutex = NULL, *g_sem_evt = NULL, *g_sem_live = NULL;

/* ---- 连接 / 断开 ----*/
int fg_shm_connect(const char *name) {
    int fd = shm_open(name, O_RDWR, 0);
    if (fd < 0) {
        log_status("shm_open %s 失败: %s", name, strerror(errno));
        return -1;
    }
    struct stat st;
    if (fstat(fd, &st) != 0) {
        log_status("fstat %s 失败: %s", name, strerror(errno));
        close(fd);
        return -1;
    }
    /* 这里只能查「太小」，查不出「太大」：v3 的 sizeof 比 v2 小 400 KB，
     * 一个残留的 v2 shm（820320 B）会顺利通过这一关。
     * 真正拦住它的是下面 header 里的 `shm_size` 三查 —— 别以为有 fstat 就够了。 */
    if (st.st_size < (off_t)sizeof(fg_shm_t)) {
        log_status("shm %s 尺寸异常 (size=%lld, 期望 >=%zu)", name,
                   (long long)st.st_size, sizeof(fg_shm_t));
        close(fd);
        return -1;
    }
    void *p = mmap(NULL, sizeof(fg_shm_t), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (p == MAP_FAILED) {
        log_status("mmap %s 失败: %s", name, strerror(errno));
        return -1;
    }
    fg_shm_t *shm = (fg_shm_t *)p;
    /* ---- v3 读端三查：magic → version → shm_size ----
     * 只有 magic 一道时，旧版 gateway 读新版 sensor 会静默错位：
     * 结构变大后所有字段偏移都变了，读到的是垃圾但不报错。 */
    if (shm->magic != FG_SHM_MAGIC) {
        log_status("shm %s 未初始化 (magic=0x%08X)：先启动事件源(mock/感知侧)", name, shm->magic);
        munmap(p, sizeof(fg_shm_t));
        return -1;
    }
    if (shm->version != FG_SHM_VERSION) {
        log_status("shm %s 版本不匹配 (板上 v%u, 本端 v%d)：新旧版本不能共存，"
                   "请同时重启 sensor 与 gateway 并 rm /dev/shm/fallguard_shm",
                   name, shm->version, FG_SHM_VERSION);
        munmap(p, sizeof(fg_shm_t));
        return -1;
    }
    if (shm->shm_size != (uint32_t)sizeof(fg_shm_t)) {
        log_status("shm %s 结构漂移 (记录 size=%u, 本端 %zu)：拒绝启动",
                   name, shm->shm_size, sizeof(fg_shm_t));
        munmap(p, sizeof(fg_shm_t));
        return -1;
    }
    g_shm = shm;
    /* O_CREAT：事件源先建则复用；网关先建则按初值建（mutex=1, evt/live=0），语义一致*/
    g_sem_mutex = sem_open("/fg_shm_mutex", O_CREAT, 0660, 1);
    g_sem_evt   = sem_open("/fg_evt_sem",   O_CREAT, 0660, 0);
    g_sem_live  = sem_open("/fg_live_sem",  O_CREAT, 0660, 0);
    if (!g_sem_mutex || !g_sem_evt || !g_sem_live) {
        log_status("sem_open 失败: %s", strerror(errno));
        fg_shm_disconnect();
        return -1;
    }
    log_status("已连接共享内存 %s v%d size=%zu", name, shm->version, sizeof(fg_shm_t));
    return 0;
}

void fg_shm_disconnect(void) {
    if (g_sem_mutex) { sem_close(g_sem_mutex); g_sem_mutex = NULL; }
    if (g_sem_evt)   { sem_close(g_sem_evt);   g_sem_evt = NULL; }
    if (g_sem_live)  { sem_close(g_sem_live);  g_sem_live = NULL; }
    if (g_shm) { munmap(g_shm, sizeof(fg_shm_t)); g_shm = NULL; }
    /* 不 shm_unlink/sem_unlink：事件源(mock/感知侧)可能仍在用*/
}

void fg_mark_gateway_pid(void) {
    if (g_shm) g_shm->gateway_pid = getpid();
}

/* ---- 消费者读：事件环（与 shm_ipc.py take_event 语义一致） ----*/
int fg_take_event(fg_event_t *out) {
    int rc = -1;
    if (sem_wait(g_sem_mutex) != 0)
        return -1;
    if (g_shm->evt_count > 0) {
        uint32_t head = g_shm->evt_head;
        *out = g_shm->events[head];
        g_shm->evt_head = (head + 1) % FG_EVT_RING_CAP;
        g_shm->evt_count--;
        g_shm->status_seq++;          /* 每次消费递增，保持事件源/网关状态可见*/
        rc = 0;
    }
    sem_post(g_sem_mutex);
    return rc;
}

/* ---- 消费者读：双缓冲帧，取 frame_seq 较大者（与 pick_latest_frame 一致） ----
 * v3 起 shm 里只剩告警快照（alarm_snap），live_frame[2] 已删 —— 直播走
 * GStreamer/RTSP，不再进共享内存。want_alarm=0 直接返回 -1。 */
int fg_pick_latest_frame(int want_alarm, fg_frame_t *out) {
    if (!g_shm) return -1;
    if (!want_alarm) return -1;          /* v3：无直播帧区域*/
    const fg_frame_t *frames = g_shm->alarm_snap;
    const fg_frame_t *fr = (frames[0].frame_seq >= frames[1].frame_seq)
                               ? &frames[0] : &frames[1];
    if (fr->frame_seq == 0 || fr->jpeg_len == 0)
        return -1;
    *out = *fr;
    return 0;
}

/* ---- CRC32(IEEE 802.3, 与 zlib.crc32 一致) + 配置区大端打包(对齐 struct.pack(">iifii")) ----*/
static uint32_t g_crc_table[256];
static int g_crc_ready = 0;

static void crc32_table_init(void) {
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t c = i;
        for (int k = 0; k < 8; k++)
            c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
        g_crc_table[i] = c;
    }
    g_crc_ready = 1;
}

static uint32_t crc32(const void *data, size_t len) {
    if (!g_crc_ready)
        crc32_table_init();
    const uint8_t *p = (const uint8_t *)data;
    uint32_t crc = 0xFFFFFFFFu;
    while (len--)
        crc = g_crc_table[(crc ^ *p++) & 0xFFu] ^ (crc >> 8);
    return crc ^ 0xFFFFFFFFu;
}

static uint32_t config_crc32(void) {
    uint8_t buf[20];
    uint32_t u;
    fg_shm_t *s = g_shm;

    u = htonl(s->cfg_seq);                       memcpy(&buf[0], &u, 4);
    u = htonl((uint32_t)s->sensitivity_level);   memcpy(&buf[4], &u, 4);
    float f = s->push_threshold;
    memcpy(&u, &f, 4);  u = htonl(u);            memcpy(&buf[8], &u, 4);
    u = htonl((uint32_t)s->push_cooldown_ms);    memcpy(&buf[12], &u, 4);
    u = htonl((uint32_t)s->push_enabled);        memcpy(&buf[16], &u, 4);
    return crc32(buf, sizeof(buf));
}

/* ---- 写配置区（事件源轮询 cfg_seq 变化后重读） ----*/
void fg_write_config(const fg_cfg_t *cfg) {
    if (!g_shm) return;
    g_shm->cfg_seq++;
    g_shm->sensitivity_level = cfg->sensitivity_level;
    g_shm->push_threshold    = cfg->push_threshold;
    g_shm->push_cooldown_ms  = cfg->push_cooldown_ms;
    g_shm->push_enabled      = cfg->push_enabled;
    g_shm->cfg_crc32         = config_crc32();
    log_status("config->shm: cfg_seq=%u sens=%d threshold=%.2f cooldown=%d enabled=%d crc=0x%08X",
               g_shm->cfg_seq, g_shm->sensitivity_level, (double)g_shm->push_threshold,
               g_shm->push_cooldown_ms, g_shm->push_enabled, g_shm->cfg_crc32);
}
