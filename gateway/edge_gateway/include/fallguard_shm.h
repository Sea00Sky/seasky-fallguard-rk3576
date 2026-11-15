/*
 * fallguard_shm.h — FallGuard 共享内存契约（唯一事实来源）
 *
 * Python 侧逐字段镜像见 shm_proto/shm_ipc.py（ctypes）。
 * 约定：C 与 Python 均使用【自然对齐】（不 packed），任何平台布局必须一致。
 * 布局由编译期断言（typedef 尺寸/offset 检查）兜底；改动结构必须同步两边并更新断言。
 *
 * 进程模型：
 * 事件源(mock / 用户跌倒代码) 写：状态区 / 事件环 / 告警快照 + sem_post
 * C 网关(edge_gateway) 读：事件环 / 告警快照；写：配置区
 * 命名信号量：/fg_shm_mutex(初值1) /fg_evt_sem(初值0) /fg_live_sem(初值0，v3 起无消费者)
 *
 * v3 相对 v2 的改动：事件结构 120 → 144 B（state / window_* / prob_fall /
 * coverage / model_ver）；事件环 8 → 32 条（多人场景下 8 条会被覆盖）；
 * 删掉 fg_frame_t live_frame[2]（400 KB），直播帧不再进 shm。
 * sizeof(fg_shm_t) 由 820320 变为 414344。新旧版本不能共存，
 * 切换时必须同时重启 sensor 与 gateway，并先 `rm /dev/shm/fallguard_shm`。
 */
#ifndef FALLGUARD_SHM_H
#define FALLGUARD_SHM_H

#include <stdint.h>
#include <sys/types.h>   /* pid_t*/
#include <stddef.h>      /* offsetof*/

#define FG_SHM_MAGIC    0x4647414CUL   /* "FGAL"*/
#define FG_SHM_VERSION  3
#define FG_EVT_RING_CAP 32             /* 事件环形队列容量（v3: 8 → 32）*/
#define FG_JPEG_MAX     (200*1024)     /* 单帧JPEG上限 200KB*/
#define FG_LIVE_W       320            /* 快照帧宽（沿用直播帧尺寸做默认）*/
#define FG_LIVE_H       240            /* 快照帧高*/

/* 事件状态机（v3 新增；V1 只产生 confirmed / retracted 两态）*/
#define FG_EVT_ST_SUSPECTED  0
#define FG_EVT_ST_CONFIRMED  1
#define FG_EVT_ST_RETRACTED  2

/* 单条跌倒/姿态事件*/
typedef struct {
    uint64_t seq;          /* 全局事件序号(单调递增)*/
    int64_t  ts_sec;       /* 事件时刻(epoch秒)*/
    int32_t  ts_ms;        /* 毫秒余量*/
    int32_t  person_id;
    int32_t  class_id;     /* 0=walk 1=sit 2=squat 3=bend 4=lie 5=fall*/
    float    confidence;   /* 分类置信度*/
    float    bbox[4];      /* x1 y1 x2 y2 像素坐标*/
    float    hybrid_score; /* 混合融合分数*/
    char     reason[64];   /* dual_confirm / ml_rescue / ml_temporal_confirm / ...*/
    uint32_t snapshot_seq; /* 关联告警快照的 frame_seq*/

    /* ---- v3 新增 ----*/
    uint8_t  state;        /* FG_EVT_ST_*：0=suspected 1=confirmed 2=retracted*/
    uint8_t  _pad[3];      /* 显式补齐，保证 window_start 落在 4 字节边界*/
    uint32_t window_start; /* 窗口起始帧号 —— 幂等键的一部分*/
    uint32_t window_end;   /* 窗口结束帧号*/
    float    prob_fall;    /* ST-GCN 原始概率（未经阈值/融合修饰）*/
    float    coverage;
    uint32_t model_ver;    /* 模型版本 hash 低 32 位 —— 幂等键的一部分*/
} fg_event_t;              /* 120 → 144 字节*/

/* 单帧 JPEG（仅告警快照；v3 起直播帧不再进 shm）*/
typedef struct {
    uint32_t frame_seq;    /* 单调递增，读者按它判新帧（写全后最后发布）*/
    int32_t  w, h;
    uint32_t jpeg_len;
    uint8_t  jpeg[FG_JPEG_MAX];
} fg_frame_t;

/* 共享内存总布局*/
typedef struct {
    /* ---- 固定头部 ----*/
    uint32_t magic;
    uint32_t version;
    uint32_t shm_size;     /* v3 新增：= sizeof(fg_shm_t)，读端第三道校验*/
    pid_t    sensor_pid;   /* 事件源进程PID，看门狗用*/
    pid_t    gateway_pid;
    uint64_t sensor_uptime;

    /* ---- 配置区: 网关写, 事件源读 ----*/
    uint32_t cfg_seq;          /* 配置版本, 事件源轮询发现变化才重读*/
    int32_t  sensitivity_level;/* 0=low 1=medium 2=high*/
    float    push_threshold;   /* v3 起不参与推送判定，仅保留字段兼容*/
    int32_t  push_cooldown_ms; /* 相邻两次推送最小间隔*/
    int32_t  push_enabled;     /* 总开关*/
    uint32_t cfg_crc32;

    /* ---- 状态区: 事件源写, 网关/上位机读 ----*/
    uint64_t status_seq;       /* 每次更新递增(最后发布)*/
    int32_t  is_falling;
    int32_t  people_count;
    float    fps;
    uint64_t frame_count;
    uint32_t last_evt_seq;

    /* ---- 事件环形队列: 事件源写, 网关读 ----*/
    uint32_t evt_head, evt_tail, evt_count;
    fg_event_t events[FG_EVT_RING_CAP];

    /* ---- 告警快照双缓冲: 事件源写, 网关读 ----*/
    fg_frame_t alarm_snap[2];
} fg_shm_t;   /* 自然对齐；Python ctypes 必须逐字段镜像，禁止 _pack_ 差异*/

/* ---- 布局编译期断言（改动结构必须同步） ----*/
typedef char fg_ck_event_sz[(sizeof(fg_event_t) == 144)          ? 1 : -1];
typedef char fg_ck_frame_sz[(sizeof(fg_frame_t) == 204816)       ? 1 : -1];
typedef char fg_ck_shm_sz[(sizeof(fg_shm_t) == 414344)           ? 1 : -1];
typedef char fg_ck_evt_bbox[(offsetof(fg_event_t, bbox) == 32)   ? 1 : -1];
typedef char fg_ck_evt_reason[(offsetof(fg_event_t, reason) == 52) ? 1 : -1];
typedef char fg_ck_evt_snapseq[(offsetof(fg_event_t, snapshot_seq) == 116) ? 1 : -1];
typedef char fg_ck_evt_state[(offsetof(fg_event_t, state) == 120) ? 1 : -1];
typedef char fg_ck_evt_wstart[(offsetof(fg_event_t, window_start) == 124) ? 1 : -1];
typedef char fg_ck_evt_modelver[(offsetof(fg_event_t, model_ver) == 140) ? 1 : -1];
typedef char fg_ck_shm_sizeoff[(offsetof(fg_shm_t, shm_size) == 8) ? 1 : -1];
typedef char fg_ck_shm_events[(offsetof(fg_shm_t, events) == 104) ? 1 : -1];
typedef char fg_ck_shm_alarm[(offsetof(fg_shm_t, alarm_snap) == 4712) ? 1 : -1];

#endif /* FALLGUARD_SHM_H*/
