
#include "gateway.h"

#include <string.h>

#define EQ_CAP 256

typedef struct {
    fg_event_t items[EQ_CAP];
    size_t head, tail, count;
    unsigned long dropped;          /* 满时丢最旧的累计条数（metric）*/
    pthread_mutex_t mtx;
    pthread_cond_t not_empty, not_full;
} event_queue_t;

static event_queue_t g_eq;

void eq_init(void) {
    pthread_mutex_init(&g_eq.mtx, NULL);
    pthread_cond_init(&g_eq.not_empty, NULL);
    pthread_cond_init(&g_eq.not_full, NULL);
    g_eq.head = g_eq.tail = g_eq.count = 0;
    g_eq.dropped = 0;
}

void eq_shutdown(void) {
    /* 唤醒所有阻塞在 push/pop 的线程，让其在 g_stop 下自行退出*/
    pthread_mutex_lock(&g_eq.mtx);
    pthread_cond_broadcast(&g_eq.not_empty);
    pthread_cond_broadcast(&g_eq.not_full);
    pthread_mutex_unlock(&g_eq.mtx);
}

void eq_push(const fg_event_t *evt) {
    pthread_mutex_lock(&g_eq.mtx);
    /* 满时丢最旧，不再 cond_wait 阻塞。
     *
     * 阻塞的后果：推送线程一旦卡住（网络慢、对端不回），生产者就被钉死，
     * 共享内存事件环随之填满，之后每来一条新事件就静默丢掉一条旧的，
     * 而且丢的是哪条完全不可控。
     *
     * 现在：丢队列里最老的那条，生产者永不阻塞。
     * 对跌倒告警这是正确的取舍 —— 新的比旧的紧急。
     * 但必须记 metric（`eq_dropped_total`），否则又是一次静默丢。
     * 该计数由 T5 心跳 / metrics 行上报。 */
    if (g_eq.count == EQ_CAP) {
        g_eq.head = (g_eq.head + 1) % EQ_CAP;   /* 丢掉最旧*/
        g_eq.count--;
        g_eq.dropped++;
    }
    g_eq.items[g_eq.tail] = *evt;
    g_eq.tail = (g_eq.tail + 1) % EQ_CAP;
    g_eq.count++;
    pthread_cond_signal(&g_eq.not_empty);
    pthread_mutex_unlock(&g_eq.mtx);
}

unsigned long eq_dropped_total(void) {
    pthread_mutex_lock(&g_eq.mtx);
    unsigned long d = g_eq.dropped;
    pthread_mutex_unlock(&g_eq.mtx);
    return d;
}

int eq_pop(fg_event_t *out) {
    pthread_mutex_lock(&g_eq.mtx);
    while (g_eq.count == 0 && !g_stop)
        pthread_cond_wait(&g_eq.not_empty, &g_eq.mtx);
    if (g_eq.count == 0) {
        /* 停止且空：消费者退出*/
        pthread_mutex_unlock(&g_eq.mtx);
        return -1;
    }
    *out = g_eq.items[g_eq.head];
    g_eq.head = (g_eq.head + 1) % EQ_CAP;
    g_eq.count--;
    pthread_cond_signal(&g_eq.not_full);
    pthread_mutex_unlock(&g_eq.mtx);
    return 0;
}

int eq_depth(void) {
    pthread_mutex_lock(&g_eq.mtx);
    int d = (int)g_eq.count;
    pthread_mutex_unlock(&g_eq.mtx);
    return d;
}
