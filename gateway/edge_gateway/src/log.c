/* log.c — 统一日志封装，打到 syslog（journald 自动收集）*/
#include "gateway.h"

#include <syslog.h>
#include <stdarg.h>

void log_init(void) {
    openlog("fallguard-gateway", LOG_PID | LOG_NDELAY, LOG_DAEMON);
}

void log_close(void) {
    closelog();
}

void log_status(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsyslog(LOG_DAEMON | LOG_INFO, fmt, ap);
    va_end(ap);
}

static const char *class_name(int c) {
    switch (c) {
        case 0: return "walk";
        case 1: return "sit";
        case 2: return "squat";
        case 3: return "bend";
        case 4: return "lie";
        case 5: return "fall";
        default: return "?";
    }
}

void log_event(const fg_event_t *evt) {
    /* %.63s 限制宽度，即使 reason 无 null 结尾也不越界*/
    syslog(LOG_DAEMON | LOG_INFO,
           "EVENT seq=%llu ts=%lld.%03d pid=%d cls=%d(%s) conf=%.2f "
           "bbox=[%.1f,%.1f,%.1f,%.1f] hybrid=%.2f reason=%.63s snap=%u "
           "state=%u win=[%u,%u] mv=%u p=%.3f cov=%.3f",
           (unsigned long long)evt->seq, (long long)evt->ts_sec, evt->ts_ms,
           evt->person_id, evt->class_id, class_name(evt->class_id),
           (double)evt->confidence,
           (double)evt->bbox[0], (double)evt->bbox[1],
           (double)evt->bbox[2], (double)evt->bbox[3],
           (double)evt->hybrid_score, evt->reason, evt->snapshot_seq,
           (unsigned)evt->state, evt->window_start, evt->window_end,
           evt->model_ver, (double)evt->prob_fall, (double)evt->coverage);
}
