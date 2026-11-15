
#include "gateway.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <unistd.h>
#include <poll.h>
#include <sys/inotify.h>

/* 默认配置（medium 档，与契约/ mock 默认一致）*/
#define CFG_DEFAULT_SENS       1
#define CFG_DEFAULT_THRESHOLD  0.60f
#define CFG_DEFAULT_COOLDOWN   3000
#define CFG_DEFAULT_ENABLED    1

int  g_http_port = 8080;
char g_shm_name[64] = "/fallguard_shm";

/* ---- 极简 JSON 取值（schema 固定，不引第三方库） ----*/
static const char *skip_ws(const char *p) {
    while (p && *p && isspace((unsigned char)*p)) p++;
    return p;
}

/* 定位 key 的冒号后的值起点*/
static const char *value_after(const char *json, const char *key) {
    const char *p = strstr(json, key);
    if (!p) return NULL;
    p = strchr(p, ':');
    if (!p) return NULL;
    return skip_ws(p + 1);
}

static int read_string(const char *json, const char *key, char *out, size_t n) {
    const char *v = value_after(json, key);
    if (!v || *v != '"') return -1;
    v++;
    size_t i = 0;
    while (*v && *v != '"' && i < n - 1) out[i++] = *v++;
    out[i] = '\0';
    return 0;
}

static int read_int(const char *json, const char *key, int *out) {
    const char *v = value_after(json, key);
    if (!v) return -1;
    char *end;
    long val = strtol(v, &end, 10);
    if (end == v) return -1;
    *out = (int)val;
    return 0;
}

static int read_float(const char *json, const char *key, float *out) {
    const char *v = value_after(json, key);
    if (!v) return -1;
    char *end;
    double val = strtod(v, &end);
    if (end == v) return -1;
    *out = (float)val;
    return 0;
}

static int read_bool(const char *json, const char *key, int *out) {
    const char *v = value_after(json, key);
    if (!v) return -1;
    if (strncmp(v, "true", 4) == 0)  { *out = 1; return 0; }
    if (strncmp(v, "false", 5) == 0) { *out = 0; return 0; }
    return read_int(json, key, out);
}

static int sensitivity_from_str(const char *s) {
    if (strcmp(s, "low") == 0)    return 0;
    if (strcmp(s, "medium") == 0) return 1;
    if (strcmp(s, "high") == 0)   return 2;
    return CFG_DEFAULT_SENS;
}

/* ---- 加载 config.json（缺失/非法字段用默认值） ----*/
int fg_config_load(const char *path, fg_cfg_t *cfg) {
    cfg->sensitivity_level = CFG_DEFAULT_SENS;
    cfg->push_threshold    = CFG_DEFAULT_THRESHOLD;
    cfg->push_cooldown_ms  = CFG_DEFAULT_COOLDOWN;
    cfg->push_enabled      = CFG_DEFAULT_ENABLED;

    FILE *fp = fopen(path, "r");
    if (!fp) {
        log_status("config %s 不可读(%s)，使用默认配置", path, strerror(errno));
        return 0;
    }
    char *buf = NULL;
    if (fseek(fp, 0, SEEK_END) == 0) {
        long sz = ftell(fp);
        if (sz > 0 && sz < (1L << 20)) {
            rewind(fp);
            buf = malloc((size_t)sz + 1);
            if (buf) {
                size_t r = fread(buf, 1, (size_t)sz, fp);
                buf[r] = '\0';
            }
        }
    }
    fclose(fp);
    if (!buf) return 0;

    char sens[16] = "medium";
    if (read_string(buf, "sensitivity", sens, sizeof(sens)) == 0)
        cfg->sensitivity_level = sensitivity_from_str(sens);
    if (read_float(buf, "push_threshold", &cfg->push_threshold) != 0)
        cfg->push_threshold = CFG_DEFAULT_THRESHOLD;
    if (read_int(buf, "push_cooldown_ms", &cfg->push_cooldown_ms) != 0)
        cfg->push_cooldown_ms = CFG_DEFAULT_COOLDOWN;
    if (read_bool(buf, "push_enabled", &cfg->push_enabled) != 0)
        cfg->push_enabled = CFG_DEFAULT_ENABLED;
    int port = 0;
    if (read_int(buf, "http_port", &port) == 0 && port > 0 && port < 65536)
        g_http_port = port;
    if (read_string(buf, "shm_name", g_shm_name, sizeof(g_shm_name)) != 0)
        snprintf(g_shm_name, sizeof(g_shm_name), "/fallguard_shm");
    /* 网关本地项（不进契约）：推送 URL 与落盘目录*/
    snprintf(g_push_url, sizeof(g_push_url), "%s", "");   /* 默认不推*/
    read_string(buf, "push_url", g_push_url, sizeof(g_push_url));
    snprintf(g_data_dir, sizeof(g_data_dir), "%s", "/mnt/sdcard/fallguard_data");
    read_string(buf, "data_dir", g_data_dir, sizeof(g_data_dir));
    free(buf);

    log_status("config 已加载: sens=%d threshold=%.2f cooldown=%d enabled=%d "
               "port=%d shm=%s push_url=%s data_dir=%s",
               cfg->sensitivity_level, (double)cfg->push_threshold,
               cfg->push_cooldown_ms, cfg->push_enabled, g_http_port, g_shm_name,
               g_push_url[0] ? g_push_url : "(空)", g_data_dir);
    return 0;
}

/* ---- inotify 监听 ----*/
static void split_path(const char *path, char *dir, size_t dsz, char *base, size_t bsz) {
    const char *slash = strrchr(path, '/');
    if (slash) {
        size_t dlen = (size_t)(slash - path);
        if (dlen == 0) dlen = 1;              /* "/config.json" -> dir="/"*/
        snprintf(dir, dsz, "%.*s", (int)dlen, path);
        snprintf(base, bsz, "%s", slash + 1);
    } else {
        snprintf(dir, dsz, ".");
        snprintf(base, bsz, "%s", path);
    }
}

void *t4_config_watch(void *arg) {
    const char *path = (const char *)arg;
    fg_cfg_t cur;
    fg_config_load(path, &cur);
    fg_write_config(&cur);
    log_status("T4 config_watch 启动 (%s)", path);

    char dir[512], base[256];
    split_path(path, dir, sizeof(dir), base, sizeof(base));

    int ifd = inotify_init1(IN_NONBLOCK);
    if (ifd < 0) {
        log_status("inotify_init1 失败: %s", strerror(errno));
        return NULL;
    }
    /* 监听目录 + 按文件名匹配，兼容编辑器原子替换(IN_MOVED_TO)*/
    if (inotify_add_watch(ifd, dir, IN_CLOSE_WRITE | IN_MOVED_TO) < 0) {
        log_status("inotify_add_watch %s 失败: %s", dir, strerror(errno));
        close(ifd);
        return NULL;
    }

    struct pollfd pfd;
    pfd.fd = ifd;
    pfd.events = POLLIN;
    while (!g_stop) {
        int pr = poll(&pfd, 1, 500);
        if (pr <= 0) continue;
        char buf[4096];
        ssize_t n;
        while ((n = read(ifd, buf, sizeof(buf))) > 0) {
            const char *p = buf;
            while (p < buf + n) {
                const struct inotify_event *ev = (const struct inotify_event *)p;
                if (ev->len > 0 && strcmp(ev->name, base) == 0) {
                    char old_url[sizeof(g_push_url)], old_dir[sizeof(g_data_dir)];
                    snprintf(old_url, sizeof(old_url), "%s", g_push_url);
                    snprintf(old_dir, sizeof(old_dir), "%s", g_data_dir);
                    fg_cfg_t nc;
                    fg_config_load(path, &nc);
                    int cfg_changed = memcmp(&nc, &cur, sizeof(fg_cfg_t)) != 0;
                    int url_changed = strcmp(old_url, g_push_url) != 0;
                    int dir_changed = strcmp(old_dir, g_data_dir) != 0;
                    if (cfg_changed) {
                        cur = nc;
                        fg_write_config(&cur);   /* 只有契约字段变更才推进 cfg_seq*/
                    }
                    if (cfg_changed || url_changed || dir_changed)
                        log_status("config 变更已生效: sens=%d threshold=%.2f "
                                   "cooldown=%d enabled=%d push_url=%s data_dir=%s",
                                   cur.sensitivity_level, (double)cur.push_threshold,
                                   cur.push_cooldown_ms, cur.push_enabled,
                                   g_push_url[0] ? g_push_url : "(空)", g_data_dir);
                }
                p += sizeof(struct inotify_event) + ev->len;
            }
        }
    }
    close(ifd);
    log_status("T4 退出");
    return NULL;
}
