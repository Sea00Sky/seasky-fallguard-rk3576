CC      ?= gcc
# -D_GNU_SOURCE：`sem_clockwait`（T1 的 CLOCK_MONOTONIC 超时）需要它。
# 缺了只会拿到隐式声明警告，运行时参数传递恰好还能跑 —— 属于 UB，必须加。
CFLAGS  ?= -O2 -Wall -Wextra -D_GNU_SOURCE -pthread -Iinclude -Ithird_party/sqlite3
TARGET  := fallguard-gateway
SRCS    := src/main.c src/shm_ipc.c src/event_queue.c src/config.c src/db.c src/log.c
OBJS    := $(SRCS:.c=.o)

# 板上无 libsqlite3-dev（无 .so 符号链接），直接链版本化运行库；
# 头文件用 third_party/sqlite3/sqlite3.h（3.45.x，与 Ubuntu 24.04 运行库同系）
SQLITE3_LIB ?= $(firstword $(wildcard /lib/aarch64-linux-gnu/libsqlite3.so*))

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(SQLITE3_LIB)

%.o: %.c include/gateway.h include/fallguard_shm.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

install: $(TARGET)
	install -m 0755 $(TARGET) /usr/local/bin/$(TARGET)

.PHONY: all clean install
