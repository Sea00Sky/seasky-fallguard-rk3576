# 逐窗对账 —— 硬门
#
# 用法
#   make regress-selftest                 # 只验比较逻辑，不碰 NPU/视频（秒级）
#   make regress                          # 全量跑一遍 + 跨臂对账（约 11 min）
#   make regress-determinism              # 同臂确定性：连跑两次，必须恰好 0
#   make regress-diff NEW=... BASE=...    # 任意两份对账
#
# 长跑请写日志文件，不要用管道过滤（会缓冲，看不到进度）。

PY      ?= python
RUN_ID  ?= $(shell $(PY) -c "import time;print(time.strftime('%%Y%%m%%d_%%H%%M'))")
OUT     ?= ../results/regress/$(RUN_ID)
BASE    ?= ../reference/g3_windows.csv
REG     := $(PY) p8/regress.py

export PYTHONIOENCODING = utf-8

.PHONY: help regress regress-selftest regress-determinism regress-diff clean

help:
	@echo "make regress-selftest    只验比较逻辑（秒级，先跑这个）"
	@echo "make regress             全量跑 + 跨臂对账  -> $(OUT)"
	@echo "make regress-determinism 同臂确定性（连跑两次，必须恰好 0）"
	@echo "make regress-diff NEW=dir BASE=csv"

regress-selftest:
	$(REG) selftest

regress:
	$(REG) run --out $(OUT) --base $(BASE)

# 同臂确定性：真判据。两次独立进程，max|Δp| 必须恰好 0。
regress-determinism:
	@echo "== run 1 =="
	$(REG) run --out $(OUT)/a --base $(OUT)/a/windows.csv
	@echo "== run 2 =="
	$(REG) run --out $(OUT)/b --base $(OUT)/a/windows.csv --expect-zero
	@echo "== 判定见 $(OUT)/b/diff.md =="

regress-diff:
	@test -n "$(NEW)" || { echo "用法: make regress-diff NEW=<dir> BASE=<csv>"; exit 2; }
	$(REG) diff --new $(NEW) --base $(BASE) $(if $(EXPECT_ZERO),--expect-zero,)

clean:
	rm -rf ../results/regress
