#!/usr/bin/env python3
"""dump_shm.py — 诊断工具：只读打印共享内存当前内容。"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shm_ipc import FallGuardShm  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="打印 FallGuard 共享内存")
    ap.add_argument("--name", default=FallGuardShm.SHM_DEFAULT)
    args = ap.parse_args()
    try:
        shm = FallGuardShm(name=args.name, create=False)
        print(shm.dump())
        return 0
    except OSError as e:
        print("无法打开共享内存 %s: %s" % (args.name, e))
        print("（先运行 mock_fall_generator.py 生成 shm）")
        return 1
    except RuntimeError as e:
        print(e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
