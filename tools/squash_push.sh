#!/bin/sh
# 把工作区的改动并回仓库唯一那条提交，再强推上去。
#
# 这个仓库对外只保留一条提交（Initial commit），为的是让每个文件的历史里都不出现逐次
# 改动记录。所以日常改动不要新开提交，一律 amend 进那条提交再推。
#
# 用法:
#   tools/squash_push.sh            并回并推送
#   tools/squash_push.sh --dry-run  只看会改什么，不提交不推送
#
# 注意:
#   - 用 --force-with-lease，远端若被别人推过会直接失败，不会覆盖。
#   - `git add -A` 会带上所有未跟踪且没被忽略的文件；被 .gitignore 兜住的（例如
#     training/reports/**/runs/ 下的历史残留）不会被带进来。

set -eu

BRANCH="${FG_BRANCH:-main}"

case "${1:-}" in
    ""|--dry-run) ;;
    *) echo "用法: $0 [--dry-run]" >&2; exit 2 ;;
esac

DRY=0
if [ "${1:-}" = "--dry-run" ]; then
    DRY=1
fi

cd "$(git rev-parse --show-toplevel)"

n=$(git rev-list --count HEAD)
if [ "$n" -ne 1 ]; then
    echo "拒绝执行：当前分支有 $n 条提交，这个仓库应当只有 1 条。" >&2
    echo "先把历史收拢成一条，再回来用它。" >&2
    exit 1
fi

git add -A
if git diff --cached --quiet; then
    echo "没有改动，退出。"
    exit 0
fi

echo "并回这条提交:"
git log -1 --format='  %h  %s'
echo "本次改动:"
git diff --cached --stat | sed 's/^/  /'

if [ "$DRY" -eq 1 ]; then
    echo "--dry-run: 没有提交，也没有推送。"
    exit 0
fi

git commit --amend --no-edit
git push --force-with-lease origin "$BRANCH"

echo
echo "已推送:"
git log -1 --format='  %H%n  %an <%ae>%n  %s'
files=$(git ls-tree -r HEAD --name-only | wc -l | tr -d ' ')
echo "  跟踪文件 $files 个"
