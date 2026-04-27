#!/usr/bin/env bash
set -euo pipefail

# ===== 配置区 =====
# 这个脚本的目标：
#   将本仓库里的 vllm_dual/vllm Python 源码同步到某个 conda 环境中
#   已安装的 site-packages/vllm 目录里。
#
# 常用范式说明：
#   1. 用 ${VAR:-default} 读取环境变量，如果用户没有显式设置，就使用 default。
#      这样脚本既能开箱即用，也方便临时覆盖配置：
#        CONDA_ENV_NAME=adistill-dual bash sync_vllm.sh
#   2. 用脚本自身位置定位项目根目录，而不是依赖“你当前在哪个目录运行脚本”。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SRC_VLLM="${SRC_VLLM:-$SCRIPT_DIR/vllm_dual/vllm}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-adistill-unified}"

# 默认会同步新增的 Python 文件，例如 dual_worker.py。
# 如果你明确只想更新目标目录中已经存在的文件，可以这样运行：
#   SYNC_EXISTING_ONLY=1 bash sync_vllm.sh
# 注意：第一次安装 dual 版 vLLM 时不建议打开它，否则新增的 dual_worker.py
# 可能不会被复制到 site-packages/vllm/worker/ 下。
SYNC_EXISTING_ONLY="${SYNC_EXISTING_ONLY:-0}"
DRY_RUN="${DRY_RUN:-0}"
ASSUME_YES="${ASSUME_YES:-0}"
BACKUP_ROOT="${BACKUP_ROOT:-$SCRIPT_DIR/.sync_backups}"
SKIP_BACKUP="${SKIP_BACKUP:-0}"

if command -v conda >/dev/null 2>&1; then
  PYTHON_CMD=(conda run -n "$CONDA_ENV_NAME" python)
else
  PYTHON_CMD=("$HOME/miniconda3/envs/$CONDA_ENV_NAME/bin/python")
fi

# ===== 安全检查 =====
if [ ! -d "$SRC_VLLM" ]; then
  echo "源目录不存在: $SRC_VLLM"
  exit 1
fi

if ! DST_VLLM="$("${PYTHON_CMD[@]}" -c 'import pathlib, vllm; print(pathlib.Path(vllm.__file__).parent)' | tail -n 1)"; then
  echo "无法在 conda 环境 '$CONDA_ENV_NAME' 中 import vllm。"
  echo "请先确认该环境已经安装 vllm==0.8.5，或用 CONDA_ENV_NAME 指定正确环境。"
  exit 1
fi

if [ ! -d "$DST_VLLM" ]; then
  echo "目标 vllm 目录不存在: $DST_VLLM"
  exit 1
fi

REVERSE="${1:-0}"
if [ "$REVERSE" = "1" ] || [ "$REVERSE" = "--reverse" ]; then
  TMP="$SRC_VLLM"
  SRC_VLLM="$DST_VLLM"
  DST_VLLM="$TMP"
  echo "已启用反向同步：这会把环境里的 vLLM Python 文件同步回仓库，请确认你真的需要。"
fi

echo "conda 环境: $CONDA_ENV_NAME"
echo "从: $SRC_VLLM"
echo "到: $DST_VLLM"
echo "SYNC_EXISTING_ONLY=$SYNC_EXISTING_ONLY"
echo "DRY_RUN=$DRY_RUN"
echo "SKIP_BACKUP=$SKIP_BACKUP"

if [ "$ASSUME_YES" = "1" ]; then
  echo "ASSUME_YES=1，跳过交互确认。"
else
  read -p "确认同步吗？(y/n): " confirm
  if [ "$confirm" != "y" ]; then
    echo "同步已取消"
    exit 1
  fi
fi

# ===== 目标源码备份 =====
# 常用范式说明：
#   正式覆盖环境里的包源码前，先把目标目录中的 Python 源码备份出来。
#   这里故意只备份 .py / .pyi / py.typed：
#   - vLLM 的 CUDA/C++ 扩展是二进制文件，体积大且不应该由这个源码同步脚本改写。
#   - 本次同步主要覆盖 Python 调度逻辑，备份 Python 源码足够支持快速 diff/回退。
if [ "$DRY_RUN" != "1" ] && [ "$SKIP_BACKUP" != "1" ]; then
  BACKUP_DIR="$BACKUP_ROOT/vllm_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$BACKUP_DIR"
  echo "备份目标 Python 源码到: $BACKUP_DIR"
  rsync -a \
    --include='*/' \
    --include='*.py' \
    --include='*.pyi' \
    --include='py.typed' \
    --exclude='*' \
    "$DST_VLLM/" "$BACKUP_DIR/"
fi

# ===== rsync 同步 =====
RSYNC_ARGS=(
  -av
  --checksum \
  --exclude='*.so' \
  --exclude='*.o' \
  --exclude='*.a' \
  --exclude='*.bin' \
  --exclude='*.exe' \
  --exclude='*.dll' \
  --exclude='*.dylib' \
  --exclude='*.whl' \
  --exclude='*.pyc' \
  --exclude='__pycache__/' \
  --exclude='*.pyd'
)

if [ "$SYNC_EXISTING_ONLY" = "1" ]; then
  RSYNC_ARGS+=(--existing)
fi

if [ "$DRY_RUN" = "1" ]; then
  RSYNC_ARGS+=(--dry-run)
fi

rsync "${RSYNC_ARGS[@]}" "$SRC_VLLM/" "$DST_VLLM/"

echo "同步完成（二进制文件已保留）"
