#!/bin/bash
#
# GPU 监控脚本：监控指定 GPU（可多个），当全部空闲时自动运行指定脚本
#
# 用法:
#   ./gpu_monitor_run.sh <GPU_IDS> <SCRIPT_PATH> [POLL_INTERVAL]
#
# 参数:
#   GPU_IDS      - 要监控的 GPU 编号，多个用逗号分隔 (如 0 或 0,1,2,3,4,5)
#   SCRIPT_PATH  - 全部 GPU 空闲时要执行的脚本路径
#   POLL_INTERVAL - 可选，轮询间隔秒数，默认 60
#
# 示例:
#   ./gpu_monitor_run.sh 0 ./0302-run-qwen3-8B-token-kl-weighted_inverse-openthoughts.sh
#   ./gpu_monitor_run.sh 0,1,2,3,4,5 ./0302-run-qwen3-8B-token-kl-weighted_inverse-openthoughts.sh
#   ./gpu_monitor_run.sh 0,1,2 /path/to/train.sh 120
#

set -e


GPU_IDS="0,1,2,3,4,5"
SCRIPT_PATH="examples/on_policy_distillation/0302-run-qwen3-8B-token-kl-weighted_inverse-openthoughts.sh"
POLL_INTERVAL="30"

# 校验脚本存在
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "[ERROR] 脚本不存在: $SCRIPT_PATH"
    exit 1
fi

# 校验 GPU 存在（nvidia-smi -i 支持逗号分隔的多个 GPU）
if ! nvidia-smi -i "$GPU_IDS" &>/dev/null; then
    echo "[ERROR] GPU $GPU_IDS 不存在或不可访问"
    exit 1
fi

# 检查指定 GPU 上是否有用户进程
# --query-compute-apps 返回使用 GPU 的进程，无输出表示全部空闲
is_gpu_free() {
    local procs
    procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$GPU_IDS" 2>/dev/null || echo "")
    # 去除空行和空白
    procs=$(echo "$procs" | tr -d ' \n')
    [ -z "$procs" ]
}

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 开始监控 GPU $GPU_IDS，轮询间隔 ${POLL_INTERVAL}s"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 目标脚本: $SCRIPT_PATH"
echo ""

while true; do
    if is_gpu_free; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU $GPU_IDS 已全部空闲，开始执行脚本..."
        echo ""
        # 限定脚本只使用指定的 GPU
        export CUDA_VISIBLE_DEVICES="$GPU_IDS"
        # 使用 bash 执行，确保可执行任意脚本
        exec bash "$SCRIPT_PATH"
        # exec 会替换当前进程，不会到达这里
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU $GPU_IDS 忙碌中，${POLL_INTERVAL}s 后重试..."
    sleep "$POLL_INTERVAL"
done
