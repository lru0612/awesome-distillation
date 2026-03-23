#!/bin/bash
#
# GPU 监控脚本：自动从集群中找到 k 张空闲 GPU，设置 CUDA_VISIBLE_DEVICES 后启动目标脚本
#
# 用法:
#   ./gpu_monitor_run.sh <NUM_GPUS> <SCRIPT_PATH> [POLL_INTERVAL]
#
# 参数:
#   NUM_GPUS      - 需要的 GPU 数量
#   SCRIPT_PATH   - 全部 GPU 空闲时要执行的脚本路径
#   POLL_INTERVAL - 可选，轮询间隔秒数，默认 60
#
# 示例:
#   ./gpu_monitor_run.sh 4 examples/on_policy_distillation/0316-run-OLMO3-7B-openthoughts-jsd.sh
#   ./gpu_monitor_run.sh 4 examples/on_policy_distillation/0316-run-OLMO3-7B-openthoughts-jsd.sh 15
#

set -e

NUM_GPUS=${1:-4}
SCRIPT_PATH=${2:-"examples/on_policy_distillation/start.sh"}
POLL_INTERVAL=${3:-60}

# 校验脚本存在
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "[ERROR] 脚本不存在: $SCRIPT_PATH"
    exit 1
fi

# 自动发现系统中 GPU 总数
TOTAL_GPUS=$(nvidia-smi --list-gpus | wc -l)
if [ "$TOTAL_GPUS" -eq 0 ]; then
    echo "[ERROR] 未检测到任何 GPU"
    exit 1
fi

echo "[INFO] 检测到 ${TOTAL_GPUS} 张 GPU，需要 ${NUM_GPUS} 张空闲"

# 显存占用阈值（MiB），低于此值才视为空闲
MEM_THRESHOLD_MIB=1000

# 遍历所有 GPU，收集空闲的，若满足数量要求则返回逗号分隔的编号字符串
# 返回值写入全局变量 FOUND_GPUS
find_free_gpus() {
    local free_list=()
    local gpu mem_used
    for gpu in $(seq 0 $((TOTAL_GPUS - 1))); do
        mem_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" 2>/dev/null)
        if [ $? -ne 0 ] || [ -z "$mem_used" ]; then
            echo "[WARN] GPU $gpu 显存查询失败，跳过"
            continue
        fi
        mem_used=$(echo "$mem_used" | tr -d ' ')
        if [ "$mem_used" -lt "$MEM_THRESHOLD_MIB" ]; then
            free_list+=("$gpu")
        else
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU $gpu 显存占用 ${mem_used} MiB > 阈值 ${MEM_THRESHOLD_MIB} MiB，忙碌"
        fi
        if [ "${#free_list[@]}" -ge "$NUM_GPUS" ]; then
            break
        fi
    done

    if [ "${#free_list[@]}" -ge "$NUM_GPUS" ]; then
        FOUND_GPUS=$(IFS=','; echo "${free_list[*]:0:$NUM_GPUS}")
        return 0
    else
        FOUND_GPUS=""
        return 1
    fi
}

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 开始监控，需要 ${NUM_GPUS} 张空闲 GPU，轮询间隔 ${POLL_INTERVAL}s"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 目标脚本: $SCRIPT_PATH"
echo ""

while true; do
    if find_free_gpus; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 找到 ${NUM_GPUS} 张空闲 GPU: ${FOUND_GPUS}，开始执行脚本..."
        echo ""
        export CUDA_VISIBLE_DEVICES="$FOUND_GPUS"
        exec bash "$SCRIPT_PATH"
        # exec 会替换当前进程，不会到达这里
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 空闲 GPU 不足 ${NUM_GPUS} 张，${POLL_INTERVAL}s 后重试..."
    sleep "$POLL_INTERVAL"
done
