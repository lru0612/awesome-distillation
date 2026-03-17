#!/bin/bash

# usage: bash examples/on_policy_distillation/run-qwen3-8B-opd.sh

set -ex

ACTOR_NUM_NODES=1
ACTOR_NUM_GPUS_PER_NODE=2
ROLLOUT_NUM_GPUS=2
TOTAL_RAY_GPUS=$((ACTOR_NUM_NODES * ACTOR_NUM_GPUS_PER_NODE + ROLLOUT_NUM_GPUS))

# Start the teacher model server
TEACHER_IP="127.0.0.1" # Use localhost here, you can change it to your IP
TEACHER_PORT=${TEACHER_PORT:-13141}
LOG_FILE="/tmp/sglang_$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c 6).log"
is_port_in_use() {
    python3 - "$1" <<'PY'
import socket
import sys

port = int(sys.argv[1])
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    s.bind(("0.0.0.0", port))
except OSError:
    print("1")
else:
    print("0")
finally:
    s.close()
PY
}

echo "Starting teacher model server..."

if curl -sf --max-time 2 "http://$TEACHER_IP:$TEACHER_PORT/health_generate" > /dev/null; then
    echo "Teacher model server already running at $TEACHER_IP:$TEACHER_PORT. Reusing it."
else
    # If target port is occupied by another process, pick the next free port.
    PORT_PROBE_TRIES=20
    PORT_PROBE_COUNT=0
    while [ "$(is_port_in_use "$TEACHER_PORT")" = "1" ]; do
        if curl -sf --max-time 2 "http://$TEACHER_IP:$TEACHER_PORT/health_generate" > /dev/null; then
            echo "Teacher model server already running at $TEACHER_IP:$TEACHER_PORT. Reusing it."
            break
        fi
        PORT_PROBE_COUNT=$((PORT_PROBE_COUNT + 1))
        if [ "$PORT_PROBE_COUNT" -ge "$PORT_PROBE_TRIES" ]; then
            echo "ERROR: failed to find an available teacher port after ${PORT_PROBE_TRIES} attempts."
            exit 1
        fi
        echo "Port $TEACHER_PORT is in use by another process, trying $((TEACHER_PORT + 1))..."
        TEACHER_PORT=$((TEACHER_PORT + 1))
    done

    if curl -sf --max-time 2 "http://$TEACHER_IP:$TEACHER_PORT/health_generate" > /dev/null; then
        echo "Teacher model server already running at $TEACHER_IP:$TEACHER_PORT. Reusing it."
    else
    CUDA_VISIBLE_DEVICES=5 python3 -m sglang.launch_server \
        --model-path /root/Qwen3-8B \
        --host 0.0.0.0 \
        --port $TEACHER_PORT \
        --tp 1 \
        --chunked-prefill-size 4096 \
        --mem-fraction-static 0.6 \
        > "$LOG_FILE" 2>&1 &
    TEACHER_PID=$!
    echo "Teacher model server PID: $TEACHER_PID, log: $LOG_FILE"

    MAX_WAIT_SECONDS=600
    ELAPSED=0
    until curl -sf --max-time 2 "http://$TEACHER_IP:$TEACHER_PORT/health_generate" > /dev/null; do
        if ! kill -0 "$TEACHER_PID" 2>/dev/null; then
            echo "ERROR: Teacher model server exited before becoming healthy."
            tail -n 50 "$LOG_FILE" || true
            exit 1
        fi
        if [ "$ELAPSED" -ge "$MAX_WAIT_SECONDS" ]; then
            echo "ERROR: Teacher model server not ready after ${MAX_WAIT_SECONDS}s."
            tail -n 50 "$LOG_FILE" || true
            exit 1
        fi
        echo "Waiting for teacher model server... (${ELAPSED}s/${MAX_WAIT_SECONDS}s)"
        tail -n 10 "$LOG_FILE" || true
        sleep 5
        ELAPSED=$((ELAPSED + 5))
    done
    fi
fi

curl http://$TEACHER_IP:$TEACHER_PORT/get_model_info
echo "Teacher model server is up and running at $TEACHER_IP:$TEACHER_PORT."
sleep 10


export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

source "/root/slime_siqi/scripts/models/qwen3-4B.sh"


CKPT_ARGS=(
   --hf-checkpoint /root/Qwen3-4B
   --ref-load /root/Qwen3-4B_torch_dist
   --load /root/Qwen3-4B_slime/
   --save /root/Qwen3-4B_slime/
   --save-interval 20
   --ref-update-interval 20
)
# /root/dapo-math-17k/dapo-math-17k.jsonl
ROLLOUT_ARGS=(
   --prompt-data /root/math/data/train_opsd.jsonl
   --input-key prompt
   --apply-chat-template
   --rollout-shuffle
   --num-rollout 300
   --rollout-batch-size 16
   --n-samples-per-prompt 4
   --rollout-max-response-len 16384
   --rollout-temperature 1

   --global-batch-size 64
   --balance-data
)

RM_ARGS=(
   --custom-rm-path slime.rollout.on_policy_distillation.reward_func
   --custom-reward-post-process-path slime.rollout.on_policy_distillation.post_process_rewards
   --reward-key accuracy
   --eval-reward-key accuracy
   --rm-url http://$TEACHER_IP:$TEACHER_PORT/generate
)

# EVAL_ARGS=(
#    --eval-interval 5
#    --eval-prompt-data aime ${DATA_DIR}/aime-2024/aime-2024.jsonl
#    --n-samples-per-eval-prompt 1
#    --eval-max-response-len 16384
#    --eval-top-p 1
# )

EVAL_ARGS=(
    --eval-interval 5
    --eval-config examples/on_policy_distillation/eval_config.yaml
    --log-passrate
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 16384
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-opd
   --opd-type sglang
   --opd-kl-coef 1.0
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-dev
   --wandb-group qwen3-4B-osd
   --wandb-key 2ed6f8544ac3e30d5c08879166cc10d9c6232448
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.4
)


MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)




# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8265}
RAY_STARTED_BY_SCRIPT=0

export CUDA_VISIBLE_DEVICES=6,7,8,9

if ray status > /dev/null 2>&1; then
    echo "Ray cluster already running. Reusing existing cluster."
else
    ray start --head \
      --node-ip-address "${MASTER_ADDR}" \
      --num-gpus "${TOTAL_RAY_GPUS}" \
      --disable-usage-stats \
      --dashboard-host=0.0.0.0 \
      --dashboard-port "${RAY_DASHBOARD_PORT}"
    RAY_STARTED_BY_SCRIPT=1
fi

AVAILABLE_RAY_GPUS=$(python3 - <<'PY'
import ray
ray.init(address="auto")
print(int(ray.cluster_resources().get("GPU", 0)))
ray.shutdown()
PY
)

if [ "${AVAILABLE_RAY_GPUS}" -lt "${TOTAL_RAY_GPUS}" ]; then
    echo "ERROR: Ray cluster only has ${AVAILABLE_RAY_GPUS} GPUs, but this job requires ${TOTAL_RAY_GPUS} GPUs."
    echo "Please stop and restart Ray with enough GPUs, or reduce --actor-num-gpus-per-node / --rollout-num-gpus."
    exit 1
fi

ray job submit --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" \
   --runtime-env-json='{
     "env_vars": {
        "PYTHONPATH": "/root/Megatron-LM/",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1"
     }
   }' \
   -- python3 train.py \
   --actor-num-nodes "${ACTOR_NUM_NODES}" \
   --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}" \
   --rollout-num-gpus "${ROLLOUT_NUM_GPUS}" \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${RM_ARGS[@]}



####clear after training
pkill -9 sglang
sleep 3
if [ "${RAY_STARTED_BY_SCRIPT}" -eq 1 ]; then
    ray stop --force
    pkill -9 ray || true
    pkill -9 python || true
    sleep 3
    pkill -9 ray || true
    pkill -9 python || true
fi
