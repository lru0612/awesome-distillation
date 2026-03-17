#!/bin/bash
#
# Tau-Bench On-Policy Self-Distillation (OPSD)
#
# Teacher = same model conditioned on [wiki + efficiency guideline + first_user_turn]
# Student = same model conditioned on [wiki + first_user_turn]   (standard)
#
# Learning signal: KL(teacher || student) on agent-generated tokens (loss_mask=1),
# weighted by GRPO tau-bench task success advantage.
#
# Compared to run_qwen3_4B.sh, this script adds:
#   --custom-rm-path / --custom-reward-post-process-path  (tau_opsd_reward.py)
#   OPSD training args (--use-opd, --opd-type opsd, --opsd-loss-type reverse_kl, ...)
#

# pkill -9 sglang || true
sleep 2
ray stop --force || true
# pkill -9 ray || true
# pkill -9 python || true
sleep 2

set -ex
export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
HAS_NVLINK=$([ "$NVLINK_COUNT" -gt 0 ] && echo 1 || echo 0)
echo "HAS_NVLINK: $HAS_NVLINK"

source "/root/slime_siqi/scripts/models/qwen3-1.7B.sh"

###############################################################################
# Checkpoint / save paths
###############################################################################


CKPT_ARGS=(
    --hf-checkpoint "/root/checkpoints_siqi/Qwen3-1.7B"
    --ref-load "/root/checkpoints_siqi/Qwen3-1.7B_torch_dist"
    --save "/root/slime_siqi/output/Qwen3-1.7B_pi_safety/"
    --save-interval 500
)

###############################################################################
# Rollout
###############################################################################

ROLLOUT_ARGS=(
    --prompt-data /root/tau-bench/retail_train_tasks.jsonl
    --input-key index
    --rollout-shuffle
    --num-rollout 500
    --rollout-batch-size 32
    --n-samples-per-prompt 1
    --rollout-max-response-len 2048
    --rollout-max-context-len 40960
    --rollout-temperature 1.0
    --balance-data
)

###############################################################################
# Reward + OPSD teacher construction
###############################################################################

RM_ARGS=(
    --custom-reward-post-process-path examples.tau-bench.tau_opsd_reward.post_process_rewards
    --reward-key tau_reward
)

###############################################################################
# OPSD / GRPO
#
# --opsd-pure-mode: training signal is KL only (task reward used for GRPO
#   normalisation only, not added to loss). Remove this flag to also
#   directly optimise task reward alongside KL distillation.
###############################################################################

GRPO_ARGS=(
    --advantage-estimator grpo
    --use-opd
    --opd-type opsd
    --opd-kl-coef 0.0
    --opsd-loss-type reverse_kl
    --opsd-jsd-coef 1.0
    --opsd-pure-mode
    --use-kl-loss
    --kl-loss-coef 0.01
    --opsd-use-ref-as-teacher
    --entropy-coef 0.00
)

###############################################################################
# Eval
###############################################################################

EVAL_ARGS=(
    --eval-interval 5
    --eval-prompt-data retail-dev /root/tau-bench/retail_dev_tasks.jsonl
    --eval-prompt-data retail-test /root/tau-bench/retail_test_tasks.jsonl
    --n-samples-per-eval-prompt 1
    --eval-max-response-len 2048
    --eval-top-k 1
)

###############################################################################
# Performance / parallelism
###############################################################################

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
    --use-dynamic-batch-size
    --max-tokens-per-gpu 9216
)

###############################################################################
# Optimiser
###############################################################################

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
)

###############################################################################
# Custom generate (tau-bench interactive env)
###############################################################################

CUSTOM_ARGS=(
    --custom-generate-function-path examples.tau-bench.generate_with_tau.generate
)

###############################################################################
# W&B  (uncomment to enable)
###############################################################################

WANDB_ARGS=(
    --use-wandb
    --wandb-project slime-dev
    --wandb-group qwen3-1.7B-tau-bench-opsd
    --wandb-key 2ed6f8544ac3e30d5c08879166cc10d9c6232448
)

###############################################################################
# SGLang
###############################################################################

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 1
    --sglang-mem-fraction-static 0.7
)

###############################################################################
# Misc
###############################################################################

MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
)

###############################################################################
# Launch
###############################################################################

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
NUM_GPUS=6
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
ray start --head \
    --node-ip-address "${MASTER_ADDR}" \
    --num-gpus "${NUM_GPUS}" \
    --disable-usage-stats \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265 \
    --temp-dir "/tmp/ray_opsd_$(date +%s)"

# RUNTIME_ENV_JSON="{
#   \"env_vars\": {
#     \"PYTHONPATH\": \"/root/Megatron-LM/:${SCRIPT_DIR}\",
#     \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\"
#   }
# }"


RUNTIME_ENV_JSON=$(python3 -c "
import json, os
env = {
    'PYTHONPATH': '/root/Megatron-LM/',
    'CUDA_DEVICE_MAX_CONNECTIONS': '1',
    'CUDA_VISIBLE_DEVICES': '0,1,2,3,4,5',
}
print(json.dumps({'env_vars': env}))
")
echo "Ray runtime-env: ${RUNTIME_ENV_JSON}"

ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    --working-dir /root/slime_siqi \
    -- python3 train.py \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node 2 \
    --rollout-num-gpus 4 \
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
    ${CUSTOM_ARGS[@]} \
    ${RM_ARGS[@]}

RAY_EXIT_CODE=$?
set -e
echo "Ray job exited with code: ${RAY_EXIT_CODE}"
sleep 10

ray stop --force