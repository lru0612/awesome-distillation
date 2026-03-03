#!/bin/bash

# Token-KL-Weighted OPSD on OpenThoughts
#
# Same algorithm as 0301 (competition) but trains on OpenThoughts dataset.
# Eval on AIME 2024/2025, AMO Bench, HMMT 2025.
#
# Data: train uses /root/data/openthoughts_train.jsonl
#       eval uses aime2024, aime2025, amo_bench, hmmt2025

# ---------------------------------------------------------------------------
# Logging: tee all output (stdout + stderr) to a timestamped log file
# ---------------------------------------------------------------------------
LOG_DIR="/root/awesome-distillation/output/run_log"
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +"%Y%m%d_%H%M")
LOG_FILE="${LOG_DIR}/${TIMESTAMP}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "Logging to ${LOG_FILE}"

set -ex
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=0
export NCCL_DEBUG=INFO

export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

source "/root/awesome-distillation/scripts/models/qwen3-8B.sh"

# Use same path for save/load: enables auto-resume when checkpoint exists.
# On first run (no checkpoint): falls back to ref-load, trains from scratch.
CKPT_SAVE_DIR="/root/output/Qwen3-8B_token_kl_weighted_inverse_openthoughts"
CKPT_ARGS=(
   --hf-checkpoint /root/models/Qwen3-8B
   --ref-load "/root/models/Qwen3-8B_torch_dist"
   --load "${CKPT_SAVE_DIR}"
   --save "${CKPT_SAVE_DIR}"
   --save-interval 20
   --max-save 1
)

ROLLOUT_ARGS=(
   --prompt-data /root/data/openthoughts_train.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --num-rollout 200
   --rollout-batch-size 4
   --n-samples-per-prompt 8
   --rollout-max-response-len 2048
   --rollout-temperature 1.0
   --over-sampling-batch-size 64
   --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std

   --global-batch-size 32
   --balance-data
)

RM_ARGS=(
   --custom-rm-path examples.on_policy_distillation.segmented_distillation.reward_func
   --custom-reward-post-process-path examples.on_policy_distillation.segmented_distillation.post_process_rewards
   --reward-key math_reward
)

EVAL_ARGS=(
    --eval-interval 5
    --eval-prompt-data aime2024 /root/data/aime2024_eval.jsonl aime2025 /root/data/aime2025_eval.jsonl amo_bench /root/data/amo_bench_eval.jsonl hmmt2025 /root/data/hmmt2025_eval.jsonl
    --eval-max-response-len 4096
    --eval-top-p 1.0
    --n-samples-per-eval-prompt 1
    --log-passrate
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 4096
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-opd
   --opd-type opsd
   --opd-kl-coef 0.0
   --opsd-jsd-coef 1.0
   --opsd-jsd-beta 0.5
   --opsd-pure-mode
   --use-kl-loss
   --kl-loss-coef 0.05
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 5e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-dev
   --wandb-group qwen3-8B-token-kl-weighted-inverse-openthoughts
   --wandb-key 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
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.78
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --log-probs-chunk-size 512
)

HOOK_ARGS=(
   --custom-megatron-before-train-step-hook-path examples.on_policy_distillation.segmented_opsd_forward.register_segmented_opsd
)


echo "Starting Ray job..."

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
unset RAY_ADDRESS
ray stop --force || true
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 6 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

set +e
echo "Submitting Ray job..."
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{
     "env_vars": {
        "PYTHONPATH": "/root/Megatron-LM/",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5",

        "SEG_STRATEGY": "fixed_length",
        "SEG_CHUNK_SIZE": "4096",

        "KL_WEIGHT_MODE": "inverse",
        "KL_WEIGHT_TEMP": "1.0",
        "KL_CONFIDENCE_THRESHOLD": "",

        "SEG_WEIGHT_MODE": "uniform"
     }
   }' \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 4 \
   --rollout-num-gpus 2 \
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
   ${RM_ARGS[@]} \
   ${HOOK_ARGS[@]}

RAY_EXIT_CODE=$?
set -e
echo "Ray job exited with code: ${RAY_EXIT_CODE}"
sleep 10

####clear after training
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python
