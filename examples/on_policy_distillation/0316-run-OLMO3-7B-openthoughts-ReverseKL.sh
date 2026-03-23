#!/bin/bash

# OPSD (Reverse KL) on DAPO-Math-17k + OpenThoughts mixed dataset, with OLMo3-7B
#
# Algorithm: OPSD with pure reverse KL loss.
#   --opsd-loss-type reverse_kl → KL(p_S ‖ p_T), mode-seeking.
# Token-level KL weighting: uniform (no weighting).
#
# Training data: DAPO-Math-17k + OpenThoughts merged into a single jsonl.
# Eval: AIME 2024, AIME 2025, HMMT 2025, AMO Bench
#
# GPU: 4 × 90 GB  (colocate: rollout + training on same 4 GPUs)
#
# Usage: bash examples/on_policy_distillation/0316-run-OLMO3-7B-openthoughts-ReverseKL.sh

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

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
source "${REPO_ROOT}/scripts/models/olmo3-7B.sh"

###############################################################################
# Step 0: Preprocess datasets
###############################################################################

PREPROCESS="python3 examples/on_policy_distillation/preprocess_dataset.py"
ANSWER_FORMAT="${ANSWER_FORMAT:-boxed}"

# ---- OpenThoughts training data ---------------------------------------------
$PREPROCESS --dataset open-thoughts/OpenThoughts-114k --config metadata --split train \
    --output /root/math/data/train_openthoughts.jsonl --answer-format "$ANSWER_FORMAT"

# ---- DAPO-Math-17k training data --------------------------------------------
# $PREPROCESS --dataset math-ai/DAPO-Math-17k --split train \
#     --output /root/math/data/train_dapo.jsonl --answer-format "$ANSWER_FORMAT"

# ---- Merge into mixed training file -----------------------------------------
# cat /root/math/data/train_dapo.jsonl /root/math/data/train_openthoughts.jsonl \
#     > /root/math/data/train_openthoughts.jsonl

# ---- Eval datasets ----------------------------------------------------------
$PREPROCESS --dataset math-ai/aime24             --split test  --output /root/math/data/eval_aime24.jsonl    --answer-format "$ANSWER_FORMAT"
$PREPROCESS --dataset math-ai/aime25             --split test  --output /root/math/data/eval_aime25.jsonl    --answer-format "$ANSWER_FORMAT"
$PREPROCESS --dataset FlagEval/HMMT_2025         --split train --output /root/math/data/eval_hmmt.jsonl      --answer-format "$ANSWER_FORMAT"
$PREPROCESS --dataset meituan-longcat/AMO-Bench  --split test  --output /root/math/data/eval_amo_bench.jsonl --answer-format "$ANSWER_FORMAT"
$PREPROCESS --dataset HuggingFaceH4/MATH-500     --split test  --output /root/math/data/eval_math500.jsonl   --answer-format "$ANSWER_FORMAT"

###############################################################################
# Training arguments
###############################################################################

CKPT_SAVE_DIR="/root/output/OLMo3-7B_opsd_reverseKL_openthoughts"
CKPT_ARGS=(
   --hf-checkpoint /root/models/Olmo-3-7B-Instruct
   --ref-load "/root/models/Olmo-3-7B-Instruct_torch_dist"
   --load "${CKPT_SAVE_DIR}"
   --save "${CKPT_SAVE_DIR}"
   --save-interval 20
   --max-save 1
)

ROLLOUT_ARGS=(
   --prompt-data /root/math/data/train_openthoughts.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --apply-chat-template-kwargs '{"enable_thinking":false}'
   --rollout-shuffle
   --num-rollout 100
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
   --custom-rm-path examples.on_policy_distillation.on_policy_self_distillation.reward_func
   --custom-reward-post-process-path examples.on_policy_distillation.on_policy_self_distillation.post_process_rewards
   --reward-key math_reward
)

EVAL_ARGS=(
    --eval-interval 20
    --eval-config examples/on_policy_distillation/eval_config.yaml
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
   --opsd-loss-type reverse_kl
   --opsd-jsd-coef 1.0
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
   --wandb-group olmo3-7B-opsd-dapo-openthoughts-ReverseKL
   --wandb-key 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
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.78
   --sglang-attention-backend triton
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --log-probs-chunk-size 512
)


echo "Starting Ray job..."

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
unset RAY_ADDRESS

# 从 gpu_monitor 注入的环境变量读取，未设置时 fallback 到默认值
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
export CUDA_VISIBLE_DEVICES

ray stop --force || true
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus "$NUM_GPUS" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

set +e
echo "Submitting Ray job..."
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="{
     \"env_vars\": {
        \"PYTHONPATH\": \"/root/Megatron-LM/\",
        \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
        \"CUDA_VISIBLE_DEVICES\": \"${CUDA_VISIBLE_DEVICES}\",

        \"KL_WEIGHT_MODE\": \"uniform\",
        \"KL_WEIGHT_TEMP\": \"1.0\",
        \"KL_CONFIDENCE_THRESHOLD\": \"\"
     }
   }" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 4 \
   --colocate \
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
