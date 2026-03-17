#!/bin/bash

# OPSDC: On-Policy Self-Distillation for Reasoning Compression (arXiv 2603.05433)
# Usage: bash examples/on_policy_distillation/run-qwen3-8B-opsdc.sh
#
# Key idea: The SAME model acts as both student and teacher.
#   - Student:  generates on-policy rollouts from the ORIGINAL prompt
#   - Teacher:  SAME model conditioned on a CONCISENESS INSTRUCTION + original prompt
#               (no ground-truth solution needed)
#   - Objective: minimize per-token reverse KL divergence between student and teacher
#
# This compresses reasoning traces without any ground-truth supervision.
# The compression signal emerges from the KL objective and adapts automatically
# to problem difficulty: easy problems compress aggressively, hard ones conservatively.
#
# Differences from OPSD (run-qwen3-8B-opsd.sh):
#   - Teacher prompt: conciseness instruction + student_user_content  (vs. problem + reference solution)
#   - Loss: reverse KL (KL(student || teacher))                       (vs. symmetric JSD)
#   - No ground-truth answers required for training data
#   - --opsd-use-ref-as-teacher + --ref-update-interval 50:
#     implements Algorithm 1's periodic teacher refresh (θ̄ ← θ every M=50 steps)
#   - Rollout temperature: 1.0  (vs 1.2 in OPSD)
#   - Max response length: 8192 (vs 1024 in OPSD)
#
# Reference: "On-Policy Self-Distillation for Reasoning Compression" (arXiv 2603.05433)

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

source "/root/slime_siqi/scripts/models/qwen3-1.7B.sh"

###############################################################################
# Step 0: Build JSONL data using the unified preprocess_dataset.py utility.
#
# Supported training datasets (set TRAIN_DATASET below):
#   BytedTsinghua-SIA/DAPO-Math-17k      -- format auto-detected as 'dapo'
#   open-thoughts/OpenThoughts-114k      -- format auto-detected as 'openthoughts'
#                                           (pass --config metadata)
#
# Supported eval datasets (set EVAL_DATASET below):
#   math-ai/aime25                       -- format auto-detected as 'simple'
#   math-ai/MATH-500                     -- format auto-detected as 'simple'
#   Any dataset with problem+answer fields
#
# OPSDC does not require ground-truth solutions for training (label is used
# only for logging accuracy).  The preprocessor stores student_user_content
# in metadata so the teacher prompt can be built as:
#   conciseness_instruction + student_user_content
###############################################################################

PREPROCESS="python3 examples/on_policy_distillation/preprocess_dataset.py"

# ---- Training dataset -------------------------------------------------------
# Switch between supported datasets by changing TRAIN_DATASET / TRAIN_CONFIG.
TRAIN_DATASET="${TRAIN_DATASET:-open-thoughts/OpenThoughts-114k}"
TRAIN_CONFIG="${TRAIN_CONFIG:-metadata}"  # "metadata" for OpenThoughts-114k; leave empty for DAPO
TRAIN_OUT="/root/math/data/train_openthoughts.jsonl"

# Answer format style for training and eval, independently configurable.
#   "boxed"  – model wraps final answer in \boxed{}  (natural for thinking mode)
#   "answer" – model outputs "Answer: $Answer" on the last line (DAPO style)
#
# Note: DAPO datasets embed the Answer: format in their prompt text and ignore
# TRAIN_ANSWER_FORMAT (the preprocess script detects this and stores the correct
# format_instruction in metadata automatically).
#
# Examples:
#   DAPO train + boxed eval (default):
#     bash run-qwen3-8B-opsdc.sh
#   Both boxed:
#     TRAIN_ANSWER_FORMAT=boxed EVAL_ANSWER_FORMAT=boxed bash run-qwen3-8B-opsdc.sh
#   Both answer:
#     TRAIN_ANSWER_FORMAT=answer EVAL_ANSWER_FORMAT=answer bash run-qwen3-8B-opsdc.sh
TRAIN_ANSWER_FORMAT="${TRAIN_ANSWER_FORMAT:-boxed}"
EVAL_ANSWER_FORMAT="${EVAL_ANSWER_FORMAT:-boxed}"

TRAIN_ARGS=(--dataset "$TRAIN_DATASET" --split train --output "$TRAIN_OUT" --answer-format "$TRAIN_ANSWER_FORMAT")
[ -n "$TRAIN_CONFIG" ] && TRAIN_ARGS+=(--config "$TRAIN_CONFIG")
$PREPROCESS "${TRAIN_ARGS[@]}"

# ---- Eval datasets ----------------------------------------------------------
# Four competition benchmarks used as validation sets.
# Dataset notes:
#   aime24:    math-ai/aime24     – answer in `solution` as \boxed{N}; auto-extracted by generic format
#   aime25:    math-ai/aime25     – answer in `answer` as plain integer
#   hmmt:      FlagEval/HMMT_2025 – answer in `answer`, mixed (int/frac/expr); only `train` split exists
#   amo_bench: meituan-longcat/AMO-Bench – answer in `answer`, highly variable; `answer_type` field present
#              (descriptive/set answers will score 0 — acceptable for trend monitoring)
$PREPROCESS --dataset math-ai/aime24             --split test  --output /root/math/data/eval_aime24.jsonl    --answer-format "$EVAL_ANSWER_FORMAT"
$PREPROCESS --dataset math-ai/aime25             --split test  --output /root/math/data/eval_aime25.jsonl    --answer-format "$EVAL_ANSWER_FORMAT"
$PREPROCESS --dataset FlagEval/HMMT_2025         --split train --output /root/math/data/eval_hmmt.jsonl      --answer-format "$EVAL_ANSWER_FORMAT"
$PREPROCESS --dataset meituan-longcat/AMO-Bench  --split test  --output /root/math/data/eval_amo_bench.jsonl --answer-format "$EVAL_ANSWER_FORMAT"
$PREPROCESS --dataset HuggingFaceH4/MATH-500     --split test  --output /root/math/data/eval_math500.jsonl   --answer-format "$EVAL_ANSWER_FORMAT"


###############################################################################
# Training arguments
###############################################################################

CKPT_ARGS=(
   --hf-checkpoint Qwen/Qwen3-1.7B
   --ref-load "/root/checkpoints_siqi/Qwen3-1.7B_torch_dist"
   --save /root/slime_siqi/output/Qwen3-1.7B_opsdc_slime/
   --save-interval 2000
   # Paper Algorithm 1: teacher is synced with student every M=50 training steps.
   # --ref-update-interval controls how often the "ref" weight backup is refreshed.
   # Combined with --opsd-use-ref-as-teacher, this implements the periodic refresh.
   # --ref-update-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data "$TRAIN_OUT"
   --input-key prompt
   --label-key label
   --apply-chat-template
   --apply-chat-template-kwargs '{"enable_thinking":false}'
   --rollout-shuffle
   --num-rollout 300
   --rollout-batch-size 32
   --n-samples-per-prompt 1
   --rollout-max-response-len 8192
   --rollout-temperature 1.0
   --over-sampling-batch-size 64

   --global-batch-size 32
   --balance-data
)

RM_ARGS=(
   # Reuse the same reward_func/post_process_rewards as OPSD.
   # reward_func computes math accuracy for logging only (no training signal).
   # post_process_rewards builds teacher tokens using --opsd-teacher-info-mode=conciseness.
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
   --max-tokens-per-gpu 4096
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-opd
   --opd-type opsd
   --opd-kl-coef 0.0

   # OPSDC-specific: use conciseness instruction as teacher privileged context
   # Teacher = same model + conciseness instruction prepended to problem
   # Student = same model + problem only (no conciseness instruction)
   --opsd-teacher-info-mode answer_only

   # Reverse KL: KL(student || teacher) -- mode-seeking, per OPSDC paper Eq.(1)
   --opsd-loss-type forward_kl
   --opsd-jsd-coef 1.0           # α: teacher KL 权重，与 pg_loss 并存时适当降低

   # pg_loss 正常参与训练
   # 最终损失 = pg_loss + α·KL_teacher + β·KL_ref - ε·entropy

   --opsd-pure-mode
   # --opsd-pure-mode 已移除：pg_loss 正常参与训练
   # 最终损失 = pg_loss + α·KL_teacher + β·KL_ref - ε·entropy
   # 若需恢复纯蒸馏模式（无 pg_loss），可重新加回 --opsd-pure-mode

   # Paper Algorithm 1: θ̄ ← θ every M steps.
   # --opsd-use-ref-as-teacher: teacher uses the "ref" weight backup (not live weights).
   # --ref-update-interval 50:  refresh that backup every 50 rollout steps.
   # Together these implement exactly the periodic teacher refresh from the paper.
   # Without this flag, teacher = live student weights at every step (M=1 approx),
   # which loses the stabilizing effect of a frozen teacher window.
   --opsd-use-ref-as-teacher
   --entropy-coef 0.00

   # --- opsd + 原来的KL 的实验 ---
   --use-kl-loss
   --kl-loss-coef 0.01       # 这是你的 beta，按需调整
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style cosine
   --lr-warmup-fraction 0.1
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-dev
   --wandb-group qwen3-1.7B-opsdc-forward_kl+_ref_kl_answeronly_openthought
   --wandb-key 2ed6f8544ac3e30d5c08879166cc10d9c6232448
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


echo "Starting Ray job..."

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
unset RAY_ADDRESS
ray stop --force || true
export CUDA_VISIBLE_DEVICES=1,2,3,4
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 4 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265


set +e
echo "Submitting Ray job..."
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{
     "env_vars": {
        "PYTHONPATH": "/root/Megatron-LM/",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "CUDA_VISIBLE_DEVICES": "2,3,4,5"
     }
   }' \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 2 \
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
