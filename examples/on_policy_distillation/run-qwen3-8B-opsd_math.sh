#!/bin/bash

# On-Policy Self-Distillation (OPSD) with combined PG + JSD + Ref-KL loss
# Usage: bash examples/on_policy_distillation/run-qwen3-8B-opsd.sh
#
# A single Qwen3-8B model acts as both student and teacher:
#   - Student:  generates on-policy rollouts from the ORIGINAL prompt
#   - Teacher:  SAME model conditioned on a PRIVILEGED prompt
#               (original problem + full reference solution, arXiv 2601.18734)
#   - Objective (combined):
#       L = pg_loss + α·JSD(student || teacher) + β·KL(student || ref)
#
# Key differences from pure-distillation (--opsd-pure-mode):
#   - pg_loss is active (no --opsd-pure-mode)
#   - Two KL terms act as soft regularisers alongside the RL signal
#   - Dataset must have full reference solutions → use OpenThoughts-114k
#
# Teacher prompt format (--opsd-teacher-info-mode full):
#   [original problem]
#   Here is a reference solution: [reference_solution]
#   After understanding the reference solution ... [transition prompt]
#   [format_instruction]          ← SAME suffix as student prompt ✓
#
# Student prompt format (built by preprocess_dataset.py):
#   [formatted problem with format_instruction suffix]
#
# Format instruction consistency:
#   preprocess_dataset.py stores metadata['format_instruction'] for both.
#   on_policy_self_distillation.py reads that same field for the teacher.
#   Both student and teacher therefore end with the identical format suffix.
#
# Teacher sequence length budget (Qwen3-8B max_position_embeddings = 32768):
#   teacher_prompt ≈ 1500–2600 tokens (problem + reference_solution + boilerplate)
#   student_response ≤ --rollout-max-response-len = 4096
#   teacher_total ≈ 5600–6700 tokens  →  well within 32 K ✓
#   (--max-tokens-per-gpu is lowered to 2048 to leave room for teacher pass)
#
# Reference: "Self-Distilled Reasoner" (arXiv 2601.18734)

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

source "/root/slime_siqi/scripts/models/qwen3-8B.sh"

###############################################################################
# Step 0: Build JSONL data using the unified preprocess_dataset.py utility.
#
# We use open-thoughts/OpenThoughts-114k (config=metadata) because it supplies
# full reference solutions (ground_truth_solution field).  These are needed for
# --opsd-teacher-info-mode=full: the teacher prompt embeds the entire solution.
#
# DAPO-Math-17k is NOT suitable here: its "ground_truth" field is only the
# final answer (no reasoning chain), so the teacher would receive almost no
# privileged information.
#
# Format notes:
#   --answer-format boxed   : student wraps final answer in \boxed{}
#                             metadata['format_instruction'] = "Please reason
#                             step by step, and put your final answer within
#                             \boxed{}."
#   The teacher prompt builder (on_policy_self_distillation.py) reads this
#   same field and appends it verbatim → student & teacher format are identical.
###############################################################################

PREPROCESS="python3 examples/on_policy_distillation/preprocess_dataset.py"

# ---- Training dataset -------------------------------------------------------
TRAIN_DATASET="${TRAIN_DATASET:-open-thoughts/OpenThoughts-114k}"
TRAIN_CONFIG="${TRAIN_CONFIG:-metadata}"
TRAIN_OUT="/root/math/data/train_opsd_full.jsonl"
TRAIN_ANSWER_FORMAT="${TRAIN_ANSWER_FORMAT:-boxed}"

TRAIN_ARGS=(--dataset "$TRAIN_DATASET" --config "$TRAIN_CONFIG" --split train --output "$TRAIN_OUT" --answer-format "$TRAIN_ANSWER_FORMAT")
$PREPROCESS "${TRAIN_ARGS[@]}"

# ---- Eval dataset -----------------------------------------------------------
EVAL_DATASET="${EVAL_DATASET:-math-ai/aime25}"
EVAL_CONFIG="${EVAL_CONFIG:-}"
EVAL_OUT="/root/math/data/eval_opsd_full.jsonl"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-100}"
EVAL_ANSWER_FORMAT="${EVAL_ANSWER_FORMAT:-boxed}"

EVAL_ARGS=(--dataset "$EVAL_DATASET" --split test --output "$EVAL_OUT" --max-samples "$MAX_EVAL_SAMPLES" --answer-format "$EVAL_ANSWER_FORMAT")
[ -n "$EVAL_CONFIG" ] && EVAL_ARGS+=(--config "$EVAL_CONFIG")
$PREPROCESS "${EVAL_ARGS[@]}"


###############################################################################
# Training arguments
###############################################################################

CKPT_ARGS=(
   --hf-checkpoint /root/Qwen3-8B
   --ref-load "/root/Qwen3-8B_torch_dist"
   --save /root/slime_siqi/output/Qwen3-8B_opsd_full_slime/
   --save-interval 2000
   # Paper Algorithm 1: teacher weights refreshed every M steps.
   # --opsd-use-ref-as-teacher uses the "ref" backup as frozen teacher.
   # --ref-update-interval 20:  sync backup every 20 rollout steps.
   --ref-update-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data /root/math/data/train_opsd_full.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   # enable_thinking=true: student generates full reasoning traces.
   # This is appropriate when distilling from full reference solutions.
   --apply-chat-template-kwargs '{"enable_thinking":true}'
   --rollout-shuffle
   --num-rollout 300
   --rollout-batch-size 32
   --n-samples-per-prompt 1
   # 4096 response tokens: teacher total ≈ teacher_prompt(~2000t) + 4096 ≈ 6100t
   # Increasing to 8192 is possible but requires further reducing max-tokens-per-gpu.
   --rollout-max-response-len 8192
   --rollout-temperature 1.2
   --over-sampling-batch-size 64

   --global-batch-size 32
   --balance-data
)

RM_ARGS=(
   # reward_func:          computes math accuracy (used for pg_loss and logging)
   # post_process_rewards: builds teacher tokens using --opsd-teacher-info-mode=full
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
   # Lowered from 4096 to 2048 to accommodate the extra teacher forward pass.
   # Each batch now processes both student sequences AND teacher sequences
   # (teacher_prompt ≈ 1500–2600 tokens longer than the student prompt).
   --max-tokens-per-gpu 2048
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-opd
   --opd-type opsd
   --opd-kl-coef 0.0

   # Full OPSD mode: teacher receives problem + complete reference solution.
   # Requires reference_solution in metadata (OpenThoughts-114k has this).
   --opsd-teacher-info-mode full

   # JSD loss: symmetric KL between student and teacher distributions.
   # β=0.5 is the standard 50/50 weighting (original OPSD paper default).
   --opsd-loss-type reverse_kl
#    --opsd-jsd-beta 0.5
   # α: teacher-KL weight. Reduced from 1.0 so pg_loss can contribute meaningfully.
   # Combined loss: pg_loss + 0.1·JSD + 0.01·KL_ref
   --opsd-jsd-coef 0.1

   # No --opsd-pure-mode: pg_loss is active alongside both KL terms.
   # Remove this comment and add --opsd-pure-mode to revert to distillation-only.

   # Frozen teacher: use ref model backup instead of live student weights.
   # Provides a stable distillation target (Algorithm 1 of arXiv 2601.18734).
   --opsd-use-ref-as-teacher
   --entropy-coef 0.00

   # β: ref-KL weight. Penalises drift from the initial policy (ref model).
   --use-kl-loss
   --kl-loss-coef 0.01
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
   --wandb-group qwen3-8B-opsd-full-pg+jsd+ref_kl
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
export CUDA_VISIBLE_DEVICES=6,7,8,9
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 4 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265


set +e
echo "Submitting Ray job..."
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{
     "env_vars": {
        "PYTHONPATH": "/root/Megatron-LM/",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "CUDA_VISIBLE_DEVICES": "6,7,8,9"
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
