#!/bin/bash
#
# On-Policy Self-Distillation — Prompt Internalization (Safety Edition)
#
# Key idea: The SAME model acts as both student and teacher.
#   - Student:  generates rollouts from the BARE user prompt (no constitution).
#   - Teacher:  SAME model conditioned on [constitution] + [bare prompt].
#   - Objective: KL(teacher || student), weighted by GRPO safety advantage.
#
# Over training the student internalises the alignment rules from the
# constitution, exhibiting safe behaviour even when the constitution is
# absent at inference time.
#
# Datasets (choose training difficulty level)
#
#   [easy]   PKU-SafeRLHF — straightforward harmful/benign prompts
#   [medium] BeaverTails 30k-train — broader harm categories, more varied phrasing
#   [hard]   WildGuardMix (allenai/wildguardmix) — adversarial/jailbreak-style prompts;
#                        models easily saturate easy datasets but struggle here
#
# Current training set: BeaverTails 30k-train  (TRAIN_DATA below)
# Eval: BeaverTails 30k-test + WildGuardMix test
#
# Preprocessing (run once before training):
#
#   # Easy (PKU-SafeRLHF):
#   python examples/prompt_internalization/preprocess_pi_safety.py \
#       --dataset PKU-Alignment/PKU-SafeRLHF --split train \
#       --output /root/pi_safety/train_pku.jsonl
#
#   # Medium — BeaverTails train:
#   python examples/prompt_internalization/preprocess_pi_safety.py \
#       --dataset PKU-Alignment/BeaverTails --split 30k-train \
#       --output /root/pi_safety/train_beavertails.jsonl

#   # Hard — WildGuardMix (adversarial jailbreaks, dataset: allenai/wildguardmix):
#   python examples/prompt_internalization/preprocess_pi_safety.py \
#       --dataset allenai/wildguardmix --config wildguardtrain --split train \
#       --output /root/pi_safety/train_wildguardmix.jsonl

#   # Eval — BeaverTails test:
#   python examples/prompt_internalization/preprocess_pi_safety.py \
#       --dataset PKU-Alignment/BeaverTails --split 30k-test \
#       --output /root/pi_safety/eval_beavertails.jsonl

#   # Eval — WildGuardMix test set:
#   python examples/prompt_internalization/preprocess_pi_safety.py \
#       --dataset allenai/wildguardmix --config wildguardtest \
#       --split test \
#       --output /root/pi_safety/eval_wildguardtest.jsonl

# LLM judge (optional, for higher-quality eval):
#   export JUDGE_MODE=llm
#   export JUDGE_API_BASE=http://0.0.0.0:30000/v1
#   export JUDGE_API_KEY=EMPTY
#   export JUDGE_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507

set -ex
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=0
export NCCL_DEBUG=INFO
export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
HAS_NVLINK=$([ "$NVLINK_COUNT" -gt 0 ] && echo 1 || echo 0)
echo "HAS_NVLINK: $HAS_NVLINK"

source "/root/slime_siqi/scripts/models/qwen3-4B.sh"

###############################################################################
# Checkpoint / save paths
###############################################################################

CKPT_ARGS=(
    --hf-checkpoint Qwen/Qwen3-4B
    --ref-load "/root/Qwen3-4B_torch_dist"
    --save /root/slime_siqi/output/Qwen3-4B_pi_safety/
    --save-interval 500
)

###############################################################################
# Rollout
###############################################################################

# Training data — switch path to select difficulty:
#   easy:   /root/pi_safety/train_pku.jsonl
#   medium: /root/pi_safety/train_beavertails.jsonl  (current)
#   hard:   /root/pi_safety/train_wildguardmix.jsonl
TRAIN_DATA=/root/pi_safety/train_wildguardmix.jsonl

ROLLOUT_ARGS=(
    --prompt-data ${TRAIN_DATA}
    --input-key prompt
    --label-key label
    --apply-chat-template
    # No chain-of-thought needed for safety tasks
    --apply-chat-template-kwargs '{"enable_thinking":false}'
    --rollout-shuffle
    --num-rollout 300
    --rollout-batch-size 32
    # Multiple samples per prompt give GRPO variance on binary safety rewards
    --n-samples-per-prompt 1
    --rollout-max-response-len 2048
    --rollout-temperature 1.0
    --over-sampling-batch-size 32

    --global-batch-size 32
    --balance-data
)

###############################################################################
# Reward
###############################################################################

RM_ARGS=(
    --custom-rm-path
        examples.prompt_internalization.safety_reward.reward_func
    --custom-reward-post-process-path
        examples.prompt_internalization.safety_reward.post_process_rewards
    --reward-key safety_reward
)

###############################################################################
# Eval  (in-training, BeaverTails test)
###############################################################################

EVAL_ARGS=(
    --eval-interval 10
    --eval-config examples/prompt_internalization/eval_config_pi_safety.yaml
    --log-passrate
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
    --max-tokens-per-gpu 4096
)

###############################################################################
# OPSD / GRPO
###############################################################################

GRPO_ARGS=(
    --advantage-estimator grpo
    --use-opd
    --opd-type opsd
    --opd-kl-coef 0.0

    # pi mode: teacher = constitution + student_user_content
    --opsd-teacher-info-mode pi

    # Wiener KL distillation loss (same as OPSDC)
    --opsd-loss-type forward_kl
    --opsd-jsd-coef 1.0
    --opsd-pure-mode


    # KL toward reference model for stability
    --use-kl-loss
    --kl-loss-coef 0.01

    --opd-kl-coef 0.0
    --entropy-coef 0.00

    # Periodic teacher refresh (Algorithm 1 in OPSD paper)
    --opsd-use-ref-as-teacher
    --entropy-coef 0.00
)

###############################################################################
# Optimiser
###############################################################################

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-6
    --lr-decay-style cosine
    --lr-warmup-fraction 0.1
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
)

###############################################################################
# W&B
###############################################################################

WANDB_ARGS=(
    --use-wandb
    --wandb-project slime-dev
    --wandb-group qwen3-4B-pi-safety
    --wandb-key 2ed6f8544ac3e30d5c08879166cc10d9c6232448
)

###############################################################################
# SGLang
###############################################################################

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 1
    --sglang-mem-fraction-static 0.78
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
    --log-probs-chunk-size 512
)

###############################################################################
# Launch
###############################################################################

echo "Starting Ray job..."
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
unset RAY_ADDRESS
ray stop --force || true
export CUDA_VISIBLE_DEVICES=1,2,7,8
ray start --head \
    --node-ip-address ${MASTER_ADDR} \
    --num-gpus 4 \
    --disable-usage-stats \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265

set +e
echo "Submitting Ray job..."

# Build runtime-env JSON dynamically so that judge-related env vars set on the
# host (JUDGE_MODE, JUDGE_API_BASE, JUDGE_API_KEY, JUDGE_MODEL) are
# forwarded into the Ray worker processes.
RUNTIME_ENV=$(python3 -c "
import json, os
env = {
    'PYTHONPATH': '/root/Megatron-LM/',
    'CUDA_DEVICE_MAX_CONNECTIONS': '1',
    'CUDA_VISIBLE_DEVICES': '1,2,7,8',
}
for key in ('JUDGE_MODE', 'JUDGE_API_BASE', 'JUDGE_API_KEY', 'JUDGE_MODEL'):
    val = os.environ.get(key)
    if val:
        env[key] = val
print(json.dumps({'env_vars': env}))
")
echo "Ray runtime-env: ${RUNTIME_ENV}"

ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="${RUNTIME_ENV}" \
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

ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python
