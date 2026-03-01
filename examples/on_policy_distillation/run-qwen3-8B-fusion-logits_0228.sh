#!/bin/bash
#
# Logit Fusion RL training script — Qwen3-8B
#
# Algorithm:
#   - Rollout: token-by-token logit fusion (α·ℓ_T + (1-α)·ℓ_S) via custom HF
#     decode loop.  Stores log π_mix_old per token.
#   - Loss: IS-corrected REINFORCE with single-sided cap:
#       J(θ) = mean( min(r_i, 3.0) · A_i )  where  r_i = π_S / π_mix_old
#   - Advantages: GRPO (group normalisation per prompt).
#   - Alpha decays linearly from FUSION_ALPHA_INIT=0.5 over FUSION_ALPHA_K steps.
#
# Optional Method A (full-distribution KL at train time):
#   Uncomment the HOOK_ARGS block below to add teacher KL penalty.
#
# Differences from run-qwen3-8B-token-kl-weighted_inverse_0227.sh:
#   + --custom-generate-function-path   → lockstep HF decode
#   + --use-tis --tis-clip 3.0 --tis-clip-low 0.0
#   - Removed: --use-opd, --opd-*, --use-kl-loss, --kl-loss-*
#   - Removed: --custom-megatron-before-train-step-hook-path (base config)
#   Changed:  RM path → fusion_logits_reward
#             WANDB_GROUP, CKPT_SAVE_DIR
#
# Usage:
#   bash examples/on_policy_distillation/run-qwen3-8B-fusion-logits.sh

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_DIR="/root/awesome-distillation/output/run_log"
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +"%Y%m%d_%H%M")
LOG_FILE="${LOG_DIR}/${TIMESTAMP}_fusion_logits.log"
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

# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------
CKPT_SAVE_DIR="/root/output/Qwen3-8B_fusion_logits"
CKPT_ARGS=(
   --hf-checkpoint /root/models/Qwen3-8B
   --ref-load "/root/models/Qwen3-8B_torch_dist"
   --load "${CKPT_SAVE_DIR}"
   --save "${CKPT_SAVE_DIR}"
   --save-interval 20
   --max-save 1
)

# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------
ROLLOUT_ARGS=(
   --prompt-data /root/data/math/train_opsd.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
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

   # Logit Fusion: replace SGLang generate with lockstep HF decode
   --custom-generate-function-path \
       examples.on_policy_distillation.fusion_logits.fusion_logits_generate.generate_fusion
)

# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------
RM_ARGS=(
   --custom-rm-path \
       examples.on_policy_distillation.fusion_logits.fusion_logits_reward.reward_func
   --custom-reward-post-process-path \
       examples.on_policy_distillation.fusion_logits.fusion_logits_reward.post_process_rewards
   --reward-key math_reward
)

# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------
EVAL_ARGS=(
    --eval-interval 5
    --eval-config examples/on_policy_distillation/eval_config.yaml
    --log-passrate
)

# ---------------------------------------------------------------------------
# Performance / parallelism
# ---------------------------------------------------------------------------
PERF_ARGS=(
   --tensor-model-parallel-size 4
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

# ---------------------------------------------------------------------------
# GRPO + IS loss
# ---------------------------------------------------------------------------
GRPO_ARGS=(
   --advantage-estimator grpo

   # IS-corrected REINFORCE with single-sided upper cap (no PPO-style clipping)
   --use-tis
   --tis-clip 3.0
   --tis-clip-low 0.0

   --entropy-coef 0.00
)

# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------
OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 5e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-dev
   --wandb-group qwen3-8B-fusion-logits
   --wandb-key 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
)

# ---------------------------------------------------------------------------
# SGLang (engine still started but idle — custom generate bypasses it)
# ---------------------------------------------------------------------------
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.30
)
# NOTE: sglang-mem-fraction-static is lowered (0.78→0.30) because the HF model
# loaded by generate_fusion shares the same rollout GPUs.  Adjust based on
# available GPU memory (16 GB model + KV caches ≈ 20 GB per rollout GPU).

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --log-probs-chunk-size 512
)

# ---------------------------------------------------------------------------
# Optional Method A: full-distribution KL hook
# Uncomment the block below to add teacher KL penalty at training time.
# This runs a teacher forward pass on privileged context (= OPSD-style)
# and subtracts weighted KL from advantages alongside the IS loss.
# ---------------------------------------------------------------------------
# HOOK_ARGS=(
#    --custom-megatron-before-train-step-hook-path \
#        examples.on_policy_distillation.fusion_logits.fusion_logits_kl_hook.register_kl
# )
HOOK_ARGS=()

# ---------------------------------------------------------------------------
# Ray launch
# ---------------------------------------------------------------------------
echo "Starting Ray job..."

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
unset RAY_ADDRESS
ray stop --force || true
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
ray start --head \
    --node-ip-address ${MASTER_ADDR} \
    --num-gpus 6 \
    --disable-usage-stats \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265

set +e
echo "Submitting Ray job..."
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{
     "env_vars": {
        "PYTHONPATH": "/root/Megatron-LM/",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5",

        "FUSION_ALPHA_INIT": "0.5",
        "FUSION_ALPHA_K": "5000",

        "OPSD_JSD_BETA": "0.0",
        "OPSD_JSD_COEF": "1.0",

        "KL_WEIGHT_MODE": "",
        "KL_WEIGHT_TEMP": "1.0",
        "KL_CONFIDENCE_THRESHOLD": ""
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

#### clear after training
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python
