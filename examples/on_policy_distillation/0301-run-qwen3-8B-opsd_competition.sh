#!/bin/bash

# On-Policy Self-Distillation (OPSD) with full-vocabulary JSD loss
# Usage: bash examples/on_policy_distillation/run-qwen3-8b-opsd.sh
#
# A single Qwen2.5-1.5B model acts as both teacher and student:
#   - Student: generates on-policy rollouts from the normal prompt
#   - Teacher: the SAME model conditioned on privileged context (prompt + ground-truth)
#   - Objective: minimize full-vocabulary JSD between student and teacher distributions
#
# No external sglang teacher server is needed. The teacher forward pass happens
# inside the training step using the same model weights (under torch.no_grad).
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

source "/root/slime/scripts/models/qwen3-8B.sh"

# Data preparation (first run only):
#   python examples/on_policy_distillation/data_prepare/prepare_competition_math.py \
#       --mode train --num-samples 7500 \
#       --output /root/data/competition/comp_math_train.jsonl --seed 42
#   python examples/on_policy_distillation/data_prepare/prepare_competition_math.py \
#       --mode eval --num-samples 500 \
#       --output /root/data/competition/comp_math_eval.jsonl \
#       --train-file /root/data/competition/comp_math_train.jsonl --seed 42

###############################################################################
# Training arguments
###############################################################################

CKPT_ARGS=(
   --hf-checkpoint /root/models/Qwen3-8B
   --ref-load "/root/models/Qwen3-8B_torch_dist"
   --save /root/output/Qwen3-8B_opsd_slime/
   --save-interval 20
   --max-save 1
)

ROLLOUT_ARGS=(
   --prompt-data /root/data/competition/comp_math_train.jsonl
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
)

RM_ARGS=(
   --custom-rm-path examples.on_policy_distillation.on_policy_self_distillation.reward_func
   --custom-reward-post-process-path examples.on_policy_distillation.on_policy_self_distillation.post_process_rewards
   --reward-key math_reward
)

EVAL_ARGS=(
    --eval-interval 5
    --eval-prompt-data comp_math /root/data/competition/comp_math_eval.jsonl
    --eval-max-response-len 2048
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
   --wandb-group qwen3-8B-opsd-origin
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


echo "Starting Ray job..."

# launch the master node of ray in container
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
        "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5"
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
