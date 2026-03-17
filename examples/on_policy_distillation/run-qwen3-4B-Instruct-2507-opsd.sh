#!/bin/bash

# On-Policy Self-Distillation (OPSD) with full-vocabulary JSD loss
# Usage: bash examples/on_policy_distillation/run-qwen3-4B-opsd.sh
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

source "/root/slime_siqi/scripts/models/qwen3-4B.sh"

###############################################################################
# Step 0: Build JSONL data from HuggingFace datasets
#   - Train: open-thoughts/OpenThoughts-114k (metadata split)
#   - Eval:  math-ai/aime25 (test split)
###############################################################################

python3 -c "
import json
import os
import pathlib

try:
    from datasets import load_dataset
except ImportError as e:
    raise SystemExit('datasets package is required. Please run: pip install datasets') from e

DATA_DIR = pathlib.Path('/root/math/data')
DATA_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_REPO = 'open-thoughts/OpenThoughts-114k'
TRAIN_CONFIG = 'metadata'
EVAL_REPO = 'math-ai/aime25'
MAX_REFERENCE_SOLUTION_CHARS = int(os.environ.get('MAX_REFERENCE_SOLUTION_CHARS', '12000'))

BOXED_INSTRUCTION = (
    '\\n\\nPlease solve the problem step by step. '
    'The last line must be of the form Answer: \\\\boxed{your_answer}.'
)


def normalize_text(v):
    if v is None:
        return ''
    s = str(v).strip()
    if s.lower() in {'nan', 'none', 'null'}:
        return ''
    return s

print(f'Loading training dataset: {TRAIN_REPO} ({TRAIN_CONFIG})')
train_ds = load_dataset(TRAIN_REPO, TRAIN_CONFIG, split='train')
train_out = DATA_DIR / 'train_chat.jsonl'
train_written = 0
train_ref_truncated = 0
with train_out.open('w', encoding='utf-8') as fout:
    for row in train_ds:
        problem = normalize_text(row.get('problem'))
        if not problem:
            continue

        # Use ground-truth solution directly for both privileged info and grading label.
        ground_truth_solution = normalize_text(row.get('ground_truth_solution'))
        reference_solution = ground_truth_solution
        if not reference_solution:
            continue
        if len(reference_solution) > MAX_REFERENCE_SOLUTION_CHARS:
            reference_solution = reference_solution[:MAX_REFERENCE_SOLUTION_CHARS] + '\n\n[TRUNCATED]'
            train_ref_truncated += 1

        final_answer = ground_truth_solution
        label = ground_truth_solution
        if not label:
            continue

        entry = {
            'prompt': [{'role': 'user', 'content': problem + BOXED_INSTRUCTION}],
            'label': label,
            'metadata': {
                'solution': final_answer or reference_solution,
                'reference_solution': reference_solution,
                'final_answer': final_answer,
                'raw_problem': problem,
                'domain': normalize_text(row.get('domain')),
                'source': normalize_text(row.get('source')),
            },
        }
        fout.write(json.dumps(entry, ensure_ascii=False) + '\n')
        train_written += 1
print(
    f'Created {train_out} with {train_written} entries '
    f'(reference_solution truncated={train_ref_truncated}, '
    f'max_chars={MAX_REFERENCE_SOLUTION_CHARS})'
)

print(f'Loading eval dataset: {EVAL_REPO} (test)')
eval_ds = load_dataset(EVAL_REPO, split='test')
eval_out = DATA_DIR / 'test_chat.jsonl'
eval_written = 0
with eval_out.open('w', encoding='utf-8') as fout:
    for row in eval_ds:
        problem = normalize_text(row.get('problem'))
        answer = normalize_text(row.get('answer'))
        if not problem or not answer:
            continue
        entry = {
            'prompt': [{'role': 'user', 'content': problem + BOXED_INSTRUCTION}],
            'label': answer,
            'metadata': {
                'solution': answer,
                'reference_solution': '',
                'final_answer': answer,
                'raw_problem': problem,
                'id': normalize_text(row.get('id')),
            },
        }
        fout.write(json.dumps(entry, ensure_ascii=False) + '\n')
        eval_written += 1
print(f'Created {eval_out} with {eval_written} entries')
"

###############################################################################
# Preprocess data (add raw_content for OPSD, prepare eval data)
###############################################################################

# Add raw_content to metadata so the OPSD reward function can reconstruct
# the privileged teacher prompt (original question + ground truth)
python3 -c "
import json, pathlib

src = pathlib.Path('/root/math/data/train_chat.jsonl')
dst = pathlib.Path('/root/math/data/train_opsd.jsonl')

with src.open() as fin, dst.open('w') as fout:
    for line in fin:
        obj = json.loads(line)
        metadata = obj.get('metadata') or {}
        # Prefer raw_problem if provided by preprocessing; fallback to prompt text.
        raw_content = metadata.get('raw_problem') or ''
        if not raw_content:
            for msg in obj['prompt']:
                if msg['role'] == 'user':
                    raw_content = msg['content']
                    break
        metadata['raw_content'] = raw_content
        obj['metadata'] = metadata
        fout.write(json.dumps(obj, ensure_ascii=False) + '\n')
print(f'Created {dst} with {sum(1 for _ in dst.open())} entries')
"

# Preprocess eval data (only keep first 100 samples for fast eval)
python3 -c "
import json, pathlib
src = pathlib.Path('/root/math/data/test_chat.jsonl')
dst = pathlib.Path('/root/math/data/test_chat_eval.jsonl')
MAX_EVAL_SAMPLES = 100
count = 0
with src.open() as fin, dst.open('w') as fout:
    for line in fin:
        if count >= MAX_EVAL_SAMPLES:
            break
        obj = json.loads(line)
        metadata = obj.get('metadata') or {}
        obj['label'] = metadata.get('final_answer') or metadata.get('solution') or obj.get('label', '')
        fout.write(json.dumps(obj, ensure_ascii=False) + '\n')
        count += 1
print(f'Created {dst} with {count} samples (capped at {MAX_EVAL_SAMPLES})')
"


###############################################################################
# Training arguments
###############################################################################

CKPT_ARGS=(
   --hf-checkpoint /root/Qwen3-4B
   --ref-load "/root/Qwen3-4B_torch_dist"
   --save /root/slime/output/Qwen3-4B_opsd_slime/
   --save-interval 2000
)

ROLLOUT_ARGS=(
   --prompt-data /root/math/data/train_opsd.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --apply-chat-template-kwargs '{"enable_thinking":false}'
   --rollout-shuffle
   --num-rollout 300
   --rollout-batch-size 32
   --n-samples-per-prompt 1
   --rollout-max-response-len 1024
   --rollout-temperature 1.1
   --over-sampling-batch-size 64

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
   --opsd-jsd-coef 1.0
   --opsd-jsd-beta 0.5
   --opsd-pure-mode
   --opsd-use-ref-as-teacher
   --entropy-coef 0.00
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-5
   --lr-decay-style cosine
   --lr-warmup-fraction 0.1
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-dev
   --wandb-group qwen3-4B-opsd-jsd
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
   --log-probs-chunk-size 256
)


echo "Starting Ray job..."

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export RAY_TMPDIR=${RAY_TMPDIR:-"/dev/shm/ray_tmp"}
mkdir -p "${RAY_TMPDIR}"
unset RAY_ADDRESS
ray stop --force || true
export CUDA_VISIBLE_DEVICES=6,7,8,9
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 4 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265 --temp-dir "${RAY_TMPDIR}"


set +e
echo "Submitting Ray job..."
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{
     "env_vars": {
        "PYTHONPATH": "/root/Megatron-LM/",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "CUDA_VISIBLE_DEVICES": "6,7,8,9",
        "RAY_TMPDIR": "/dev/shm/ray_tmp"
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
