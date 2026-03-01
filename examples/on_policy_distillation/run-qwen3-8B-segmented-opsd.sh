#!/bin/bash

# Segmented On-Policy Self-Distillation (Seg-OPSD)
# Usage: bash examples/on_policy_distillation/run-qwen3-8B-segmented-opsd.sh
#
# Extends standard OPSD with:
#   1. Logical segmentation of student responses into reasoning steps.
#   2. Teacher confidence-based KL weighting (entropy-based or threshold mask).
#   3. Segment-position weighting (later segments can get higher weight).
#
# Architecture (same model serves as both teacher and student):
#   - Student: generates on-policy rollouts from the normal prompt
#   - Teacher: SAME model conditioned on privileged context (prompt + ground-truth)
#   - Objective: segment-weighted, confidence-weighted JSD
#
# New modules (in examples/on_policy_distillation/):
#   - segmenter.py             : response segmentation strategies
#   - weighted_kl.py           : teacher entropy → confidence weights / masks
#   - teacher_lookahead.py     : teacher context construction per segment
#   - segmented_distillation.py: reward func & post-process (builds segments)
#   - segmented_opsd_forward.py: custom forward (patched via hook)
#
# Configuration knobs (via env vars in runtime-env-json):
#   SEG_STRATEGY              : newline | fixed_length | token_ids
#   SEG_MIN_LEN               : min tokens per segment (default 10)
#   SEG_CHUNK_SIZE            : chunk size for fixed_length (default 128)
#   KL_WEIGHT_MODE            : inverse | exp_neg | linear | "" (disabled)
#   KL_WEIGHT_TEMP            : temperature τ for confidence weights
#   KL_CONFIDENCE_THRESHOLD   : entropy threshold (tokens above are masked)
#   SEG_WEIGHT_MODE           : uniform | linear_increasing | exponential_increasing

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

###############################################################################
# Step 0: Convert parquet files to JSONL (train_chat.jsonl & test_chat.jsonl)
###############################################################################

python3 -c "
import json, os, pathlib, sys

DATA_DIR = '/root/math/data'

try:
    import pyarrow.parquet as pq
except ImportError:
    print('pyarrow not found, trying pandas ...')
    import pandas as pd
    pq = None

parquet_files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith('.parquet'))
if not parquet_files:
    print(f'No parquet files in {DATA_DIR}, skipping conversion.')
    sys.exit(0)
print(f'Found parquet files: {parquet_files}')

train_files, test_files = [], []
for fname in parquet_files:
    lower = fname.lower()
    if 'test' in lower or 'val' in lower or 'eval' in lower:
        test_files.append(os.path.join(DATA_DIR, fname))
    else:
        train_files.append(os.path.join(DATA_DIR, fname))
print(f'  Train: {[os.path.basename(f) for f in train_files]}')
print(f'  Test:  {[os.path.basename(f) for f in test_files]}')

PROBLEM_COLS = ['problem', 'question', 'input', 'prompt', 'instruction', 'content', 'text']
SOLUTION_COLS = ['solution', 'answer', 'output', 'response', 'label', 'target']

def read_rows(path):
    if pq is not None:
        return pq.read_table(path).to_pylist()
    else:
        return pd.read_parquet(path).to_dict(orient='records')

def detect_cols(rows):
    keys = {k.lower(): k for k in rows[0].keys()}
    pcol = next((keys[c] for c in PROBLEM_COLS if c in keys), None)
    scol = next((keys[c] for c in SOLUTION_COLS if c in keys), None)
    if pcol is None:
        print(f'  ERROR: cannot detect problem column from {list(rows[0].keys())}')
        sys.exit(1)
    return pcol, scol

def convert(files, out_name):
    if not files:
        print(f'  No files for {out_name}, skipping.')
        return
    all_rows = []
    for f in files:
        rows = read_rows(f)
        all_rows.extend(rows)
        print(f'  Read {len(rows)} rows from {os.path.basename(f)}')
    pcol, scol = detect_cols(all_rows)
    print(f'  Using columns: problem=\"{pcol}\", solution=\"{scol}\"')
    out_path = os.path.join(DATA_DIR, out_name)
    with open(out_path, 'w', encoding='utf-8') as fout:
        for row in all_rows:
            problem = str(row[pcol])
            solution = str(row[scol]) if scol and row.get(scol) is not None else ''
            entry = {
                'prompt': [{'role': 'user', 'content': problem}],
                'label': solution,
                'metadata': {'solution': solution},
            }
            fout.write(json.dumps(entry, ensure_ascii=False) + '\n')
    print(f'  Created {out_path} with {len(all_rows)} entries')

convert(train_files, 'train_chat.jsonl')
convert(test_files, 'test_chat.jsonl')
print('Parquet -> JSONL conversion done.')
"

###############################################################################
# Preprocess data (add raw_content for OPSD, prepare eval data)
###############################################################################

python3 -c "
import json, pathlib

src = pathlib.Path('/root/math/data/train_chat.jsonl')
dst = pathlib.Path('/root/math/data/train_opsd.jsonl')

with src.open() as fin, dst.open('w') as fout:
    for line in fin:
        obj = json.loads(line)
        raw_content = ''
        for msg in obj['prompt']:
            if msg['role'] == 'user':
                raw_content = msg['content']
                break
        metadata = obj.get('metadata') or {}
        metadata['raw_content'] = raw_content
        obj['metadata'] = metadata
        fout.write(json.dumps(obj, ensure_ascii=False) + '\n')
print(f'Created {dst} with {sum(1 for _ in dst.open())} entries')
"

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
        obj['label'] = (obj.get('metadata') or {}).get('solution', '')
        fout.write(json.dumps(obj, ensure_ascii=False) + '\n')
        count += 1
print(f'Created {dst} with {count} samples (capped at {MAX_EVAL_SAMPLES})')
"


###############################################################################
# Training arguments
###############################################################################

CKPT_ARGS=(
   --hf-checkpoint /root/Qwen3-8B
   --ref-load "/root/Qwen3-8B_torch_dist"
   --save /root/slime/output/Qwen3-8B_segmented_opsd/
   --save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data /root/math/data/train_opsd.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --num-rollout 300
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
    --eval-config examples/on_policy_distillation/eval_config.yaml
    --log-passrate
)

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
   --wandb-group qwen3-8B-segmented-opsd
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

# Hook: monkey-patch _compute_opsd_jsd_in_forward with segmented version
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

        "SEG_STRATEGY": "newline",
        "SEG_MIN_LEN": "10",
        "SEG_CHUNK_SIZE": "128",

        "KL_WEIGHT_MODE": "inverse",
        "KL_WEIGHT_TEMP": "1.0",
        "KL_CONFIDENCE_THRESHOLD": "",

        "SEG_WEIGHT_MODE": "linear_increasing"
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
