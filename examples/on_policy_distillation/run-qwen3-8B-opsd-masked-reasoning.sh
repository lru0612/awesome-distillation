#!/bin/bash
# On-Policy Self-Distillation (OPSD) -- Answer-Only Teacher Variant
# -----------------------------------------------------------------------
# Same setup as run-qwen3-8B-opsd.sh, but the teacher privileged prompt
# contains ONLY the ground-truth final answer (no reference reasoning chain).
#
# Flag:  --opsd-teacher-info-mode answer_only
#
# Information density comparison:
#   full          : teacher sees full reference solution + step-by-step rephrasing
#   answer_only   : teacher sees ONLY the final answer  <-- THIS SCRIPT
#   masked_reasoning : teacher sees full solution; reasoning tokens partially masked
#
# Reference: "Self-Distilled Reasoner" (arXiv 2601.18734)
apt install iproute2 -y
set -ex
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=0
export NCCL_DEBUG=INFO

export PYTHONBUFFERED=16

source "/root/slime_siqi/scripts/models/qwen3-8B.sh"

###############################################################################
# Data (identical to run-qwen3-8B-opsd.sh -- reuse if already built)
###############################################################################

python3 -c "
import json, os, pathlib
try:
    from datasets import load_dataset
except ImportError as e:
    raise SystemExit('datasets package is required. pip install datasets') from e

DATA_DIR = pathlib.Path('/root/math/data')
DATA_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_REPO = 'open-thoughts/OpenThoughts-114k'
TRAIN_CONFIG = 'metadata'
EVAL_REPO = 'math-ai/aime25'
MAX_REFERENCE_SOLUTION_CHARS = int(os.environ.get('MAX_REFERENCE_SOLUTION_CHARS', '12000'))
BOXED_INSTRUCTION = '\n\nPlease solve the problem step by step. The last line must be of the form Answer: \\\\boxed{your_answer}.'

def normalize_text(v):
    if v is None: return ''
    s = str(v).strip()
    return '' if s.lower() in {'nan', 'none', 'null'} else s

print(f'Loading training dataset: {TRAIN_REPO}')
train_ds = load_dataset(TRAIN_REPO, TRAIN_CONFIG, split='train')
train_out = DATA_DIR / 'train_chat.jsonl'
train_written = 0
with train_out.open('w', encoding='utf-8') as fout:
    for row in train_ds:
        problem = normalize_text(row.get('problem'))
        if not problem: continue
        ground_truth_solution = normalize_text(row.get('ground_truth_solution'))
        reference_solution = ground_truth_solution
        if not reference_solution: continue
        if len(reference_solution) > MAX_REFERENCE_SOLUTION_CHARS:
            reference_solution = reference_solution[:MAX_REFERENCE_SOLUTION_CHARS] + '\n\n[TRUNCATED]'
        final_answer = ground_truth_solution
        label = ground_truth_solution
        if not label: continue
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
print(f'Created {train_out} with {train_written} entries')

print(f'Loading eval dataset: {EVAL_REPO}')
eval_ds = load_dataset(EVAL_REPO, split='test')
eval_out = DATA_DIR / 'test_chat.jsonl'
eval_written = 0
with eval_out.open('w', encoding='utf-8') as fout:
    for row in eval_ds:
        problem = normalize_text(row.get('problem'))
        answer = normalize_text(row.get('answer'))
        if not problem or not answer: continue
        entry = {
            'prompt': [{'role': 'user', 'content': problem + BOXED_INSTRUCTION}],
            'label': answer,
            'metadata': {'solution': answer, 'reference_solution': '', 'final_answer': answer, 'raw_problem': problem},
        }
        fout.write(json.dumps(entry, ensure_ascii=False) + '\n')
        eval_written += 1
print(f'Created {eval_out} with {eval_written} entries')
"

python3 -c "
import json, pathlib
src = pathlib.Path('/root/math/data/train_chat.jsonl')
dst = pathlib.Path('/root/math/data/train_opsd.jsonl')
with src.open() as fin, dst.open('w') as fout:
    for line in fin:
        obj = json.loads(line)
        metadata = obj.get('metadata') or {}
        raw_content = metadata.get('raw_problem') or ''
        if not raw_content:
            for msg in obj['prompt']:
                if msg['role'] == 'user':
                    raw_content = msg['content']
                    break
        metadata['raw_content'] = raw_content
        obj['metadata'] = metadata
        fout.write(json.dumps(obj, ensure_ascii=False) + '\n')
print(f'Created {dst}')
"

python3 -c "
import json, pathlib
src = pathlib.Path('/root/math/data/test_chat.jsonl')
dst = pathlib.Path('/root/math/data/test_chat_eval.jsonl')
count = 0
with src.open() as fin, dst.open('w') as fout:
    for line in fin:
        if count >= 100: break
        obj = json.loads(line)
        metadata = obj.get('metadata') or {}
        obj['label'] = metadata.get('final_answer') or metadata.get('solution') or obj.get('label', '')
        fout.write(json.dumps(obj, ensure_ascii=False) + '\n')
        count += 1
print(f'Created {dst} with {count} samples')
"

###############################################################################
# Training arguments
###############################################################################

CKPT_ARGS=(
   --hf-checkpoint /root/Qwen3-8B
   --ref-load "/root/Qwen3-8B_torch_dist"
   --save /root/slime_siqi/output/Qwen3-8B_opsd_answer_only/
   --save-interval 2000
   --ref-update-interval 20
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
   --rollout-max-response-len 2048
   --rollout-temperature 1.2
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
    --eval-interval 20
    --eval-config examples/on_policy_distillation/eval_config.yaml
    --log-passrate
    --skip-eval-before-train
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
   --opsd-loss-type reverse_kl    # choices: jsd | reverse_kl | forward_kl
   # --opsd-jsd-beta 0.5          # only used when --opsd-loss-type=jsd
   --opsd-pure-mode
   --opsd-use-ref-as-teacher
   # -------------------------------------------------------
   # KEY: teacher only provides the final answer, no CoT
   # -------------------------------------------------------
   --opsd-teacher-info-mode masked_reasoning
   --opsd-reasoning-mask-ratio 0.5
   --entropy-coef 0.00
   # -------------------------------------------------------
   # Training-inference mismatch: monitor + IS correction.
   # --get-mismatch-metrics: log mis_kl / ppl metrics to wandb.
   # --use-tis: apply IS weights to OPSD JSD loss (see loss.py).
   # -------------------------------------------------------
   --get-mismatch-metrics
   --use-tis
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
   --wandb-group qwen3-8B-opsd-masked-reasoning
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

echo "Starting Ray job (OPSD answer_only teacher)..."

RAY_GCS_PORT=6400
RAY_DASHBOARD_PORT=8269
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
unset RAY_ADDRESS
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,8,9

# Kill any leftover process on THIS cluster's ports only (do NOT touch 6380/8267).
# Uses Python for multi-method detection: ss, lsof, /proc/net/tcp (fuser may be absent).
echo "Cleaning up leftover processes on ports ${RAY_GCS_PORT} and ${RAY_DASHBOARD_PORT}..."
python3 - <<PYEOF
import subprocess, os, signal, re, socket, time, sys

def kill_port(port):
    killed = []
    # Method 1: ss (iproute2)
    try:
        r = subprocess.run(['ss', '-Htlnp', f'sport = :{port}'],
                           capture_output=True, text=True, timeout=5)
        for m in re.finditer(r'pid=(\d+)', r.stdout):
            pid = int(m.group(1))
            try: os.kill(pid, signal.SIGKILL); killed.append(('ss', pid))
            except ProcessLookupError: pass
    except Exception as e: print(f'[port {port}] ss: {e}', file=sys.stderr)
    # Method 2: lsof
    try:
        r = subprocess.run(['lsof', '-ti', f':{port}'],
                           capture_output=True, text=True, timeout=5)
        for s in set(r.stdout.split()):
            try: pid=int(s); os.kill(pid, signal.SIGKILL); killed.append(('lsof', pid))
            except (ValueError, ProcessLookupError): pass
    except Exception as e: print(f'[port {port}] lsof: {e}', file=sys.stderr)
    # Method 3: /proc/net/tcp (always available on Linux)
    try:
        hex_port = f'{port:04X}'
        with open('/proc/net/tcp') as f:
            inodes = {parts[9] for line in f if (parts:=line.split()) and
                      len(parts)>9 and parts[1].endswith(f':{hex_port}') and parts[3]=='0A'}
        for pid_dir in os.listdir('/proc'):
            if not pid_dir.isdigit(): continue
            try:
                fd_dir = f'/proc/{pid_dir}/fd'
                for fd in os.listdir(fd_dir):
                    lnk = os.readlink(f'{fd_dir}/{fd}')
                    m = re.search(r'socket:\[(\d+)\]', lnk)
                    if m and m.group(1) in inodes:
                        pid = int(pid_dir)
                        os.kill(pid, signal.SIGKILL)
                        killed.append(('/proc', pid))
                        break
            except (PermissionError, FileNotFoundError): pass
    except Exception as e: print(f'[port {port}] /proc: {e}', file=sys.stderr)
    return killed

import os as _os
GCS_PORT  = int(_os.environ.get('RAY_GCS_PORT',  '6400'))
DASH_PORT = int(_os.environ.get('RAY_DASHBOARD_PORT', '8269'))

for port in [GCS_PORT, DASH_PORT]:
    killed = kill_port(port)
    print(f'Port {port}: killed {killed}' if killed else f'Port {port}: already free')

time.sleep(5)

# Verify ports are actually free before proceeding
errors = 0
for port in [GCS_PORT, DASH_PORT]:
    s = socket.socket(); s.settimeout(1)
    if s.connect_ex(('127.0.0.1', port)) == 0:
        print(f'ERROR: port {port} still occupied after cleanup!', file=sys.stderr)
        errors += 1
    else:
        print(f'Port {port} confirmed free.')
    s.close()
sys.exit(errors)
PYEOF

ray start \
  --head \
  --port ${RAY_GCS_PORT} \
  --node-ip-address ${MASTER_ADDR} \
  --num-gpu 8 \
  --disable-usage-stats \
  --dashboard-host=0.0.0.0 \
  --dashboard-port=${RAY_DASHBOARD_PORT} \
  --dashboard-agent-grpc-port=52362 \
  --dashboard-agent-listen-port=52322 \
  --min-worker-port=20000 \
  --max-worker-port=29999

# Wait for the dashboard HTTP endpoint to come up.
echo "Waiting for Ray dashboard (port ${RAY_DASHBOARD_PORT}) to be reachable..."
for i in $(seq 1 36); do
  STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:${RAY_DASHBOARD_PORT}/api/jobs/ 2>/dev/null)
  if [ "$STATUS" = "200" ]; then
    echo "Dashboard reachable (attempt $i)."
    break
  fi
  echo "  attempt $i/36: HTTP ${STATUS} -- waiting 5s..."
  sleep 5
done
# Extra buffer for the job-submission agent to register with GCS.
sleep 15

set +e
# Retry ray job submit up to 10 times (agent may still be initialising).
for SUBMIT_ATTEMPT in $(seq 1 10); do
  echo "ray job submit attempt ${SUBMIT_ATTEMPT}/10 ..."
  ray job submit --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" \
     --runtime-env-json='{
       "env_vars": {
          "PYTHONPATH": "/root/Megatron-LM/",
          "CUDA_DEVICE_MAX_CONNECTIONS": "1",
          "CUDA_VISIBLE_DEVICES": "2,3,4,5,6,7,8,9"
       }
     }' \
     -- python3 train.py \
     --actor-num-nodes 1 \
     --actor-num-gpus-per-node 2 \
     --rollout-num-gpus 6 \
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
  if [ ${RAY_EXIT_CODE} -eq 0 ]; then
    break
  fi
  echo "Submit failed (code ${RAY_EXIT_CODE}), retrying in 15s..."
  sleep 15
done
set -e
echo "Ray job exited with code: ${RAY_EXIT_CODE}"
sleep 10

# Kill only THIS cluster's ports on exit -- do NOT touch the other experiment.
echo "Stopping Ray cluster on ports ${RAY_GCS_PORT}/${RAY_DASHBOARD_PORT}..."
fuser -k -TERM ${RAY_GCS_PORT}/tcp 2>/dev/null || true
fuser -k -TERM ${RAY_DASHBOARD_PORT}/tcp 2>/dev/null || true
sleep 5
fuser -k -KILL ${RAY_GCS_PORT}/tcp 2>/dev/null || true
fuser -k -KILL ${RAY_DASHBOARD_PORT}/tcp 2>/dev/null || true
