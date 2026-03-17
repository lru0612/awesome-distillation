# Tau-Bench OPSD Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan.

**Goal:** 在 tau-bench 多轮 agentic 任务上实现 On-Policy Self-Distillation（OPSD），teacher 通过在 system prompt 里注入 privileged hint（"conciseness" 模式，无需外部 API）来引导 student 学习更高效的 tool-calling 策略。

**Architecture:**
- 在 rollout（`generate_with_tau.py` / `trainable_agents.py`）阶段把 teacher 构建所需的中间变量（wiki、initial_user_message、tools_info、initial_prompt_token_length）写入 `sample.metadata`。
- 新建 `tau_opsd_reward.py`，在 `post_process_rewards` 里用上述 metadata 构建 `sample.teacher_tokens` / `sample.teacher_prompt_length`，供 OPSD 训练 loop 计算 KL。
- 新建训练脚本 `run_qwen3_4B_opsd.sh`，在现有 tau-bench 脚本基础上添加 OPSD 参数。

**Tech Stack:** Python, PyTorch, HuggingFace Transformers (tokenizer), SGLang, Ray, Megatron-style GRPO trainer

---

## 文件清单

| 操作 | 文件 | 职责 |
|------|------|------|
| Modify | `examples/tau-bench/trainable_agents.py` | `_build_final_result` 里把 `initial_prompt_token_length` 和 `initial_user_message` 写入 `res.info` |
| Modify | `examples/tau-bench/generate_with_tau.py` | `generate()` 里把 `wiki` / `tools_info` 写入 `result_sample.metadata` |
| Create | `examples/tau-bench/tau_opsd_reward.py` | `reward_func` + `post_process_rewards`：构建 teacher tokens，返回 GRPO 归一化 reward |
| Create | `examples/tau-bench/run_qwen3_4B_opsd.sh` | 训练启动脚本（在现有脚本基础上加 OPSD 参数） |

---

## Task 1：`trainable_agents.py` — 在 `res.info` 里存 metadata

**文件：** `examples/tau-bench/trainable_agents.py`

修改 `_build_final_result`，在返回结果前把 teacher 构建需要的字段写入 `res.info`。

- [ ] **Step 1: 读懂现有逻辑**

  确认 `_build_final_result` 里有：
  - `res.tokens = prompt_token_ids + response_token_ids`
  - `res.response_length = len(loss_masks)`

  因此 `initial_prompt_token_length = len(prompt_token_ids)`，可以直接用参数计算，无需额外传值。

- [ ] **Step 2: 修改 `_build_final_result`**

  在 `res.response_length = len(loss_masks)` 之后，`return res` 之前，添加：

  ```python
  # Store fields needed for OPSD teacher token construction
  res.info['initial_prompt_token_length'] = len(prompt_token_ids)
  # messages[0] is system (wiki), messages[1] is first user turn (task description)
  if len(messages) > 1 and messages[1].get('role') == 'user':
      res.info['initial_user_message'] = messages[1]['content']
  ```

- [ ] **Step 3: 手动检查结果**

  在本地运行一次 `tau1_mock.py`（或直接跑 generate），确认 `InteractionResult.info` 包含 `initial_prompt_token_length` 和 `initial_user_message`。

---

## Task 2：`generate_with_tau.py` — 把 wiki/tools_info 写入 metadata

**文件：** `examples/tau-bench/generate_with_tau.py`

- [ ] **Step 1: 在 `generate()` 里，`res_to_sample` 之后追加 metadata**

  找到：
  ```python
  result_sample = res_to_sample(interaction_result, task_index)
  ```

  在其后添加：
  ```python
  # Enrich metadata for OPSD teacher construction
  result_sample.metadata['wiki'] = agent.wiki
  result_sample.metadata['tools_info'] = env.tools_info
  ```

  `initial_prompt_token_length` 和 `initial_user_message` 已经通过 Task 1 写入了 `interaction_result.info`，而 `res_to_sample` 把 `res.info` 赋给了 `sample.metadata`，所以这两个字段会自动出现在 `metadata` 里，无需重复赋值。

- [ ] **Step 2: 确认字段完整性**

  `sample.metadata` 此时应包含：
  - `initial_prompt_token_length` (int) — 来自 Task 1
  - `initial_user_message` (str) — 来自 Task 1
  - `wiki` (str) — 刚写入
  - `tools_info` (list[dict]) — 刚写入
  - 原有 tau-bench env info 字段（reward_info 等）

---

## Task 3：新建 `tau_opsd_reward.py`

**文件：** `examples/tau-bench/tau_opsd_reward.py`

这是核心新文件，负责：
1. `reward_func`：从 `sample.reward` 提取 task success score
2. `post_process_rewards`：用 metadata 构建 teacher_tokens，返回 GRPO 归一化 reward

- [ ] **Step 1: 创建文件，写 `reward_func`**

  ```python
  """Tau-Bench reward + OPSD teacher token construction.

  reward_func     — async, per-sample: extracts tau-bench task success score.
  post_process_rewards — sync, batch: builds teacher_tokens for OPSD KL training.

  Teacher information mode (conciseness / oracle):
    conciseness (default): Teacher system prompt = wiki + efficiency instruction.
                           No external API needed.
    oracle: Teacher system prompt = wiki + plan from OpenAI API.
            Set metadata['oracle_plan'] in generate_with_tau.py to enable.
  """

  import logging
  from functools import lru_cache

  import torch

  logger = logging.getLogger(__name__)


  @lru_cache(maxsize=1)
  def _get_tokenizer(model_path: str):
      from transformers import AutoTokenizer
      tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
      logger.info(f"tau_opsd: loaded tokenizer from {model_path}")
      return tokenizer


  async def reward_func(args, sample, **kwargs):
      """Return tau-bench task success as a named reward dict."""
      return {"tau_reward": float(sample.reward or 0.0)}
  ```

- [ ] **Step 2: 写 `post_process_rewards` — teacher prompt 构建**

  ```python
  def post_process_rewards(args, samples, **kwargs):
      """Build OPSD teacher tokens and return GRPO-normalised rewards.

      For each sample:
        1. Extract scalar tau_reward.
        2. Build teacher system message = wiki + conciseness/oracle hint.
        3. Tokenize teacher prompt (system + first user turn + tools).
        4. Assemble teacher_tokens = teacher_prompt_tokens + response_tokens.
        5. Set sample.teacher_tokens and sample.teacher_prompt_length.
        6. GRPO group-normalise rewards.
      """
      tokenizer = _get_tokenizer(args.hf_checkpoint)

      # ---- 1. Extract raw rewards ----
      raw_rewards = []
      for sample in samples:
          r = sample.get_reward_value(args)
          if isinstance(r, dict):
              r = r.get("tau_reward", 0.0)
          raw_rewards.append(float(r))

      # ---- 2-5. Build teacher tokens ----
      for sample in samples:
          meta = sample.metadata or {}
          wiki: str = meta.get("wiki", "")
          initial_user_msg: str = meta.get("initial_user_message", "")
          tools_info = meta.get("tools_info")
          initial_prompt_len: int = meta.get("initial_prompt_token_length", 0)

          if not initial_user_msg:
              logger.warning(
                  "tau_opsd: initial_user_message missing from metadata; "
                  "teacher_tokens will be empty for this sample."
              )
              continue

          # Build privileged teacher system message
          oracle_plan: str = meta.get("oracle_plan", "")
          if oracle_plan:
              teacher_system = (
                  f"{wiki}\n\n"
                  f"--- Expert Solution Plan ---\n{oracle_plan}\n---"
              )
          else:
              # Conciseness mode: no external API required
              teacher_system = (
                  f"{wiki}\n\n"
                  "Efficiency guideline: Think carefully about which tools are needed "
                  "before calling them. Use the minimum number of tool calls to complete "
                  "the task correctly."
              )

          teacher_messages = [
              {"role": "system", "content": teacher_system},
              {"role": "user", "content": initial_user_msg},
          ]
          teacher_prompt_text = tokenizer.apply_chat_template(
              teacher_messages,
              tokenize=False,
              add_generation_prompt=True,
              tools=tools_info,
          )
          teacher_prompt_tokens = tokenizer.encode(
              teacher_prompt_text, add_special_tokens=False
          )

          # response_tokens = everything after the initial prompt
          # (includes both agent turns [loss_mask=1] and tool returns [loss_mask=0])
          # KL is only computed where loss_mask=1, so tool-return tokens are safe to include.
          response_tokens = list(sample.tokens[initial_prompt_len:])

          sample.teacher_tokens = teacher_prompt_tokens + response_tokens
          sample.teacher_prompt_length = len(teacher_prompt_tokens)

      # ---- 6. GRPO group normalisation ----
      n = getattr(args, "n_samples_per_prompt", 1)
      t = torch.tensor(raw_rewards, dtype=torch.float)
      if n > 1 and len(raw_rewards) >= n:
          t = t.view(-1, n)
          mean = t.mean(dim=-1, keepdim=True)
          std = t.std(dim=-1, keepdim=True)
          norm = (t - mean) / (std + 1e-6)
          norm[std.squeeze(-1) < 1e-8] = 0.0
          normalised_rewards = norm.flatten().tolist()
      else:
          normalised_rewards = list(raw_rewards)

      return raw_rewards, normalised_rewards
  ```

---

## Task 4：新建 `run_qwen3_4B_opsd.sh`

**文件：** `examples/tau-bench/run_qwen3_4B_opsd.sh`

在现有 `run_qwen3_4B.sh` 基础上修改：
- `CUSTOM_ARGS`：加 `--custom-rm-path` 和 `--custom-reward-post-process-path`
- 新增 `GRPO_ARGS`：OPSD 参数（与 pi-safety 脚本一致）
- `ROLLOUT_ARGS`：去掉 `--dynamic-sampling-filter-path`（OPSD pure mode 下 reward 归一化不同，filter 可能不适用，先去掉）

- [ ] **Step 1: 创建脚本**

  ```bash
  #!/bin/bash
  #
  # Tau-Bench On-Policy Self-Distillation (OPSD)
  #
  # Teacher = same model conditioned on [wiki + efficiency guideline + first_user_turn]
  # Student = same model conditioned on [wiki + first_user_turn]   (standard)
  #
  # Learning signal: KL(teacher || student) on agent-generated tokens (loss_mask=1),
  # weighted by GRPO tau-bench task success advantage.
  #

  pkill -9 sglang || true
  sleep 2
  ray stop --force || true
  pkill -9 ray || true
  pkill -9 python || true
  sleep 2

  set -ex
  export PYTHONBUFFERED=16

  NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
  HAS_NVLINK=$([ "$NVLINK_COUNT" -gt 0 ] && echo 1 || echo 0)
  echo "HAS_NVLINK: $HAS_NVLINK"

  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
  source "${SCRIPT_DIR}/../../scripts/models/qwen3-4B-Instruct-2507.sh"

  ###############################################################################
  # Checkpoint
  ###############################################################################

  CKPT_ARGS=(
      --hf-checkpoint /root/Qwen3-4B-Instruct-2507/
      --ref-load /root/Qwen3-4B-Instruct-2507_torch_dist/
      --save /root/Qwen3-4B-Instruct-2507_slime_opsd/
      --save-interval 20
  )

  ###############################################################################
  # Rollout
  ###############################################################################

  ROLLOUT_ARGS=(
      --prompt-data /root/tau-bench/retail_train_tasks.jsonl
      --input-key index
      --rollout-shuffle
      --num-rollout 500
      --rollout-batch-size 32
      --n-samples-per-prompt 8
      --rollout-max-response-len 1024
      --rollout-temperature 1.0
      --global-batch-size 256
      --balance-data
  )

  ###############################################################################
  # Reward + OPSD teacher construction
  ###############################################################################

  RM_ARGS=(
      --custom-rm-path tau_opsd_reward.reward_func
      --custom-reward-post-process-path tau_opsd_reward.post_process_rewards
      --reward-key tau_reward
  )

  ###############################################################################
  # OPSD / GRPO
  ###############################################################################

  GRPO_ARGS=(
      --advantage-estimator grpo
      --use-opd
      --opd-type opsd
      --opd-kl-coef 0.0
      --opsd-loss-type reverse_kl
      --opsd-jsd-coef 1.0
      --opsd-pure-mode
      --use-kl-loss
      --kl-loss-coef 0.01
      --opsd-use-ref-as-teacher
      --entropy-coef 0.00
  )

  ###############################################################################
  # Eval
  ###############################################################################

  EVAL_ARGS=(
      --eval-interval 5
      --eval-prompt-data retail-dev /root/tau-bench/retail_dev_tasks.jsonl
      --n-samples-per-eval-prompt 1
      --eval-max-response-len 1024
      --eval-top-k 1
  )

  ###############################################################################
  # Performance
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
      --max-tokens-per-gpu 9216
  )

  ###############################################################################
  # Optimiser
  ###############################################################################

  OPTIMIZER_ARGS=(
      --optimizer adam
      --lr 1e-6
      --lr-decay-style constant
      --weight-decay 0.1
      --adam-beta1 0.9
      --adam-beta2 0.98
  )

  ###############################################################################
  # Custom generate (tau-bench interactive env)
  ###############################################################################

  CUSTOM_ARGS=(
      --custom-generate-function-path generate_with_tau.generate
  )

  ###############################################################################
  # SGLang
  ###############################################################################

  SGLANG_ARGS=(
      --rollout-num-gpus-per-engine 1
      --sglang-mem-fraction-static 0.7
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
  )

  ###############################################################################
  # Launch
  ###############################################################################

  export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
  NUM_GPUS=2

  ray start --head \
      --node-ip-address "${MASTER_ADDR}" \
      --num-gpus "${NUM_GPUS}" \
      --disable-usage-stats \
      --dashboard-host=0.0.0.0 \
      --dashboard-port=8265 \
      --temp-dir /tmp/ray_opsd_$(date +%s)

  RUNTIME_ENV_JSON="{
    \"env_vars\": {
      \"PYTHONPATH\": \"/root/Megatron-LM/:${SCRIPT_DIR}\",
      \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\"
    }
  }"

  ray job submit --address="http://127.0.0.1:8265" \
      --runtime-env-json="${RUNTIME_ENV_JSON}" \
      -- python3 train.py \
      --actor-num-nodes 1 \
      --actor-num-gpus-per-node "${NUM_GPUS}" \
      --rollout-num-gpus "${NUM_GPUS}" \
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
      ${CUSTOM_ARGS[@]} \
      ${RM_ARGS[@]}
  ```

---

## 注意事项 / 已知风险

1. **`apply_chat_template` 与 `tools` 参数**：Qwen3 tokenizer 支持 `tools=` 参数，但如果 `tools_info` 里包含不可序列化的对象（非 dict/list/str/int），会报错。tau-bench 的 `env.tools_info` 是 list of dict，应该没问题。

2. **`initial_prompt_token_length` 的准确性**：依赖 `len(prompt_token_ids)`，在 `_build_final_result` 里直接取参数值，是准确的。

3. **teacher prompt 长度可能与 student prompt 长度不同**：这是正常的。teacher 看的是不同的 system prompt，token 数可以多也可以少。OPSD 训练 loop 只用 `teacher_prompt_length` 来切分 teacher tokens，不依赖它等于 student prompt 长度。

4. **OPSD pure mode**：`--opsd-pure-mode` 意味着 task reward 只用于 GRPO 归一化（不加进 loss），实际训练信号只有 KL。这适合 pi-style distillation。如果想兼顾 RL reward，去掉 `--opsd-pure-mode`。
