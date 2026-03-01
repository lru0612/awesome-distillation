"""Logit Fusion RL — reward function and post-processing.

Exported symbols for CLI args::

    --custom-rm-path \
        examples.on_policy_distillation.fusion_logits.fusion_logits_reward.reward_func
    --custom-reward-post-process-path \
        examples.on_policy_distillation.fusion_logits.fusion_logits_reward.post_process_rewards
    --reward-key math_reward

Design notes
------------
* The teacher's privileged forward pass happens **at rollout time** inside
  ``generate_fusion``.  No teacher computation is needed here — only math
  correctness grading and GRPO group normalisation.
* ``post_process_rewards`` additionally stores ``teacher_tokens`` /
  ``teacher_prompt_length`` in each sample for the optional
  ``fusion_logits_kl_hook`` (Method A, full-distribution reverse KL).
  These fields are computed here because the tokenizer is already loaded
  for the reward and because this is the natural post-rollout hook.
* ``sample.metadata["teacher_entropy"]`` is populated by ``generate_fusion``
  and passed through as-is; no processing needed here.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_tokenizer(model_path: str):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    logger.info("Fusion logits reward: loaded tokenizer from %s", model_path)
    return tokenizer


def _grade_math(response: str, label: str) -> float:
    from slime.rollout.rm_hub import grade_answer_verl
    if not label:
        return 0.0
    return 1.0 if grade_answer_verl(response, label) else 0.0


# ---------------------------------------------------------------------------
# reward_func  (async, per-sample, during rollout)
# ---------------------------------------------------------------------------

async def reward_func(args, sample, **kwargs):
    """Compute math correctness reward for one sample.

    Returns a dict so that ``--reward-key math_reward`` can extract the scalar.
    Teacher computation is not needed here — it already occurred inside
    ``generate_fusion`` at rollout time.
    """
    response = sample.response or ""
    label = sample.label or ""
    math_reward = _grade_math(response, label)
    return {"math_reward": math_reward}


# ---------------------------------------------------------------------------
# post_process_rewards  (batch, after rollout)
# ---------------------------------------------------------------------------

def post_process_rewards(args, samples, **kwargs):
    """Post-process rewards for the Logit Fusion RL algorithm.

    Per sample:

    1. Extract scalar math reward from ``sample.reward``.
    2. Build privileged teacher prompt tokens + concatenate with student
       response tokens.  Stored in ``sample.teacher_tokens`` and
       ``sample.teacher_prompt_length`` for the optional KL hook.
    3. GRPO group normalisation.

    Returns
    -------
    raw_rewards : list[float]
    normalised_rewards : list[float]
    """
    tokenizer = _get_tokenizer(args.hf_checkpoint)

    from examples.on_policy_distillation.teacher_lookahead import (
        build_privileged_prompt_tokens,
        build_teacher_tokens_standard,
    )

    # ---- 1. Extract raw rewards ----
    raw_rewards: list[float] = []
    for sample in samples:
        r = sample.get_reward_value(args)
        if isinstance(r, dict):
            r = r.get("math_reward", 0.0)
        raw_rewards.append(float(r))

    # ---- 2. Build teacher tokens for optional KL hook ----
    for sample in samples:
        metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
        raw_content = metadata.get("raw_content", "")
        if not raw_content and isinstance(sample.prompt, list):
            for msg in sample.prompt:
                if isinstance(msg, dict) and msg.get("role") == "user":
                    raw_content = msg.get("content", "")
                    break

        label = sample.label or ""
        privileged_prompt_tokens = build_privileged_prompt_tokens(
            raw_content, label, tokenizer,
        )
        response_tokens = list(sample.tokens[-sample.response_length:])
        teacher_tokens, teacher_prompt_length = build_teacher_tokens_standard(
            privileged_prompt_tokens, response_tokens,
        )
        sample.teacher_tokens = teacher_tokens
        sample.teacher_prompt_length = teacher_prompt_length

    # ---- 3. GRPO group normalisation ----
    n = getattr(args, "n_samples_per_prompt", 1)
    rewards_tensor = torch.tensor(raw_rewards, dtype=torch.float)

    if n > 1 and len(raw_rewards) >= n:
        rewards_tensor = rewards_tensor.view(-1, n)
        mean = rewards_tensor.mean(dim=-1, keepdim=True)
        std = rewards_tensor.std(dim=-1, keepdim=True)
        normalised = (rewards_tensor - mean) / (std + 1e-6)
        zero_std_mask = std.squeeze(-1) < 1e-8
        normalised[zero_std_mask] = 0.0
        normalised_rewards = normalised.flatten().tolist()
    else:
        normalised_rewards = list(raw_rewards)

    logger.info(
        "Fusion logits post-process: %d samples, mean_reward=%.4f",
        len(samples),
        sum(raw_rewards) / max(len(raw_rewards), 1),
    )

    return raw_rewards, normalised_rewards
