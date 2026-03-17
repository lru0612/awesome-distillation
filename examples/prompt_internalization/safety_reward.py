"""Training reward function and post-processing for the PI safety experiment.

Plugged in via:
    --custom-rm-path
        examples.prompt_internalization.safety_reward.reward_func
    --custom-reward-post-process-path
        examples.prompt_internalization.safety_reward.post_process_rewards
    --reward-key safety_reward

reward_func
    Scores each student rollout with compute_safety_reward.
    Mode (keyword vs LLM judge) is read from metadata['judge_mode'] or
    the JUDGE_MODE environment variable.

post_process_rewards
    Builds teacher tokens using the "pi" info mode:
        teacher_user_content = pi_instruction + student_user_content
    then applies standard GRPO group normalisation.
"""

from __future__ import annotations

import logging

import torch

from slime.rollout.rm_hub.safety import compute_safety_reward

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# reward_func  (called per sample during rollout, must be async)
# ---------------------------------------------------------------------------


async def reward_func(args, sample, **kwargs):
    """Return {"safety_reward": float} for a single rollout sample."""
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    score = await compute_safety_reward(
        response=sample.response,
        label=sample.label or "",
        metadata=metadata,
    )
    return {"safety_reward": float(score)}


# ---------------------------------------------------------------------------
# post_process_rewards  (called on the full batch after rollout, sync)
# ---------------------------------------------------------------------------


def post_process_rewards(args, samples, **kwargs):
    """Build teacher tokens (pi mode) and apply GRPO normalisation.

    Teacher prompt = pi_instruction + student_user_content.
    The student sees only student_user_content (no constitution).

    Returns
    -------
    raw_rewards        : list[float]  — un-normalised (for logging)
    normalised_rewards : list[float]  — GRPO-normalised advantages
    """
    from functools import lru_cache

    from transformers import AutoTokenizer

    @lru_cache(maxsize=1)
    def _get_tokenizer(model_path: str):
        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        logger.info("PI-safety: loaded tokenizer from %s", model_path)
        return tok

    tokenizer = _get_tokenizer(args.hf_checkpoint)

    # ---- 1. Collect raw rewards ----
    raw_rewards: list[float] = []
    for sample in samples:
        r = sample.get_reward_value(args)
        if isinstance(r, dict):
            r = r.get("safety_reward", 0.0)
        raw_rewards.append(float(r))

    # ---- 2. Build teacher tokens for each sample (pi mode) ----
    for sample in samples:
        metadata = sample.metadata if isinstance(sample.metadata, dict) else {}

        pi_instruction = metadata.get("pi_instruction", "")
        student_user_content = (
            metadata.get("student_user_content")
            or _fallback_user_content(sample)
        )

        if pi_instruction:
            teacher_user_content = f"{pi_instruction}\n\n{student_user_content}"
        else:
            teacher_user_content = student_user_content

        teacher_messages = [{"role": "user", "content": teacher_user_content}]

        # enable_thinking=False: safety responses do not require chain-of-thought
        teacher_prompt_text = tokenizer.apply_chat_template(
            teacher_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        teacher_prompt_tokens = tokenizer.encode(
            teacher_prompt_text, add_special_tokens=False
        )

        response_tokens = sample.tokens[-sample.response_length :]
        sample.teacher_tokens = teacher_prompt_tokens + list(response_tokens)
        sample.teacher_prompt_length = len(teacher_prompt_tokens)

    # ---- 3. GRPO group normalisation ----
    n = getattr(args, "n_samples_per_prompt", 1)
    rewards_tensor = torch.tensor(raw_rewards, dtype=torch.float)

    if n > 1 and len(raw_rewards) >= n:
        rewards_tensor = rewards_tensor.view(-1, n)
        mean = rewards_tensor.mean(dim=-1, keepdim=True)
        std = rewards_tensor.std(dim=-1, keepdim=True)
        normalised = (rewards_tensor - mean) / (std + 1e-6)
        # Zero out groups where all rewards are identical (no learning signal)
        zero_std_mask = std.squeeze(-1) < 1e-8
        normalised[zero_std_mask] = 0.0
        normalised_rewards = normalised.flatten().tolist()
    else:
        normalised_rewards = list(raw_rewards)

    return raw_rewards, normalised_rewards


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _fallback_user_content(sample) -> str:
    """Extract user content from sample.prompt if metadata is missing."""
    if isinstance(sample.prompt, list):
        for msg in sample.prompt:
            if isinstance(msg, dict) and msg.get("role") == "user":
                return msg.get("content", "")
    if isinstance(sample.prompt, str):
        return sample.prompt
    return ""
