"""Segmented On-Policy Self-Distillation — reward function & post-processing.

Extends the standard OPSD (``on_policy_self_distillation.py``) with:

1. **Logical segmentation** of student responses (via ``segmenter``).
2. **Per-segment teacher contexts** (via ``teacher_lookahead``).
3. Metadata annotations that the custom forward function
   (``segmented_opsd_forward``) uses to apply weighted KL.

The reward computation itself is identical to standard OPSD (math correctness),
but ``post_process_rewards`` additionally stores segment boundaries in the
sample metadata so they can be reconstructed during the training forward pass.

Exported symbols (for CLI args)::

    --custom-rm-path \
        examples.on_policy_distillation.segmented_distillation.reward_func
    --custom-reward-post-process-path \
        examples.on_policy_distillation.segmented_distillation.post_process_rewards
    --reward-key math_reward
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

import torch

from examples.on_policy_distillation.segmenter import (
    get_separator_token_ids,
    segment_tokens,
)
from examples.on_policy_distillation.teacher_lookahead import (
    build_privileged_prompt_tokens,
    build_teacher_tokens_standard,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_tokenizer(model_path: str):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    logger.info("Segmented OPSD: loaded tokenizer from %s", model_path)
    return tokenizer


def _grade_math(response: str, label: str) -> float:
    from slime.rollout.rm_hub import grade_answer_verl
    if not label:
        return 0.0
    return 1.0 if grade_answer_verl(response, label) else 0.0


def _get_segmentation_config() -> dict:
    """Read segmentation hyperparameters from environment variables."""
    return {
        "strategy": os.environ.get("SEG_STRATEGY", "newline"),
        "min_segment_len": int(os.environ.get("SEG_MIN_LEN", "10")),
        "chunk_size": int(os.environ.get("SEG_CHUNK_SIZE", "128")),
    }


# ---------------------------------------------------------------------------
# reward_func  (async, per-sample, during rollout)
# ---------------------------------------------------------------------------

async def reward_func(args, sample, **kwargs):
    """Compute math reward for one sample (identical to standard OPSD)."""
    response = sample.response
    label = sample.label or ""
    math_reward = _grade_math(response, label)
    return {"math_reward": math_reward}


# ---------------------------------------------------------------------------
# post_process_rewards  (batch, after rollout)
# ---------------------------------------------------------------------------

def post_process_rewards(args, samples, **kwargs):
    """Process rewards, segment responses, and build teacher tokens.

    Per sample:

    1. Extract scalar math reward.
    2. Build privileged teacher prompt (question + answer hint).
    3. Construct teacher token sequence (privileged prompt + student response).
    4. Segment the student response into logical parts and store the segment
       boundaries in ``sample.metadata["segment_boundaries"]``.
    5. Perform GRPO group normalisation on the rewards.

    Returns
    -------
    raw_rewards : list[float]
    normalised_rewards : list[float]
    """
    tokenizer = _get_tokenizer(args.hf_checkpoint)
    seg_config = _get_segmentation_config()
    separator_ids = get_separator_token_ids(tokenizer)

    # ---- 1. Extract raw math rewards ----
    raw_rewards: list[float] = []
    for sample in samples:
        r = sample.get_reward_value(args)
        if isinstance(r, dict):
            r = r.get("math_reward", 0.0)
        raw_rewards.append(float(r))

    # ---- 2-4. Teacher tokens + segmentation per sample ----
    for sample in samples:
        metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
        raw_content = metadata.get("raw_content", "")
        label = sample.label or ""

        if not raw_content:
            if isinstance(sample.prompt, list):
                for msg in sample.prompt:
                    if msg.get("role") == "user":
                        raw_content = msg.get("content", "")
                        break
            if not raw_content:
                logger.warning(
                    "Segmented OPSD: raw_content missing — teacher tokens empty."
                )

        # Privileged prompt
        privileged_prompt_tokens = build_privileged_prompt_tokens(
            raw_content, label, tokenizer,
        )

        # Student response tokens
        response_tokens = list(sample.tokens[-sample.response_length :])

        # Standard teacher tokens (one forward pass covers all segments)
        teacher_tokens, teacher_prompt_length = build_teacher_tokens_standard(
            privileged_prompt_tokens, response_tokens,
        )
        sample.teacher_tokens = teacher_tokens
        sample.teacher_prompt_length = teacher_prompt_length

        # Segment boundaries (stored for logging; recomputed in forward pass)
        segments = segment_tokens(
            response_tokens,
            strategy=seg_config["strategy"],
            separator_ids=separator_ids,
            min_segment_len=seg_config["min_segment_len"],
            chunk_size=seg_config["chunk_size"],
        )
        metadata["segment_boundaries"] = segments
        metadata["num_segments"] = len(segments)
        sample.metadata = metadata

    # ---- 5. GRPO group normalisation ----
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
        "Segmented OPSD post-process: %d samples, seg_strategy=%s, "
        "avg_segments=%.1f",
        len(samples),
        seg_config["strategy"],
        sum(
            (s.metadata or {}).get("num_segments", 1) for s in samples
        ) / max(len(samples), 1),
    )

    return raw_rewards, normalised_rewards
