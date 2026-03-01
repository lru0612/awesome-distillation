"""Logit Fusion RL — custom generate function.

Implements token-by-token logit fusion between a teacher and student model.
Both teacher and student use the **same** HuggingFace weights (frozen at
rollout time); only their prompts differ:

* **Student prompt:** standard chat prompt (``sample.tokens``).
* **Teacher prompt:** privileged prompt (question + ground-truth answer hint).

At each decoding step:

1. Run two sequential HF forward passes (teacher then student) using their
   respective KV caches.  Both are updated with the **same** sampled token.
2. Mix logits:  ``ℓ_mix = α·ℓ_T + (1-α)·ℓ_S``.
3. Sample ``y_i ~ softmax(ℓ_mix)``, record ``log π_mix(y_i)``.

The stored ``sample.rollout_log_probs`` (= ``log π_mix_old`` per token) feeds
directly into ``vanilla_tis_function`` at training time (``--use-tis``).

CLI registration::

    --custom-generate-function-path \\
        examples.on_policy_distillation.fusion_logits.fusion_logits_generate.generate_fusion

Alpha schedule (env vars)::

    FUSION_ALPHA_INIT   float  (default 0.5)   initial teacher mixing weight
    FUSION_ALPHA_K      int    (default 5000)  linear-decay steps (per-sample invocations)
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

# Tracks total generate_fusion invocations (used as rollout step proxy for
# alpha decay).  K should be tuned to account for n_samples_per_prompt.
_rollout_step: int = 0

# Model / tokenizer cache: {model_path: (model, tokenizer)}
_model_cache: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _get_model(model_path: str):
    """Load and cache the HF model + tokenizer for this process.

    The model is placed on the current CUDA device.  Loading happens only
    once per process regardless of how many times ``generate_fusion`` is called.
    """
    if model_path in _model_cache:
        return _model_cache[model_path]

    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Fusion logits: loading HF model from %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
    )
    model.eval()

    param_gb = sum(p.numel() for p in model.parameters()) * 2 / 1e9
    logger.info(
        "Fusion logits: model loaded (%.2f GB bfloat16) on device %s",
        param_gb, device,
    )

    _model_cache[model_path] = (model, tokenizer)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Alpha schedule
# ---------------------------------------------------------------------------

def _compute_alpha(
    rollout_step: int,
    difficulty: float | None,
    alpha_init: float,
    K: int,
) -> float:
    """Compute dynamic mixing weight for the teacher.

    Implements linear decay with optional prompt-difficulty scaling::

        alpha_base = alpha_init * max(0, 1 - rollout_step / K)
        s(d)       = (d - d_min) / (d_max - d_min)   [clamped to [0, 1]]
        alpha      = alpha_base * s(d)  if difficulty given  else  alpha_base

    The difficulty range ``[d_min, d_max]`` is read from env vars
    ``FUSION_DIFFICULTY_MIN`` / ``FUSION_DIFFICULTY_MAX`` (defaults 0 / 1).
    """
    alpha_base = alpha_init * max(0.0, 1.0 - rollout_step / max(K, 1))
    if difficulty is not None:
        d_min = float(os.environ.get("FUSION_DIFFICULTY_MIN", "0.0"))
        d_max = float(os.environ.get("FUSION_DIFFICULTY_MAX", "1.0"))
        if d_max > d_min:
            s = max(0.0, min(1.0, (difficulty - d_min) / (d_max - d_min)))
            return alpha_base * s
    return alpha_base


# ---------------------------------------------------------------------------
# Privileged teacher prompt construction
# ---------------------------------------------------------------------------

def _extract_raw_content(sample) -> str:
    """Extract original user question text from the sample."""
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    raw_content = metadata.get("raw_content", "")
    if not raw_content and isinstance(sample.prompt, list):
        for msg in sample.prompt:
            if isinstance(msg, dict) and msg.get("role") == "user":
                raw_content = msg.get("content", "")
                break
    return raw_content


def _build_teacher_prompt_ids(raw_content: str, label: str, tokenizer) -> list[int]:
    """Build the privileged teacher prompt token IDs (question + answer hint)."""
    from examples.on_policy_distillation.teacher_lookahead import (
        build_privileged_prompt_tokens,
    )
    return build_privileged_prompt_tokens(raw_content, label, tokenizer)


# ---------------------------------------------------------------------------
# Lockstep decode loop
# ---------------------------------------------------------------------------

def _lockstep_decode(
    model,
    tokenizer,
    student_prompt_ids: list[int],
    teacher_prompt_ids: list[int],
    alpha: float,
    max_new_tokens: int,
    temperature: float,
) -> tuple[list[int], list[float], list[float]]:
    """Run the lockstep logit-fusion decode loop.

    Both teacher and student use the same shared model weights.  At each step
    the same sampled token ``y_i`` updates both KV caches, keeping the two
    streams in lockstep.  Only the **prompt prefixes** differ.

    Args:
        model: Frozen HF causal LM (bfloat16, on GPU).
        tokenizer: Corresponding HF tokenizer.
        student_prompt_ids: Standard student prompt token IDs.
        teacher_prompt_ids: Privileged teacher prompt token IDs.
        alpha: Teacher mixing weight for this sample.
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature (applied to both logit streams).

    Returns:
        ``(generated_ids, log_probs_mix, teacher_entropies)`` where:
        * ``generated_ids``: sampled token IDs (response).
        * ``log_probs_mix``: ``log π_mix(y_i)`` per token.
        * ``teacher_entropies``: ``H_T(y_i)`` per token (full vocab).
    """
    device = next(model.parameters()).device

    # Student vocab size — used to truncate teacher logits for safety
    V_S: int = model.config.vocab_size

    # EOS token IDs from tokenizer
    eos_ids: set[int] = set()
    if tokenizer.eos_token_id is not None:
        eos_ids.add(tokenizer.eos_token_id)
    # Some tokenizers expose a list of EOS ids
    for attr in ("eos_token_ids", "additional_special_tokens_ids"):
        for tid in getattr(tokenizer, attr, None) or []:
            eos_ids.add(int(tid))

    student_input = torch.tensor(
        student_prompt_ids, dtype=torch.long, device=device,
    ).unsqueeze(0)  # [1, S]
    teacher_input = torch.tensor(
        teacher_prompt_ids, dtype=torch.long, device=device,
    ).unsqueeze(0)  # [1, T]

    generated_ids: list[int] = []
    log_probs_mix: list[float] = []
    teacher_entropies: list[float] = []

    with torch.no_grad():
        # ----------------------------------------------------------------
        # Step 0: prefill both prompts
        # ----------------------------------------------------------------
        out_s = model(input_ids=student_input, use_cache=True)
        past_s = out_s.past_key_values
        logits_s = out_s.logits[0, -1].float()  # [V]

        out_t = model(input_ids=teacher_input, use_cache=True)
        past_t = out_t.past_key_values
        logits_t = out_t.logits[0, -1].float()[:V_S]  # [V_S]

        # Sample y_0 from the mixed distribution
        if temperature != 1.0:
            logits_s = logits_s / temperature
            logits_t = logits_t / temperature

        logits_mix = alpha * logits_t + (1.0 - alpha) * logits_s[:V_S]
        log_p_mix = F.log_softmax(logits_mix, dim=-1)
        probs_mix = log_p_mix.exp()

        y_i = int(torch.multinomial(probs_mix, num_samples=1).item())
        log_probs_mix.append(float(log_p_mix[y_i]))

        # Teacher entropy H_T = -sum(p_T * log(p_T))
        p_t = F.softmax(logits_t, dim=-1)
        h_t = float(-(p_t * (p_t.clamp(min=1e-10).log())).sum())
        teacher_entropies.append(h_t)

        generated_ids.append(y_i)
        if y_i in eos_ids:
            return generated_ids, log_probs_mix, teacher_entropies

        # ----------------------------------------------------------------
        # Decode loop: step 1 … max_new_tokens-1
        # ----------------------------------------------------------------
        for _ in range(1, max_new_tokens):
            y_prev = torch.tensor([[y_i]], dtype=torch.long, device=device)

            # Student forward: next-token logits given y_{i-1}
            out_s = model(
                input_ids=y_prev,
                past_key_values=past_s,
                use_cache=True,
            )
            past_s = out_s.past_key_values
            logits_s = out_s.logits[0, -1].float()  # [V]

            # Teacher forward: same token, different KV prefix
            out_t = model(
                input_ids=y_prev,
                past_key_values=past_t,
                use_cache=True,
            )
            past_t = out_t.past_key_values
            logits_t = out_t.logits[0, -1].float()[:V_S]  # truncate

            if temperature != 1.0:
                logits_s = logits_s / temperature
                logits_t = logits_t / temperature

            logits_mix = alpha * logits_t + (1.0 - alpha) * logits_s[:V_S]
            log_p_mix = F.log_softmax(logits_mix, dim=-1)
            probs_mix = log_p_mix.exp()

            y_i = int(torch.multinomial(probs_mix, num_samples=1).item())
            log_probs_mix.append(float(log_p_mix[y_i]))

            p_t = F.softmax(logits_t, dim=-1)
            h_t = float(-(p_t * (p_t.clamp(min=1e-10).log())).sum())
            teacher_entropies.append(h_t)

            generated_ids.append(y_i)
            if y_i in eos_ids:
                break

    return generated_ids, log_probs_mix, teacher_entropies


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def generate_fusion(args, sample, sampling_params) -> Any:
    """Logit Fusion custom generate function (``--custom-generate-function-path``).

    Replaces the standard SGLang-based generate with a lockstep HF decode loop
    that fuses teacher and student logits at every token.

    Args:
        args: Training argument namespace (must have ``hf_checkpoint``,
              ``rollout_max_response_len``, ``rollout_temperature``).
        sample: ``slime.utils.types.Sample`` with prompt tokens pre-filled.
        sampling_params: SGLang sampling params (not used; params taken from
                         ``args`` instead).

    Returns:
        Modified sample with populated fields:
        ``tokens``, ``response``, ``response_length``,
        ``rollout_log_probs``, ``status``,
        ``metadata["alpha_used"]``, ``metadata["teacher_entropy"]``.
    """
    global _rollout_step

    from slime.utils.types import Sample as SlimeSample

    # ---- Config ----
    alpha_init = float(os.environ.get("FUSION_ALPHA_INIT", "0.5"))
    K = int(os.environ.get("FUSION_ALPHA_K", "5000"))

    # ---- Alpha ----
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    difficulty = metadata.get("difficulty", None)
    alpha = _compute_alpha(_rollout_step, difficulty, alpha_init, K)
    _rollout_step += 1

    # ---- Generation params from args ----
    max_new_tokens = getattr(args, "rollout_max_response_len", 2048)
    temperature = getattr(args, "rollout_temperature", 1.0)

    # ---- Load model (singleton per process) ----
    model, tokenizer = _get_model(args.hf_checkpoint)

    # ---- Student prompt ----
    student_prompt_ids = list(sample.tokens)

    # ---- Teacher privileged prompt ----
    raw_content = _extract_raw_content(sample)
    if not raw_content:
        logger.warning(
            "Fusion logits: raw_content missing for sample; teacher prompt = student prompt.",
        )
    label = sample.label or ""
    teacher_prompt_ids = _build_teacher_prompt_ids(raw_content, label, tokenizer)

    # ---- Lockstep decode ----
    generated_ids, log_probs_mix, teacher_entropies = _lockstep_decode(
        model=model,
        tokenizer=tokenizer,
        student_prompt_ids=student_prompt_ids,
        teacher_prompt_ids=teacher_prompt_ids,
        alpha=alpha,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )

    # ---- Populate sample ----
    response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    sample.tokens = student_prompt_ids + generated_ids
    sample.response = response_text
    sample.response_length = len(generated_ids)
    sample.rollout_log_probs = log_probs_mix

    eos_id = tokenizer.eos_token_id
    if generated_ids and eos_id is not None and generated_ids[-1] == eos_id:
        sample.status = SlimeSample.Status.COMPLETED
    elif len(generated_ids) >= max_new_tokens:
        sample.status = SlimeSample.Status.TRUNCATED
    else:
        sample.status = SlimeSample.Status.COMPLETED

    # Store per-token teacher entropy for optional entropy-weighted IS / KL
    metadata["alpha_used"] = alpha
    metadata["teacher_entropy"] = teacher_entropies  # list[float], len = resp_len
    sample.metadata = metadata

    logger.debug(
        "Fusion logits: invocation=%d alpha=%.4f resp_len=%d",
        _rollout_step - 1, alpha, len(generated_ids),
    )

    return sample
