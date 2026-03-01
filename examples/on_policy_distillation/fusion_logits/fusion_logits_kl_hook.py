"""Logit Fusion RL — optional Method A: full-distribution reverse KL hook.

This module is **optional**.  The base Logit Fusion algorithm uses only the
IS-corrected REINFORCE loss (``--use-tis``).  Method A augments it with a
full-vocabulary reverse KL penalty computed at training time:

    KL(π_S || π_T)_i = Σ_v π_S(v|x, y_{<i}; θ) · log(π_S / π_T)(v)

This gives a lower-variance dense signal on every token position rather than
only the sampled token's IS ratio.

The hook:

1. Runs a **single** teacher forward pass on the full privileged context
   ``teacher_tokens`` (efficient: causal attention means one pass suffices).
2. Computes per-token JSD/KL between student and teacher on response tokens.
3. Optionally applies **teacher-entropy weighting** to downweight uncertain
   teacher positions (``KL_WEIGHT_MODE`` env var).
4. Subtracts the weighted KL from advantages:
   ``advantages[i] -= kl_coef * w_i * KL_i``

Integration::

    --custom-megatron-before-train-step-hook-path \\
        examples.on_policy_distillation.fusion_logits.fusion_logits_kl_hook.register_kl

Configuration (env vars, same convention as OPSD scripts)::

    OPSD_JSD_BETA        float  (default 0.0)  JSD beta; 0 = pure KL(S||T)
    OPSD_JSD_COEF        float  (default 1.0)  weight of the KL term in total loss
    KL_WEIGHT_MODE       str    (default "")   entropy weighting: inverse|exp_neg|linear|""
    KL_WEIGHT_TEMP       float  (default 1.0)  temperature τ for entropy weights
    KL_CONFIDENCE_THRESHOLD  str (default "")  entropy threshold or "" to skip masking

Note: ``teacher_tokens`` and ``teacher_prompt_length`` must be present in the
batch (populated by ``fusion_logits_reward.post_process_rewards``).
"""

from __future__ import annotations

import logging
import os
from argparse import Namespace
from functools import lru_cache

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_CONFIG: dict = {}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _load_config(args: Namespace) -> dict:
    global _CONFIG
    if _CONFIG:
        return _CONFIG

    kl_mode = os.environ.get("KL_WEIGHT_MODE", "")
    kl_thresh = os.environ.get("KL_CONFIDENCE_THRESHOLD", "")

    _CONFIG = {
        "jsd_beta": float(os.environ.get("OPSD_JSD_BETA", "0.0")),
        "jsd_coef": float(os.environ.get("OPSD_JSD_COEF", "1.0")),
        "confidence_weight_mode": kl_mode if kl_mode else None,
        "confidence_weight_temp": float(os.environ.get("KL_WEIGHT_TEMP", "1.0")),
        "confidence_threshold": float(kl_thresh) if kl_thresh else None,
    }
    logger.info("Fusion logits KL hook config: %s", _CONFIG)
    return _CONFIG


# ---------------------------------------------------------------------------
# Core forward function
# ---------------------------------------------------------------------------

def compute_fusion_kl_in_forward(
    args: Namespace,
    model: torch.nn.Module,
    batch: dict,
    output_tensor: torch.Tensor,
) -> None:
    """Teacher forward + reverse KL penalty subtracted from advantages.

    This function is monkey-patched as
    ``slime.backends.megatron_utils.model._compute_opsd_jsd_in_forward``
    so that the existing ``policy_loss_function`` picks up
    ``batch["opsd_jsd_values"]`` automatically.

    Args:
        args: Training args namespace.
        model: The Megatron student model (used for teacher pass under no_grad).
        batch: Training micro-batch dict.  Must contain ``teacher_tokens``,
               ``teacher_prompt_lengths``, ``response_lengths``, ``total_lengths``.
        output_tensor: Student logits from the current forward pass.
    """
    from megatron.core import mpu
    from megatron.core.packed_seq_params import PackedSeqParams

    from examples.on_policy_distillation.weighted_kl import (
        apply_weighted_kl_to_jsd,
    )
    from slime.utils.ppo_utils import compute_vocab_parallel_jsd

    config = _load_config(args)

    teacher_tokens_list = batch["teacher_tokens"]
    teacher_prompt_lengths = batch["teacher_prompt_lengths"]
    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]

    tp_group = mpu.get_tensor_model_parallel_group()
    pad_size = (
        mpu.get_tensor_model_parallel_world_size() * args.data_pad_size_multiplier
    )
    pad_token_id = 0

    # ------------------------------------------------------------------
    # 1. Pack teacher tokens and run teacher forward
    # ------------------------------------------------------------------
    teacher_cu_seqlens = [0]
    teacher_tokens_cat_list: list[torch.Tensor] = []
    for t_tok in teacher_tokens_list:
        teacher_tokens_cat_list.append(t_tok)
        teacher_cu_seqlens.append(teacher_cu_seqlens[-1] + t_tok.size(0))

    teacher_tokens_cat = torch.cat(teacher_tokens_cat_list)
    pad = (pad_size - teacher_tokens_cat.size(0) % pad_size) % pad_size
    if pad > 0:
        teacher_tokens_cat = F.pad(teacher_tokens_cat, (0, pad), value=pad_token_id)
        teacher_cu_seqlens.append(teacher_cu_seqlens[-1] + pad)

    cp_size = mpu.get_context_parallel_world_size()
    device = torch.cuda.current_device()
    teacher_cu_seqlens_tensor = (
        torch.tensor(teacher_cu_seqlens, dtype=torch.int, device=device) * cp_size
    )
    teacher_max_seqlen = int(
        (teacher_cu_seqlens_tensor[1:] - teacher_cu_seqlens_tensor[:-1]).max().item()
    )
    teacher_packed_seq_params = PackedSeqParams(
        cu_seqlens_q=teacher_cu_seqlens_tensor,
        cu_seqlens_kv=teacher_cu_seqlens_tensor,
        max_seqlen_q=teacher_max_seqlen,
        max_seqlen_kv=teacher_max_seqlen,
        qkv_format="thd",
    )
    teacher_tokens_input = teacher_tokens_cat.unsqueeze(0)

    with torch.no_grad():
        teacher_output = model(
            input_ids=teacher_tokens_input,
            position_ids=None,
            attention_mask=None,
            labels=None,
            packed_seq_params=teacher_packed_seq_params,
            loss_mask=None,
        )
    teacher_logits = teacher_output.detach().float()

    # ------------------------------------------------------------------
    # 2. Flatten logit tensors to 2-D
    # ------------------------------------------------------------------
    student_logits = output_tensor.float()
    student_logits_2d = (
        student_logits.squeeze(0)
        if student_logits.dim() == 3 and student_logits.size(0) == 1
        else student_logits.view(-1, student_logits.size(-1))
    )
    teacher_logits_2d = (
        teacher_logits.squeeze(0)
        if teacher_logits.dim() == 3 and teacher_logits.size(0) == 1
        else teacher_logits.view(-1, teacher_logits.size(-1))
    )

    if args.rollout_temperature != 1.0:
        student_logits_2d = student_logits_2d / args.rollout_temperature
        teacher_logits_2d = teacher_logits_2d / args.rollout_temperature

    # ------------------------------------------------------------------
    # 3. Per-sample: compute (weighted) JSD → store in batch
    # ------------------------------------------------------------------
    opsd_jsd_values: list[torch.Tensor] = []
    all_metrics: dict[str, list[torch.Tensor]] = {}

    student_end = 0
    teacher_end = 0

    for i in range(len(response_lengths)):
        resp_len = response_lengths[i]
        total_len = total_lengths[i]
        t_prompt_len = teacher_prompt_lengths[i]
        t_total_len = t_prompt_len + resp_len

        student_end += total_len
        s_logits = student_logits_2d[student_end - resp_len - 1 : student_end - 1]

        teacher_end += t_total_len
        t_logits = teacher_logits_2d[teacher_end - resp_len - 1 : teacher_end - 1]

        jsd = compute_vocab_parallel_jsd(
            s_logits, t_logits, tp_group, beta=config["jsd_beta"],
        )

        # Entropy-based confidence weighting (reuses weighted_kl utility)
        weighted_jsd, metrics = apply_weighted_kl_to_jsd(
            jsd,
            t_logits,
            tp_group,
            segment_boundaries=None,
            confidence_weight_mode=config["confidence_weight_mode"],
            confidence_weight_temp=config["confidence_weight_temp"],
            confidence_threshold=config["confidence_threshold"],
            segment_weight_mode="uniform",
        )

        opsd_jsd_values.append(weighted_jsd)
        for k, v in metrics.items():
            all_metrics.setdefault(k, []).append(v)

    batch["opsd_jsd_values"] = opsd_jsd_values

    for k, vs in all_metrics.items():
        batch[f"fusion_kl_{k}"] = torch.stack(vs).mean().detach()


# ---------------------------------------------------------------------------
# Hook entry point
# ---------------------------------------------------------------------------

def register_kl(
    args: Namespace,
    rollout_id: int,
    step_id: int,
    model,
    optimizer,
    opt_param_scheduler,
) -> None:
    """``--custom-megatron-before-train-step-hook-path`` entry point.

    Monkey-patches ``_compute_opsd_jsd_in_forward`` with
    ``compute_fusion_kl_in_forward``.  Idempotent — patches only once.

    Also forces the required OPSD flags so that ``policy_loss_function``
    will consume ``batch["opsd_jsd_values"]``::

        args.use_opd  = True
        args.opd_type = "opsd"
        args.opsd_jsd_coef = config["jsd_coef"]
        args.opsd_jsd_beta = config["jsd_beta"]
        args.opsd_pure_mode = False  (let IS loss also run)
    """
    if getattr(args, "_fusion_kl_registered", False):
        return

    config = _load_config(args)

    # Force OPSD mode so policy_loss_function processes opsd_jsd_values
    args.use_opd = True
    if not hasattr(args, "opd_type") or not args.opd_type:
        args.opd_type = "opsd"
    args.opsd_jsd_coef = config["jsd_coef"]
    args.opsd_jsd_beta = config["jsd_beta"]
    args.opsd_pure_mode = False  # Keep IS loss active alongside KL

    import slime.backends.megatron_utils.model as model_module
    model_module._compute_opsd_jsd_in_forward = compute_fusion_kl_in_forward

    args._fusion_kl_registered = True
    logger.info(
        "Fusion logits KL hook: patched _compute_opsd_jsd_in_forward "
        "(jsd_beta=%.2f, jsd_coef=%.3f, kl_weight=%s)",
        config["jsd_beta"],
        config["jsd_coef"],
        config["confidence_weight_mode"],
    )
