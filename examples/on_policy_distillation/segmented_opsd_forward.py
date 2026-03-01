"""Segmented OPSD forward: per-segment teacher forward + weighted JSD.

Replaces the standard ``_compute_opsd_jsd_in_forward`` in
``slime.backends.megatron_utils.model`` with a version that:

1. Runs a **single** teacher forward pass on the full privileged context
   (identical to standard OPSD — efficient thanks to causal attention).
2. **Segments** each student response into logical parts.
3. Computes per-token JSD between student and teacher.
4. Applies **teacher-confidence weighting** (entropy-based or threshold mask).
5. Applies **segment-position weighting** (e.g., later steps get more weight).
6. Stores the weighted JSD in ``batch["opsd_jsd_values"]`` so that the
   unchanged ``policy_loss_function`` aggregates it into the total loss.

Integration
-----------
This module is activated via the ``--custom-megatron-before-train-step-hook-path``
argument, which monkey-patches the forward function **once** at the first
training step::

    --custom-megatron-before-train-step-hook-path \
        examples.on_policy_distillation.segmented_opsd_forward.register_segmented_opsd

Configuration is read from environment variables (set in the ray
``runtime-env-json``)::

    SEG_STRATEGY           newline | fixed_length | token_ids  (default: newline)
    SEG_MIN_LEN            minimum tokens per segment           (default: 10)
    SEG_CHUNK_SIZE         chunk size for fixed_length           (default: 128)
    KL_WEIGHT_MODE         inverse | exp_neg | linear | ""      (default: "" = off)
    KL_WEIGHT_TEMP         temperature for confidence weights    (default: 1.0)
    KL_CONFIDENCE_THRESHOLD  entropy threshold or ""             (default: "" = off)
    SEG_WEIGHT_MODE        uniform | linear_increasing | exponential_increasing
                                                                 (default: uniform)
"""

from __future__ import annotations

import logging
import os
from argparse import Namespace
from functools import lru_cache

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Module-level config cache — populated once by ``_load_config``.
_CONFIG: dict = {}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_separator_ids(model_path: str) -> set[int]:
    """Derive newline separator token IDs from the tokenizer (cached)."""
    from examples.on_policy_distillation.segmenter import get_separator_token_ids
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return get_separator_token_ids(tokenizer)


def _load_config(args: Namespace) -> dict:
    """Read and cache config from environment variables."""
    global _CONFIG
    if _CONFIG:
        return _CONFIG

    kl_mode = os.environ.get("KL_WEIGHT_MODE", "")
    kl_thresh = os.environ.get("KL_CONFIDENCE_THRESHOLD", "")

    _CONFIG = {
        "strategy": os.environ.get("SEG_STRATEGY", "newline"),
        "min_segment_len": int(os.environ.get("SEG_MIN_LEN", "10")),
        "chunk_size": int(os.environ.get("SEG_CHUNK_SIZE", "128")),
        "confidence_weight_mode": kl_mode if kl_mode else None,
        "confidence_weight_temp": float(os.environ.get("KL_WEIGHT_TEMP", "1.0")),
        "confidence_threshold": float(kl_thresh) if kl_thresh else None,
        "segment_weight_mode": os.environ.get("SEG_WEIGHT_MODE", "uniform"),
        "separator_ids": _get_separator_ids(args.hf_checkpoint),
    }

    logger.info("Segmented OPSD config: %s", {
        k: v for k, v in _CONFIG.items() if k != "separator_ids"
    })
    return _CONFIG


# ---------------------------------------------------------------------------
# Core forward function (replaces _compute_opsd_jsd_in_forward)
# ---------------------------------------------------------------------------

def compute_segmented_opsd_jsd_in_forward(
    args: Namespace,
    model: torch.nn.Module,
    batch: dict,
    output_tensor: torch.Tensor,
) -> None:
    """Segmented OPSD teacher forward + weighted per-token JSD.

    Drop-in replacement for
    ``slime.backends.megatron_utils.model._compute_opsd_jsd_in_forward``.
    Same signature, same side-effect (populates ``batch["opsd_jsd_values"]``).

    **Algorithm overview** (for each sample):

    1. Run teacher forward on ``privileged_prompt + student_response``
       (single efficient forward pass; causal attention ensures that logits
       at position *t* depend only on tokens 0..*t*−1).
    2. Compute raw per-token JSD between student and teacher on response
       tokens (using ``compute_vocab_parallel_jsd``).
    3. Segment the response tokens into *K* logical parts.
    4. Compute teacher per-token entropy and derive confidence weights.
    5. Apply confidence weighting / threshold masking.
    6. Apply segment-position weighting.
    7. Store weighted JSD as ``batch["opsd_jsd_values"]``.
    """
    from megatron.core import mpu
    from megatron.core.packed_seq_params import PackedSeqParams

    from examples.on_policy_distillation.segmenter import segment_tokens
    from examples.on_policy_distillation.weighted_kl import apply_weighted_kl_to_jsd
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

    # ==================================================================
    # 1. Pack teacher tokens into THD format and run teacher forward
    # ==================================================================
    teacher_cu_seqlens = [0]
    teacher_tokens_cat_list: list[torch.Tensor] = []
    for t_tok in teacher_tokens_list:
        teacher_tokens_cat_list.append(t_tok)
        teacher_cu_seqlens.append(teacher_cu_seqlens[-1] + t_tok.size(0))

    teacher_tokens_cat = torch.cat(teacher_tokens_cat_list)
    pad = (pad_size - teacher_tokens_cat.size(0) % pad_size) % pad_size
    if pad > 0:
        teacher_tokens_cat = F.pad(
            teacher_tokens_cat, (0, pad), value=pad_token_id,
        )
        teacher_cu_seqlens.append(teacher_cu_seqlens[-1] + pad)

    cp_size = mpu.get_context_parallel_world_size()
    device = torch.cuda.current_device()
    teacher_cu_seqlens_tensor = (
        torch.tensor(teacher_cu_seqlens, dtype=torch.int, device=device) * cp_size
    )
    teacher_max_seqlen = (
        (teacher_cu_seqlens_tensor[1:] - teacher_cu_seqlens_tensor[:-1])
        .max()
        .item()
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

    # ==================================================================
    # 2. Prepare 2-D logit tensors
    # ==================================================================
    student_logits = output_tensor.float()
    if student_logits.dim() == 3 and student_logits.size(0) == 1:
        student_logits_2d = student_logits.squeeze(0)
    else:
        student_logits_2d = student_logits.view(-1, student_logits.size(-1))

    if teacher_logits.dim() == 3 and teacher_logits.size(0) == 1:
        teacher_logits_2d = teacher_logits.squeeze(0)
    else:
        teacher_logits_2d = teacher_logits.view(-1, teacher_logits.size(-1))

    if args.rollout_temperature != 1.0:
        student_logits_2d = student_logits_2d / args.rollout_temperature
        teacher_logits_2d = teacher_logits_2d / args.rollout_temperature

    # ==================================================================
    # 3-7. Per-sample: segment → JSD → weight
    # ==================================================================
    opsd_jsd_values: list[torch.Tensor] = []
    all_metrics: dict[str, list[torch.Tensor]] = {}

    student_end = 0
    teacher_end = 0

    for i in range(len(response_lengths)):
        resp_len = response_lengths[i]
        total_len = total_lengths[i]
        t_prompt_len = teacher_prompt_lengths[i]
        t_total_len = t_prompt_len + resp_len

        # -- Student response logits --
        student_end += total_len
        s_logits = student_logits_2d[
            student_end - resp_len - 1 : student_end - 1
        ]

        # -- Teacher response logits --
        teacher_end += t_total_len
        t_logits = teacher_logits_2d[
            teacher_end - resp_len - 1 : teacher_end - 1
        ]

        # -- 3a. Raw per-token JSD --
        jsd = compute_vocab_parallel_jsd(
            s_logits, t_logits, tp_group, beta=args.opsd_jsd_beta,
        )

        # -- 3b. Segment the response tokens --
        response_toks = teacher_tokens_list[i][t_prompt_len:].tolist()
        segments = segment_tokens(
            response_toks,
            strategy=config["strategy"],
            separator_ids=config["separator_ids"],
            min_segment_len=config["min_segment_len"],
            chunk_size=config["chunk_size"],
        )

        # -- 3c-3f. Apply weighted KL --
        weighted_jsd, metrics = apply_weighted_kl_to_jsd(
            jsd,
            t_logits,
            tp_group,
            segment_boundaries=segments,
            confidence_weight_mode=config["confidence_weight_mode"],
            confidence_weight_temp=config["confidence_weight_temp"],
            confidence_threshold=config["confidence_threshold"],
            segment_weight_mode=config["segment_weight_mode"],
        )

        opsd_jsd_values.append(weighted_jsd)

        for k, v in metrics.items():
            all_metrics.setdefault(k, []).append(v)

    # ==================================================================
    # Store results
    # ==================================================================
    batch["opsd_jsd_values"] = opsd_jsd_values

    # Store aggregate metrics for logging (optional, picked up if loss_function checks)
    for k, vs in all_metrics.items():
        batch[f"seg_{k}"] = torch.stack(vs).mean().detach()


# ---------------------------------------------------------------------------
# Hook entry point
# ---------------------------------------------------------------------------

def register_segmented_opsd(
    args: Namespace,
    rollout_id: int,
    step_id: int,
    model,
    optimizer,
    opt_param_scheduler,
) -> None:
    """``--custom-megatron-before-train-step-hook-path`` entry point.

    Monkey-patches ``_compute_opsd_jsd_in_forward`` in the Megatron model
    module with the segmented version.  Idempotent — only patches once.
    """
    if getattr(args, "_segmented_opsd_registered", False):
        return

    _load_config(args)

    import slime.backends.megatron_utils.model as model_module

    model_module._compute_opsd_jsd_in_forward = (
        compute_segmented_opsd_jsd_in_forward
    )
    args._segmented_opsd_registered = True
    logger.info(
        "Segmented OPSD: patched _compute_opsd_jsd_in_forward "
        "(strategy=%s, kl_weight=%s, seg_weight=%s)",
        _CONFIG.get("strategy"),
        _CONFIG.get("confidence_weight_mode"),
        _CONFIG.get("segment_weight_mode"),
    )
