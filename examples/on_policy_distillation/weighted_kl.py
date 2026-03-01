"""Weighted KL / JSD computation with teacher-confidence metrics.

Provides two complementary weighting mechanisms for distillation losses:

1. **Entropy-based weighting**: Tokens where the teacher is *confident*
   (low entropy) receive higher weight, so the student focuses on tokens
   where the teacher's signal is most reliable.

2. **Threshold-based masking**: Tokens where the teacher's entropy exceeds
   a threshold are masked out entirely, discarding unreliable supervision.

Both mechanisms can be combined with segment-level position weighting
(later reasoning steps receive higher weight).

Usage::

    from examples.on_policy_distillation.weighted_kl import (
        apply_weighted_kl_to_jsd,
    )
    weighted_jsd = apply_weighted_kl_to_jsd(
        jsd_values, teacher_logits, tp_group,
        segment_boundaries=segments,
        confidence_weight_mode="inverse",
        confidence_threshold=2.0,
        segment_weight_mode="linear_increasing",
    )
"""

from __future__ import annotations

import torch
import torch.distributed as dist


# ---------------------------------------------------------------------------
# Teacher entropy
# ---------------------------------------------------------------------------

def compute_teacher_token_entropy(
    teacher_logits: torch.Tensor,
    process_group: dist.ProcessGroup | None,
) -> torch.Tensor:
    r"""Per-token entropy of the teacher distribution (TP-aware).

    .. math::
        H(p_T)_t = -\sum_{v} p_T(v \mid t) \log p_T(v \mid t)

    The computation is numerically stable (log-sum-exp) and handles
    tensor-parallel sharding: each rank holds ``V_local`` columns
    and the global softmax / entropy is recovered via all-reduce.

    Args:
        teacher_logits: Shape ``[seq_len, V_local]``.
        process_group: TP process group (``None`` for single-GPU).

    Returns:
        Per-token entropy, shape ``[seq_len]``.
    """
    t_max = teacher_logits.max(dim=-1, keepdim=True).values
    if process_group is not None:
        dist.all_reduce(t_max, op=dist.ReduceOp.MAX, group=process_group)

    t_shifted = teacher_logits - t_max
    t_exp = t_shifted.exp()
    t_sum_exp = t_exp.sum(dim=-1, keepdim=True)
    if process_group is not None:
        dist.all_reduce(t_sum_exp, group=process_group)

    t_probs = t_exp / t_sum_exp

    # H = log(Z) + max - E[logits]  (numerically stable form)
    # Equivalent to: -sum(p * log(p))
    log_probs_local = t_shifted - t_sum_exp.log()  # local log-softmax
    local_entropy = -(t_probs * log_probs_local).sum(dim=-1)
    if process_group is not None:
        dist.all_reduce(local_entropy, group=process_group)

    return local_entropy


# ---------------------------------------------------------------------------
# Confidence weighting
# ---------------------------------------------------------------------------

def entropy_to_confidence_weights(
    entropy: torch.Tensor,
    mode: str = "inverse",
    temperature: float = 1.0,
) -> torch.Tensor:
    r"""Map per-token entropy to ``[0, 1]`` confidence weights.

    Low entropy  → high confidence → high weight.

    Supported modes:

    * ``"inverse"``:  :math:`w_t = 1 / (1 + \tau \cdot H_t)`
    * ``"exp_neg"``:  :math:`w_t = \exp(-\tau \cdot H_t)`
    * ``"linear"``:   :math:`w_t = \max(0,\; 1 - \tau \cdot H_t)`

    Args:
        entropy: Per-token entropy, shape ``[T]``.
        mode: One of ``"inverse"``, ``"exp_neg"``, ``"linear"``.
        temperature: Sensitivity to entropy (:math:`\tau`).

    Returns:
        Per-token weights in ``[0, 1]``, shape ``[T]``.
    """
    if mode == "inverse":
        return 1.0 / (1.0 + temperature * entropy)
    if mode == "exp_neg":
        return torch.exp(-temperature * entropy)
    if mode == "linear":
        return torch.clamp(1.0 - temperature * entropy, min=0.0)
    raise ValueError(f"Unknown confidence weight mode: {mode!r}")


def threshold_confidence_mask(
    entropy: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """Binary mask that zeros out tokens above an entropy threshold.

    Args:
        entropy: Per-token entropy, shape ``[T]``.
        threshold: Maximum tolerated entropy.

    Returns:
        Binary mask ``[T]``:  ``1`` where ``H_t <= threshold``, else ``0``.
    """
    return (entropy <= threshold).float()


# ---------------------------------------------------------------------------
# Segment-position weighting
# ---------------------------------------------------------------------------

def compute_segment_weights(
    num_segments: int,
    mode: str = "uniform",
    device: torch.device | None = None,
) -> torch.Tensor:
    r"""Per-segment aggregation weights (sum to 1).

    Modes:

    * ``"uniform"``: :math:`w_k = 1/K`
    * ``"linear_increasing"``: later segments get linearly more weight.
    * ``"exponential_increasing"``: later segments get exponentially more weight.

    Args:
        num_segments: Number of segments *K*.
        mode: Weighting strategy name.
        device: Target device.

    Returns:
        Weight tensor of shape ``[K]``, summing to 1.
    """
    if num_segments <= 0:
        return torch.ones(1, device=device)

    if mode == "uniform":
        return torch.ones(num_segments, device=device) / num_segments

    if mode == "linear_increasing":
        raw = torch.arange(1, num_segments + 1, dtype=torch.float, device=device)
        return raw / raw.sum()

    if mode == "exponential_increasing":
        raw = torch.exp(
            torch.arange(num_segments, dtype=torch.float, device=device)
        )
        return raw / raw.sum()

    raise ValueError(f"Unknown segment weight mode: {mode!r}")


# ---------------------------------------------------------------------------
# Combined application
# ---------------------------------------------------------------------------

def apply_weighted_kl_to_jsd(
    jsd_values: torch.Tensor,
    teacher_logits: torch.Tensor,
    process_group: dist.ProcessGroup | None,
    *,
    segment_boundaries: list[tuple[int, int]] | None = None,
    confidence_weight_mode: str | None = None,
    confidence_weight_temp: float = 1.0,
    confidence_threshold: float | None = None,
    segment_weight_mode: str = "uniform",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Apply confidence + segment weighting to raw per-token JSD values.

    This is the main entry point that composes the two weighting schemes.

    Args:
        jsd_values: Raw per-token JSD, shape ``[resp_len]``.
        teacher_logits: Teacher logits for entropy, ``[resp_len, V_local]``.
        process_group: TP process group.
        segment_boundaries: ``[(s0, e0), (s1, e1), ...]`` in token space.
        confidence_weight_mode: Entropy weighting mode or ``None`` to skip.
        confidence_weight_temp: Temperature :math:`\\tau` for entropy weights.
        confidence_threshold: Entropy threshold or ``None`` to skip masking.
        segment_weight_mode: Segment-position weighting mode.

    Returns:
        Tuple of ``(weighted_jsd, metrics)`` where ``metrics`` contains
        detached diagnostic tensors (mean entropy, fraction masked, etc.).
    """
    weighted = jsd_values.clone()
    metrics: dict[str, torch.Tensor] = {}

    # ---- 1. Teacher confidence weighting / masking ----
    need_entropy = (confidence_weight_mode is not None) or (confidence_threshold is not None)
    if need_entropy:
        teacher_entropy = compute_teacher_token_entropy(teacher_logits, process_group)
        metrics["teacher_entropy_mean"] = teacher_entropy.mean().detach()

        if confidence_weight_mode is not None:
            conf_weights = entropy_to_confidence_weights(
                teacher_entropy,
                mode=confidence_weight_mode,
                temperature=confidence_weight_temp,
            )
            weighted = weighted * conf_weights
            metrics["confidence_weight_mean"] = conf_weights.mean().detach()

        if confidence_threshold is not None:
            conf_mask = threshold_confidence_mask(teacher_entropy, confidence_threshold)
            weighted = weighted * conf_mask
            frac_masked = 1.0 - conf_mask.mean()
            metrics["confidence_masked_frac"] = frac_masked.detach()

    # ---- 2. Segment-position weighting ----
    if segment_boundaries is not None and len(segment_boundaries) > 1:
        K = len(segment_boundaries)
        seg_weights = compute_segment_weights(
            K, mode=segment_weight_mode, device=weighted.device,
        )
        for seg_idx, (seg_start, seg_end) in enumerate(segment_boundaries):
            actual_end = min(seg_end, weighted.size(0))
            if seg_start < actual_end:
                # Rescale so that the overall mean is preserved:
                #   weight_k * K  (since uniform would give 1/K * K = 1)
                weighted[seg_start:actual_end] *= seg_weights[seg_idx] * K
        metrics["num_segments"] = torch.tensor(float(K), device=weighted.device)

    return weighted, metrics
