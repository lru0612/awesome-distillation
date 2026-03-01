"""Teacher context construction for segmented on-policy self-distillation.

Builds the privileged teacher prompt and constructs per-segment teacher
token sequences for the segmented distillation algorithm:

* **Standard mode** (teacher-forcing): One teacher forward pass on the full
  privileged context + student response.  Since causal attention makes
  logits at position *t* depend only on tokens 0..*t*-1, this is equivalent
  to running separate forward passes per segment — but far more efficient.

* **Lookahead generation mode** (future extension): At each segment
  boundary, the teacher autoregressively generates *L* tokens.  The KL
  loss is then computed on *teacher-generated* tokens rather than student
  tokens.  This is more expensive but allows the student to learn the
  teacher's preferred continuation path.

Usage::

    from examples.on_policy_distillation.teacher_lookahead import (
        build_privileged_prompt_tokens,
        build_teacher_tokens_standard,
        build_segment_teacher_contexts,
    )
"""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def _get_tokenizer(model_path: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def build_privileged_prompt_tokens(
    raw_content: str,
    label: str,
    tokenizer,
) -> list[int]:
    """Tokenize the privileged teacher prompt (question + answer hint).

    The prompt ends with ``add_generation_prompt=True`` so the assistant
    turn is *open*, followed by a hint containing the ground-truth answer.

    Args:
        raw_content: Original user question text.
        label: Ground-truth answer string.
        tokenizer: HuggingFace tokenizer.

    Returns:
        Token IDs for the full privileged prompt (no response tokens).
    """
    teacher_messages = [{"role": "user", "content": raw_content}]
    prompt_text = tokenizer.apply_chat_template(
        teacher_messages, tokenize=False, add_generation_prompt=True,
    )
    prompt_text += f"The answer is {label}. Let me verify this step by step.\n"
    return tokenizer.encode(prompt_text, add_special_tokens=False)


def build_teacher_tokens_standard(
    privileged_prompt_tokens: list[int],
    student_response_tokens: list[int],
) -> tuple[list[int], int]:
    """Build a single teacher token sequence (standard OPSD).

    ``teacher_tokens = privileged_prompt + student_response``

    Args:
        privileged_prompt_tokens: From :func:`build_privileged_prompt_tokens`.
        student_response_tokens: The student's on-policy response tokens.

    Returns:
        ``(teacher_tokens, teacher_prompt_length)``
    """
    teacher_tokens = privileged_prompt_tokens + list(student_response_tokens)
    return teacher_tokens, len(privileged_prompt_tokens)


def build_segment_teacher_contexts(
    privileged_prompt_tokens: list[int],
    student_response_tokens: list[int],
    segment_boundaries: list[tuple[int, int]],
    lookahead_tokens: int = 0,
) -> list[dict]:
    """Build per-segment teacher token sequences.

    For each segment *k*, the teacher context is::

        privileged_prompt + student_response[0 : end_of_segment_k + lookahead]

    The KL loss is computed only on the segment's own tokens (from
    ``seg_start`` to ``seg_end``).

    .. note::
        For causal models a **single** forward pass on the full context
        yields identical logits at each position.  This function is
        primarily useful for the *lookahead generation* variant where the
        teacher actually generates new tokens at each boundary.

    Args:
        privileged_prompt_tokens: Teacher prompt tokens (with answer hint).
        student_response_tokens: Student response tokens.
        segment_boundaries: ``[(s0, e0), (s1, e1), ...]`` in response space.
        lookahead_tokens: Extra tokens beyond the segment end to include.

    Returns:
        List of dicts (one per segment)::

            {
                "teacher_tokens": [...],
                "teacher_prompt_length": int,
                "loss_start": int,   # in teacher sequence coords
                "loss_end": int,
                "segment_idx": int,
            }
    """
    total_resp_len = len(student_response_tokens)
    pp_len = len(privileged_prompt_tokens)
    results: list[dict] = []

    for seg_idx, (seg_start, seg_end) in enumerate(segment_boundaries):
        context_end = min(seg_end + lookahead_tokens, total_resp_len)
        teacher_tokens = (
            privileged_prompt_tokens + student_response_tokens[:context_end]
        )
        results.append({
            "teacher_tokens": teacher_tokens,
            "teacher_prompt_length": pp_len + seg_start,
            "loss_start": pp_len + seg_start,
            "loss_end": pp_len + seg_end,
            "segment_idx": seg_idx,
        })

    return results
