"""Logical segmentation of response token sequences into reasoning steps.

Provides multiple strategies to split a response into logical segments:
- ``newline``: Split by newline token IDs (double-newline preferred).
- ``fixed_length``: Split into equal-size token chunks.
- ``token_ids``: Split by user-specified separator token IDs.

All functions operate directly on token ID sequences for efficiency and
robustness (no round-trip through text decoding).

Usage::

    from examples.on_policy_distillation.segmenter import (
        segment_tokens,
        get_separator_token_ids,
    )
    sep_ids = get_separator_token_ids(tokenizer)
    segments = segment_tokens(tokens, strategy="newline", separator_ids=sep_ids)
    # segments: [(0, 34), (34, 80), (80, 128)]
"""

from __future__ import annotations


def get_separator_token_ids(
    tokenizer,
    separators: list[str] | None = None,
) -> set[int]:
    """Derive separator token IDs from a HuggingFace tokenizer.

    Args:
        tokenizer: A HuggingFace ``AutoTokenizer`` instance.
        separators: Strings that mark segment boundaries.
            Defaults to ``["\\n\\n", "\\n"]``.

    Returns:
        Set of token IDs that act as segment separators.
    """
    if separators is None:
        separators = ["\n\n", "\n"]
    ids: set[int] = set()
    for sep in separators:
        encoded = tokenizer.encode(sep, add_special_tokens=False)
        if encoded:
            ids.add(encoded[-1])
    return ids


def segment_by_separator_ids(
    tokens: list[int] | tuple[int, ...],
    separator_ids: set[int],
    min_segment_len: int = 10,
) -> list[tuple[int, int]]:
    """Segment a token sequence by separator token IDs.

    A new segment boundary is created when a separator token is encountered
    **and** the current segment already has at least ``min_segment_len`` tokens.
    Short trailing fragments are merged into the last segment.

    Args:
        tokens: Flat list of token IDs (response only).
        separator_ids: Token IDs that trigger a split.
        min_segment_len: Minimum tokens per segment.

    Returns:
        List of ``(start, end)`` half-open intervals in token space.
    """
    n = len(tokens)
    if n == 0:
        return [(0, 0)]

    segments: list[tuple[int, int]] = []
    start = 0
    for i, tok in enumerate(tokens):
        if tok in separator_ids and (i + 1 - start) >= min_segment_len:
            segments.append((start, i + 1))
            start = i + 1

    if start < n:
        if segments and (n - start) < min_segment_len:
            segments[-1] = (segments[-1][0], n)
        else:
            segments.append((start, n))

    return segments if segments else [(0, n)]


def segment_by_fixed_length(
    total_len: int,
    chunk_size: int,
) -> list[tuple[int, int]]:
    """Segment into fixed-size token chunks.

    Args:
        total_len: Total number of response tokens.
        chunk_size: Target tokens per segment.

    Returns:
        List of ``(start, end)`` half-open intervals.
    """
    if total_len == 0:
        return [(0, 0)]
    segments: list[tuple[int, int]] = []
    for start in range(0, total_len, chunk_size):
        end = min(start + chunk_size, total_len)
        segments.append((start, end))
    return segments


def segment_tokens(
    tokens: list[int] | tuple[int, ...],
    *,
    tokenizer=None,
    strategy: str = "newline",
    separator_ids: set[int] | None = None,
    min_segment_len: int = 10,
    chunk_size: int = 128,
) -> list[tuple[int, int]]:
    """Segment a response token sequence.

    Main entry point. Dispatches to the selected strategy.

    Args:
        tokens: Response token IDs.
        tokenizer: Optional tokenizer (used to derive separator IDs when
            ``separator_ids`` is not provided and strategy needs them).
        strategy: ``"newline"`` | ``"fixed_length"`` | ``"token_ids"``.
        separator_ids: Explicit separator token IDs (overrides tokenizer lookup).
        min_segment_len: Minimum tokens per segment (for separator strategies).
        chunk_size: Chunk size for ``"fixed_length"`` strategy.

    Returns:
        List of ``(start, end)`` half-open intervals covering all response tokens.
    """
    if strategy == "fixed_length":
        return segment_by_fixed_length(len(tokens), chunk_size)

    if strategy in ("newline", "token_ids", "separator"):
        if separator_ids is None and tokenizer is not None:
            separator_ids = get_separator_token_ids(tokenizer)
        if not separator_ids:
            return segment_by_fixed_length(len(tokens), chunk_size)
        return segment_by_separator_ids(tokens, separator_ids, min_segment_len)

    raise ValueError(
        f"Unknown segmentation strategy: {strategy!r}. "
        f"Supported: 'newline', 'fixed_length', 'token_ids'."
    )
