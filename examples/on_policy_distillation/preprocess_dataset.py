"""Unified dataset preprocessing for OPSDC training and evaluation.

Converts HuggingFace datasets to a common JSONL format understood by slime:
    {
        "prompt":    [{"role": "user", "content": "<full student user message>"}],
        "label":     "<final answer string, for reward logging>",
        "metadata":  {
            "raw_content":          "<original problem text, no format instruction>",
            "student_user_content": "<full student user message (same as prompt content)>",
            "raw_problem":          "<original problem text>",
            "source":               "<dataset source tag>",
            # train-only fields (used by OPSD full/answer_only/masked_reasoning modes):
            "reference_solution":   "<full reference solution, may be empty>",
            "solution":             "<same as reference_solution>",
        }
    }

Supported dataset formats (auto-detected from column names):
  - dapo:          BytedTsinghua-SIA/DAPO-Math-17k
                   prompt=list[dict], reward_model={'ground_truth':...}
  - openthoughts:  open-thoughts/OpenThoughts-114k  (config='metadata')
                   problem=str, ground_truth_solution=str
  - simple:        math-ai/aime25, math-ai/MATH-500, etc.
                   problem=str, answer=str  (or solution=str)
  - generic:       any other dataset -- field names guessed automatically

Usage:
    python preprocess_dataset.py \\
        --dataset BytedTsinghua-SIA/DAPO-Math-17k \\
        --split train \\
        --output /root/math/data/dapo_train.jsonl

    python preprocess_dataset.py \\
        --dataset open-thoughts/OpenThoughts-114k \\
        --config metadata \\
        --split train \\
        --output /root/math/data/openthoughts_train.jsonl

    python preprocess_dataset.py \\
        --dataset math-ai/aime25 \\
        --split test \\
        --output /root/math/data/aime25_eval.jsonl \\
        --max-samples 100
"""

import argparse
import ast
import json
import logging
import pathlib
import re
import sys

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Answer-format instruction templates appended to bare problem texts.
# Used for datasets whose problem field does NOT already include one.
#
# Two styles are supported (select with --answer-format):
#   "answer"  (default) – DAPO style: model outputs "Answer: <value>" on the last line.
#   "boxed"             – LaTeX style: model wraps its final answer in \boxed{}.
#
# Both templates use the {problem} placeholder for the raw problem text.
# ---------------------------------------------------------------------------
_ANSWER_FORMAT_INSTRUCTION = (
    "Solve the following math problem step by step. "
    "The last line of your response should be of the form "
    "Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n"
    "{problem}\n\n"
    "Remember to put your answer on its own line after \"Answer:\"."
)

_BOXED_FORMAT_INSTRUCTION = (
    "{problem}\n\n"
    "Please reason step by step, and put your final answer within \\boxed{{}}."
)

# Keep the legacy name as an alias so other code that imported _DAPO_FORMAT_INSTRUCTION still works.
_DAPO_FORMAT_INSTRUCTION = _ANSWER_FORMAT_INSTRUCTION

# Short tail-only strings stored in metadata['format_instruction'].
# Used by the teacher prompt builder in on_policy_self_distillation.py so the
# teacher always appends the same format instruction as the student.
_FORMAT_SUFFIX: dict[str, str] = {
    "answer": (
        "Please reason step by step. "
        "The last line of your response should be of the form "
        "Answer: $Answer (without quotes) where $Answer is the answer to the problem. "
        "Remember to put your answer on its own line after \"Answer:\"."
    ),
    "boxed": "Please reason step by step, and put your final answer within \\boxed{}.",
}


def _normalize(v) -> str:
    if v is None:
        return ""
    s = str(v).strip()
    return "" if s.lower() in {"nan", "none", "null"} else s


def _safe_literal(v):
    """Parse a string that might be a Python literal (dict/list)."""
    if isinstance(v, (dict, list)):
        return v
    if isinstance(v, str):
        try:
            return ast.literal_eval(v)
        except Exception:
            pass
    return v


def _extract_boxed_answer(text: str) -> str:
    """Pull the last \\boxed{...} expression, handling nested braces."""
    last = ""
    i = 0
    while i < len(text):
        idx = text.find(r"\boxed{", i)
        if idx == -1:
            break
        # Walk forward tracking brace depth
        depth = 0
        start = idx + len(r"\boxed{") - 1  # points at the opening '{'
        j = start
        while j < len(text):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    last = text[start + 1 : j].strip()
                    break
            j += 1
        i = idx + 1
    return last


# ---------------------------------------------------------------------------
# Per-format extractors
# Each returns (problem_raw, student_user_content, label, reference_solution, source)
# ---------------------------------------------------------------------------

def _extract_dapo(row: dict):
    """BytedTsinghua-SIA/DAPO-Math-17k format.

    prompt: list[{'role':'user','content': <full formatted problem>}]
    reward_model: {'ground_truth': '<answer>', 'style': ...}
    """
    raw_prompt = _safe_literal(row.get("prompt", []))
    student_user_content = ""
    for msg in raw_prompt:
        if isinstance(msg, dict) and msg.get("role") == "user":
            student_user_content = _normalize(msg.get("content", ""))
            break

    reward_model = _safe_literal(row.get("reward_model", {}))
    label = _normalize(reward_model.get("ground_truth", "")) if isinstance(reward_model, dict) else ""

    # raw problem = student content (already includes format instruction)
    problem_raw = student_user_content
    source = _normalize(row.get("data_source", "dapo"))
    return problem_raw, student_user_content, label, "", source


def _extract_openthoughts(row: dict):
    """open-thoughts/OpenThoughts-114k (metadata config) format.

    problem: str  (bare problem, no format instruction)
    ground_truth_solution: str  (full solution, may contain \\boxed{})
    """
    problem_raw = _normalize(row.get("problem", ""))
    reference_solution = _normalize(
        row.get("ground_truth_solution") or row.get("deepseek_solution") or ""
    )

    # Wrap bare problem in standard DAPO-style format instruction (may be overridden in preprocess())
    student_user_content = _ANSWER_FORMAT_INSTRUCTION.format(problem=problem_raw)

    # Try to extract final boxed answer; fall back to full solution for grading
    label = _extract_boxed_answer(reference_solution) or _normalize(
        row.get("ground_truth_solution") or ""
    )

    source = _normalize(row.get("source") or row.get("domain") or "openthoughts")
    return problem_raw, student_user_content, label, reference_solution, source


def _extract_simple(row: dict):
    """Simple format: problem + answer (aime25, MATH-500, etc.)

    Fields: problem, answer  (or solution)
    """
    problem_raw = _normalize(row.get("problem") or row.get("question") or "")
    label = _normalize(row.get("answer") or row.get("solution") or "")

    student_user_content = _ANSWER_FORMAT_INSTRUCTION.format(problem=problem_raw)
    source = _normalize(row.get("source") or row.get("id") or "eval")
    return problem_raw, student_user_content, label, "", source


def _extract_generic(row: dict):
    """Best-effort extraction for unknown schemas."""
    # problem
    problem_raw = _normalize(
        row.get("problem") or row.get("question") or row.get("prompt") or ""
    )
    if not problem_raw:
        # Try DAPO-style prompt list
        raw_prompt = _safe_literal(row.get("prompt", []))
        if isinstance(raw_prompt, list):
            for msg in raw_prompt:
                if isinstance(msg, dict) and msg.get("role") == "user":
                    problem_raw = _normalize(msg.get("content", ""))
                    break

    # label / answer
    label = _normalize(
        row.get("answer")
        or row.get("label")
        or row.get("ground_truth")
        or ""
    )
    if not label:
        reward_model = _safe_literal(row.get("reward_model", {}))
        if isinstance(reward_model, dict):
            label = _normalize(reward_model.get("ground_truth", ""))
    if not label:
        solution = _normalize(row.get("solution") or row.get("ground_truth_solution") or "")
        label = _extract_boxed_answer(solution) or solution

    reference_solution = _normalize(
        row.get("ground_truth_solution") or row.get("solution") or ""
    )
    student_user_content = problem_raw  # keep as-is for unknown formats
    source = _normalize(row.get("source") or row.get("data_source") or "unknown")
    return problem_raw, student_user_content, label, reference_solution, source


def _detect_format(column_names: list[str]) -> str:
    cols = set(column_names)
    if "reward_model" in cols and "data_source" in cols:
        return "dapo"
    if "ground_truth_solution" in cols or "deepseek_reasoning" in cols:
        return "openthoughts"
    if "problem" in cols and ("answer" in cols or "solution" in cols):
        return "simple"
    return "generic"


_EXTRACTORS = {
    "dapo": _extract_dapo,
    "openthoughts": _extract_openthoughts,
    "simple": _extract_simple,
    "generic": _extract_generic,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def preprocess(
    dataset: str,
    output: str,
    split: str = "train",
    config: str | None = None,
    max_samples: int | None = None,
    fmt: str | None = None,
    answer_format: str = "answer",
) -> int:
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("datasets package is required: pip install datasets")
        sys.exit(1)

    logger.info(f"Loading {dataset}" + (f" ({config})" if config else "") + f" split={split}")
    load_kwargs = {"split": split}
    if config:
        load_kwargs["name"] = config
    ds = load_dataset(dataset, **load_kwargs)

    detected = fmt or _detect_format(ds.column_names)
    logger.info(f"Detected format: {detected}  (columns: {ds.column_names})")
    extractor = _EXTRACTORS[detected]

    output_path = pathlib.Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resolve the format template and its short suffix for this run.
    if answer_format not in _FORMAT_SUFFIX:
        raise ValueError(f"--answer-format must be 'answer' or 'boxed', got: {answer_format!r}")
    fmt_template = _ANSWER_FORMAT_INSTRUCTION if answer_format == "answer" else _BOXED_FORMAT_INSTRUCTION
    fmt_suffix = _FORMAT_SUFFIX[answer_format]
    logger.info(f"Answer format: {answer_format!r}")

    written = skipped = 0
    with output_path.open("w", encoding="utf-8") as fout:
        for row in ds:
            if max_samples is not None and written >= max_samples:
                break

            problem_raw, student_user_content, label, reference_solution, source = extractor(row)

            if not problem_raw and not student_user_content:
                skipped += 1
                continue

            # For formats that expose a bare problem (openthoughts, simple, generic),
            # rebuild student_user_content using the chosen answer-format template.
            # DAPO content is pre-formatted by the dataset and is left unchanged.
            if detected != "dapo" and problem_raw:
                student_user_content = fmt_template.format(problem=problem_raw)
                effective_fmt_suffix = fmt_suffix
            else:
                # DAPO: format instruction is baked into the dataset content (always Answer: style).
                # Override fmt_suffix so metadata['format_instruction'] matches what the student
                # actually sees, regardless of what --answer-format was passed.
                effective_fmt_suffix = _FORMAT_SUFFIX["answer"]

            entry = {
                "prompt": [{"role": "user", "content": student_user_content}],
                "label": label,
                "metadata": {
                    "raw_content": problem_raw,
                    "student_user_content": student_user_content,
                    "raw_problem": problem_raw,
                    "source": source,
                    "reference_solution": reference_solution,
                    "solution": reference_solution,
                    # Stored so the teacher prompt builder uses the same format at runtime.
                    "format_instruction": effective_fmt_suffix,
                },
            }
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
            written += 1

    logger.info(f"Written {written} entries to {output}  (skipped {skipped})")
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="HuggingFace dataset repo id")
    parser.add_argument("--config", default=None, help="Dataset config/subset name")
    parser.add_argument("--split", default="train", help="Dataset split (default: train)")
    parser.add_argument("--output", required=True, help="Output JSONL file path")
    parser.add_argument("--max-samples", type=int, default=None, help="Truncate to N samples")
    parser.add_argument(
        "--format",
        dest="fmt",
        choices=["dapo", "openthoughts", "simple", "generic"],
        default=None,
        help="Force a specific extraction format (default: auto-detect)",
    )
    parser.add_argument(
        "--answer-format",
        dest="answer_format",
        choices=["answer", "boxed"],
        default="answer",
        help=(
            "Format instruction style appended to each problem. "
            "'answer' (default): DAPO style – model outputs 'Answer: <value>' on the last line. "
            "'boxed': LaTeX style – model wraps its final answer in \\boxed{}. "
            "Note: DAPO datasets already embed the format instruction; this flag affects "
            "openthoughts, simple, and generic formats only."
        ),
    )
    args = parser.parse_args()
    preprocess(
        dataset=args.dataset,
        output=args.output,
        split=args.split,
        config=args.config,
        max_samples=args.max_samples,
        fmt=args.fmt,
        answer_format=args.answer_format,
    )


if __name__ == "__main__":
    main()
