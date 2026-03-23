"""Filter open-thoughts/OpenThoughts-114k to clean math samples only.

Applies two filters:
  1. Domain filter: keep only rows with domain == "math"
     (removes code, biology, physics, chemistry, puzzle domains)
  2. Answer filter: keep only rows where a \\boxed{} answer can be extracted
     from ground_truth_solution (or deepseek_solution as fallback)

Output JSONL format matches preprocess_dataset.py:
    {
        "prompt":   [{"role": "user", "content": "<formatted problem>"}],
        "label":    "<boxed answer>",
        "metadata": {
            "raw_problem":          "<original problem text>",
            "student_user_content": "<same as prompt content>",
            "source":               "<source tag>",
            "reference_solution":   "<full reference solution>",
            "solution":             "<same as reference_solution>",
            "format_instruction":   "<tail format instruction>",
        }
    }

Usage:
    python filter_openthoughts_math.py --output /root/math/data/train_openthoughts_math.jsonl
    python filter_openthoughts_math.py --output /tmp/out.jsonl --stats-only
"""

import argparse
import json
import logging
import pathlib
import sys
from collections import Counter

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Format templates (same as preprocess_dataset.py)
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

_FORMAT_SUFFIX = {
    "answer": (
        "Please reason step by step. "
        "The last line of your response should be of the form "
        "Answer: $Answer (without quotes) where $Answer is the answer to the problem. "
        "Remember to put your answer on its own line after \"Answer:\"."
    ),
    "boxed": "Please reason step by step, and put your final answer within \\boxed{}.",
}


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def _extract_boxed_answers(text: str) -> list[str]:
    """Return all top-level \\boxed{...} expressions in order, handling nested braces."""
    if not text:
        return []
    results = []
    i = 0
    while i < len(text):
        idx = text.find(r"\boxed{", i)
        if idx == -1:
            break
        depth = 0
        start = idx + len(r"\boxed{") - 1  # points at the opening '{'
        j = start
        while j < len(text):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    results.append(text[start + 1 : j].strip())
                    break
            j += 1
        i = idx + 1
    return results


def _normalize(v) -> str:
    if v is None:
        return ""
    s = str(v).strip()
    return "" if s.lower() in {"nan", "none", "null"} else s


# ---------------------------------------------------------------------------
# Main filter
# ---------------------------------------------------------------------------

def filter_math(
    output: str,
    answer_format: str = "boxed",
    max_samples: int | None = None,
    stats_only: bool = False,
) -> None:
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("datasets package is required: pip install datasets")
        sys.exit(1)

    if answer_format not in _FORMAT_SUFFIX:
        raise ValueError(f"--answer-format must be 'answer' or 'boxed', got: {answer_format!r}")
    fmt_template = _ANSWER_FORMAT_INSTRUCTION if answer_format == "answer" else _BOXED_FORMAT_INSTRUCTION
    fmt_suffix = _FORMAT_SUFFIX[answer_format]

    logger.info("Loading open-thoughts/OpenThoughts-114k (config=metadata, split=train)...")
    ds = load_dataset("open-thoughts/OpenThoughts-114k", name="metadata", split="train")
    logger.info(f"Total samples: {len(ds)}")

    # ---- Collect stats ------------------------------------------------
    domain_counts: Counter = Counter()
    kept = skipped_domain = skipped_no_answer = skipped_multi_answer = 0
    skipped_domain_breakdown: Counter = Counter()

    output_path = pathlib.Path(output)
    if not stats_only:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fout = output_path.open("w", encoding="utf-8")
    else:
        fout = None

    try:
        for row in ds:
            if max_samples is not None and kept >= max_samples:
                break

            domain = _normalize(row.get("domain", ""))
            domain_counts[domain] += 1

            # Filter 1: math domain only
            if domain != "math":
                skipped_domain += 1
                skipped_domain_breakdown[domain] += 1
                continue

            # Filter 2: exactly one extractable boxed answer
            reference_solution = _normalize(
                row.get("ground_truth_solution") or row.get("deepseek_solution") or ""
            )
            boxed_answers = _extract_boxed_answers(reference_solution)
            if len(boxed_answers) == 0:
                skipped_no_answer += 1
                continue
            if len(boxed_answers) > 1:
                skipped_multi_answer += 1
                continue
            label = boxed_answers[0]

            if stats_only:
                kept += 1
                continue

            problem_raw = _normalize(row.get("problem", ""))
            if not problem_raw:
                skipped_no_answer += 1
                continue

            student_user_content = fmt_template.format(problem=problem_raw)
            source = _normalize(row.get("source") or row.get("domain") or "openthoughts")

            entry = {
                "prompt": [{"role": "user", "content": student_user_content}],
                "label": label,
                "metadata": {
                    "raw_content": problem_raw,
                    "raw_problem": problem_raw,
                    "student_user_content": student_user_content,
                    "source": source,
                    "reference_solution": reference_solution,
                    "solution": reference_solution,
                    "format_instruction": fmt_suffix,
                },
            }
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
            kept += 1

    finally:
        if fout is not None:
            fout.close()

    # ---- Report -------------------------------------------------------
    logger.info("")
    logger.info("=== Domain breakdown ===")
    for domain, count in domain_counts.most_common():
        logger.info(f"  {domain or '(empty)'}: {count}")

    logger.info("")
    logger.info("=== Filter results ===")
    logger.info(f"  Total samples:                {len(ds)}")
    logger.info(f"  Skipped (non-math domain):    {skipped_domain}")
    for d, c in skipped_domain_breakdown.most_common():
        logger.info(f"    - {d or '(empty)'}: {c}")
    logger.info(f"  Skipped (no boxed answer):    {skipped_no_answer}")
    logger.info(f"  Skipped (multiple boxed):     {skipped_multi_answer}")
    logger.info(f"  Kept (math + single answer):  {kept}")
    if not stats_only:
        logger.info(f"  Output written to:          {output}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output",
        default="/root/math/data/train_openthoughts_math.jsonl",
        help="Output JSONL file path (default: /root/math/data/train_openthoughts_math.jsonl)",
    )
    parser.add_argument(
        "--answer-format",
        dest="answer_format",
        choices=["answer", "boxed"],
        default="boxed",
        help=(
            "Format instruction style appended to each problem. "
            "'boxed' (default): model wraps answer in \\boxed{}. "
            "'answer': DAPO style – model outputs 'Answer: <value>' on last line."
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Stop after writing N samples (for quick testing)",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Print filter statistics without writing any output file",
    )
    args = parser.parse_args()
    filter_math(
        output=args.output,
        answer_format=args.answer_format,
        max_samples=args.max_samples,
        stats_only=args.stats_only,
    )


if __name__ == "__main__":
    main()
