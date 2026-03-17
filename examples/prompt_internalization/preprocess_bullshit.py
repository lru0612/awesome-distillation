"""Preprocess bullshit-benchmark for Prompt Internalization eval.

Converts questions from questions.v2.json into the standard slime eval JSONL format.
This is an eval-only dataset — no teacher/constitution fields are included.

Output format:
    {
        "prompt":   [{"role": "user", "content": "<question text>"}],
        "label":    "bullshit" | "safe",
        "metadata": {
            "prompt_text":         "<question text>",
            "rm_type":             "bullshit",
            "judge_mode":          "keyword" | "llm",
            "source":              "bullshit-benchmark",
            "nonsensical_element": "<ground truth: what is fabricated/nonsensical>",
            "technique":           "<technique id>",
            "domain":              "<domain>",
            "question_id":         "<id>",
        }
    }

Label convention:
  label = "bullshit"  →  question contains a fabricated/nonsensical premise;
                         reward=1 if the model identifies the incoherence.
  label = "safe"      →  question is legitimate (control);
                         reward=1 if the model answers normally without false refusal.

Usage:
    # All questions (bullshit + controls), keyword judge:
    python examples/prompt_internalization/preprocess_bullshit.py \\
        --questions /path/to/bullshit-benchmark/questions.v2.json \\
        --output /root/pi_safety/eval_bullshit.jsonl

    # LLM judge (recommended for final eval):
    python examples/prompt_internalization/preprocess_bullshit.py \\
        --questions /path/to/bullshit-benchmark/questions.v2.json \\
        --output /root/pi_safety/eval_bullshit.jsonl \\
        --judge-mode llm

    # Filter to specific techniques only:
    python examples/prompt_internalization/preprocess_bullshit.py \\
        --questions /path/to/bullshit-benchmark/questions.v2.json \\
        --output /root/pi_safety/eval_bullshit_pnf.jsonl \\
        --techniques plausible_nonexistent_framework
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def preprocess(
    questions_path: str,
    output: str,
    techniques_filter: list[str],
    include_controls: bool,
    judge_mode: str,
    max_samples: int | None,
) -> int:
    with open(questions_path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)

    techniques = payload.get("techniques")
    if not isinstance(techniques, list):
        logger.error("questions file must contain a top-level 'techniques' array.")
        sys.exit(1)

    allowed = set(techniques_filter)
    bullshit_rows: list[dict] = []
    safe_rows: list[dict] = []

    for technique in techniques:
        technique_id = str(technique.get("technique", "")).strip()
        if allowed and technique_id not in allowed:
            continue

        is_control_technique = technique_id == "control_legitimate"

        for question in technique.get("questions", []):
            is_control = bool(question.get("is_control", False)) or is_control_technique
            q_text = question.get("question", "").strip()
            if not q_text:
                continue

            entry = {
                "prompt": [{"role": "user", "content": q_text}],
                "label": "safe" if is_control else "bullshit",
                "metadata": {
                    "prompt_text": q_text,
                    "rm_type": "bullshit",
                    "judge_mode": judge_mode,
                    "source": "bullshit-benchmark",
                    "nonsensical_element": question.get("nonsensical_element", ""),
                    "technique": technique_id,
                    "domain": question.get("domain", ""),
                    "question_id": question.get("id", ""),
                },
            }

            if is_control:
                safe_rows.append(entry)
            else:
                bullshit_rows.append(entry)

    logger.info("Loaded — bullshit: %d, control/safe: %d", len(bullshit_rows), len(safe_rows))

    rows = bullshit_rows + (safe_rows if include_controls else [])
    if max_samples is not None:
        rows = rows[:max_samples]

    output_path = pathlib.Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    logger.info(
        "Written %d entries → %s  (bullshit=%d, safe=%d)",
        len(rows), output, len(bullshit_rows),
        len(safe_rows) if include_controls else 0,
    )
    return len(rows)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--questions", required=True,
        help="Path to questions.v2.json from bullshit-benchmark.",
    )
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument(
        "--techniques", nargs="*", default=[],
        help="Only include these technique IDs. Default: all.",
    )
    parser.add_argument(
        "--no-controls", action="store_true",
        help="Exclude control (legitimate) questions.",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--judge-mode", choices=["keyword", "llm"], default="keyword",
        help="Stored in metadata['judge_mode']. Use 'llm' for final eval.",
    )
    args = parser.parse_args()
    preprocess(
        questions_path=args.questions,
        output=args.output,
        techniques_filter=args.techniques or [],
        include_controls=not args.no_controls,
        judge_mode=args.judge_mode,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
