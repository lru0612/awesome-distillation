#!/usr/bin/env python3
"""
Prepare qwedsacf/competition_math for OPSD training or evaluation.

Dataset schema:
  problem  : problem text
  level    : "Level 1" ... "Level 5"
  type     : category (e.g. "Algebra", "Counting & Probability")
  solution : full solution with answer in \\boxed{...}

The ground-truth label is extracted from the last \\boxed{} in the solution.
The full solution is stored in metadata["solution"].

Output format (one JSON per line):
  {
    "prompt"  : [{"role": "user", "content": "<problem>"}],
    "label"   : "<answer extracted from \\\\boxed{}>",
    "metadata": {
      "solution"    : "<full solution>",
      "raw_content" : "<problem>",
      "level"       : "Level 3",
      "type"        : "Algebra"
    }
  }

Usage:
  # Train: 5000 examples from all levels
  python prepare_competition_math.py --mode train --num-samples 5000 \\
      --output /root/data/comp_math_train.jsonl --seed 42

  # Train: only Level 4 and 5
  python prepare_competition_math.py --mode train --num-samples 2000 \\
      --level 4 5 --output /root/data/comp_math_hard_train.jsonl --seed 42

  # Eval: 500 examples, no overlap with train file
  python prepare_competition_math.py --mode eval --num-samples 500 \\
      --output /root/data/comp_math_eval.jsonl \\
      --train-file /root/data/comp_math_train.jsonl --seed 42

  # China mirror
  python prepare_competition_math.py --mode train --num-samples 5000 \\
      --output /root/data/comp_math_train.jsonl --seed 42 \\
      --hf-endpoint https://hf-mirror.com
"""

import argparse
import json
import os
import pathlib
import random
import sys

DATASET_REPO = "qwedsacf/competition_math"

VALID_LEVELS = {"Level 1", "Level 2", "Level 3", "Level 4", "Level 5"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_hf_endpoint(url: str):
    if url:
        os.environ["HF_ENDPOINT"] = url
        print(f"  Using HF endpoint: {url}")


def _import_datasets():
    try:
        import datasets as hf_datasets
        return hf_datasets
    except ImportError:
        print("ERROR: 'datasets' library not installed. Run: pip install datasets")
        sys.exit(1)


def _extract_boxed_answer(solution: str) -> str:
    """Extract the content of the last \\boxed{...} in the solution string.

    Handles arbitrarily nested braces, e.g. \\boxed{\\frac{1}{4}}.
    Falls back to \\boxed <token> (no braces) if present.
    Returns an empty string when no boxed answer is found.
    """
    marker = r"\boxed{"
    idx = solution.rfind(marker)
    if idx == -1:
        # Try \\boxed <single-token> form (rare but present in some problems)
        marker2 = r"\boxed "
        idx2 = solution.rfind(marker2)
        if idx2 != -1:
            token = solution[idx2 + len(marker2):].strip().split()[0]
            return token
        return ""

    start = idx + len(marker)
    depth = 1
    i = start
    while i < len(solution) and depth > 0:
        if solution[i] == "{":
            depth += 1
        elif solution[i] == "}":
            depth -= 1
        i += 1
    return solution[start : i - 1].strip()


def _normalize_level(raw: str) -> str | None:
    """Accept '1', '2', ..., 'Level 1', 'level 1', etc. → 'Level N' or None."""
    raw = raw.strip()
    if raw in VALID_LEVELS:
        return raw
    if raw.isdigit() and 1 <= int(raw) <= 5:
        return f"Level {raw}"
    lower = raw.lower()
    for v in VALID_LEVELS:
        if lower == v.lower():
            return v
    return None


def _row_to_entry(row: dict) -> dict | None:
    """Convert one dataset row to a training entry. Returns None for malformed rows."""
    problem  = str(row.get("problem",  "")).strip()
    solution = str(row.get("solution", "")).strip()
    level    = str(row.get("level",    "")).strip()
    category = str(row.get("type",     "")).strip()

    if not problem or not solution:
        return None

    answer = _extract_boxed_answer(solution)
    if not answer:
        return None  # skip problems with unparseable answers

    return {
        "prompt":   [{"role": "user", "content": problem}],
        "label":    answer,
        "metadata": {
            "solution":    solution,
            "raw_content": problem,
            "level":       level,
            "type":        category,
        },
    }


def _load_exclusion_set(train_file: str) -> set:
    """Build a set of problem strings from an existing output file."""
    path = pathlib.Path(train_file)
    if not path.exists():
        print(f"WARNING: --train-file '{train_file}' not found; skipping contamination check.")
        return set()
    exclusion = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prompt_msgs = obj.get("prompt", [])
            key = ""
            for msg in reversed(prompt_msgs):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    key = str(msg.get("content", "")).strip()
                    break
            if not key:
                key = (obj.get("metadata") or {}).get("raw_content", "").strip()
            if key:
                exclusion.add(key)
    print(f"  Loaded {len(exclusion)} problems from train file for contamination check.")
    return exclusion


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prepare competition_math dataset for OPSD training or evaluation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--mode", choices=["train", "eval"], required=True,
                        help="Output mode: 'train' or 'eval'")
    parser.add_argument("--num-samples", type=int, required=True,
                        help="Number of samples to output")
    parser.add_argument("--output", required=True,
                        help="Output JSONL file path")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffling (default: 42)")
    parser.add_argument(
        "--level", nargs="+", default=[],
        metavar="LEVEL",
        help=(
            "Filter by difficulty level. Accept level numbers (1 2 3) or full strings "
            "('Level 1' 'Level 2'). Default: all levels."
        ),
    )
    parser.add_argument("--train-file", default="",
                        help="(eval mode) Path to train JSONL to exclude for contamination check")
    parser.add_argument("--cache-dir", default="",
                        help="HuggingFace dataset cache directory")
    parser.add_argument("--hf-endpoint", default="", metavar="URL",
                        help="HuggingFace mirror, e.g. https://hf-mirror.com")
    args = parser.parse_args()

    if args.mode == "train" and args.train_file:
        print("WARNING: --train-file is only used in eval mode; ignoring.")

    # Parse and validate level filter
    level_filter: set = set()
    for raw in args.level:
        norm = _normalize_level(raw)
        if norm is None:
            print(f"ERROR: unrecognized level '{raw}'. "
                  f"Valid values: 1-5 or 'Level 1'-'Level 5'.")
            sys.exit(1)
        level_filter.add(norm)
    if level_filter:
        print(f"  Level filter: {sorted(level_filter)}")

    _set_hf_endpoint(args.hf_endpoint)
    hf = _import_datasets()

    cache_dir = args.cache_dir or None
    print(f"Loading '{DATASET_REPO}' ...")
    ds = hf.load_dataset(DATASET_REPO, cache_dir=cache_dir, trust_remote_code=True)

    # Merge all splits (train / test)
    all_rows = []
    for split in ds.keys():
        rows = [dict(r) for r in ds[split]]
        print(f"  Split '{split}': {len(rows)} rows.")
        all_rows.extend(rows)
    print(f"  Total rows: {len(all_rows)}")

    # Apply level filter
    if level_filter:
        all_rows = [r for r in all_rows if str(r.get("level", "")).strip() in level_filter]
        print(f"  After level filter: {len(all_rows)} rows.")

    if not all_rows:
        print("ERROR: no rows remain after filtering.")
        sys.exit(1)

    # Shuffle with fixed seed
    rng = random.Random(args.seed)
    rng.shuffle(all_rows)

    # Build exclusion set for contamination check
    exclusion: set = set()
    if args.mode == "eval" and args.train_file:
        exclusion = _load_exclusion_set(args.train_file)

    # Convert rows → entries, applying exclusion filter
    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = skipped_malformed = skipped_contaminated = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for row in all_rows:
            if written >= args.num_samples:
                break
            entry = _row_to_entry(row)
            if entry is None:
                skipped_malformed += 1
                continue
            if exclusion and entry["metadata"]["raw_content"] in exclusion:
                skipped_contaminated += 1
                continue
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
            written += 1

    print(f"\nDone.")
    print(f"  Output      : {out_path}  ({written} entries)")
    print(f"  Seed        : {args.seed}")
    if level_filter:
        print(f"  Levels      : {sorted(level_filter)}")
    if skipped_malformed:
        print(f"  Skipped (no \\boxed answer)  : {skipped_malformed}")
    if skipped_contaminated:
        print(f"  Skipped (contamination)     : {skipped_contaminated}")
    if written < args.num_samples:
        print(f"  WARNING: requested {args.num_samples} but only {written} available "
              f"(after filtering {skipped_contaminated} contaminated + {skipped_malformed} malformed rows).")


if __name__ == "__main__":
    main()
