#!/usr/bin/env python3
"""
Prepare meituan-longcat/AMO-Bench for OPSD evaluation (or training).

AMO-Bench has 50 human-crafted IMO-level math problems.
Fields: question_id, prompt, solution, answer, answer_type

The answer may be in \\boxed{...} format; we extract the inner content for label.

Output format (one JSON per line):
  {
    "prompt"  : [{"role": "user", "content": "<problem>"}],
    "label"   : "<answer>",
    "metadata": {"solution": "<full solution>", "raw_content": "<problem>"}
  }

Usage:
  # Eval: all 50 problems
  python prepare_amo_bench.py --mode eval --num-samples 50 \\
      --output /root/data/amo_bench_eval.jsonl --seed 42

  # China mirror
  python prepare_amo_bench.py --mode eval --num-samples 50 \\
      --output /root/data/amo_bench_eval.jsonl --seed 42 \\
      --hf-endpoint https://hf-mirror.com
"""

import argparse
import json
import os
import pathlib
import random
import sys

AMO_BENCH_REPO = "meituan-longcat/AMO-Bench"


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


def _extract_boxed_answer(answer: str) -> str:
    """Extract content from \\boxed{...} if present, otherwise return answer as-is."""
    marker = r"\boxed{"
    idx = answer.rfind(marker)
    if idx == -1:
        marker2 = r"\boxed "
        idx2 = answer.rfind(marker2)
        if idx2 != -1:
            token = answer[idx2 + len(marker2):].strip().split()[0]
            return token
        return answer.strip()
    start = idx + len(marker)
    depth = 1
    i = start
    while i < len(answer) and depth > 0:
        if answer[i] == "{":
            depth += 1
        elif answer[i] == "}":
            depth -= 1
        i += 1
    return answer[start : i - 1].strip()


def _row_to_entry(row: dict) -> dict | None:
    """Convert one row to a training entry. Returns None for malformed rows."""
    problem = str(row.get("prompt", "")).strip()
    solution = str(row.get("solution", "")).strip()
    answer_raw = str(row.get("answer", "")).strip()
    if not problem or not answer_raw:
        return None
    label = _extract_boxed_answer(answer_raw)
    if not label:
        label = answer_raw
    return {
        "prompt": [{"role": "user", "content": problem}],
        "label": label,
        "metadata": {"solution": solution, "raw_content": problem},
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
        description="Prepare AMO-Bench for OPSD evaluation or training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--mode", choices=["train", "eval"], required=True,
                        help="Output mode: 'train' or 'eval'")
    parser.add_argument("--num-samples", type=int, required=True,
                        help="Number of samples to output (AMO-Bench has 50 total)")
    parser.add_argument("--output", required=True,
                        help="Output JSONL file path")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffling (default: 42)")
    parser.add_argument("--train-file", default="",
                        help="(eval mode) Path to train JSONL to exclude for contamination check")
    parser.add_argument("--cache-dir", default="",
                        help="HuggingFace dataset cache directory")
    parser.add_argument("--hf-endpoint", default="", metavar="URL",
                        help="HuggingFace mirror, e.g. https://hf-mirror.com")
    args = parser.parse_args()

    if args.mode == "train" and args.train_file:
        print("WARNING: --train-file is only used in eval mode; ignoring.")

    _set_hf_endpoint(args.hf_endpoint)
    hf = _import_datasets()

    cache_dir = args.cache_dir or None
    print(f"Loading '{AMO_BENCH_REPO}' ...")
    ds = hf.load_dataset(AMO_BENCH_REPO, cache_dir=cache_dir, trust_remote_code=True)

    # Merge all splits
    all_rows = []
    for split in ds.keys():
        split_rows = [dict(r) for r in ds[split]]
        print(f"  Split '{split}': {len(split_rows)} rows.")
        all_rows.extend(split_rows)
    print(f"  Total rows across all splits: {len(all_rows)}")

    if not all_rows:
        print("ERROR: dataset is empty.")
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
    if skipped_malformed:
        print(f"  Skipped (malformed)     : {skipped_malformed}")
    if skipped_contaminated:
        print(f"  Skipped (contamination) : {skipped_contaminated}")
    if written < args.num_samples:
        print(f"  WARNING: requested {args.num_samples} but only {written} available "
              f"(after filtering {skipped_contaminated} contaminated rows).")


if __name__ == "__main__":
    main()
