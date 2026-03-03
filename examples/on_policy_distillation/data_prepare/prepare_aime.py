#!/usr/bin/env python3
"""
Prepare HuggingFaceH4/aime_2024 for OPSD evaluation (or training).

AIME-2024 dataset has 30 competition problems total.
All available splits are merged before sampling.

Output format (one JSON per line):
  {
    "prompt"  : [{"role": "user", "content": "<problem>"}],
    "label"   : "<answer>",
    "metadata": {"solution": "<answer>", "raw_content": "<problem>"}
  }

Usage:
  # Eval: all 30 problems
  python prepare_aime.py --mode eval --num-samples 30 \\
      --output /root/data/aime_eval.jsonl --seed 42

  # Eval with contamination check against a train file
  python prepare_aime.py --mode eval --num-samples 30 \\
      --output /root/data/aime_eval.jsonl \\
      --train-file /root/data/dapo_train.jsonl --seed 42

  # China mirror
  python prepare_aime.py --mode eval --num-samples 30 \\
      --output /root/data/aime_eval.jsonl --seed 42 \\
      --hf-endpoint https://hf-mirror.com
"""

import argparse
import json
import os
import pathlib
import random
import sys

AIME_REPO = "HuggingFaceH4/aime_2024"

# Candidate column names (checked in order, case-insensitive)
PROBLEM_COLS = ["problem", "question", "prompt", "input", "instruction", "content", "text"]
ANSWER_COLS  = ["answer", "solution", "output", "response", "label", "target"]


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


def _detect_col(row_keys: list, candidates: list):
    keys_lower = {k.lower(): k for k in row_keys}
    for c in candidates:
        if c in keys_lower:
            return keys_lower[c]
    return None


def _aime_row_to_entry(row: dict, pcol: str, acol: str) -> dict | None:
    """Convert one AIME row to a training entry. Returns None for malformed rows."""
    problem = str(row.get(pcol, "")).strip()
    answer  = str(row.get(acol, "")).strip()
    if not problem or not answer:
        return None
    return {
        "prompt":   [{"role": "user", "content": problem}],
        "label":    answer,
        "metadata": {"solution": answer, "raw_content": problem},
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
            # Use the last user message content as the dedup key
            prompt_msgs = obj.get("prompt", [])
            key = ""
            for msg in reversed(prompt_msgs):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    key = str(msg.get("content", "")).strip()
                    break
            if not key:
                # Fallback: raw_content in metadata
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
        description="Prepare AIME-2024 for OPSD evaluation or training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--mode", choices=["train", "eval"], required=True,
                        help="Output mode: 'train' or 'eval'")
    parser.add_argument("--num-samples", type=int, required=True,
                        help="Number of samples to output (AIME 2024 has 30 total)")
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
    print(f"Loading '{AIME_REPO}' ...")
    ds = hf.load_dataset(AIME_REPO, cache_dir=cache_dir, trust_remote_code=True)

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

    # Detect problem and answer columns
    keys = list(all_rows[0].keys())
    pcol = _detect_col(keys, PROBLEM_COLS)
    acol = _detect_col(keys, ANSWER_COLS)
    if pcol is None:
        print(f"ERROR: cannot detect problem column. Available: {keys}")
        sys.exit(1)
    if acol is None:
        print(f"ERROR: cannot detect answer column. Available: {keys}")
        sys.exit(1)
    print(f"  Detected columns → problem: '{pcol}', answer: '{acol}'")

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
            entry = _aime_row_to_entry(row, pcol, acol)
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
