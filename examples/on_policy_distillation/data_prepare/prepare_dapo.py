#!/usr/bin/env python3
"""
Prepare BytedTsinghua-SIA/DAPO-Math-17k for OPSD training or evaluation.

DAPO row schema:
  prompt       : [{"role": "user", "content": "<instruction + problem>"}]
  reward_model : {"ground_truth": "<answer>", "style": "..."}
  extra_info   : {"index": "..."}

Output format (one JSON per line):
  {
    "prompt"  : [{"role": "user", "content": "..."}],
    "label"   : "<answer>",
    "metadata": {"solution": "<answer>", "raw_content": "<last user msg>"}
  }

Usage:
  # Train: randomly sample 17000 examples
  python prepare_dapo.py --mode train --num-samples 17000 \\
      --output /root/data/dapo_train.jsonl --seed 42

  # Eval: 200 examples, no overlap with the train file
  python prepare_dapo.py --mode eval --num-samples 200 \\
      --output /root/data/dapo_eval.jsonl \\
      --train-file /root/data/dapo_train.jsonl --seed 42

  # China mirror
  python prepare_dapo.py --mode train --num-samples 17000 \\
      --output /root/data/dapo_train.jsonl --seed 42 \\
      --hf-endpoint https://hf-mirror.com
"""

import argparse
import json
import os
import pathlib
import random
import sys

DAPO_REPO = "BytedTsinghua-SIA/DAPO-Math-17k"


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


def _extract_last_user_content(prompt_field) -> str:
    """Return the content of the last 'user' message in a prompt list."""
    if isinstance(prompt_field, str):
        try:
            prompt_field = json.loads(prompt_field)
        except json.JSONDecodeError:
            return prompt_field.strip()
    for msg in reversed(prompt_field):
        if isinstance(msg, dict) and msg.get("role") == "user":
            return str(msg.get("content", "")).strip()
    return ""


def _dapo_row_to_entry(row: dict) -> dict | None:
    """Convert one DAPO row to a training entry. Returns None for malformed rows."""
    prompt_field = row.get("prompt", [])
    if isinstance(prompt_field, str):
        try:
            prompt_field = json.loads(prompt_field)
        except json.JSONDecodeError:
            prompt_field = [{"role": "user", "content": prompt_field}]

    raw_content = _extract_last_user_content(prompt_field)
    if not raw_content:
        return None

    reward_model = row.get("reward_model") or {}
    if isinstance(reward_model, str):
        try:
            reward_model = json.loads(reward_model)
        except json.JSONDecodeError:
            reward_model = {}

    answer = str(reward_model.get("ground_truth", "")).strip()
    if not answer:
        extra = row.get("extra_info") or {}
        if isinstance(extra, str):
            try:
                extra = json.loads(extra)
            except json.JSONDecodeError:
                extra = {}
        answer = str(extra.get("answer", extra.get("ground_truth", ""))).strip()

    return {
        "prompt":   prompt_field,
        "label":    answer,
        "metadata": {"solution": answer, "raw_content": raw_content},
    }


def _load_exclusion_set(train_file: str) -> set:
    """Build a set of raw_content strings from an existing output file."""
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
            # Use the last user message as the dedup key
            prompt_msgs = obj.get("prompt", [])
            key = ""
            for msg in reversed(prompt_msgs):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    key = str(msg.get("content", "")).strip()
                    break
            if key:
                exclusion.add(key)
    print(f"  Loaded {len(exclusion)} problems from train file for contamination check.")
    return exclusion


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prepare DAPO-Math-17k for OPSD training or evaluation.",
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
    print(f"Loading '{DAPO_REPO}' ...")
    ds = hf.load_dataset(DAPO_REPO, cache_dir=cache_dir, trust_remote_code=True)

    split_name = "train" if "train" in ds else list(ds.keys())[0]
    if split_name != "train":
        print(f"  No 'train' split found; using '{split_name}'.")

    rows = list(ds[split_name])
    total = len(rows)
    print(f"  Dataset has {total} rows in split '{split_name}'.")

    # Shuffle with fixed seed
    rng = random.Random(args.seed)
    rng.shuffle(rows)

    # Build exclusion set for eval contamination check
    exclusion: set = set()
    if args.mode == "eval" and args.train_file:
        exclusion = _load_exclusion_set(args.train_file)

    # Convert rows → entries, applying exclusion filter
    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = skipped_malformed = skipped_contaminated = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for row in rows:
            if written >= args.num_samples:
                break
            entry = _dapo_row_to_entry(dict(row))
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
