#!/usr/bin/env python3
"""
Math dataset preparation for On-Policy Segmented Distillation (OPSD).

Steps:
  0. (Optional) Download dataset from HuggingFace
  1. Parquet → train_chat.jsonl / test_chat.jsonl
  2. train_chat.jsonl → train_opsd.jsonl  (adds raw_content to metadata)
  3. test_chat.jsonl  → test_chat_eval.jsonl  (capped at --max-eval-samples)

Usage examples:
  # Download + prepare (recommended first run)
  python prepare_math.py --download

  # Custom endpoint (China mirror)
  python prepare_math.py --download --hf-endpoint https://hf-mirror.com

  # Use a specific subset instead of the full dataset
  python prepare_math.py --download --subset algebra

  # Skip download if parquet files already exist
  python prepare_math.py --data-dir /root/math/data

  # Skip parquet conversion if JSONL files already exist
  python prepare_math.py --skip-parquet
"""

import argparse
import json
import os
import pathlib
import sys


PROBLEM_COLS = ["problem", "question", "input", "prompt", "instruction", "content", "text"]
SOLUTION_COLS = ["solution", "answer", "output", "response", "label", "target"]

DATASET_REPO = "DigitalLearningGmbH/MATH-lighteval"

# Available subsets for this dataset
VALID_SUBSETS = [
    "default",
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


# ---------------------------------------------------------------------------
# Step 0: Download from HuggingFace
# ---------------------------------------------------------------------------

def step0_download(data_dir: pathlib.Path, dataset: str, subset: str, hf_endpoint: str):
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    if hf_endpoint:
        os.environ["HF_ENDPOINT"] = hf_endpoint
        print(f"[Step 0] Using HF endpoint: {hf_endpoint}")

    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # HuggingFace stores parquet files under refs/convert/parquet branch,
    # organised as {subset}/train/*.parquet and {subset}/test/*.parquet.
    allow_patterns = [
        f"{subset}/train/*.parquet",
        f"{subset}/test/*.parquet",
    ]
    print(f"[Step 0] Downloading '{dataset}' (subset={subset}) → {raw_dir}")
    print(f"         Patterns: {allow_patterns}")

    snapshot_download(
        repo_id=dataset,
        repo_type="dataset",
        revision="refs/convert/parquet",
        allow_patterns=allow_patterns,
        local_dir=str(raw_dir),
    )

    downloaded = list(raw_dir.rglob("*.parquet"))
    print(f"[Step 0] Downloaded {len(downloaded)} parquet file(s).\n")
    if not downloaded:
        print("WARNING: no parquet files found after download. Check dataset name / subset.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_parquet_reader():
    try:
        import pyarrow.parquet as pq
        return pq, None
    except ImportError:
        print("pyarrow not found, falling back to pandas ...")
        try:
            import pandas as pd
            return None, pd
        except ImportError:
            print("ERROR: neither pyarrow nor pandas is installed.")
            sys.exit(1)


def read_rows(path, pq, pd):
    if pq is not None:
        return pq.read_table(path).to_pylist()
    return pd.read_parquet(path).to_dict(orient="records")


def detect_cols(rows):
    keys = {k.lower(): k for k in rows[0].keys()}
    pcol = next((keys[c] for c in PROBLEM_COLS if c in keys), None)
    scol = next((keys[c] for c in SOLUTION_COLS if c in keys), None)
    if pcol is None:
        print(f"ERROR: cannot detect problem column from {list(rows[0].keys())}")
        sys.exit(1)
    return pcol, scol


# ---------------------------------------------------------------------------
# Step 1: Parquet → JSONL
# ---------------------------------------------------------------------------

def step1_parquet_to_jsonl(data_dir: pathlib.Path):
    # Scan recursively to handle subset subdirectories (e.g. raw/default/train/*.parquet)
    all_parquet = sorted(data_dir.rglob("*.parquet"))
    if not all_parquet:
        print(f"[Step 1] No .parquet files found under {data_dir}, skipping.")
        return

    print(f"[Step 1] Found {len(all_parquet)} parquet file(s).")
    pq, pd = _load_parquet_reader()

    # Split by train / test based on path components or filename.
    # Files downloaded from HF look like: raw/default/train/part-0.parquet
    #                                  or raw/default/test/part-0.parquet
    train_files, test_files = [], []
    for f in all_parquet:
        parts = [p.lower() for p in f.parts]
        fname_lower = f.name.lower()
        is_test = (
            "test" in parts or "val" in parts or "eval" in parts
            or any(k in fname_lower for k in ("test", "val", "eval"))
        )
        (test_files if is_test else train_files).append(f)

    print(f"  Train files ({len(train_files)}): {[f.name for f in train_files]}")
    print(f"  Test  files ({len(test_files)}): {[f.name for f in test_files]}")

    def convert(files, out_name):
        if not files:
            print(f"  No files for {out_name}, skipping.")
            return
        all_rows = []
        for f in files:
            rows = read_rows(f, pq, pd)
            all_rows.extend(rows)
            print(f"  Read {len(rows):>6} rows from {f.relative_to(data_dir)}")

        pcol, scol = detect_cols(all_rows)
        print(f'  Detected columns → problem: "{pcol}", solution: "{scol}"')

        out_path = data_dir / out_name
        with out_path.open("w", encoding="utf-8") as fout:
            for row in all_rows:
                problem = str(row[pcol])
                solution = str(row[scol]) if scol and row.get(scol) is not None else ""
                entry = {
                    "prompt": [{"role": "user", "content": problem}],
                    "label": solution,
                    "metadata": {"solution": solution},
                }
                fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"  Created {out_path} with {len(all_rows)} entries")

    convert(train_files, "train_chat.jsonl")
    convert(test_files, "test_chat.jsonl")
    print("[Step 1] Parquet → JSONL done.\n")


# ---------------------------------------------------------------------------
# Step 2: Add raw_content for OPSD
# ---------------------------------------------------------------------------

def step2_add_raw_content(data_dir: pathlib.Path):
    src = data_dir / "train_chat.jsonl"
    dst = data_dir / "train_opsd.jsonl"

    if not src.exists():
        print(f"[Step 2] {src} not found, skipping.")
        return

    print(f"[Step 2] Adding raw_content: {src.name} → {dst.name}")
    count = 0
    with src.open() as fin, dst.open("w") as fout:
        for line in fin:
            obj = json.loads(line)
            raw_content = next(
                (msg["content"] for msg in obj.get("prompt", []) if msg.get("role") == "user"),
                "",
            )
            metadata = obj.get("metadata") or {}
            metadata["raw_content"] = raw_content
            obj["metadata"] = metadata
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            count += 1
    print(f"[Step 2] Created {dst} with {count} entries.\n")


# ---------------------------------------------------------------------------
# Step 3: Prepare eval split
# ---------------------------------------------------------------------------

def step3_prepare_eval(data_dir: pathlib.Path, max_eval_samples: int):
    src = data_dir / "test_chat.jsonl"
    dst = data_dir / "test_chat_eval.jsonl"

    if not src.exists():
        print(f"[Step 3] {src} not found, skipping.")
        return

    print(f"[Step 3] Preparing eval set (max {max_eval_samples}): {src.name} → {dst.name}")
    count = 0
    with src.open() as fin, dst.open("w") as fout:
        for line in fin:
            if count >= max_eval_samples:
                break
            obj = json.loads(line)
            obj["label"] = (obj.get("metadata") or {}).get("solution", "")
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            count += 1
    print(f"[Step 3] Created {dst} with {count} samples (capped at {max_eval_samples}).\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prepare MATH-lighteval dataset for OPSD training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data-dir",
        default="/root/data/math",
        help="Working directory for all data files (default: /root/math/data)",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the dataset from HuggingFace before processing",
    )
    parser.add_argument(
        "--dataset",
        default=DATASET_REPO,
        help=f"HuggingFace dataset repo id (default: {DATASET_REPO})",
    )
    parser.add_argument(
        "--subset",
        default="default",
        choices=VALID_SUBSETS,
        help=(
            "Dataset subset to download. 'default' = full dataset (7 500 train + 5 000 test). "
            f"Choices: {VALID_SUBSETS}"
        ),
    )
    parser.add_argument(
        "--hf-endpoint",
        default="",
        metavar="URL",
        help="HuggingFace mirror endpoint, e.g. https://hf-mirror.com (useful in China)",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=100,
        help="Max samples in eval set (default: 100)",
    )
    parser.add_argument(
        "--skip-parquet",
        action="store_true",
        help="Skip Step 1 (parquet→jsonl), useful if train_chat.jsonl already exists",
    )
    args = parser.parse_args()

    data_dir = pathlib.Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Data directory: {data_dir}\n")

    if args.download:
        step0_download(data_dir, args.dataset, args.subset, args.hf_endpoint)

    if not args.skip_parquet:
        step1_parquet_to_jsonl(data_dir)

    step2_add_raw_content(data_dir)
    step3_prepare_eval(data_dir, args.max_eval_samples)

    print("All done! Output files:")
    for name in ("train_chat.jsonl", "train_opsd.jsonl", "test_chat.jsonl", "test_chat_eval.jsonl"):
        p = data_dir / name
        if p.exists():
            lines = sum(1 for _ in p.open())
            print(f"  {p}  ({lines} lines)")


if __name__ == "__main__":
    main()
