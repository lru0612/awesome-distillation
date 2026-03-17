"""Preprocess datasets for the Prompt Internalization (PI) safety experiment.

Produces JSONL files in the standard slime format:
    {
        "prompt":   [{"role": "user", "content": "<bare question>"}],
        "label":    "safe" | "unsafe",
        "metadata": {
            "student_user_content": "<bare question>",
            "pi_instruction":       "<constitution>",
            "prompt_text":          "<bare question>",
            "rm_type":              "safety",
            "judge_mode":           "keyword",   # override to "llm" for richer eval
            "source":               "<dataset tag>",
        }
    }

Supported datasets
------------------
Training — easy (PKU-SafeRLHF, straightforward harmful/benign prompts):
    python preprocess_pi_safety.py \\
        --dataset PKU-Alignment/PKU-SafeRLHF \\
        --split train \\
        --output /root/pi_safety/train_pku.jsonl

Training — medium (BeaverTails train, broader harm categories):
    python preprocess_pi_safety.py \\
        --dataset PKU-Alignment/BeaverTails \\
        --split 30k-train \\
        --output /root/pi_safety/train_beavertails.jsonl

Training — hard (WildGuardMix, includes adversarial jailbreak-style prompts):
    python preprocess_pi_safety.py \\
        --dataset allenai/wildguardmix --config wildguardtrain \\
        --split train \\
        --output /root/pi_safety/train_wildguardmix.jsonl

Evaluation (BeaverTails test):
    python preprocess_pi_safety.py \\
        --dataset PKU-Alignment/BeaverTails \\
        --split 30k-test \\
        --output /root/pi_safety/eval_beavertails.jsonl \\
        --judge-mode llm

Evaluation (WildGuardMix test set):
    python preprocess_pi_safety.py \\
        --dataset allenai/wildguardmix --config wildguardtest \\
        --split test \\
        --output /root/pi_safety/eval_wildguardtest.jsonl \\
        --judge-mode llm


Balance control (default: keep all harmful, subsample benign to 2× harmful count):
    --max-benign-ratio 2.0   # benign = ratio × harmful_count  (0 = keep all benign)
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constitution (must match slime/rollout/rm_hub/safety.py DEFAULT_CONSTITUTION)
# ---------------------------------------------------------------------------

CONSTITUTION = (
    "Follow these rules when responding:\n"
    "1. Do not assist with harmful or illegal actions.\n"
    "2. Politely refuse unsafe requests.\n"
    "3. Provide safe alternatives when possible.\n"
    "4. Remain helpful and neutral for legitimate requests."
)


# ---------------------------------------------------------------------------
# Per-dataset extractors  →  list[{"prompt": str, "label": "safe"|"unsafe"}]
# ---------------------------------------------------------------------------


def _extract_pku_saferlhf(ds) -> list[dict]:
    """PKU-Alignment/PKU-SafeRLHF.

    Relevant columns: prompt, is_response_0_safe, is_response_1_safe.
    A prompt is marked "unsafe" if either annotated response is unsafe.
    """
    rows = []
    for row in ds:
        prompt = (row.get("prompt") or "").strip()
        if not prompt:
            continue
        safe0 = bool(row.get("is_response_0_safe", True))
        safe1 = bool(row.get("is_response_1_safe", True))
        label = "unsafe" if (not safe0 or not safe1) else "safe"
        rows.append({"prompt": prompt, "label": label})
    return rows


def _extract_beavertails(ds) -> list[dict]:
    """PKU-Alignment/BeaverTails.

    Relevant columns: prompt, is_safe.
    is_safe: True → benign prompt → label "safe"
             False → harmful prompt → label "unsafe"
    """
    rows = []
    for row in ds:
        prompt = (row.get("prompt") or "").strip()
        if not prompt:
            continue
        label = "safe" if bool(row.get("is_safe", True)) else "unsafe"
        rows.append({"prompt": prompt, "label": label})
    return rows


def _extract_wildguardmix(ds) -> list[dict]:
    """allenai/wildguardmix — WildGuardMix training/test data.

    Dataset page: https://huggingface.co/datasets/allenai/wildguardmix
    Contains vanilla and adversarial (jailbreak-style) prompts with harm annotations.
    Significantly harder than PKU-SafeRLHF because harmful intent is often disguised.

    Columns: prompt, prompt_harm_label, response, response_harm_label,
             subcategory, adversarial.
    prompt_harm_label: "harmful" | "unharmful"
    """
    rows = []
    for row in ds:
        prompt = (row.get("prompt") or "").strip()
        if not prompt:
            continue
        raw = row.get("prompt_harm_label")
        if raw is None:
            label = "unsafe"  # conservative: missing annotation → treat as harmful
        else:
            label = "unsafe" if str(raw).strip().lower() == "harmful" else "safe"
        rows.append({"prompt": prompt, "label": label})
    return rows



def _extract_generic(ds) -> list[dict]:
    """Best-effort for unknown schemas.

    Looks for a "prompt"/"question"/"instruction" field and optional
    "is_safe"/"label"/"is_response_safe" field.
    """
    rows = []
    for row in ds:
        prompt = (
            row.get("prompt")
            or row.get("question")
            or row.get("instruction")
            or ""
        ).strip()
        if not prompt:
            continue
        raw_label = row.get("is_safe") or row.get("label") or row.get("is_response_safe")
        if raw_label is None:
            label = "unsafe"  # conservative default
        elif isinstance(raw_label, bool):
            label = "safe" if raw_label else "unsafe"
        else:
            label = str(raw_label).strip().lower()
            label = "safe" if label in {"safe", "1", "true"} else "unsafe"
        rows.append({"prompt": prompt, "label": label})
    return rows


def _detect_extractor(column_names: list[str], dataset_name: str = ""):
    cols = set(column_names)
    ds_lower = dataset_name.lower()

    if "is_response_0_safe" in cols or "is_response_1_safe" in cols:
        return _extract_pku_saferlhf, "pku-saferlhf"
    if "is_safe" in cols and "category" in cols:
        return _extract_beavertails, "beavertails"
    # WildGuardMix: distinctive column prompt_harm_label
    if "prompt_harm_label" in cols:
        return _extract_wildguardmix, "wildguardmix"

    return _extract_generic, "generic"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def preprocess(
    dataset: str,
    output: str,
    split: str = "train",
    config: str | None = None,
    max_samples: int | None = None,
    max_benign_ratio: float = 2.0,
    judge_mode: str = "keyword",
) -> int:
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("datasets package required: pip install datasets")
        sys.exit(1)

    # HuggingFace split names must match \w+(\.\w+)* — replace hyphens with underscores.
    # e.g. "30k-test" → "30k_test"  (BeaverTails naming convention)
    split = split.replace("-", "_")
    logger.info("Loading %s (split=%s%s)", dataset, split,
                f", config={config}" if config else "")
    load_kwargs: dict = {"split": split}
    if config:
        load_kwargs["name"] = config
    ds = load_dataset(dataset, **load_kwargs)

    extractor, source_tag = _detect_extractor(ds.column_names, dataset)
    logger.info("Detected format: %s  (columns: %s)", source_tag, ds.column_names)

    rows = extractor(ds)

    # Separate harmful / benign
    harmful = [r for r in rows if r["label"] == "unsafe"]
    benign  = [r for r in rows if r["label"] == "safe"]
    logger.info("Raw counts — harmful: %d, benign: %d", len(harmful), len(benign))

    import random as _rng
    _rng.seed(42)

    # Subsample benign to avoid imbalance
    if max_benign_ratio > 0 and len(benign) > max_benign_ratio * len(harmful):
        cap = int(max_benign_ratio * len(harmful))
        benign = _rng.sample(benign, cap)
        logger.info("Subsampled benign to %d (ratio=%.1f×)", cap, max_benign_ratio)

    combined = harmful + benign
    _rng.shuffle(combined)
    if max_samples is not None:
        combined = combined[:max_samples]

    output_path = pathlib.Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with output_path.open("w", encoding="utf-8") as fout:
        for row in combined:
            prompt_text = row["prompt"]
            label = row["label"]
            entry = {
                "prompt": [{"role": "user", "content": prompt_text}],
                "label": label,
                "metadata": {
                    "student_user_content": prompt_text,
                    "pi_instruction": CONSTITUTION,
                    "prompt_text": prompt_text,
                    "rm_type": "safety",
                    "judge_mode": judge_mode,
                    "source": source_tag,
                },
            }
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
            written += 1

    logger.info("Written %d entries → %s  (harmful=%d, benign=%d)",
                written, output, len(harmful), len(benign))
    return written


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", required=True, help="HuggingFace dataset repo id")
    parser.add_argument("--config", default=None, help="Dataset config/subset name")
    parser.add_argument("--split", default="train", help="Dataset split")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--max-benign-ratio",
        type=float,
        default=2.0,
        help="Cap benign samples at ratio × harmful count (0 = keep all)",
    )
    parser.add_argument(
        "--judge-mode",
        choices=["keyword", "llm"],
        default="keyword",
        help=(
            "Evaluation mode stored in metadata['judge_mode']. "
            "Use 'llm' for eval splits when JUDGE_API_BASE/JUDGE_MODEL are set."
        ),
    )
    args = parser.parse_args()
    preprocess(
        dataset=args.dataset,
        output=args.output,
        split=args.split,
        config=args.config,
        max_samples=args.max_samples,
        max_benign_ratio=args.max_benign_ratio,
        judge_mode=args.judge_mode,
    )


if __name__ == "__main__":
    main()
