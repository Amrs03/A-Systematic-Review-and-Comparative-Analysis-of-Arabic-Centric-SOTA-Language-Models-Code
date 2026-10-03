"""
Step 1: Download and prepare all datasets.
Run this FIRST. Takes ~5-10 minutes depending on internet speed.

Usage:
    python 01_setup_datasets.py
"""

import os
import json
from datasets import load_dataset, DatasetDict, concatenate_datasets, Value, Features

OUTPUT_DIR = "prepared_data"
RESULTS_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{RESULTS_DIR}/track_a", exist_ok=True)
os.makedirs(f"{RESULTS_DIR}/track_b", exist_ok=True)
os.makedirs(f"{RESULTS_DIR}/bridge", exist_ok=True)

# ─────────────────────────────────────────────
# Track A Datasets (for fine-tuning)
# ─────────────────────────────────────────────

TRACK_A_DATASETS = {
    "hard": {
        "path": "Elnagara/hard",
        "task": "sentiment_analysis",
        "text_col": "text",
        "label_col": "label",
        "needs_split": True,
    },
    "astd": {
        "path": "arbml/ASTD",
        "task": "sentiment_analysis",
        "text_col": "text",
        "label_col": "label",
        "needs_split": True,
    },
    "anercorp": {
        "path": "asas-ai/ANERCorp",
        "task": "ner",
        "text_col": "tokens",
        "label_col": "ner_tags",
        "needs_split": False,
    },
    "arsarcasm": {
        "path": "iabufarha/ar_sarcasm",
        "task": "sarcasm",
        "text_col": "tweet",
        "label_col": "sarcasm",
        "needs_split": False,
    },
    "qadi": {
        "path": "Abdelrahman-Rezk/Arabic_Dialect_Identification",
        "task": "dialect_id",
        "text_col": "text",
        "label_col": "label",
        "needs_split": False,
    },
}

# ─────────────────────────────────────────────
# Track B Datasets (for LLM zero-shot eval)
# ─────────────────────────────────────────────

TRACK_B_DATASETS = {
    "arabic_mmlu": {
        "path": "MBZUAI/ArabicMMLU",
        "task": "mcq",
        "configs": ["All"],           # single config — load as-is
    },
    "madinah_qa": {
        "path": "MBZUAI/MadinahQA",
        "task": "mcq",
        "configs": ["Arabic Language (General)", "Arabic Language (Grammar)"],
    },
    "exams_ar": {
        "path": "OALL/Arabic_EXAMS",
        "task": "mcq",
        "configs": None,              # no config needed
    },
    "aratrust": {
        "path": "asas-ai/AraTrust",
        "task": "trust",
        "configs": None,
    },
    "arabculture": {
        "path": "MBZUAI/ArabCulture",
        "task": "culture",
        "configs": ["Jordan", "Palestine", "Lebanon", "Syria"],
    },
    "aradice": {
        "path": "QCRI/AraDiCE-Culture",
        "task": "culture",
        "configs": ["Jordan", "Palestine", "Lebanon", "Syria"],
    },
}


def _merge_features(feat_list):
    """Merge multiple feature schemas, preferring string over numeric on conflicts."""
    merged = dict(feat_list[0])
    for features in feat_list[1:]:
        for key, ftype in features.items():
            if key not in merged:
                merged[key] = ftype
            elif merged[key] != ftype:
                merged[key] = Value("string")  # safest common type
    return Features(merged)


def download_and_prepare():
    report = {"success": [], "failed": [], "warnings": []}

    # ── Track A ──
    print("\n" + "=" * 60)
    print("TRACK A: Downloading fine-tuning datasets...")
    print("=" * 60)

    for name, config in TRACK_A_DATASETS.items():
        print(f"\n📥 Loading {name} from {config['path']}...")
        try:
            ds = load_dataset(config["path"])
            print(f"   Raw splits: {list(ds.keys())}")

            # Inspect columns
            first_split = list(ds.keys())[0]
            print(f"   Columns: {ds[first_split].column_names}")
            print(f"   Sample: {ds[first_split][0]}")

            # Create train/test split if needed
            if config["needs_split"]:
                if "train" in ds:
                    try:
                        split = ds["train"].train_test_split(
                            test_size=0.2, seed=42, stratify_by_column=config["label_col"]
                        )
                    except Exception:
                        # Fallback without stratification
                        split = ds["train"].train_test_split(test_size=0.2, seed=42)
                    ds = DatasetDict({"train": split["train"], "test": split["test"]})
                    print(f"   ✂️  Created 80/20 split: train={len(ds['train'])}, test={len(ds['test'])}")
                else:
                    report["warnings"].append(f"{name}: no 'train' split found, check manually")

            # Save to disk for fast reloading
            ds.save_to_disk(f"{OUTPUT_DIR}/{name}")
            print(f"   ✅ Saved to {OUTPUT_DIR}/{name}")
            report["success"].append(name)

        except Exception as e:
            print(f"   ❌ FAILED: {e}")
            report["failed"].append({"name": name, "error": str(e)})

    # ── Track B ──
    print("\n" + "=" * 60)
    print("TRACK B: Downloading LLM evaluation datasets...")
    print("=" * 60)

    for name, config in TRACK_B_DATASETS.items():
        print(f"\n📥 Loading {name} from {config['path']}...")
        try:
            cfg_list = config.get("configs")

            if cfg_list is None:
                # No config needed — load directly
                ds = load_dataset(config["path"])
            elif len(cfg_list) == 1:
                # Single named config
                ds = load_dataset(config["path"], cfg_list[0])
            else:
                # Multiple configs — load each and concatenate per split
                loaded = [load_dataset(config["path"], c) for c in cfg_list]
                print(f"   Loaded configs: {cfg_list}")
                all_splits = set(loaded[0].keys())
                merged = {}
                for split in all_splits:
                    parts = [d[split] for d in loaded if split in d]
                    # Build a unified schema (prefer string on type conflicts) then cast all
                    unified = _merge_features([p.features for p in parts])
                    parts = [p.cast(unified) for p in parts]
                    merged[split] = concatenate_datasets(parts)
                ds = DatasetDict(merged)

            print(f"   Splits: {list(ds.keys())}")
            first_split = list(ds.keys())[0]
            print(f"   Columns: {ds[first_split].column_names}")
            print(f"   Size: {len(ds[first_split])} rows")
            print(f"   Sample: {ds[first_split][0]}")

            ds.save_to_disk(f"{OUTPUT_DIR}/{name}")
            print(f"   ✅ Saved to {OUTPUT_DIR}/{name}")
            report["success"].append(name)

        except Exception as e:
            print(f"   ❌ FAILED: {e}")
            report["failed"].append({"name": name, "error": str(e)})

    # ── Report ──
    print("\n" + "=" * 60)
    print("DOWNLOAD REPORT")
    print("=" * 60)
    print(f"✅ Success: {len(report['success'])}/{len(TRACK_A_DATASETS) + len(TRACK_B_DATASETS)}")
    print(f"❌ Failed:  {len(report['failed'])}")
    if report["failed"]:
        for f in report["failed"]:
            print(f"   - {f['name']}: {f['error']}")
    if report["warnings"]:
        print(f"⚠️  Warnings:")
        for w in report["warnings"]:
            print(f"   - {w}")

    # Save report
    with open(f"{RESULTS_DIR}/download_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n{'🎉 ALL CLEAR — proceed to Track A!' if not report['failed'] else '⚠️  Fix failed downloads before proceeding.'}")


if __name__ == "__main__":
    download_and_prepare()
