"""
Step 2: Track A — Fine-tune AraBERT and MARBERT on classification datasets.
Optimized for RTX 4060 (8GB VRAM).

Usage:
    # Run ALL Track A experiments (12 runs):
    python 02_track_a_finetune.py --all

    # Run a single experiment:
    python 02_track_a_finetune.py --model arabert --dataset hard

    # Resume (skip already completed experiments):
    python 02_track_a_finetune.py --all --skip-existing
"""

import os
import json
import argparse
import numpy as np
import torch
from datetime import datetime
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
from sklearn.metrics import f1_score, classification_report

# ─────────────────────────────────────────────
# Configuration — RTX 4060 optimized
# ─────────────────────────────────────────────

MODELS = {
    "arabert": "aubmindlab/bert-base-arabertv02",
    "marbert": "UBC-NLP/MARBERT",
}

DATASETS = {
    "hard": {"task": "classification", "text_col": "text", "label_col": "label"},
    "astd": {"task": "classification", "text_col": "tweet", "label_col": "label"},
    "arsarcasm": {"task": "classification", "text_col": "tweet", "label_col": "sarcasm"},
    "qadi": {"task": "classification", "text_col": "text", "label_col": "label"},
    # ANERCorp on HF is word-per-row (one word + one tag per example), not sentence-level lists.
    # Treat as token-level classification: each word is its own input.
    "anercorp": {"task": "classification", "text_col": "word", "label_col": "tag"},
}

DATA_DIR = "prepared_data"
RESULTS_DIR = "results/track_a"
CHECKPOINTS_DIR = "checkpoints/track_a"

# RTX 4060 settings: small batch + gradient accumulation = effective batch 32
TRAINING_DEFAULTS = {
    "num_train_epochs": 3,
    "per_device_train_batch_size": 8,       # fits in 8GB VRAM
    "per_device_eval_batch_size": 16,
    "gradient_accumulation_steps": 4,        # effective batch = 8 * 4 = 32
    "learning_rate": 2e-5,
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "eval_strategy": "epoch",
    "save_strategy": "epoch",
    "load_best_model_at_end": True,
    "metric_for_best_model": "f1_macro",
    "seed": 42,
    "fp16": True,                            # use mixed precision on RTX 4060
    "dataloader_num_workers": 0,             # 0 required on Windows (avoids multiprocessing errors)
    "dataloader_pin_memory": False,          # pin_memory needs CUDA device, set explicitly to avoid warning
    "logging_steps": 50,
    "report_to": "none",                     # disable wandb/tensorboard
}

MAX_LENGTH = 128  # increase to 256 if results seem low


# ─────────────────────────────────────────────
# Classification Pipeline
# ─────────────────────────────────────────────

def run_classification(model_key, dataset_key, dataset_config, eval_only=False):
    print(f"\n{'='*60}")
    action = "Evaluating" if eval_only else "Fine-tuning"
    print(f"🔧 {action} {model_key.upper()} on {dataset_key.upper()}")
    print(f"{'='*60}")

    model_name = MODELS[model_key]
    text_col = dataset_config["text_col"]
    label_col = dataset_config["label_col"]
    output_dir = f"{CHECKPOINTS_DIR}/{model_key}_{dataset_key}"

    # Fail fast if eval_only and no checkpoint exists
    if eval_only:
        ckpt = find_best_checkpoint(output_dir)
        if ckpt is None:
            raise FileNotFoundError(f"No checkpoint found in {output_dir}. Run training first.")
    else:
        ckpt = None

    # Load data
    ds = load_from_disk(f"{DATA_DIR}/{dataset_key}")
    print(f"   Train: {len(ds['train'])}, Test: {len(ds['test'])}")

    # Build label mapping
    all_labels = sorted(set(ds["train"][label_col]))
    label2id = {str(l): i for i, l in enumerate(all_labels)}
    id2label = {i: str(l) for i, l in enumerate(all_labels)}
    num_labels = len(all_labels)
    print(f"   Labels ({num_labels}): {all_labels[:10]}{'...' if len(all_labels) > 10 else ''}")

    # Remap labels to contiguous integers if they aren't already
    def remap_labels(example):
        example["label_id"] = label2id[str(example[label_col])]
        return example

    ds = ds.map(remap_labels)

    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(batch[text_col], truncation=True, padding="max_length", max_length=MAX_LENGTH)

    ds = ds.map(tokenize, batched=True, remove_columns=[c for c in ds["train"].column_names if c not in ["input_ids", "attention_mask", "label_id"]])
    ds = ds.rename_column("label_id", "labels")
    ds.set_format("torch")

    # Model — load from checkpoint if eval_only, else from base model
    if eval_only:
        print(f"   📂 Loading checkpoint: {ckpt}")
        load_from = ckpt
    else:
        load_from = model_name

    model = AutoModelForSequenceClassification.from_pretrained(
        load_from, num_labels=num_labels, id2label=id2label, label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    # Metrics
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        f1_macro = f1_score(labels, preds, average="macro")
        f1_micro = f1_score(labels, preds, average="micro")
        return {"f1_macro": f1_macro, "f1_micro": f1_micro}

    training_args = TrainingArguments(output_dir=output_dir, **TRAINING_DEFAULTS)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        compute_metrics=compute_metrics,
    )

    if not eval_only:
        trainer.train()
    eval_result = trainer.evaluate()

    # Detailed classification report
    preds_output = trainer.predict(ds["test"])
    preds = np.argmax(preds_output.predictions, axis=-1)
    labels = preds_output.label_ids
    report = classification_report(labels, preds, target_names=[str(id2label[i]) for i in range(num_labels)], output_dict=True)

    return {
        "f1_macro": eval_result["eval_f1_macro"],
        "f1_micro": eval_result["eval_f1_micro"],
        "classification_report": report,
    }


# ─────────────────────────────────────────────
# NER Pipeline
# ─────────────────────────────────────────────

def run_ner(model_key, dataset_key, dataset_config, eval_only=False):
    print(f"\n{'='*60}")
    action = "Evaluating" if eval_only else "Fine-tuning"
    print(f"🔧 {action} {model_key.upper()} on {dataset_key.upper()} (NER)")
    print(f"{'='*60}")

    model_name = MODELS[model_key]
    output_dir = f"{CHECKPOINTS_DIR}/{model_key}_{dataset_key}"

    # Fail fast if eval_only and no checkpoint exists
    if eval_only:
        ckpt = find_best_checkpoint(output_dir)
        if ckpt is None:
            raise FileNotFoundError(f"No checkpoint found in {output_dir}. Run training first.")
    else:
        ckpt = None

    # Load data
    ds = load_from_disk(f"{DATA_DIR}/{dataset_key}")
    print(f"   Train: {len(ds['train'])}, Test: {len(ds['test'])}")

    # Get NER tag info from dataset features
    label_col = dataset_config["label_col"]
    token_col = dataset_config["text_col"]

    # Build label list — try to get from features first
    try:
        label_list = ds["train"].features[label_col].feature.names
    except Exception:
        # Fallback: collect unique tags
        all_tags = set()
        for example in ds["train"]:
            all_tags.update(example[label_col])
        label_list = sorted(all_tags)

    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for i, l in enumerate(label_list)}
    num_labels = len(label_list)
    print(f"   NER tags ({num_labels}): {label_list}")

    # Tokenize with label alignment
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_and_align(examples):
        tokenized = tokenizer(
            examples[token_col],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            is_split_into_words=True,
        )

        all_labels = []
        for i, labels in enumerate(examples[label_col]):
            word_ids = tokenized.word_ids(batch_index=i)
            label_ids = []
            prev_word_id = None
            for word_id in word_ids:
                if word_id is None:
                    label_ids.append(-100)
                elif word_id != prev_word_id:
                    label_ids.append(labels[word_id])
                else:
                    label_ids.append(-100)  # ignore non-first subword tokens
                prev_word_id = word_id
            all_labels.append(label_ids)

        tokenized["labels"] = all_labels
        return tokenized

    ds = ds.map(tokenize_and_align, batched=True, remove_columns=ds["train"].column_names)
    ds.set_format("torch")

    # Model — load from checkpoint if eval_only, else from base model
    if eval_only:
        print(f"   📂 Loading checkpoint: {ckpt}")
        load_from = ckpt
    else:
        load_from = model_name

    model = AutoModelForTokenClassification.from_pretrained(
        load_from, num_labels=num_labels, id2label=id2label, label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    # Metrics using seqeval
    from evaluate import load as load_metric
    seqeval = load_metric("seqeval")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        true_labels = []
        true_preds = []
        for pred_seq, label_seq in zip(preds, labels):
            t_labels = []
            t_preds = []
            for p, l in zip(pred_seq, label_seq):
                if l != -100:
                    t_labels.append(id2label[l])
                    t_preds.append(id2label[p])
            true_labels.append(t_labels)
            true_preds.append(t_preds)

        results = seqeval.compute(predictions=true_preds, references=true_labels)
        return {
            "f1_macro": results["overall_f1"],
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
        }

    training_args = TrainingArguments(output_dir=output_dir, **TRAINING_DEFAULTS)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        compute_metrics=compute_metrics,
        data_collator=DataCollatorForTokenClassification(tokenizer),
    )

    if not eval_only:
        trainer.train()
    eval_result = trainer.evaluate()

    return {
        "f1_macro": eval_result["eval_f1_macro"],
        "precision": eval_result["eval_precision"],
        "recall": eval_result["eval_recall"],
    }


# ─────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────

def find_best_checkpoint(output_dir):
    """Return the best checkpoint path from trainer_state.json, or the latest checkpoint as fallback."""
    state_file = os.path.join(output_dir, "trainer_state.json")
    if os.path.exists(state_file):
        with open(state_file) as f:
            state = json.load(f)
        best = state.get("best_model_checkpoint")
        if best and os.path.isdir(best):
            return best

    # Fallback: pick the highest-numbered checkpoint
    checkpoints = sorted(
        [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")],
        key=lambda x: int(x.split("-")[1]),
    )
    if checkpoints:
        return os.path.join(output_dir, checkpoints[-1])
    return None


def save_result(model_key, dataset_key, result):
    result_entry = {
        "model": model_key,
        "dataset": dataset_key,
        "timestamp": datetime.now().isoformat(),
        **result,
    }
    path = f"{RESULTS_DIR}/{model_key}_{dataset_key}.json"
    with open(path, "w") as f:
        json.dump(result_entry, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n   💾 Result saved: {path}")
    f1 = result.get('f1_macro')
    print(f"   📊 F1 (macro): {f1:.4f}" if f1 is not None else "   📊 F1 (macro): N/A")


def run_experiment(model_key, dataset_key, skip_existing=False, eval_only=False):
    result_path = f"{RESULTS_DIR}/{model_key}_{dataset_key}.json"
    if skip_existing and os.path.exists(result_path):
        print(f"\n⏭️  Skipping {model_key}/{dataset_key} — result already exists")
        return

    config = DATASETS[dataset_key]

    if config["task"] == "ner":
        result = run_ner(model_key, dataset_key, config, eval_only=eval_only)
    else:
        result = run_classification(model_key, dataset_key, config, eval_only=eval_only)

    save_result(model_key, dataset_key, result)

    # Free GPU memory
    torch.cuda.empty_cache()


def run_all(skip_existing=False, eval_only=False):
    total = len(MODELS) * len(DATASETS)
    done = 0

    for model_key in MODELS:
        for dataset_key in DATASETS:
            done += 1
            print(f"\n{'#'*60}")
            print(f"# RUN {done}/{total}: {model_key} × {dataset_key}")
            print(f"{'#'*60}")

            try:
                run_experiment(model_key, dataset_key, skip_existing, eval_only=eval_only)
            except Exception as e:
                print(f"\n   ❌ FAILED: {e}")
                save_result(model_key, dataset_key, {"f1_macro": None, "error": str(e)})

    print(f"\n\n{'='*60}")
    print(f"🏁 TRACK A COMPLETE — {done} experiments finished")
    print(f"   Results in: {RESULTS_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track A: Fine-tune encoder models")
    parser.add_argument("--all", action="store_true", help="Run all 12 experiments")
    parser.add_argument("--model", choices=list(MODELS.keys()), help="Single model to run")
    parser.add_argument("--dataset", choices=list(DATASETS.keys()), help="Single dataset to run")
    parser.add_argument("--skip-existing", action="store_true", help="Skip experiments with existing results")
    parser.add_argument("--eval-only", action="store_true", help="Skip training — load saved checkpoints and evaluate only")
    args = parser.parse_args()

    if args.all:
        run_all(args.skip_existing, eval_only=args.eval_only)
    elif args.model and args.dataset:
        run_experiment(args.model, args.dataset, args.skip_existing, eval_only=args.eval_only)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python 02_track_a_finetune.py --all")
        print("  python 02_track_a_finetune.py --model arabert --dataset hard")
        print("  python 02_track_a_finetune.py --all --skip-existing")
