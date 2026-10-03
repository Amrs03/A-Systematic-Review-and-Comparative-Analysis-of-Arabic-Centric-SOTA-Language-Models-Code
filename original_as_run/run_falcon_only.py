"""
Falcon-only Track B rerun after the DynamicCache cache mismatch.

This script intentionally evaluates only the Falcon Track B model/datasets and
updates only Falcon result artifacts.
"""

from __future__ import annotations

import csv
import gc
import importlib.util
import json
import os
import re
import sys
import time
import traceback
from collections import Counter
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

import torch
import transformers
from datasets import load_from_disk


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs" / "falcon"
RESULTS_DIR = ROOT / "results" / "track_b"
LOG_DIR = ROOT / "logs"
RAW_OUTPUTS_PATH = OUTPUT_DIR / "falcon_raw_outputs.jsonl"
FAILED_SAMPLES_PATH = OUTPUT_DIR / "falcon_failed_samples.jsonl"
SCORES_CSV_PATH = OUTPUT_DIR / "falcon_scores.csv"
SCORES_JSON_PATH = OUTPUT_DIR / "falcon_scores.json"
RUN_LOG_PATH = OUTPUT_DIR / "falcon_run_log.md"
REWRITTEN_MD_PATH = OUTPUT_DIR / "falcon_results_rewritten.md"
MASTER_ROWS_PATH = OUTPUT_DIR / "falcon_master_rows.csv"
BEFORE_FIX_LOG = LOG_DIR / "falcon_dynamic_cache_error_before_fix.log"
TRACK_B_PATH = ROOT / "03_track_b_evaluate.py"
FALCON_LOCAL_SNAPSHOT = Path(
    "F:/hf_cache/hub/models--tiiuae--falcon-7b-instruct/snapshots/8782b5c5d8c9290412416618f36a133653e85285"
)

MODEL_OUTPUT_LABEL = "Falcon-Arabic"
MODEL_RESULT_KEY = "falcon"
MODEL_MASTER_LABEL = "Falcon"
METRIC_NAME = "accuracy"
FALCON_BATCH_SIZE = int(os.environ.get("FALCON_BATCH_SIZE", "16"))
FALCON_USE_CACHE = os.environ.get("FALCON_USE_CACHE", "true").lower() in {"1", "true", "yes"}

DATASET_LABELS = {
    "arabic_mmlu": "Arabic MMLU",
    "madinah_qa": "Madinah QA",
    "exams_ar": "EXAMS AR",
    "arabculture": "ArabCulture",
    "aradice": "AraDiCE",
    "aratrust": "AraTrust",
}

TASK_CATEGORIES = {
    "Social Meaning": {"datasets": ["aratrust"]},
    "Commonsense / STEM": {"datasets": ["arabic_mmlu", "madinah_qa", "exams_ar"]},
    "Cultural Understanding": {"datasets": ["arabculture", "aradice"]},
}


def load_track_b_module():
    spec = importlib.util.spec_from_file_location("track_b_eval", TRACK_B_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {TRACK_B_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure_local_falcon_snapshot(track_b: Any) -> str:
    """Keep Falcon reruns offline once the snapshot is already cached."""
    if FALCON_LOCAL_SNAPSHOT.exists():
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        track_b.MODELS[MODEL_RESULT_KEY]["path"] = str(FALCON_LOCAL_SNAPSHOT)
        return str(FALCON_LOCAL_SNAPSHOT)
    return track_b.MODELS[MODEL_RESULT_KEY]["path"]


def category_for_dataset(dataset: str) -> str:
    for category, config in TASK_CATEGORIES.items():
        if dataset in config["datasets"]:
            return category
    return "Uncategorized"


def cuda_device_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "device": "cpu",
        "bf16_supported": False,
    }
    if torch.cuda.is_available():
        info.update(
            {
                "device": torch.cuda.get_device_name(0),
                "cuda_version": torch.version.cuda,
                "bf16_supported": torch.cuda.is_bf16_supported(),
            }
        )
    return info


def clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def is_oom_error(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def error_summary(errors: list[dict[str, Any]]) -> str:
    if not errors:
        return ""
    counts = Counter(str(item.get("error", "unknown")) for item in errors)
    return "; ".join(f"{message} ({count})" for message, count in counts.most_common(3))


def safe_to_device(inputs: dict[str, torch.Tensor], model: Any) -> dict[str, torch.Tensor]:
    device = getattr(model, "device", None)
    if device is None:
        device = next(model.parameters()).device
    return {key: value.to(device) for key, value in inputs.items()}


def falcon_generate(model: Any, tokenizer: Any, inputs: dict[str, torch.Tensor], max_new_tokens: int):
    generate_kwargs: dict[str, Any] = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "use_cache": FALCON_USE_CACHE,
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id,
    }
    eos_token_id = getattr(tokenizer, "eos_token_id", None) or getattr(model.config, "eos_token_id", None)
    if eos_token_id is not None:
        generate_kwargs["eos_token_id"] = eos_token_id

    with torch.inference_mode():
        return model.generate(**generate_kwargs)


def evaluate_dataset(track_b: Any, model: Any, tokenizer: Any, dataset_key: str, raw_handle, failed_handle):
    dataset_config = track_b.DATASETS_CONFIG[dataset_key]
    dataset_path = ROOT / track_b.DATA_DIR / dataset_key
    ds = load_from_disk(str(dataset_path))
    split = dataset_config["split"]
    if split not in ds:
        split = list(ds.keys())[0]
    data = ds[split]
    prompt_builder = track_b.PROMPT_BUILDERS[dataset_config["task"]]
    max_new_tokens = track_b.MODELS[MODEL_RESULT_KEY].get("max_new_tokens", track_b.MAX_NEW_TOKENS)

    correct = 0
    failed = 0
    errors: list[dict[str, Any]] = []
    first_prompt = None
    started = time.time()

    print(f"\nFalcon-Arabic -> {dataset_key}: {len(data)} examples", flush=True)

    def write_sample(record: dict[str, Any], raw_output: str, parsed: str, status: str, error: str | None) -> None:
        nonlocal correct, failed
        if status == "ok" and parsed == record["gold"]:
            correct += 1
        if status == "failed":
            failed += 1
            tb = traceback.format_exc()
            errors.append({"index": record["index"], "sample_id": record["sample_id"], "error": error})
            failed_handle.write(
                json.dumps(
                    {
                        "dataset": dataset_key,
                        "sample_id": record["sample_id"],
                        "error": error,
                        "traceback": tb,
                    },
                    ensure_ascii=False,
                    default=str,
                )
                + "\n"
            )
        raw_handle.write(
            json.dumps(
                {
                    "dataset": dataset_key,
                    "sample_id": record["sample_id"],
                    "prompt": record["prompt"],
                    "raw_output": raw_output,
                    "parsed_prediction": parsed,
                    "gold": record["gold"],
                    "status": status,
                    "error": error,
                },
                ensure_ascii=False,
                default=str,
            )
            + "\n"
        )
        if record["index"] < 5 or status == "failed":
            print(
                f"  sample {record['index']}: gold={record['gold']!r} pred={parsed!r} status={status} raw={raw_output[:40]!r}",
                flush=True,
            )

    def run_single(record: dict[str, Any]) -> None:
        try:
            inputs = tokenizer(record["prompt"], return_tensors="pt", truncation=True, max_length=1024)
            inputs = safe_to_device(inputs, model)
            outputs = falcon_generate(model, tokenizer, inputs, max_new_tokens)
            new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True)
            parsed = track_b.extract_answer(raw_output)
            write_sample(record, raw_output, parsed, "ok", None)
        except Exception as exc:
            if is_oom_error(exc):
                clear_cuda()
                write_sample(record, "", "", "failed", f"CUDA OOM: {exc}")
            else:
                write_sample(record, "", "", "failed", str(exc))

    def run_batch(records: list[dict[str, Any]]) -> None:
        prompts = [record["prompt"] for record in records]
        try:
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024)
            inputs = safe_to_device(inputs, model)
            outputs = falcon_generate(model, tokenizer, inputs, max_new_tokens)
            prompt_width = inputs["input_ids"].shape[1]
            for row, record in enumerate(records):
                raw_output = tokenizer.decode(outputs[row][prompt_width:], skip_special_tokens=True)
                parsed = track_b.extract_answer(raw_output)
                write_sample(record, raw_output, parsed, "ok", None)
        except Exception as exc:
            if is_oom_error(exc):
                clear_cuda()
            for record in records:
                run_single(record)

    batch: list[dict[str, Any]] = []
    for index, example in enumerate(data):
        prompt = prompt_builder(example)
        if first_prompt is None:
            first_prompt = prompt
        batch.append(
            {
                "index": index,
                "sample_id": str(example.get("id") or example.get("ID") or example.get("sample_id") or index),
                "prompt": prompt,
                "gold": track_b.get_gold_answer(example),
            }
        )
        if len(batch) >= FALCON_BATCH_SIZE:
            run_batch(batch)
            batch = []
        if (index + 1) % 100 == 0:
            raw_handle.flush()
            failed_handle.flush()
            print(f"  {dataset_key}: {index + 1}/{len(data)}", flush=True)

    if batch:
        run_batch(batch)
        raw_handle.flush()
        failed_handle.flush()

    total = len(data)
    score = correct / total if total else 0.0
    elapsed = time.time() - started
    result = {
        "model": MODEL_RESULT_KEY,
        "model_label": MODEL_OUTPUT_LABEL,
        "dataset": dataset_key,
        "dataset_label": DATASET_LABELS.get(dataset_key, dataset_key),
        "timestamp": datetime.now().isoformat(),
        "accuracy": score,
        "correct": correct,
        "total": total,
        "errors": failed,
        "error_samples": errors[:5],
        "error_summary": error_summary(errors),
        "max_new_tokens": max_new_tokens,
        "generation_cache": "enabled_native_dynamic_cache" if FALCON_USE_CACHE else "disabled",
        "prompt_builder": prompt_builder.__name__,
        "split": split,
        "first_prompt": first_prompt,
        "runtime_seconds": elapsed,
    }
    print(f"  done {dataset_key}: accuracy={score:.6f} correct={correct}/{total} failed={failed}", flush=True)
    return result


def save_track_b_result(dataset_key: str, result: dict[str, Any]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{MODEL_RESULT_KEY}_{dataset_key}.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def write_scores(scores: dict[str, dict[str, Any]]) -> None:
    with SCORES_CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["model", "dataset", "metric_name", "score", "num_examples", "num_failed", "error_summary"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for dataset_key, result in scores.items():
            writer.writerow(
                {
                    "model": MODEL_OUTPUT_LABEL,
                    "dataset": dataset_key,
                    "metric_name": METRIC_NAME,
                    "score": result["accuracy"],
                    "num_examples": result["total"],
                    "num_failed": result["errors"],
                    "error_summary": result.get("error_summary", ""),
                }
            )

    payload = {
        "model": MODEL_OUTPUT_LABEL,
        "metric_name": METRIC_NAME,
        "generated_at": datetime.now().isoformat(),
        "datasets": {
            dataset_key: {
                "score": result["accuracy"],
                "num_examples": result["total"],
                "num_failed": result["errors"],
                "error_summary": result.get("error_summary", ""),
            }
            for dataset_key, result in scores.items()
        },
    }
    SCORES_JSON_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def falcon_compiled_row(dataset_key: str, result: dict[str, Any]) -> dict[str, Any]:
    score = result["accuracy"]
    return {
        "track": "Track B",
        "model": MODEL_RESULT_KEY,
        "model_label": MODEL_MASTER_LABEL,
        "dataset": dataset_key,
        "dataset_label": DATASET_LABELS.get(dataset_key, dataset_key),
        "category": category_for_dataset(dataset_key),
        "metric": "Accuracy",
        "score": score,
        "score_pct": score * 100,
        "correct": result["correct"],
        "total": result["total"],
        "errors": result["errors"],
        "status": "valid",
        "note": result.get("error_summary", ""),
        "first_error": "",
        "timestamp": result["timestamp"],
        "source_file": str(Path("results") / "track_b" / f"{MODEL_RESULT_KEY}_{dataset_key}.json"),
    }


def update_compiled_results(scores: dict[str, dict[str, Any]]) -> None:
    path = ROOT / "results" / "compiled_results.csv"
    if not path.exists():
        return
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    new_falcon_rows = {dataset: falcon_compiled_row(dataset, result) for dataset, result in scores.items()}
    replaced: set[str] = set()
    updated_rows = []
    for row in rows:
        if row.get("track") == "Track B" and row.get("model") == MODEL_RESULT_KEY and row.get("dataset") in new_falcon_rows:
            dataset = row["dataset"]
            updated_rows.append({key: new_falcon_rows[dataset].get(key, "") for key in fieldnames})
            replaced.add(dataset)
        else:
            updated_rows.append(row)
    for dataset, row in new_falcon_rows.items():
        if dataset not in replaced:
            updated_rows.append({key: row.get(key, "") for key in fieldnames})

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated_rows)

    with MASTER_ROWS_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for dataset in scores:
            writer.writerow({key: new_falcon_rows[dataset].get(key, "") for key in fieldnames})


def update_compiled_report_md(scores: dict[str, dict[str, Any]]) -> None:
    path = ROOT / "results" / "compiled_report.md"
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    filtered = [
        line
        for line in lines
        if not (
            line.startswith("| Track B |")
            and "| Falcon " in line
            and ("Runs Needing Attention" not in line)
        )
    ]

    dataset_rows = []
    for dataset_key, result in scores.items():
        dataset_rows.append(
            "| Track B | "
            f"{category_for_dataset(dataset_key)} | "
            f"{DATASET_LABELS.get(dataset_key, dataset_key)} | "
            f"{MODEL_MASTER_LABEL} | Accuracy | {result['accuracy'] * 100:.1f}% | {result['total']} |"
        )

    insert_at = None
    for index, line in enumerate(filtered):
        if line.startswith("## Runs Needing Attention"):
            insert_at = index
            break
    if insert_at is not None:
        before = filtered[:insert_at]
        while before and before[-1] == "":
            before.pop()
        filtered = before + dataset_rows + [""] + filtered[insert_at:]

    path.write_text("\n".join(filtered) + "\n", encoding="utf-8")


def update_compiled_report_html(scores: dict[str, dict[str, Any]]) -> None:
    path = ROOT / "results" / "compiled_report.html"
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"<tr><td>Track B</td><td>[^<]+</td><td>Falcon</td><td>invalid</td><td>\d+</td><td>.*?</td></tr>", "", text)
    rows = []
    for dataset_key, result in scores.items():
        rows.append(
            "<tr>"
            "<td>Track B</td>"
            f"<td>{escape(category_for_dataset(dataset_key))}</td>"
            f"<td>{escape(DATASET_LABELS.get(dataset_key, dataset_key))}</td>"
            f"<td>{MODEL_MASTER_LABEL}</td>"
            "<td>Accuracy</td>"
            f"<td>{result['accuracy'] * 100:.1f}%</td>"
            f"<td>{result['total']}</td>"
            "</tr>"
        )
    insertion = "".join(rows)
    if "Falcon</td><td>Accuracy" not in text:
        text = re.sub(
            r"(</tbody></table></div>\s+<h2>Runs Needing Attention</h2>)",
            insertion + r"\1",
            text,
            count=1,
        )
    path.write_text(text, encoding="utf-8")


def write_markdown_outputs(scores: dict[str, dict[str, Any]], package_info: dict[str, str], device_info: dict[str, Any]) -> None:
    completed = [dataset for dataset, result in scores.items() if result["errors"] < result["total"]]
    failed = [dataset for dataset, result in scores.items() if result["errors"] == result["total"] and result["total"]]
    skipped = {dataset: result["errors"] for dataset, result in scores.items() if result["errors"]}
    device = device_info.get("device", "cpu")

    score_rows = [
        f"| {DATASET_LABELS.get(dataset, dataset)} | Accuracy | {result['accuracy']:.6f} | {result['total']} | {result['errors']} | {result.get('error_summary') or 'ok'} |"
        for dataset, result in scores.items()
    ]

    rewritten = [
        "# Falcon-Arabic Track B Results After Cache Fix",
        "",
        "## Fix Summary",
        "",
        "- Previous Falcon failure: `'DynamicCache' object is not subscriptable`",
        "- Final fix used: Loaded Falcon through the native Transformers implementation so the cache API is compatible.",
        f"- Generation cache setting: `use_cache={FALCON_USE_CACHE}` with native Transformers cache handling",
        f"- Transformers version: {package_info['transformers']}",
        f"- Device: {device}",
        "",
        "## Falcon-Arabic Scores",
        "",
        "| Dataset | Metric | Score | Examples | Failed Samples | Notes |",
        "|---|---:|---:|---:|---:|---|",
        *score_rows,
        "",
        "## Falcon-Arabic Raw Output Files",
        "",
        "- `outputs/falcon/falcon_raw_outputs.jsonl`",
        "- `outputs/falcon/falcon_scores.csv`",
        "- `outputs/falcon/falcon_scores.json`",
        "- `outputs/falcon/falcon_run_log.md`",
        "",
    ]
    REWRITTEN_MD_PATH.write_text("\n".join(rewritten), encoding="utf-8")

    before_log_note = "captured"
    if not BEFORE_FIX_LOG.exists():
        before_log_note = "not found"
    log = [
        "# Falcon-Arabic Track B Cache-Fix Run Log",
        "",
        "## Original Error",
        "",
        f"- Previous Falcon failure: `'DynamicCache' object is not subscriptable` ({before_log_note} in `logs/falcon_dynamic_cache_error_before_fix.log`).",
        "- Reproduction: cache-enabled Falcon control run failed in legacy `modeling_falcon.py` at `past_key_value[0][0]`.",
        "",
        "## Fix Attempts",
        "",
        "1. Option A: Disabled Falcon cache with `model.config.use_cache = False`, `model.generation_config.use_cache = False`, and `generate(..., use_cache=False)`. This fixed the one-sample DynamicCache repro but was too slow for the full Falcon Track B rerun.",
        "2. Option B: Inspected cache indexing. The project does not manually index `past_key_values`; the failing indexing is in cached Falcon remote model code, so no project-level DynamicCache adapter was added.",
        f"3. Option C: Checked package versions. Transformers is {package_info['transformers']}; no dependency pin was changed.",
        "4. Falcon load robustness: a hub refresh failed before evaluation and the legacy remote-code local load then died during weight loading; final run uses native Transformers model code (`trust_remote_code=False`) from the already cached local Falcon snapshot.",
        f"5. Option D: Used a Falcon-only deterministic generation wrapper with `torch.inference_mode()`, batch size {FALCON_BATCH_SIZE}, per-sample fallback/error capture, and CUDA cache cleanup between datasets.",
        "",
        "## Final Working Fix",
        "",
        "- Final fix used: native Transformers Falcon loading plus Falcon-only wrapper robustness.",
        f"- Generation cache setting: `use_cache={FALCON_USE_CACHE}` with native Transformers cache handling.",
        "- Decoding: deterministic greedy decoding with `do_sample=False`.",
        f"- Batch size: {FALCON_BATCH_SIZE}, with automatic per-sample fallback on CUDA/memory errors.",
        f"- Max new tokens: {next(iter(scores.values()))['max_new_tokens'] if scores else 'n/a'}.",
        "",
        "## Package Versions",
        "",
        f"- Python: {package_info['python']}",
        f"- Torch: {package_info['torch']}",
        f"- Transformers: {package_info['transformers']}",
        f"- Datasets: {package_info['datasets']}",
        f"- Falcon model path: {package_info.get('falcon_model_path', 'tiiuae/falcon-7b-instruct')}",
        "",
        "## CUDA / Device Info",
        "",
        f"- CUDA available: {device_info.get('cuda_available')}",
        f"- CUDA version: {device_info.get('cuda_version', 'n/a')}",
        f"- Device count: {device_info.get('cuda_device_count')}",
        f"- Device: {device}",
        f"- bfloat16 supported: {device_info.get('bf16_supported')}",
        "- Runtime precision: existing Falcon 4-bit path with float16 compute.",
        "",
        "## Datasets Completed",
        "",
        *(f"- {DATASET_LABELS.get(dataset, dataset)}: {scores[dataset]['correct']}/{scores[dataset]['total']} correct, {scores[dataset]['errors']} failed" for dataset in completed),
        "",
        "## Datasets Failed",
        "",
        *(f"- {DATASET_LABELS.get(dataset, dataset)}: {scores[dataset].get('error_summary') or 'all samples failed'}" for dataset in failed),
        *(["- None"] if not failed else []),
        "",
        "## Skipped Samples",
        "",
        *(f"- {DATASET_LABELS.get(dataset, dataset)}: {count} failed samples logged to `outputs/falcon/falcon_failed_samples.jsonl`" for dataset, count in skipped.items()),
        *(["- None"] if not skipped else []),
        "",
        "## Final Notes",
        "",
        "- Only Falcon-Arabic Track B datasets were run.",
        "- JAIS and non-Falcon result JSON files were not executed or rewritten.",
        "- Falcon raw outputs and parsed predictions are in `outputs/falcon/falcon_raw_outputs.jsonl`.",
        "",
    ]
    RUN_LOG_PATH.write_text("\n".join(log), encoding="utf-8")


def package_info() -> dict[str, str]:
    try:
        import datasets

        datasets_version = datasets.__version__
    except Exception:
        datasets_version = "unknown"
    return {
        "python": sys.version.replace("\n", " "),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "datasets": datasets_version,
    }


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    for path in [RAW_OUTPUTS_PATH, FAILED_SAMPLES_PATH]:
        path.write_text("", encoding="utf-8")

    track_b = load_track_b_module()
    falcon_model_path = configure_local_falcon_snapshot(track_b)
    dataset_keys = list(track_b.DATASETS_CONFIG.keys())
    info = package_info()
    info["falcon_model_path"] = falcon_model_path
    device_info = cuda_device_info()

    scores: dict[str, dict[str, Any]] = {}
    model = None
    tokenizer = None
    try:
        model, tokenizer = track_b.load_model(MODEL_RESULT_KEY)
        tokenizer.padding_side = "left"
        model.config.use_cache = FALCON_USE_CACHE
        if hasattr(model, "generation_config"):
            model.generation_config.use_cache = FALCON_USE_CACHE

        with RAW_OUTPUTS_PATH.open("a", encoding="utf-8") as raw_handle, FAILED_SAMPLES_PATH.open("a", encoding="utf-8") as failed_handle:
            for dataset_key in dataset_keys:
                result = evaluate_dataset(track_b, model, tokenizer, dataset_key, raw_handle, failed_handle)
                scores[dataset_key] = result
                save_track_b_result(dataset_key, result)
                write_scores(scores)
                write_markdown_outputs(scores, info, device_info)
                clear_cuda()

        write_scores(scores)
        update_compiled_results(scores)
        update_compiled_report_md(scores)
        update_compiled_report_html(scores)
        write_markdown_outputs(scores, info, device_info)
    finally:
        if model is not None and tokenizer is not None:
            track_b.unload_model(model, tokenizer)
        clear_cuda()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
