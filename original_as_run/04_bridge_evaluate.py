"""
Step 4: Bridge Tasks — Evaluate LLMs on sentiment analysis datasets
so all 5 models can be compared on the same task.

Usage:
    python 04_bridge_evaluate.py --all
    python 04_bridge_evaluate.py --model allam --dataset hard
"""

import os

# ── Redirect ALL HuggingFace/model downloads to F: drive ──────────────────────
os.environ.setdefault("HF_HOME", "F:/hf_cache")
os.environ.setdefault("TRANSFORMERS_CACHE", "F:/hf_cache/hub")
os.environ.setdefault("HF_DATASETS_CACHE", "F:/hf_cache/datasets")
# ──────────────────────────────────────────────────────────────────────────────

# ── Patch huggingface_hub validator to accept Windows absolute paths ───────────
import re as _re2
import huggingface_hub.utils._validators as _hfv
_orig_validate = _hfv.validate_repo_id
def _patched_validate_repo_id(repo_id, *args, **kwargs):
    if _re2.match(r'^[A-Za-z]:[\\/]', str(repo_id)) or os.path.isabs(str(repo_id)):
        return
    return _orig_validate(repo_id, *args, **kwargs)
_hfv.validate_repo_id = _patched_validate_repo_id
import huggingface_hub.file_download as _hffd
if hasattr(_hffd, 'validate_repo_id'):
    _hffd.validate_repo_id = _patched_validate_repo_id
# ──────────────────────────────────────────────────────────────────────────────

import re
import json
import argparse
import torch
from datetime import datetime
from tqdm import tqdm
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sklearn.metrics import f1_score, classification_report

# ── Transformers 5.x shim for Jais custom modeling_jais.py ────────────────────
from transformers.configuration_utils import PreTrainedConfig as _PTC
_orig_ga = _PTC.__getattribute__
def _patched_ga(self, name):
    if name == "add_cross_attention":
        try: return _orig_ga(self, name)
        except AttributeError: return False
    return _orig_ga(self, name)
_PTC.__getattribute__ = _patched_ga

import types as _t, sys as _s
_mpu = _t.ModuleType("transformers.utils.model_parallel_utils")
_mpu.assert_device_map = lambda device_map, num_blocks: None
_mpu.get_device_map = lambda num_blocks, device_ids: {i: device_ids[i % len(device_ids)] for i in range(num_blocks)}
_s.modules["transformers.utils.model_parallel_utils"] = _mpu
import transformers.utils as _tu; _tu.model_parallel_utils = _mpu

import transformers.pytorch_utils as _pt
if not hasattr(_pt, "find_pruneable_heads_and_indices"):
    def _fph(heads, n_heads, head_size, already_pruned):
        mask = torch.ones(n_heads, head_size)
        heads = set(heads) - already_pruned
        for h in sorted(heads):
            h -= sum(1 if p < h else 0 for p in already_pruned)
            mask[h] = 0
        mask = mask.view(-1).contiguous().eq(1)
        return heads, torch.arange(len(mask))[mask].long()
    _pt.find_pruneable_heads_and_indices = _fph
if not hasattr(_pt, "prune_conv1d_layer"):
    def _pcl(layer, index, dim=1):
        index = index.to(layer.weight.device)
        W = layer.weight.index_select(dim, index).clone().detach()
        b = layer.bias.clone().detach() if dim == 0 else layer.bias[index].clone().detach()
        sz = list(layer.weight.size()); sz[dim] = len(index)
        nl = _pt.Conv1D(sz[1], sz[0]).to(layer.weight.device)
        nl.weight.data.copy_(W.contiguous()); nl.bias.data.copy_(b.contiguous())
        return nl
    _pt.prune_conv1d_layer = _pcl
# ──────────────────────────────────────────────────────────────────────────────


# Reuse model configs from Track B
MODELS = {
    "allam": {
        "path": "ALLaM-AI/ALLaM-7B-Instruct-preview",
        "use_4bit": True,
        "cpu_offload": False,
        "use_fast_tokenizer": False,
    },
    # "jais": skipped — see 03_track_b_evaluate.py comment
    "qwen": {
        "path": "Qwen/Qwen2.5-7B-Instruct",
        "use_4bit": True,
        "cpu_offload": False,
        "use_fast_tokenizer": True,
    },
    "falcon": {
        "path": "tiiuae/falcon-7b-instruct",
        "use_4bit": True,
        "cpu_offload": False,
        "use_fast_tokenizer": True,
    },
}

BRIDGE_DATASETS = {
    "hard": {"label_col": "label"},
    "astd": {"label_col": "label"},
}

DATA_DIR = "prepared_data"
RESULTS_DIR = "results/bridge"


def load_model(model_key):
    config = MODELS[model_key]
    model_path = config["path"]
    import re as _re, os as _os
    _is_local = bool(_re.match(r'^[A-Za-z]:[\\/]', model_path)) or _os.path.isabs(model_path)
    if _is_local:
        model_path = _os.path.normpath(model_path)
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    else:
        os.environ["TRANSFORMERS_OFFLINE"] = "0"
    print(f"\n🔄 Loading {model_key} from {model_path}...")

    use_fast = config.get("use_fast_tokenizer", True)
    _local_kwargs = {"local_files_only": True} if _is_local else {}
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=use_fast, **_local_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {"trust_remote_code": True, "torch_dtype": torch.float16}
    if _is_local:
        load_kwargs["local_files_only"] = True

    if config["cpu_offload"]:
        load_kwargs["torch_dtype"] = torch.float16   # cast fp32→fp16 on load, halves peak RAM
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
            llm_int8_enable_fp32_cpu_offload=True,
        )
        load_kwargs["quantization_config"] = bnb_config
        load_kwargs["device_map"] = "auto"
        load_kwargs["max_memory"] = {0: "5GiB", "cpu": "56GiB"}
        load_kwargs["offload_folder"] = "F:/hf_cache/offload"
        load_kwargs["offload_buffers"] = True
        load_kwargs["low_cpu_mem_usage"] = True
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
            llm_int8_enable_fp32_cpu_offload=True,
        )
        load_kwargs["quantization_config"] = bnb_config
        load_kwargs["device_map"] = "auto"
        load_kwargs["max_memory"] = {0: "7GiB", "cpu": "56GiB"}

    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    model.eval()
    return model, tokenizer


def build_sentiment_prompt(text):
    return f"""صنف المراجعة التالية إلى واحدة من الفئات: إيجابي، سلبي، محايد.
أجب بكلمة واحدة فقط.

المراجعة: {text}

التصنيف:"""


def parse_sentiment(generated_text):
    text = generated_text.strip().lower()

    positive_keywords = ["إيجابي", "positive", "ايجابي", "pos"]
    negative_keywords = ["سلبي", "negative", "neg"]
    neutral_keywords = ["محايد", "neutral", "neu"]

    for kw in positive_keywords:
        if kw in text:
            return "positive"
    for kw in negative_keywords:
        if kw in text:
            return "negative"
    for kw in neutral_keywords:
        if kw in text:
            return "neutral"

    return text[:20]  # for manual inspection


def evaluate_sentiment(model, tokenizer, dataset_key, config):
    print(f"\n   📊 Bridge eval: sentiment on {dataset_key}")

    ds = load_from_disk(f"{DATA_DIR}/{dataset_key}")
    data = ds["test"]

    label_col = config["label_col"]
    all_labels = sorted(set(data[label_col]))
    print(f"   Labels: {all_labels}")
    print(f"   Test size: {len(data)}")

    # Determine text column
    text_col = "text" if "text" in data.column_names else data.column_names[0]

    all_preds = []
    all_golds = []
    errors = []

    # Cap at 1000 samples for speed (these datasets are large)
    max_samples = min(len(data), 1000)
    print(f"   Evaluating on {max_samples} samples (capped for speed)")

    for i in tqdm(range(max_samples), desc=f"   {dataset_key}"):
        example = data[i]
        try:
            prompt = build_sentiment_prompt(example[text_col])
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=10, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )

            new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            generated = tokenizer.decode(new_tokens, skip_special_tokens=True)

            pred = parse_sentiment(generated)
            gold = str(example[label_col])

            all_preds.append(pred)
            all_golds.append(gold)

            if i < 5:
                print(f"\n   [Sample {i}] Gold={gold} | Pred={pred} | Raw='{generated[:40]}'")

        except Exception as e:
            errors.append(str(e))

    # Calculate F1
    # Need to handle label mismatch — map predictions to gold label space
    unique_golds = sorted(set(all_golds))
    unique_preds = sorted(set(all_preds))
    print(f"\n   Gold labels: {unique_golds}")
    print(f"   Pred labels: {unique_preds}")

    f1_macro = f1_score(all_golds, all_preds, average="macro", zero_division=0)
    f1_micro = f1_score(all_golds, all_preds, average="micro", zero_division=0)

    print(f"   ✅ F1 macro: {f1_macro:.4f}")

    return {
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "samples_evaluated": max_samples,
        "errors": len(errors),
    }


def save_result(model_key, dataset_key, result):
    entry = {"model": model_key, "dataset": dataset_key,
             "timestamp": datetime.now().isoformat(), **result}
    path = f"{RESULTS_DIR}/{model_key}_{dataset_key}.json"
    with open(path, "w") as f:
        json.dump(entry, f, indent=2, ensure_ascii=False, default=str)
    print(f"   💾 Saved: {path}")


def run_all(skip_existing=False):
    for model_key in MODELS:
        model, tokenizer = load_model(model_key)

        for dataset_key, config in BRIDGE_DATASETS.items():
            result_path = f"{RESULTS_DIR}/{model_key}_{dataset_key}.json"
            if skip_existing and os.path.exists(result_path):
                print(f"⏭️  Skipping {model_key}/{dataset_key}")
                continue

            try:
                result = evaluate_sentiment(model, tokenizer, dataset_key, config)
                save_result(model_key, dataset_key, result)
            except Exception as e:
                print(f"❌ FAILED: {e}")
                save_result(model_key, dataset_key, {"f1_macro": None, "error": str(e)})

        del model, tokenizer
        torch.cuda.empty_cache()

    print(f"\n🏁 BRIDGE TASKS COMPLETE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--model", choices=list(MODELS.keys()))
    parser.add_argument("--dataset", choices=list(BRIDGE_DATASETS.keys()))
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    if args.all:
        run_all(args.skip_existing)
    elif args.model and args.dataset:
        model, tokenizer = load_model(args.model)
        result = evaluate_sentiment(model, tokenizer, args.dataset, BRIDGE_DATASETS[args.dataset])
        save_result(args.model, args.dataset, result)
    else:
        parser.print_help()
