"""
Step 3: Track B — Evaluate generative LLMs (ALLaM, Jais, Falcon) via zero-shot prompting.
Optimized for RTX 4060 (8GB VRAM) + 64GB RAM.

Usage:
    # Run ALL Track B experiments (18 runs):
    python 03_track_b_evaluate.py --all

    # Run a single experiment:
    python 03_track_b_evaluate.py --model allam --dataset arabic_mmlu

    # Resume (skip completed):
    python 03_track_b_evaluate.py --all --skip-existing
"""

import os, re as _re

# ── Redirect ALL HuggingFace/model downloads to F: drive ──────────────────────
os.environ.setdefault("HF_HOME", "F:/hf_cache")
os.environ.setdefault("TRANSFORMERS_CACHE", "F:/hf_cache/hub")
os.environ.setdefault("HF_DATASETS_CACHE", "F:/hf_cache/datasets")
# ──────────────────────────────────────────────────────────────────────────────

# ── Patch huggingface_hub validator to accept Windows absolute paths ───────────
# transformers 5.x calls validate_repo_id before checking if the path is local,
# so Windows paths like F:\... are rejected. Skip validation for absolute paths.
import huggingface_hub.utils._validators as _hfv
_orig_validate = _hfv.validate_repo_id
def _patched_validate_repo_id(repo_id, *args, **kwargs):
    if _re.match(r'^[A-Za-z]:[\\/]', str(repo_id)) or os.path.isabs(str(repo_id)):
        return  # local path — skip hub validation
    return _orig_validate(repo_id, *args, **kwargs)
_hfv.validate_repo_id = _patched_validate_repo_id
# also patch on the decorated wrapper used in file_download
import huggingface_hub.file_download as _hffd
if hasattr(_hffd, 'validate_repo_id'):
    _hffd.validate_repo_id = _patched_validate_repo_id
# ──────────────────────────────────────────────────────────────────────────────

import re
import json
import argparse
import torch
import sys
from datetime import datetime
from tqdm import tqdm
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

try:
    from bitsandbytes.nn import Params4bit as _Params4bit
    _orig_params4bit_new = _Params4bit.__new__
    def _patched_params4bit_new(cls, *args, **kwargs):
        kwargs.pop("_is_hf_initialized", None)
        return _orig_params4bit_new(cls, *args, **kwargs)
    _Params4bit.__new__ = staticmethod(_patched_params4bit_new)
except Exception:
    pass

# ── Transformers 5.x shim for Jais custom modeling_jais.py ────────────────────
from transformers.configuration_utils import PreTrainedConfig as _PTC
_orig_ga = _PTC.__getattribute__
def _patched_ga(self, name):
    if name == "add_cross_attention":
        try: return _orig_ga(self, name)
        except AttributeError: return False
    return _orig_ga(self, name)
_PTC.__getattribute__ = _patched_ga

from transformers.modeling_utils import PreTrainedModel as _PTM
_orig_initialize_missing_keys = _PTM._initialize_missing_keys
def _patched_initialize_missing_keys(self, is_quantized):
    if is_quantized:
        return
    return _orig_initialize_missing_keys(self, is_quantized)
_PTM._initialize_missing_keys = _patched_initialize_missing_keys
if not hasattr(_PTM, "get_head_mask"):
    def _get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
        if head_mask is None:
            return [None] * num_hidden_layers
        return head_mask
    _PTM.get_head_mask = _get_head_mask

import types as _t, sys as _s
_mpu = _t.ModuleType("transformers.utils.model_parallel_utils")
_mpu.assert_device_map = lambda device_map, num_blocks: None
_mpu.get_device_map = lambda num_blocks, device_ids: {i: device_ids[i % len(device_ids)] for i in range(num_blocks)}
_s.modules["transformers.utils.model_parallel_utils"] = _mpu
import transformers.utils as _tu; _tu.model_parallel_utils = _mpu

import transformers.pytorch_utils as _pt
if not hasattr(_pt, "find_pruneable_heads_and_indices"):
    def _fph(heads, n_heads, head_size, already_pruned):
        import torch as _torch
        mask = _torch.ones(n_heads, head_size)
        heads = set(heads) - already_pruned
        for h in sorted(heads):
            h -= sum(1 if p < h else 0 for p in already_pruned)
            mask[h] = 0
        mask = mask.view(-1).contiguous().eq(1)
        return heads, _torch.arange(len(mask))[mask].long()
    _pt.find_pruneable_heads_and_indices = _fph
if not hasattr(_pt, "prune_conv1d_layer"):
    def _pcl(layer, index, dim=1):
        import torch as _torch
        index = index.to(layer.weight.device)
        W = layer.weight.index_select(dim, index).clone().detach()
        b = layer.bias.clone().detach() if dim == 0 else layer.bias[index].clone().detach()
        sz = list(layer.weight.size()); sz[dim] = len(index)
        nl = _pt.Conv1D(sz[1], sz[0]).to(layer.weight.device)
        nl.weight.data.copy_(W.contiguous()); nl.bias.data.copy_(b.contiguous())
        return nl
    _pt.prune_conv1d_layer = _pcl
# ──────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

MODELS = {
    "allam": {
        "path": "ALLaM-AI/ALLaM-7B-Instruct-preview",
        "use_4bit": True,
        "cpu_offload": False,
        "use_fast_tokenizer": False,
    },
    # JAIS runs from converted fp16 local weights to avoid the fp32 load-time RAM spike.
    "jais": {
        "path": "F:/hf_cache/jais-family-13b-chat-fp16",
        "use_4bit": False,
        "cpu_offload": True,
        "use_fast_tokenizer": False,
        "max_new_tokens": 1,
    },
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
        "use_native_transformers": True,
        "generation_use_cache": True,
    },
}

DATASETS_CONFIG = {
    "arabic_mmlu": {"task": "mcq", "split": "test"},
    "madinah_qa": {"task": "mcq", "split": "test"},
    "exams_ar": {"task": "mcq", "split": "test"},
    "aratrust": {"task": "trust", "split": "test"},
    "arabculture": {"task": "culture", "split": "test"},
    "aradice": {"task": "culture", "split": "test"},
}

DATA_DIR = "prepared_data"
RESULTS_DIR = "results/track_b"

MAX_NEW_TOKENS = 20  # we only need the answer letter/word


# ─────────────────────────────────────────────
# Model Loading (RTX 4060 optimized)
# ─────────────────────────────────────────────

def load_model(model_key):
    config = MODELS[model_key]
    model_path = config["path"]
    # Windows drive-letter paths are rejected by HF hub validator in transformers 5.x.
    # Detect local paths and pass local_files_only=True to bypass hub validation.
    import re as _re, os as _os
    _is_local = bool(_re.match(r'^[A-Za-z]:[\\/]', model_path)) or _os.path.isabs(model_path)
    if _is_local:
        model_path = _os.path.normpath(model_path)
        os.environ["TRANSFORMERS_OFFLINE"] = "1"   # prevent network calls for local models
    else:
        os.environ["TRANSFORMERS_OFFLINE"] = "0"

    print(f"\n🔄 Loading {model_key} from {model_path}...")

    use_fast = config.get("use_fast_tokenizer", True)
    _local_kwargs = {"local_files_only": True} if _is_local else {}
    trust_remote_code = not config.get("use_native_transformers", False)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        use_fast=use_fast,
        **_local_kwargs,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {"trust_remote_code": trust_remote_code, "torch_dtype": torch.float16}
    if _is_local:
        load_kwargs["local_files_only"] = True

    if config["cpu_offload"]:
        # JAIS-13B is converted to fp16 first so CPU offload avoids the original fp32 RAM spike.
        # Keep JAIS unquantized here: bitsandbytes 4-bit CPU/disk offload is unstable on this stack.
        load_kwargs["torch_dtype"] = torch.float16
        if config.get("use_4bit", False):
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                llm_int8_enable_fp32_cpu_offload=True,
            )
            load_kwargs["quantization_config"] = bnb_config
        load_kwargs["device_map"] = "auto"
        load_kwargs["max_memory"] = {0: "5GiB", "cpu": "56GiB"}
        load_kwargs["offload_folder"] = "F:/hf_cache/offload"
        load_kwargs["offload_buffers"] = True
        load_kwargs["low_cpu_mem_usage"] = True
        print("   ⚠️  Using CPU/disk offload — inference will be slower")
    elif config["use_4bit"]:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_enable_fp32_cpu_offload=True,  # allow CPU fallback if VRAM is fragmented
        )
        load_kwargs["quantization_config"] = bnb_config
        load_kwargs["device_map"] = "auto"
        if model_key == "falcon":
            load_kwargs["max_memory"] = {0: "6GiB", "cpu": "56GiB"}
            load_kwargs["low_cpu_mem_usage"] = True
        else:
            load_kwargs["max_memory"] = {0: "7GiB", "cpu": "56GiB"}
    else:
        load_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    if model_key == "falcon":
        model.config.use_cache = bool(config.get("generation_use_cache", False))
        if hasattr(model, "generation_config"):
            model.generation_config.use_cache = bool(config.get("generation_use_cache", False))
    elif model_key == "jais":
        model.config.use_cache = False
    model.eval()

    print(f"   ✅ {model_key} loaded successfully")
    return model, tokenizer


def unload_model(model, tokenizer):
    del model
    del tokenizer
    torch.cuda.empty_cache()
    import gc; gc.collect()
    print("   🗑️  Model unloaded, GPU memory freed")


# ─────────────────────────────────────────────
# Prompt Templates
# ─────────────────────────────────────────────

def build_mcq_prompt(example):
    """
    Build prompt for MCQ datasets.
    Adapts to whatever column names exist in the dataset.
    """
    # Try common column name patterns
    question = (
        example.get("Question") or
        example.get("question") or
        example.get("stem") or
        example.get("input") or
        ""
    )

    # Try to find options
    options = []
    # Pattern 1: separate option columns
    for key in [
        "A", "B", "C", "D", "E",
        "option_a", "option_b", "option_c", "option_d", "option_e",
        "option_1", "option_2", "option_3", "option_4", "option_5",
        "Option A", "Option B", "Option C", "Option D", "Option E",
        "Option 1", "Option 2", "Option 3", "Option 4", "Option 5",
    ]:
        if key in example and example[key]:
            options.append(str(example[key]))

    # Pattern 2: choices list
    if not options and "choices" in example:
        options = [str(c) for c in example["choices"]]

    # Pattern 3: options list/dict
    if not options and "options" in example:
        raw_options = example["options"]
        if isinstance(raw_options, dict) and "text" in raw_options:
            options = [str(o) for o in raw_options["text"]]
        else:
            options = [str(o) for o in raw_options]

    # Build prompt
    option_labels = ["أ", "ب", "ج", "د", "هـ"]  # Arabic A, B, C, D, E
    option_text = "\n".join(
        f"{option_labels[i]}) {opt}" for i, opt in enumerate(options)
    )

    prompt = f"""السؤال: {question}
{option_text}

اختر الإجابة الصحيحة (أ، ب، ج، أو د). أجب بحرف واحد فقط.
الإجابة:"""

    return prompt


def build_trust_prompt(example):
    """Prompt for AraTrust — adapt based on actual dataset format."""
    question = example.get("question") or example.get("Question") or example.get("input") or str(example)

    prompt = f"""السؤال التالي يتعلق بالثقة والسلامة. أجب بأفضل إجابة.

{question}

الإجابة:"""
    return prompt


def build_culture_prompt(example):
    """Prompt for ArabCulture / AraDiCE — adapt based on actual dataset format."""
    question = (
        example.get("Question") or
        example.get("question") or
        example.get("first_statement") or
        example.get("input") or
        ""
    )

    # Check if MCQ-style
    options = []
    for key in ["A", "B", "C", "D", "E", "Option A", "Option B", "Option C", "Option D", "Option E", "options"]:
        if key in example and example[key]:
            options.append(str(example[key]))

    if options:
        return build_mcq_prompt(example)  # reuse MCQ template

    prompt = f"""السؤال: {question}

الإجابة:"""
    return prompt


PROMPT_BUILDERS = {
    "mcq": build_mcq_prompt,
    "trust": build_trust_prompt,
    "culture": build_culture_prompt,
}


# ─────────────────────────────────────────────
# Answer Parsing
# ─────────────────────────────────────────────

def extract_answer(generated_text):
    """Extract the answer from model output. Returns normalized answer."""
    text = generated_text.strip()

    # Map Arabic letters to A/B/C/D
    ar_to_en = {"أ": "A", "ا": "A", "ب": "B", "ج": "C", "د": "D", "هـ": "E", "ه": "E"}

    # Try to find single letter answer (Arabic or English)
    for ar, en in ar_to_en.items():
        if text.startswith(ar):
            return en

    for letter in ["A", "B", "C", "D", "E"]:
        if text.upper().startswith(letter):
            return letter

    # Try regex for "الإجابة: X" pattern
    match = re.search(r'[أابجدھهـABCDE]', text)
    if match:
        char = match.group()
        return ar_to_en.get(char, char.upper())

    # Try numeric
    num_map = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}
    if text[0:1] in num_map:
        return num_map[text[0]]

    return text[:10]  # return first 10 chars for manual inspection


def get_gold_answer(example):
    """Extract the gold/correct answer from the dataset example."""
    # Try common column names
    answer = (
        example.get("answer") or
        example.get("Answer") or
        example.get("Answer Key") or
        example.get("answer_key") or
        example.get("correct_answer") or
        example.get("label") or
        example.get("gold") or
        ""
    )

    if isinstance(answer, dict):
        answer = (
            answer.get("english_answer_key") or
            answer.get("arabic_answer_key") or
            answer.get("answer") or
            ""
        )

    answer = str(answer).strip()

    # Normalize: map index to letter
    index_map = {"0": "A", "1": "B", "2": "C", "3": "D", "4": "E"}
    if answer in index_map:
        return index_map[answer]

    # Map Arabic
    ar_to_en = {"أ": "A", "ا": "A", "ب": "B", "ج": "C", "د": "D", "هـ": "E", "ه": "E"}
    if answer in ar_to_en:
        return ar_to_en[answer]

    return answer.upper()


# ─────────────────────────────────────────────
# Evaluation Loop
# ─────────────────────────────────────────────

def evaluate_dataset(model, tokenizer, dataset_key, dataset_config, model_key=None):
    print(f"\n   📊 Evaluating on {dataset_key}...")

    ds = load_from_disk(f"{DATA_DIR}/{dataset_key}")

    # Find the right split
    split = dataset_config["split"]
    if split not in ds:
        # Try alternatives
        available = list(ds.keys())
        split = available[0]
        print(f"   ⚠️  '{dataset_config['split']}' split not found, using '{split}'")

    data = ds[split]
    print(f"   Dataset size: {len(data)} examples")
    print(f"   Columns: {data.column_names}")
    print(f"   Sample: {data[0]}")

    prompt_builder = PROMPT_BUILDERS[dataset_config["task"]]
    max_new_tokens = MODELS.get(model_key, {}).get("max_new_tokens", MAX_NEW_TOKENS)

    correct = 0
    total = 0
    errors = []

    for i, example in enumerate(tqdm(data, desc=f"   {dataset_key}")):
        try:
            prompt = prompt_builder(example)

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    use_cache=MODELS.get(model_key, {}).get("generation_use_cache", False),
                    do_sample=False,        # greedy — deterministic
                    temperature=1.0,
                    pad_token_id=tokenizer.pad_token_id,
                )

            # Decode only the NEW tokens
            new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            generated = tokenizer.decode(new_tokens, skip_special_tokens=True)

            pred = extract_answer(generated)
            gold = get_gold_answer(example)

            if pred == gold:
                correct += 1
            total += 1

            # Log first 5 predictions for sanity check
            if i < 5:
                print(f"\n   [Sample {i}] Gold={gold} | Pred={pred} | Raw='{generated[:50]}'")

        except Exception as e:
            errors.append({"index": i, "error": str(e)})
            if len(errors) <= 3:
                print(f"\n   ⚠️  Error at index {i}: {e}")

    accuracy = correct / total if total > 0 else 0
    print(f"\n   ✅ {dataset_key}: {correct}/{total} = {accuracy:.4f} accuracy")

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "errors": len(errors),
        "error_samples": errors[:5],
    }


# ─────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────

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
    print(f"   💾 Saved: {path}")


def run_experiment(model_key, dataset_key, skip_existing=False):
    result_path = f"{RESULTS_DIR}/{model_key}_{dataset_key}.json"
    if skip_existing and os.path.exists(result_path):
        print(f"\n⏭️  Skipping {model_key}/{dataset_key}")
        return

    model, tokenizer = load_model(model_key)
    result = evaluate_dataset(model, tokenizer, dataset_key, DATASETS_CONFIG[dataset_key], model_key)
    save_result(model_key, dataset_key, result)
    unload_model(model, tokenizer)


def run_all(skip_existing=False, model_filter=None):
    model_keys = [model_filter] if model_filter else list(MODELS.keys())
    total = len(model_keys) * len(DATASETS_CONFIG)
    done = 0

    for model_key in model_keys:
        # Load model ONCE, run all datasets, then unload
        print(f"\n{'#'*60}")
        print(f"# LOADING {model_key.upper()}")
        print(f"{'#'*60}")

        model, tokenizer = load_model(model_key)

        for dataset_key in DATASETS_CONFIG:
            done += 1
            result_path = f"{RESULTS_DIR}/{model_key}_{dataset_key}.json"
            if skip_existing and os.path.exists(result_path):
                print(f"\n⏭️  [{done}/{total}] Skipping {model_key}/{dataset_key}")
                continue

            print(f"\n{'─'*40}")
            print(f"  [{done}/{total}] {model_key} × {dataset_key}")
            print(f"{'─'*40}")

            try:
                result = evaluate_dataset(model, tokenizer, dataset_key, DATASETS_CONFIG[dataset_key], model_key)
                save_result(model_key, dataset_key, result)
            except Exception as e:
                print(f"\n   ❌ FAILED: {e}")
                save_result(model_key, dataset_key, {"accuracy": None, "error": str(e)})

        unload_model(model, tokenizer)

    print(f"\n\n{'='*60}")
    print(f"🏁 TRACK B COMPLETE — {done} experiments finished")
    print(f"   Results in: {RESULTS_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track B: Evaluate generative LLMs")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--model", choices=list(MODELS.keys()))
    parser.add_argument("--dataset", choices=list(DATASETS_CONFIG.keys()))
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    if args.all:
        run_all(args.skip_existing, args.model)
    elif args.model and args.dataset:
        run_experiment(args.model, args.dataset, args.skip_existing)
    else:
        parser.print_help()
