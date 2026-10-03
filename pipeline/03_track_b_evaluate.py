"""
Step 3 (v2): Track B — zero-shot evaluation of instruction-tuned decoders.
RTX 4060 (8 GB) + 64 GB RAM, same loading recipe as the as-run script (original_as_run/03_track_b_evaluate.py).

What changed vs. the as-run script (see ../AUDIT_REPORT.md for the evidence behind each item):
  * data adapters, prompts, gold labels and answer extraction come from harness.py (unit-tested on CPU);
    a dataset whose gold cannot be resolved aborts the run instead of silently scoring 0;
  * the same generation budget for every model (the as-run config gave Jais max_new_tokens=1);
  * batch size 1 for every model (the Falcon rerun used batched left-padded generation);
  * chat template applied when the tokenizer has one (instruct models), recorded in the result;
  * every generation is logged to <out>/raw/<model>_<dataset>.jsonl (prompt, raw output, parsed option, rule, gold);
  * parse rate, prediction distribution, degenerate-output rate, chance and constant-letter baselines
    are reported next to accuracy;
  * --mode loglik scores each option label by log-likelihood (no extractor at all), as recommended in Sec. VII.

Usage:
    python 03_track_b_evaluate.py --model allam --dataset arabic_mmlu --limit 300          # quick check
    python 03_track_b_evaluate.py --all --mode loglik                                     # primary protocol
    python 03_track_b_evaluate.py --all --mode generate                                   # secondary protocol
    python 03_track_b_evaluate.py --model falcon --all --harness legacy --limit 200       # reproduce as-run behaviour
"""
import compat_shims  # noqa: F401  (must run before model classes are imported)

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import transformers
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import harness as H

MODELS = {
    "allam":  {"path": "ALLaM-AI/ALLaM-7B-Instruct-preview", "use_4bit": True, "cpu_offload": False, "use_fast_tokenizer": False},
    "jais":   {"path": "F:/hf_cache/jais-family-13b-chat-fp16", "use_4bit": False, "cpu_offload": True, "use_fast_tokenizer": False},
    "qwen":   {"path": "Qwen/Qwen2.5-7B-Instruct", "use_4bit": True, "cpu_offload": False, "use_fast_tokenizer": True},
    "falcon": {"path": "tiiuae/falcon-7b-instruct", "use_4bit": True, "cpu_offload": False, "use_fast_tokenizer": True,
               "use_native_transformers": True},
}
DATASETS = ["arabic_mmlu", "madinah_qa", "exams_ar", "aratrust", "arabculture", "aradice"]
DATA_DIR = Path(os.environ.get("PREPARED_DATA", "prepared_data"))
OUT_DIR = Path(os.environ.get("TRACK_B_OUT", "results/track_b_v2"))
MAX_NEW_TOKENS = 20
MAX_PROMPT_TOKENS = 2048   # as-run used 1024 with right truncation, which could cut the "الإجابة:" cue


# ─────────────────────────── model loading (as-run recipe) ───────────────────────────

def load_model(model_key):
    cfg = MODELS[model_key]
    path = cfg["path"]
    is_local = bool(os.path.isabs(path) or (len(path) > 2 and path[1] == ":"))
    if is_local:
        path = os.path.normpath(path)
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    trust_remote_code = not cfg.get("use_native_transformers", False)
    local = {"local_files_only": True} if is_local else {}
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=trust_remote_code,
                                        use_fast=cfg.get("use_fast_tokenizer", True), **local)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    kw = {"trust_remote_code": trust_remote_code, "torch_dtype": torch.float16, "device_map": "auto", **local}
    if cfg["cpu_offload"]:
        kw.update(max_memory={0: "5GiB", "cpu": "56GiB"}, offload_folder="F:/hf_cache/offload",
                  offload_buffers=True, low_cpu_mem_usage=True)
    elif cfg["use_4bit"]:
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4", llm_int8_enable_fp32_cpu_offload=True)
        kw["max_memory"] = {0: "6GiB" if model_key == "falcon" else "7GiB", "cpu": "56GiB"}
        kw["low_cpu_mem_usage"] = True
    model = AutoModelForCausalLM.from_pretrained(path, **kw)
    model.config.use_cache = model_key != "jais"
    model.eval()
    return model, tok


def unload(model, tok):
    del model, tok
    import gc
    gc.collect()
    torch.cuda.empty_cache()


# ─────────────────────────── prompt → tokens ───────────────────────────

def render(tok, prompt: str, use_chat: bool) -> str:
    if use_chat and getattr(tok, "chat_template", None):
        return tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    return prompt


def encode(tok, text: str, device):
    ids = tok(text, return_tensors="pt", add_special_tokens=not text.startswith(tok.bos_token or "\0"))
    n = ids["input_ids"].shape[1]
    if n > MAX_PROMPT_TOKENS:  # truncate from the LEFT so the answer cue survives
        ids = {k: v[:, -MAX_PROMPT_TOKENS:] for k, v in ids.items()}
    return {k: v.to(device) for k, v in ids.items()}, n > MAX_PROMPT_TOKENS


# ─────────────────────────── scoring ───────────────────────────

@torch.inference_mode()
def generate_one(model, tok, text, max_new_tokens):
    enc, truncated = encode(tok, text, model.device)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tok.pad_token_id, use_cache=model.config.use_cache)
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True), truncated


@torch.inference_mode()
def loglik_one(model, tok, text, labels):
    """Log-likelihood of each option label as the continuation of the prompt (rank classification)."""
    enc, truncated = encode(tok, text, model.device)
    base = enc["input_ids"]
    cont_ids = [tok(" " + lab, add_special_tokens=False)["input_ids"] for lab in labels]
    if all(len(c) == 1 for c in cont_ids) and len({c[0] for c in cont_ids}) == len(cont_ids):
        logp = torch.log_softmax(model(input_ids=base).logits[0, -1].float(), -1)
        scores = [logp[c[0]].item() for c in cont_ids]
    else:
        scores = []
        for c in cont_ids:
            ids = torch.cat([base, torch.tensor([c], device=base.device)], 1)
            logp = torch.log_softmax(model(input_ids=ids).logits[0].float(), -1)
            start = base.shape[1] - 1
            scores.append(sum(logp[start + j, t].item() for j, t in enumerate(c)))
    return scores, truncated


# ─────────────────────────── legacy (as-run) harness, for reproduction only ───────────────────────────

def legacy_functions():
    src = (Path(__file__).resolve().parent.parent / "original_as_run" / "03_track_b_evaluate.py").read_text(encoding="utf-8")
    ns = {"re": __import__("re")}
    exec(src[src.index("def build_mcq_prompt"):src.index("# ─────────────────────────────────────────────\n# Evaluation Loop")], ns)
    return ns


# ─────────────────────────── evaluation loop ───────────────────────────

def evaluate(model, tok, model_key, dataset_key, args):
    data = load_from_disk(str(DATA_DIR / dataset_key))["test"]
    if args.limit:
        data = data.select(range(min(args.limit, len(data))))
    raw_dir = OUT_DIR / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{model_key}_{dataset_key}_{args.mode}{'_legacy' if args.harness == 'legacy' else ''}"
    started = time.time()

    if args.harness == "legacy":
        L = legacy_functions()
        task = {"arabic_mmlu": "mcq", "madinah_qa": "mcq", "exams_ar": "mcq", "aratrust": "trust",
                "arabculture": "culture", "aradice": "culture"}[dataset_key]
        correct = 0
        with open(raw_dir / f"{tag}.jsonl", "w", encoding="utf-8") as fh:
            for i, ex in enumerate(tqdm(data, desc=tag)):
                prompt = L["PROMPT_BUILDERS"][task](ex)
                raw, _ = generate_one(model, tok, prompt, 20)
                pred, gold = L["extract_answer"](raw), L["get_gold_answer"](ex)
                correct += pred == gold
                fh.write(json.dumps({"i": i, "raw": raw, "pred": pred, "gold": gold}, ensure_ascii=False) + "\n")
        return {"harness": "legacy(as-run)", "n": len(data), "correct": correct, "accuracy": correct / len(data)}

    items = H.load_items(dataset_key, data, shuffle=args.shuffle_all, seed=args.seed)  # raises on missing gold
    preds, rules, n_trunc = [], [], 0
    use_chat = not args.no_chat_template
    with open(raw_dir / f"{tag}.jsonl", "w", encoding="utf-8") as fh:
        for it in tqdm(items, desc=tag):
            prompt = H.build_prompt(it)
            text = render(tok, prompt, use_chat)
            rec = {"id": it.item_id, "gold": it.gold_letter, "permutation": it.permutation}
            if args.mode == "generate":
                raw, tr = generate_one(model, tok, text, args.max_new_tokens)
                p, rule = H.extract_choice(raw, it)
                rec.update(raw=raw)
            else:
                scores, tr = loglik_one(model, tok, text, H.option_labels(it))
                p, rule = max(range(len(scores)), key=scores.__getitem__), "loglik"
                rec.update(scores=scores)
            n_trunc += tr
            preds.append(p)
            rules.append(rule)
            rec.update(pred=None if p is None else H.EN_LABELS[p], rule=rule, prompt_truncated=tr)
            if len(preds) <= args.log_prompts:
                rec["prompt"] = text
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    s = H.summarize(items, preds, rules)
    s.update(harness="v2", mode=args.mode, prompts_truncated=n_trunc, runtime_seconds=time.time() - started,
             chat_template=bool(use_chat and getattr(tok, "chat_template", None)),
             max_new_tokens=args.max_new_tokens if args.mode == "generate" else None,
             first_prompt=render(tok, H.build_prompt(items[0]), use_chat),
             prompt_template_sha1=hashlib.sha1(Path(H.__file__).read_bytes()).hexdigest()[:12])
    flag = "" if s["above_best_constant"] else "  <-- NOT above best constant-letter baseline"
    print(f"   {tag}: acc={s['accuracy']:.4f} parse={s['parse_rate']:.3f} "
          f"best_const={s['best_constant_acc']:.4f} dist={s['prediction_distribution']}{flag}")
    return s


def save(model_key, dataset_key, args, result):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{model_key}_{dataset_key}_{args.mode}{'_legacy' if args.harness == 'legacy' else ''}"
    entry = {"model": model_key, "model_path": MODELS[model_key]["path"], "dataset": dataset_key,
             "timestamp": datetime.now().isoformat(), "limit": args.limit, "seed": args.seed,
             "env": {"python": platform.python_version(), "torch": torch.__version__,
                     "transformers": transformers.__version__,
                     "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"},
             **result}
    (OUT_DIR / f"{tag}.json").write_text(json.dumps(entry, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Track B v2")
    ap.add_argument("--model", choices=list(MODELS))
    ap.add_argument("--dataset", choices=DATASETS)
    ap.add_argument("--all", action="store_true", help="all datasets (and all models unless --model is given)")
    ap.add_argument("--mode", choices=["loglik", "generate"], default="loglik")
    ap.add_argument("--harness", choices=["v2", "legacy"], default="v2")
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    ap.add_argument("--no-chat-template", action="store_true")
    ap.add_argument("--shuffle-all", action="store_true", help="shuffle options for every dataset (AraDiCE is always shuffled)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--log-prompts", type=int, default=20, help="store the full prompt for the first N items")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()
    if not (args.all or (args.model and args.dataset)):
        ap.error("give --all, or --model and --dataset")

    torch.manual_seed(args.seed)
    models = [args.model] if args.model else list(MODELS)
    datasets = DATASETS if args.all and not args.dataset else [args.dataset]
    for m in models:
        model, tok = load_model(m)
        for d in datasets:
            tag = f"{m}_{d}_{args.mode}{'_legacy' if args.harness == 'legacy' else ''}"
            if args.skip_existing and (OUT_DIR / f"{tag}.json").exists():
                continue
            try:
                save(m, d, args, evaluate(model, tok, m, d, args))
            except H.GoldLabelError:
                raise  # a data defect must stop the run, never become a 0.0 score
            except Exception as e:  # noqa: BLE001
                print(f"   FAILED {tag}: {e}", file=sys.stderr)
                save(m, d, args, {"accuracy": None, "error": repr(e)})
        unload(model, tok)


if __name__ == "__main__":
    main()
