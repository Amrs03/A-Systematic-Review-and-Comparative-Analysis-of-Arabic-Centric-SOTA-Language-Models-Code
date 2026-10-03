"""
Post-hoc audit of the Track B (zero-shot decoder) results.

CPU only: it downloads no models and runs no generation. Everything is recomputed from
the artifacts that were actually produced by the original runs:

  <root>/results/prepared_data/<dataset>/test      (the exact data every run used, prepared 2026-04-01)
  <root>/results/track_b/<model>_<dataset>.json     (the reported scores)
  <root>/results/03_track_b_evaluate.py            (the as-run prompt builders / extractor / gold lookup)
  <root>/logs/falcon_only_rerun_stdout.log          (the only surviving raw generations)

For every (model, dataset) cell it reports:
  * gold coverage   - share of items for which the as-run get_gold_answer() returns a letter A-E
  * prompt coverage - whether the as-run prompt builder shows the stem, the context, and the options
  * chance level    - mean 1/k over items (k = number of options actually shown)
  * constant-letter baselines - the accuracy of always answering A, B, C, D or E
  * an exact binomial test of the reported score against the best constant-letter baseline
  * a status label with the reason

Usage:
    python audit_track_b.py --root /path/to/ARABIC --out ./results_audit
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

import pandas as pd
from datasets import load_from_disk
from scipy.stats import binomtest

DATASETS = {
    "arabic_mmlu": "mcq",
    "madinah_qa": "mcq",
    "exams_ar": "mcq",
    "aratrust": "trust",
    "arabculture": "culture",
    "aradice": "culture",
}
DATASET_LABELS = {
    "arabic_mmlu": "Arabic MMLU", "madinah_qa": "Madinah QA", "exams_ar": "EXAMS AR",
    "aratrust": "AraTrust", "arabculture": "ArabCulture", "aradice": "AraDiCE",
}
MODELS = {"allam": "ALLaM-7B", "falcon": "Falcon-7B", "jais": "Jais-13B", "qwen": "Qwen-7B"}
# Generation budget actually configured in the as-run code (03_track_b_evaluate.py, MODELS dict).
# ALLaM/Qwen ran on 2026-04-04/06 with an earlier revision of the script that is not preserved;
# the budget for them is the global default in every surviving revision.
DECODE_BUDGET = {"allam": 20, "qwen": 20, "falcon": 20, "jais": 1}
LETTERS = ["A", "B", "C", "D", "E"]
OPTION_MARKERS = ("أ)", "ب)", "ج)", "د)", "هـ)")


def load_as_run_functions(script: Path) -> dict:
    """Execute only the pure prompt/extract/gold functions of the as-run evaluator (no torch import)."""
    src = script.read_text(encoding="utf-8")
    start = src.index("def build_mcq_prompt")
    end = src.index("# ─────────────────────────────────────────────\n# Evaluation Loop")
    ns: dict = {"re": re}
    exec(src[start:end], ns)  # noqa: S102 - trusted local file, audited by hand
    return ns


def empty(v) -> bool:
    return v is None or (isinstance(v, str) and v.strip() in ("", "None"))


def dataset_profile(name: str, task: str, fn: dict, data) -> dict:
    builder = fn["PROMPT_BUILDERS"][task]
    golds, n_opts = [], []
    empty_stem = dropped_context = dropped_stem_field = 0
    for ex in data:
        prompt = builder(ex)
        shown = sum(1 for line in prompt.split("\n") if line.startswith(OPTION_MARKERS))
        n_opts.append(shown)
        if re.match(r"^السؤال: *\n(أ|ب|ج|د|هـ)\)", prompt):  # stem line is empty, options follow directly
            empty_stem += 1
        if "Context" in ex and not empty(ex["Context"]) and str(ex["Context"]).strip()[:30] not in prompt:
            dropped_context += 1
        if "first_statement" in ex and not empty(ex["first_statement"]) and ex["first_statement"] not in prompt:
            dropped_stem_field += 1
        golds.append(fn["get_gold_answer"](ex))
    n = len(golds)
    gold_counts = collections.Counter(golds)
    valid_gold = sum(gold_counts[L] for L in LETTERS)
    # options that exist in the data but were not shown (AraTrust: columns A/B/C never rendered)
    opts_in_data = 0
    for ex in data:
        k = sum(1 for key in ("A", "B", "C", "D", "Option A", "Option B", "Option C",
                               "Option 1", "Option 2", "Option 3", "Option 4", "Option 5")
                if key in ex and not empty(ex[key]))
        if not k and "options" in ex and isinstance(ex["options"], dict):
            k = len(ex["options"].get("text", []))
        opts_in_data += k
    chance = sum((1.0 / k) if k else 1.0 / max(1, round(opts_in_data / n)) for k in n_opts) / n
    return {
        "dataset": name,
        "n": n,
        "columns": ",".join(data.column_names),
        "gold_coverage": valid_gold / n,
        "gold_counts": {L: gold_counts.get(L, 0) for L in LETTERS},
        "empty_gold": gold_counts.get("", 0),
        "prompts_with_empty_stem": empty_stem,
        "prompts_dropping_context": dropped_context,
        "prompts_showing_no_options": sum(1 for k in n_opts if k == 0),
        "options_shown_hist": dict(collections.Counter(n_opts)),
        "chance_level": chance,
    }


def parse_falcon_log(log: Path) -> pd.DataFrame:
    rows = []
    cur = None
    pat = re.compile(r"sample (\d+): gold='([^']*)' pred='([^']*)' status=(\w+) raw='(.*)'$")
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"Falcon-Arabic -> (\w+):", line)
        if m:
            cur = m.group(1)
            continue
        m = pat.search(line)
        if m and cur:
            raw = m.group(5)
            body = raw.strip()
            degenerate = len(body) >= 8 and collections.Counter(body).most_common(1)[0][1] / len(body) >= 0.8
            rows.append({"dataset": cur, "sample": int(m.group(1)), "gold": m.group(2),
                         "pred": m.group(3), "raw": raw, "degenerate_repetition": degenerate})
    return pd.DataFrame(rows)


def classify(model: str, ds: str, prof: dict, res: dict | None, falcon: pd.DataFrame | None = None) -> tuple[str, str, dict]:
    if res is None:
        return "NO_RESULT", "No result file was produced (run did not complete).", {}
    n, correct = res["total"], res["correct"]
    acc = correct / n if n else 0.0
    gc = prof["gold_counts"]
    best_L = max(gc, key=gc.get)
    best_const = gc[best_L] / n
    exact_const = [L for L in LETTERS if gc[L] and gc[L] == correct]
    near_const = sorted([L for L in LETTERS if gc[L] and abs(gc[L] - correct) <= max(3, 0.03 * gc[L]) and correct <= gc[L]],
                        key=lambda L: abs(gc[L] - correct))
    p_vs_const = binomtest(correct, n, best_const, alternative="greater").pvalue if best_const > 0 else float("nan")
    p_vs_chance = binomtest(correct, n, prof["chance_level"], alternative="greater").pvalue
    extra = {"best_constant_letter": best_L, "best_constant_acc": best_const,
             "p_above_best_constant": p_vs_const, "p_above_chance": p_vs_chance,
             "exact_constant_match": ",".join(exact_const), "near_constant_match": ",".join(near_const)}

    if prof["gold_coverage"] == 0:
        return ("GOLD_MISSING",
                f"Prepared data has no answer column ({prof['columns']}); get_gold_answer() returns '' for all "
                f"{n} items, so no prediction can match. Score is structurally 0 for every model.", extra)
    if prof["prompts_showing_no_options"] == n:
        also = (f" The correct count ({correct}) also exactly equals the number of gold='{exact_const[0]}' items."
                if exact_const else "")
        return ("PROMPT_DEFECT",
                "build_trust_prompt() renders only the question; the three answer options (columns A/B/C) are "
                "never shown, yet the item is scored against option letters. Score is not a capability measurement."
                + also, extra)
    if ds == "arabculture" and prof["prompts_with_empty_stem"] == n:
        stem_note = ("All prompts omit the scenario sentence (first_statement): build_mcq_prompt() looks for "
                     "Question/question/stem/input only, so every prompt is 'السؤال: ' followed by options.")
    else:
        stem_note = ""
    if correct == 0 and prof["gold_coverage"] == 1.0 and n >= 100:
        return ("ZERO_ANOMALY_GOLD_LOOKUP",
                "Exactly 0 correct although gold labels resolve for 100% of items under the surviving code and the "
                "surviving extractor maps almost any Arabic text to a letter. Pattern across models: 0 only on datasets "
                "whose gold field is 'Answer Key' or 'answer_key' (dict); non-zero on 'answer'/'Answer'. Consistent with "
                "the pre-2026-04-21 get_gold_answer() not resolving those fields. " + stem_note, extra)
    if model == "falcon" and falcon is not None and len(falcon[falcon.dataset == ds]) and p_vs_const > 0.05:
        fs = falcon[falcon.dataset == ds]
        return ("DEGENERATE_OUTPUT",
                f"Logged generations are degenerate repetition ({int(fs.degenerate_repetition.sum())}/{len(fs)} logged samples, "
                f"e.g. {fs.raw.iloc[0].strip()[:12]!r}); {int((fs.pred == 'A').sum())}/{len(fs)} were mapped to 'A' by the "
                f"leading-alef rule of extract_answer(). Accuracy {acc:.1%} is not above always-A ({gc['A'] / n:.1%}). "
                + stem_note, extra)
    if exact_const or (near_const and p_vs_const > 0.05):
        L = (exact_const or near_const)[0]
        how = "exactly equals" if exact_const else "is within 3% of (and not above)"
        return ("CONSTANT_PREDICTOR",
                f"Reported correct count ({correct}) {how} the number of items whose gold is '{L}' ({gc[L]}); "
                f"not significantly above the best constant-letter baseline (p={p_vs_const:.2g}). "
                + stem_note, extra)
    if acc < prof["chance_level"] and p_vs_chance > 0.5 and acc < 0.05:
        return ("DEGENERATE_NEAR_ZERO",
                f"{correct}/{n} correct, far below chance ({prof['chance_level']:.1%}); with a "
                f"{DECODE_BUDGET[model]}-token budget the output could not be interpreted.", extra)
    if stem_note:
        return ("PROMPT_DEFECT", stem_note, extra)
    if p_vs_const < 0.001:
        partial = ""
        if prof["prompts_dropping_context"]:
            partial = f" Note: {prof['prompts_dropping_context']} prompts dropped their Context passage."
        return ("ABOVE_BASELINE", f"Significantly above the best constant-letter baseline "
                f"({best_L}={best_const:.1%}, p={p_vs_const:.1g}) and chance ({prof['chance_level']:.1%}).{partial}", extra)
    return ("NOT_DISTINGUISHABLE", "Not significantly above the best constant-letter baseline.", extra)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="ARABIC project folder (contains results/ and logs/)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    fn = load_as_run_functions(root / "results" / "03_track_b_evaluate.py")
    profiles = {}
    for ds, task in DATASETS.items():
        data = load_from_disk(str(root / "results" / "prepared_data" / ds))["test"]
        profiles[ds] = dataset_profile(ds, task, fn, data)

    falcon = parse_falcon_log(root / "logs" / "falcon_only_rerun_stdout.log")
    rows = []
    for m in MODELS:
        for ds in DATASETS:
            p = root / "results" / "track_b" / f"{m}_{ds}.json"
            res = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
            status, reason, extra = classify(m, ds, profiles[ds], res, falcon)
            prof = profiles[ds]
            rows.append({
                "model": MODELS[m], "model_key": m, "dataset": DATASET_LABELS[ds], "dataset_key": ds,
                "reported_acc_pct": round(100 * res["accuracy"], 2) if res else None,
                "correct": res["correct"] if res else None, "total": res["total"] if res else prof["n"],
                "run_date": res["timestamp"][:10] if res else None,
                "decode_budget_tokens": DECODE_BUDGET[m],
                "chance_pct": round(100 * prof["chance_level"], 1),
                "best_constant_letter": extra.get("best_constant_letter"),
                "best_constant_pct": round(100 * extra["best_constant_acc"], 1) if extra.get("best_constant_acc") is not None else None,
                "p_above_best_constant": extra.get("p_above_best_constant"),
                "exact_constant_match": extra.get("exact_constant_match"),
                "status": status, "reason": reason,
            })
    audit = pd.DataFrame(rows)
    audit.to_csv(out / "track_b_audit.csv", index=False, encoding="utf-8")

    prof_df = pd.DataFrame([{**{k: v for k, v in p.items() if k != "gold_counts"},
                             **{f"gold_{L}": p["gold_counts"][L] for L in LETTERS}} for p in profiles.values()])
    prof_df.to_csv(out / "dataset_profiles.csv", index=False, encoding="utf-8")

    falcon.to_csv(out / "falcon_logged_samples.csv", index=False, encoding="utf-8")

    summary = {
        "profiles": profiles,
        "falcon_logged_samples": {
            "n": int(len(falcon)),
            "pred_A": int((falcon["pred"] == "A").sum()),
            "degenerate_repetition": int(falcon["degenerate_repetition"].sum()),
            "raw_starting_with_alef_lam_run": int(falcon["raw"].str.contains("أللل").sum()),
        },
        "status_counts": audit["status"].value_counts().to_dict(),
    }
    (out / "audit_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    pd.set_option("display.width", 250, "display.max_colwidth", 60)
    print(prof_df.drop(columns=["columns"]).to_string(index=False))
    print()
    print(audit[["model", "dataset", "reported_acc_pct", "chance_pct", "best_constant_letter", "best_constant_pct",
                 "exact_constant_match", "status"]].to_string(index=False))
    print()
    print(json.dumps(summary["falcon_logged_samples"], indent=2))
    print(summary["status_counts"])


if __name__ == "__main__":
    main()
