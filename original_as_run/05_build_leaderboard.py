"""
Step 5: Assemble all results into the final leaderboard.
Reads all JSON result files and produces a unified leaderboard + per-task tables.

Usage:
    python 05_build_leaderboard.py
"""

import os
import json
import csv
from collections import defaultdict

RESULTS_DIRS = {
    "track_a": "results/track_a",
    "track_b": "results/track_b",
    "bridge": "results/bridge",
}

OUTPUT_DIR = "results"

TASK_CATEGORIES = {
    "Sentiment Analysis": {
        "datasets": ["hard", "astd"],
        "metric": "f1_macro",
        "metric_label": "F1",
    },
    "Named Entity Recognition": {
        "datasets": ["anercorp"],
        "metric": "f1_macro",
        "metric_label": "F1",
    },
    "Dialect Identification": {
        "datasets": ["qadi", "aoc"],
        "metric": "f1_macro",
        "metric_label": "F1",
    },
    "Social Meaning": {
        "datasets": ["arsarcasm", "aratrust"],
        "metric": "f1_macro",  # arsarcasm uses f1, aratrust uses accuracy
        "metric_label": "F1/Acc",
    },
    "Commonsense / STEM": {
        "datasets": ["arabic_mmlu", "madinah_qa", "exams_ar"],
        "metric": "accuracy",
        "metric_label": "Accuracy",
    },
    "Cultural Understanding": {
        "datasets": ["arabculture", "aradice"],
        "metric": "accuracy",
        "metric_label": "Accuracy",
    },
}

ALL_MODELS = ["arabert", "marbert", "allam", "qwen", "falcon"]  # jais skipped for now


def load_all_results():
    """Load every JSON result file into a nested dict: results[model][dataset] = {...}"""
    results = defaultdict(dict)

    for track, dir_path in RESULTS_DIRS.items():
        if not os.path.exists(dir_path):
            continue
        for fname in os.listdir(dir_path):
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(dir_path, fname)) as f:
                data = json.load(f)

            model = data.get("model", "unknown")
            dataset = data.get("dataset", fname.replace(".json", ""))
            data["_track"] = track
            results[model][dataset] = data

    return results


def get_score(result, metric):
    """Extract the score from a result dict, handling both f1_macro and accuracy."""
    if result is None:
        return None

    # Try the requested metric first
    score = result.get(metric)
    if score is not None:
        return score

    # Fallback: try the other metric
    for key in ["f1_macro", "accuracy", "f1_micro"]:
        score = result.get(key)
        if score is not None:
            return score

    return None


def build_leaderboard():
    results = load_all_results()

    print("=" * 80)
    print("ARABIC NLP LEADERBOARD — FINAL RESULTS")
    print("=" * 80)

    if not results:
        print("\n❌ No results found! Make sure you've run the evaluation scripts first.")
        print(f"   Expected results in: {list(RESULTS_DIRS.values())}")
        return

    print(f"\nFound results for models: {list(results.keys())}")
    for model, datasets in results.items():
        print(f"  {model}: {list(datasets.keys())}")

    # ── Per-Task Tables ──
    all_task_scores = defaultdict(dict)  # model -> category -> avg_score

    for category, config in TASK_CATEGORIES.items():
        print(f"\n\n{'─'*60}")
        print(f"📊 {category}")
        print(f"{'─'*60}")

        metric = config["metric"]
        datasets = config["datasets"]

        # Header
        header = f"{'Model':<12}"
        for ds in datasets:
            header += f" | {ds:<15}"
        header += f" | {'Average':<10}"
        print(header)
        print("-" * len(header))

        for model in ALL_MODELS:
            row = f"{model:<12}"
            scores = []

            for ds in datasets:
                result = results.get(model, {}).get(ds)
                score = get_score(result, metric) if result else None

                if score is not None:
                    row += f" | {score*100:>13.1f}%"
                    scores.append(score)
                else:
                    row += f" | {'N/A':>14}"

            avg = sum(scores) / len(scores) if scores else None
            if avg is not None:
                row += f" | {avg*100:>8.1f}%"
                all_task_scores[model][category] = avg
            else:
                row += f" | {'N/A':>9}"

            print(row)

    # ── Unified Leaderboard ──
    print(f"\n\n{'='*80}")
    print("🏆 UNIFIED LEADERBOARD")
    print(f"{'='*80}")

    categories = list(TASK_CATEGORIES.keys())
    header = f"{'Model':<12}"
    for cat in categories:
        short = cat[:8]
        header += f" | {short:<10}"
    header += f" | {'OVERALL':<10}"
    print(header)
    print("-" * len(header))

    model_overalls = {}
    for model in ALL_MODELS:
        row = f"{model:<12}"
        scores = []

        for cat in categories:
            score = all_task_scores.get(model, {}).get(cat)
            if score is not None:
                row += f" | {score*100:>8.1f}%"
                scores.append(score)
            else:
                row += f" | {'—':>9}"

        overall = sum(scores) / len(scores) if scores else None
        model_overalls[model] = overall
        if overall is not None:
            row += f" | {overall*100:>8.1f}%"
        else:
            row += f" | {'—':>9}"

        print(row)

    # ── Rankings ──
    print(f"\n\n{'─'*40}")
    print("🥇 RANKINGS (by overall average)")
    print(f"{'─'*40}")

    ranked = sorted(
        [(m, s) for m, s in model_overalls.items() if s is not None],
        key=lambda x: x[1], reverse=True
    )
    for i, (model, score) in enumerate(ranked, 1):
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(i, "  ")
        cats_participated = len(all_task_scores.get(model, {}))
        print(f"  {medal} {i}. {model:<12} {score*100:.1f}% (across {cats_participated} categories)")

    # ── Export CSV ──
    csv_path = f"{OUTPUT_DIR}/leaderboard.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model"] + categories + ["Overall"])
        for model in ALL_MODELS:
            row = [model]
            for cat in categories:
                score = all_task_scores.get(model, {}).get(cat)
                row.append(f"{score*100:.1f}%" if score else "N/A")
            overall = model_overalls.get(model)
            row.append(f"{overall*100:.1f}%" if overall else "N/A")
            writer.writerow(row)

    print(f"\n📁 CSV exported: {csv_path}")

    # ── Export full JSON ──
    json_path = f"{OUTPUT_DIR}/leaderboard_full.json"
    export = {
        "models": ALL_MODELS,
        "categories": {cat: {"datasets": cfg["datasets"], "metric": cfg["metric"]}
                       for cat, cfg in TASK_CATEGORIES.items()},
        "per_task_scores": dict(all_task_scores),
        "overall_scores": model_overalls,
        "raw_results": {m: {d: {k: v for k, v in r.items() if k != "classification_report"}
                            for d, r in ds.items()}
                        for m, ds in results.items()},
    }
    with open(json_path, "w") as f:
        json.dump(export, f, indent=2, ensure_ascii=False, default=str)

    print(f"📁 Full JSON: {json_path}")
    print(f"\n🎉 LEADERBOARD COMPLETE!")


if __name__ == "__main__":
    build_leaderboard()
