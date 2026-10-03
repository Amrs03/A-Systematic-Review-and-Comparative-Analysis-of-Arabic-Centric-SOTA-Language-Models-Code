"""
Compile evaluation JSON files into readable tables, reports, and visuals.

Outputs:
    results/compiled_results.csv
    results/compiled_category_scores.csv
    results/compiled_leaderboard.csv
    results/compiled_report.md
    results/compiled_report.html
    results/visuals/*.png
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime
from html import escape
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
TRACK_DIRS = {
    "Track A": RESULTS_DIR / "track_a",
    "Track B": RESULTS_DIR / "track_b",
    "Bridge": RESULTS_DIR / "bridge",
}
VISUALS_DIR = RESULTS_DIR / "visuals"

TASK_CATEGORIES = {
    "Sentiment Analysis": {
        "datasets": ["hard", "astd"],
        "metric_label": "F1 macro",
    },
    "Named Entity Recognition": {
        "datasets": ["anercorp"],
        "metric_label": "F1 macro",
    },
    "Dialect Identification": {
        "datasets": ["qadi", "aoc"],
        "metric_label": "F1 macro",
    },
    "Social Meaning": {
        "datasets": ["arsarcasm", "aratrust"],
        "metric_label": "F1 macro / Accuracy",
    },
    "Commonsense / STEM": {
        "datasets": ["arabic_mmlu", "madinah_qa", "exams_ar"],
        "metric_label": "Accuracy",
    },
    "Cultural Understanding": {
        "datasets": ["arabculture", "aradice"],
        "metric_label": "Accuracy",
    },
}

DATASET_LABELS = {
    "anercorp": "ANERcorp",
    "arsarcasm": "ArSarcasm",
    "astd": "ASTD",
    "hard": "HARD",
    "qadi": "QADI",
    "aoc": "AOC",
    "arabic_mmlu": "Arabic MMLU",
    "madinah_qa": "Madinah QA",
    "exams_ar": "EXAMS AR",
    "arabculture": "ArabCulture",
    "aradice": "AraDiCE",
    "aratrust": "AraTrust",
}

MODEL_LABELS = {
    "arabert": "AraBERT",
    "marbert": "MARBERT",
    "allam": "ALLAM",
    "qwen": "Qwen",
    "falcon": "Falcon",
    "jais": "Jais",
}

PALETTE = {
    "arabert": "#276FBF",
    "marbert": "#2A9D8F",
    "allam": "#E76F51",
    "qwen": "#F4A261",
    "falcon": "#6D597A",
    "jais": "#8AB17D",
}


def pct(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value * 100:.1f}%"


def md_table(headers: list[str], rows: list[list[object]]) -> str:
    values = [[str(cell) for cell in row] for row in rows]
    widths = []
    for idx, header in enumerate(headers):
        column_values = [row[idx] for row in values] if values else []
        widths.append(max(len(str(header)), *(len(v) for v in column_values)))

    def fmt_row(row: list[object]) -> str:
        return "| " + " | ".join(str(cell).ljust(widths[idx]) for idx, cell in enumerate(row)) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([fmt_row(headers), separator, *(fmt_row(row) for row in values)])


def category_for_dataset(dataset: str) -> str:
    for category, config in TASK_CATEGORIES.items():
        if dataset in config["datasets"]:
            return category
    return "Uncategorized"


def score_from_result(data: dict) -> tuple[float | None, str]:
    if data.get("f1_macro") is not None:
        return float(data["f1_macro"]), "F1 macro"
    if data.get("accuracy") is not None:
        return float(data["accuracy"]), "Accuracy"
    if data.get("f1_micro") is not None:
        return float(data["f1_micro"]), "F1 micro"
    return None, "N/A"


def load_results() -> pd.DataFrame:
    rows = []
    for track, directory in TRACK_DIRS.items():
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.json")):
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)

            score, metric = score_from_result(data)
            model = str(data.get("model", "unknown"))
            dataset = str(data.get("dataset", path.stem.replace(f"{model}_", "")))
            total = data.get("total")
            correct = data.get("correct")
            errors = data.get("errors", 0)
            error_samples = data.get("error_samples") or []
            first_error = ""
            if error_samples and isinstance(error_samples[0], dict):
                first_error = str(error_samples[0].get("error", ""))

            status = "valid"
            note = ""
            if total == 0 and errors:
                status = "invalid"
                note = "All examples errored during evaluation."
                if first_error:
                    note = f"{note} First error: {first_error}"
                score = None
            elif score is None:
                status = "missing_score"
                note = "No supported score field found."

            rows.append(
                {
                    "track": track,
                    "model": model,
                    "model_label": MODEL_LABELS.get(model, model),
                    "dataset": dataset,
                    "dataset_label": DATASET_LABELS.get(dataset, dataset),
                    "category": category_for_dataset(dataset),
                    "metric": metric,
                    "score": score,
                    "score_pct": score * 100 if score is not None else np.nan,
                    "correct": correct,
                    "total": total,
                    "errors": errors,
                    "status": status,
                    "note": note,
                    "first_error": first_error,
                    "timestamp": data.get("timestamp", ""),
                    "source_file": str(path.relative_to(ROOT)),
                }
            )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["track", "category", "dataset", "model"])


def build_category_scores(results: pd.DataFrame) -> pd.DataFrame:
    valid = results[results["status"].eq("valid") & results["score"].notna()].copy()
    if valid.empty:
        return pd.DataFrame()

    grouped = (
        valid.groupby(["model", "model_label", "category"], as_index=False)
        .agg(
            average_score=("score", "mean"),
            datasets=("dataset_label", lambda values: ", ".join(sorted(values))),
            result_count=("dataset", "count"),
        )
        .sort_values(["category", "average_score"], ascending=[True, False])
    )
    grouped["average_score_pct"] = grouped["average_score"] * 100
    return grouped


def build_leaderboard(category_scores: pd.DataFrame) -> pd.DataFrame:
    if category_scores.empty:
        return pd.DataFrame()
    leaderboard = (
        category_scores.groupby(["model", "model_label"], as_index=False)
        .agg(
            overall_score=("average_score", "mean"),
            categories_completed=("category", "count"),
        )
        .sort_values("overall_score", ascending=False)
    )
    leaderboard["overall_score_pct"] = leaderboard["overall_score"] * 100
    leaderboard.insert(0, "rank", range(1, len(leaderboard) + 1))
    return leaderboard


def find_missing_runs(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for track in ["Track A", "Track B", "Bridge"]:
        subset = results[results["track"].eq(track)]
        if subset.empty:
            continue
        models = sorted(subset["model"].unique())
        datasets = sorted(subset["dataset"].unique())
        existing = set(zip(subset["model"], subset["dataset"]))
        for model in models:
            for dataset in datasets:
                if (model, dataset) not in existing:
                    rows.append(
                        {
                            "track": track,
                            "model": model,
                            "model_label": MODEL_LABELS.get(model, model),
                            "dataset": dataset,
                            "dataset_label": DATASET_LABELS.get(dataset, dataset),
                        }
                    )
    return pd.DataFrame(rows)


def save_overall_chart(leaderboard: pd.DataFrame) -> Path | None:
    if leaderboard.empty:
        return None
    chart = leaderboard.sort_values("overall_score_pct", ascending=True)
    colors = [PALETTE.get(model, "#777777") for model in chart["model"]]

    fig, ax = plt.subplots(figsize=(9, 5.2))
    bars = ax.barh(chart["model_label"], chart["overall_score_pct"], color=colors)
    ax.set_xlabel("Average score across completed categories (%)")
    ax.set_title("Overall Leaderboard")
    ax.set_xlim(0, max(100, chart["overall_score_pct"].max() + 8))
    ax.grid(axis="x", alpha=0.22)
    ax.spines[["top", "right", "left"]].set_visible(False)
    for bar, value, count in zip(bars, chart["overall_score_pct"], chart["categories_completed"]):
        ax.text(
            value + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}% ({count} cat.)",
            va="center",
            fontsize=9,
        )
    fig.tight_layout()
    path = VISUALS_DIR / "overall_leaderboard.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_heatmap(category_scores: pd.DataFrame) -> Path | None:
    if category_scores.empty:
        return None
    pivot = category_scores.pivot(index="model_label", columns="category", values="average_score_pct")
    category_order = [cat for cat in TASK_CATEGORIES if cat in pivot.columns]
    pivot = pivot.reindex(columns=category_order)
    pivot = pivot.reindex(pivot.mean(axis=1).sort_values(ascending=False).index)

    values = pivot.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(values)
    cmap = plt.colormaps["YlGnBu"].copy()
    cmap.set_bad("#f1f1f1")

    fig_width = max(9, len(pivot.columns) * 1.9)
    fig_height = max(4.8, len(pivot.index) * 0.65)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(masked, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    ax.set_title("Category Score Heatmap")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(pivot.index)), pivot.index)

    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            if math.isnan(values[row, col]):
                label = "N/A"
                color = "#666666"
            else:
                label = f"{values[row, col]:.1f}"
                color = "white" if values[row, col] > 55 else "#202020"
            ax.text(col, row, label, ha="center", va="center", color=color, fontsize=8.5)

    fig.colorbar(image, ax=ax, label="Score (%)", shrink=0.88)
    fig.tight_layout()
    path = VISUALS_DIR / "category_heatmap.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_dataset_grouped_chart(results: pd.DataFrame, track: str, file_name: str, title: str) -> Path | None:
    subset = results[
        results["track"].eq(track) & results["status"].eq("valid") & results["score_pct"].notna()
    ].copy()
    if subset.empty:
        return None

    dataset_order = list(dict.fromkeys(subset.sort_values(["category", "dataset_label"])["dataset_label"]))
    model_order = list(
        subset.groupby("model_label")["score_pct"].mean().sort_values(ascending=False).index
    )
    pivot = subset.pivot_table(index="dataset_label", columns="model_label", values="score_pct")
    pivot = pivot.reindex(index=dataset_order, columns=model_order)

    fig, ax = plt.subplots(figsize=(max(9, len(dataset_order) * 1.25), 5.6))
    x = np.arange(len(dataset_order))
    width = min(0.75 / max(len(model_order), 1), 0.22)

    for idx, model_label in enumerate(model_order):
        model_key = next((key for key, value in MODEL_LABELS.items() if value == model_label), model_label)
        offset = (idx - (len(model_order) - 1) / 2) * width
        ax.bar(
            x + offset,
            pivot[model_label],
            width,
            label=model_label,
            color=PALETTE.get(model_key, "#777777"),
        )

    ax.set_title(title)
    ax.set_ylabel("Score (%)")
    ax.set_xticks(x, dataset_order, rotation=25, ha="right")
    ax.set_ylim(0, max(100, np.nanmax(pivot.to_numpy()) + 8))
    ax.grid(axis="y", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(ncols=min(len(model_order), 4), frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    path = VISUALS_DIR / file_name
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_error_chart(results: pd.DataFrame) -> Path | None:
    subset = results[results["errors"].fillna(0).gt(0)].copy()
    if subset.empty:
        return None
    errors_by_model = subset.groupby(["model", "model_label"], as_index=False)["errors"].sum()
    errors_by_model = errors_by_model.sort_values("errors", ascending=True)
    colors = [PALETTE.get(model, "#777777") for model in errors_by_model["model"]]

    fig, ax = plt.subplots(figsize=(8, 4.6))
    bars = ax.barh(errors_by_model["model_label"], errors_by_model["errors"], color=colors)
    ax.set_title("Evaluation Errors by Model")
    ax.set_xlabel("Errored examples")
    ax.grid(axis="x", alpha=0.22)
    ax.spines[["top", "right", "left"]].set_visible(False)
    for bar, value in zip(bars, errors_by_model["errors"]):
        ax.text(value + max(errors_by_model["errors"].max() * 0.015, 1), bar.get_y() + bar.get_height() / 2, f"{int(value)}", va="center", fontsize=9)
    fig.tight_layout()
    path = VISUALS_DIR / "evaluation_errors.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_markdown_report(
    results: pd.DataFrame,
    category_scores: pd.DataFrame,
    leaderboard: pd.DataFrame,
    charts: list[Path],
    missing_runs: pd.DataFrame,
) -> Path:
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# Arabic NLP Evaluation Results",
        "",
        f"Generated: {generated}",
        "",
        "This report compiles all JSON files found under `results/track_a`, `results/track_b`, and `results/bridge`.",
        "",
        "Important reading note: scores are averaged only across categories where a model has a valid completed result. Track A and Track B evaluate different model families and tasks, so use the per-track and per-category tables for the fairest comparisons.",
        "",
    ]

    valid_count = int(results["status"].eq("valid").sum())
    invalid_count = int(results["status"].ne("valid").sum())
    all_models = ", ".join(sorted(results["model_label"].unique()))
    ranked_models = ", ".join(leaderboard["model_label"]) if not leaderboard.empty else "N/A"
    lines.extend(
        [
            "## Snapshot",
            "",
            md_table(
                ["Item", "Value"],
                [
                    ["Result files read", len(results)],
                    ["Valid scored runs", valid_count],
                    ["Invalid / missing-score runs", invalid_count],
                    ["Models found in files", all_models],
                    ["Models with valid ranked scores", ranked_models],
                ],
            ),
            "",
        ]
    )

    if not leaderboard.empty:
        rows = [
            [
                int(row.rank),
                row.model_label,
                f"{row.overall_score_pct:.1f}%",
                int(row.categories_completed),
            ]
            for row in leaderboard.itertuples(index=False)
        ]
        lines.extend(["## Overall Leaderboard", "", md_table(["Rank", "Model", "Average", "Categories"], rows), ""])

    if charts:
        lines.extend(["## Visuals", ""])
        for chart in charts:
            rel = chart.relative_to(RESULTS_DIR).as_posix()
            title = chart.stem.replace("_", " ").title()
            lines.append(f"![{title}]({rel})")
            lines.append("")

    if not category_scores.empty:
        rows = [
            [
                row.category,
                row.model_label,
                f"{row.average_score_pct:.1f}%",
                row.datasets,
            ]
            for row in category_scores.sort_values(["category", "average_score"], ascending=[True, False]).itertuples(index=False)
        ]
        lines.extend(["## Category Scores", "", md_table(["Category", "Model", "Average", "Datasets"], rows), ""])

    valid_results = results[results["status"].eq("valid")].sort_values(["track", "category", "dataset_label", "score"], ascending=[True, True, True, False])
    rows = [
        [
            row.track,
            row.category,
            row.dataset_label,
            row.model_label,
            row.metric,
            f"{row.score_pct:.1f}%",
            "" if pd.isna(row.total) else int(row.total),
        ]
        for row in valid_results.itertuples(index=False)
    ]
    lines.extend(["## Dataset-Level Results", "", md_table(["Track", "Category", "Dataset", "Model", "Metric", "Score", "Total"], rows), ""])

    invalid = results[results["status"].ne("valid")].copy()
    if not invalid.empty:
        rows = [
            [
                row.track,
                row.dataset_label,
                row.model_label,
                row.status,
                int(row.errors) if not pd.isna(row.errors) else "",
                row.note,
            ]
            for row in invalid.itertuples(index=False)
        ]
        lines.extend(["## Runs Needing Attention", "", md_table(["Track", "Dataset", "Model", "Status", "Errors", "Note"], rows), ""])

    if not missing_runs.empty:
        rows = [
            [row.track, row.dataset_label, row.model_label]
            for row in missing_runs.itertuples(index=False)
        ]
        lines.extend(["## Missing Result Files", "", md_table(["Track", "Dataset", "Model"], rows), ""])

    path = RESULTS_DIR / "compiled_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def html_table(headers: list[str], rows: list[list[object]]) -> str:
    head = "".join(f"<th>{escape(str(header))}</th>" for header in headers)
    body_rows = []
    for row in rows:
        cells = "".join(f"<td>{escape(str(cell))}</td>" for cell in row)
        body_rows.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def write_html_report(
    results: pd.DataFrame,
    category_scores: pd.DataFrame,
    leaderboard: pd.DataFrame,
    charts: list[Path],
    missing_runs: pd.DataFrame,
) -> Path:
    chart_html = "\n".join(
        f'<figure><img src="{escape(chart.relative_to(RESULTS_DIR).as_posix())}" alt="{escape(chart.stem)}"><figcaption>{escape(chart.stem.replace("_", " ").title())}</figcaption></figure>'
        for chart in charts
    )
    leaderboard_rows = [
        [int(row.rank), row.model_label, f"{row.overall_score_pct:.1f}%", int(row.categories_completed)]
        for row in leaderboard.itertuples(index=False)
    ]
    category_rows = [
        [row.category, row.model_label, f"{row.average_score_pct:.1f}%", row.datasets]
        for row in category_scores.sort_values(["category", "average_score"], ascending=[True, False]).itertuples(index=False)
    ]
    dataset_rows = [
        [row.track, row.category, row.dataset_label, row.model_label, row.metric, f"{row.score_pct:.1f}%", "" if pd.isna(row.total) else int(row.total)]
        for row in results[results["status"].eq("valid")].sort_values(["track", "category", "dataset_label", "score"], ascending=[True, True, True, False]).itertuples(index=False)
    ]
    invalid_rows = [
        [row.track, row.dataset_label, row.model_label, row.status, int(row.errors) if not pd.isna(row.errors) else "", row.note]
        for row in results[results["status"].ne("valid")].itertuples(index=False)
    ]
    missing_rows = [
        [row.track, row.dataset_label, row.model_label]
        for row in missing_runs.itertuples(index=False)
    ]

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Arabic NLP Evaluation Results</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #1d2733;
      --muted: #607083;
      --line: #d8dee6;
      --panel: #f7f9fb;
      --accent: #276fbf;
    }}
    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      color: var(--ink);
      background: #ffffff;
      line-height: 1.5;
    }}
    main {{
      width: min(1180px, calc(100% - 40px));
      margin: 32px auto 56px;
    }}
    header {{
      border-bottom: 2px solid var(--line);
      padding-bottom: 18px;
      margin-bottom: 26px;
    }}
    h1, h2 {{
      margin: 0 0 12px;
      line-height: 1.15;
    }}
    h1 {{ font-size: 32px; }}
    h2 {{ font-size: 22px; margin-top: 34px; }}
    p.note {{
      color: var(--muted);
      max-width: 860px;
    }}
    .visual-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(330px, 1fr));
      gap: 20px;
      align-items: start;
    }}
    figure {{
      margin: 0;
      padding: 12px;
      border: 1px solid var(--line);
      background: var(--panel);
    }}
    img {{
      width: 100%;
      height: auto;
      display: block;
      background: white;
    }}
    figcaption {{
      color: var(--muted);
      font-size: 13px;
      margin-top: 8px;
    }}
    .table-wrap {{
      overflow-x: auto;
      border: 1px solid var(--line);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
      background: white;
    }}
    th, td {{
      padding: 9px 10px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
      white-space: nowrap;
    }}
    th {{
      background: #edf3f8;
      color: #203040;
      font-weight: 700;
    }}
    tr:nth-child(even) td {{ background: #fbfcfd; }}
  </style>
</head>
<body>
<main>
  <header>
    <h1>Arabic NLP Evaluation Results</h1>
    <p class="note">Compiled from JSON result files in the local results folders. Scores are averaged only across categories where a model has valid completed results, so per-track and per-category tables are the fairest comparison surfaces.</p>
  </header>

  <h2>Overall Leaderboard</h2>
  <div class="table-wrap">{html_table(["Rank", "Model", "Average", "Categories"], leaderboard_rows)}</div>

  <h2>Visuals</h2>
  <div class="visual-grid">{chart_html}</div>

  <h2>Category Scores</h2>
  <div class="table-wrap">{html_table(["Category", "Model", "Average", "Datasets"], category_rows)}</div>

  <h2>Dataset-Level Results</h2>
  <div class="table-wrap">{html_table(["Track", "Category", "Dataset", "Model", "Metric", "Score", "Total"], dataset_rows)}</div>

  <h2>Runs Needing Attention</h2>
  <div class="table-wrap">{html_table(["Track", "Dataset", "Model", "Status", "Errors", "Note"], invalid_rows)}</div>

  <h2>Missing Result Files</h2>
  <div class="table-wrap">{html_table(["Track", "Dataset", "Model"], missing_rows)}</div>
</main>
</body>
</html>
"""
    path = RESULTS_DIR / "compiled_report.html"
    path.write_text(html, encoding="utf-8")
    return path


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    VISUALS_DIR.mkdir(exist_ok=True)

    results = load_results()
    if results.empty:
        raise SystemExit("No JSON results found.")

    category_scores = build_category_scores(results)
    leaderboard = build_leaderboard(category_scores)
    missing_runs = find_missing_runs(results)

    results.to_csv(RESULTS_DIR / "compiled_results.csv", index=False, encoding="utf-8")
    category_scores.to_csv(RESULTS_DIR / "compiled_category_scores.csv", index=False, encoding="utf-8")
    leaderboard.to_csv(RESULTS_DIR / "compiled_leaderboard.csv", index=False, encoding="utf-8")
    missing_runs.to_csv(RESULTS_DIR / "compiled_missing_runs.csv", index=False, encoding="utf-8")

    charts = [
        save_overall_chart(leaderboard),
        save_heatmap(category_scores),
        save_dataset_grouped_chart(results, "Track A", "track_a_dataset_scores.png", "Track A Dataset Scores"),
        save_dataset_grouped_chart(results, "Track B", "track_b_dataset_scores.png", "Track B Dataset Scores"),
        save_error_chart(results),
    ]
    charts = [chart for chart in charts if chart is not None]

    markdown_path = write_markdown_report(results, category_scores, leaderboard, charts, missing_runs)
    html_path = write_html_report(results, category_scores, leaderboard, charts, missing_runs)

    print(f"Read {len(results)} result files.")
    print(f"Wrote {RESULTS_DIR / 'compiled_results.csv'}")
    print(f"Wrote {RESULTS_DIR / 'compiled_category_scores.csv'}")
    print(f"Wrote {RESULTS_DIR / 'compiled_leaderboard.csv'}")
    print(f"Wrote {RESULTS_DIR / 'compiled_missing_runs.csv'}")
    print(f"Wrote {markdown_path}")
    print(f"Wrote {html_path}")
    print(f"Wrote {len(charts)} chart(s) to {VISUALS_DIR}")


if __name__ == "__main__":
    main()
