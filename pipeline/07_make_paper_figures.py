"""
Regenerate the Track B figures of the paper from audit/track_b_audit.csv.

    python 07_make_paper_figures.py --audit ../results_audit/track_b_audit.csv --out ../paper/figures

fig_trackb_status.(pdf|png)    replaces Fig. 2 (grid: reported score + audit status in every cell)
fig_trackb_vs_baseline.(pdf|png) replaces Fig. 3 (reported accuracy minus best constant-letter baseline)
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#d9d8d3", "#ffffff"
# status roles (dataviz reference status palette) + neutral for "no result"
ROLE = {
    "ABOVE_BASELINE": ("V", "#0ca30c", "Valid: above constant-letter baseline"),
    "DEGENERATE_OUTPUT": ("K", "#fab219", "Constant / degenerate predictor"),
    "CONSTANT_PREDICTOR": ("K", "#fab219", "Constant / degenerate predictor"),
    "DEGENERATE_NEAR_ZERO": ("K", "#fab219", "Constant / degenerate predictor"),
    "PROMPT_DEFECT": ("P", "#ec835a", "Prompt defect (options/stem not shown)"),
    "ZERO_ANOMALY_GOLD_LOOKUP": ("G", "#d03b3b", "Gold-label defect"),
    "GOLD_MISSING": ("G", "#d03b3b", "Gold-label defect"),
    "NO_RESULT": ("–", "#e8e7e2", "Run did not complete"),
}
MODEL_ORDER = ["ALLaM-7B", "Falcon-7B", "Jais-13B", "Qwen-7B"]
DS_ORDER = ["Arabic MMLU", "EXAMS AR", "Madinah QA", "ArabCulture", "AraDiCE", "AraTrust"]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})


def status_grid(a: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(7.0, 2.35))
    for r, m in enumerate(MODEL_ORDER):
        for c, d in enumerate(DS_ORDER):
            row = a[(a.model == m) & (a.dataset == d)].iloc[0]
            code, color, _ = ROLE[row.status]
            ax.add_patch(plt.Rectangle((c + 0.03, r + 0.05), 0.94, 0.9, facecolor=color, edgecolor=SURF, lw=2,
                                       alpha=0.9 if code != "–" else 1))
            score = "n/a" if pd.isna(row.reported_acc_pct) else f"{row.reported_acc_pct:.1f}"
            ink = "#ffffff" if code in ("G", "V") else INK
            ax.text(c + 0.5, r + 0.42, score, ha="center", va="center", fontsize=9, fontweight="bold", color=ink)
            ax.text(c + 0.5, r + 0.75, code, ha="center", va="center", fontsize=7, color=ink)
    ax.set_xlim(0, len(DS_ORDER)); ax.set_ylim(len(MODEL_ORDER), 0)
    ax.set_xticks([i + 0.5 for i in range(len(DS_ORDER))], DS_ORDER)
    ax.set_yticks([i + 0.5 for i in range(len(MODEL_ORDER))], MODEL_ORDER)
    ax.tick_params(length=0); ax.xaxis.tick_top()
    for s in ax.spines.values():
        s.set_visible(False)
    seen, handles = set(), []
    for code, color, label in ROLE.values():
        if label not in seen:
            seen.add(label)
            handles.append(Patch(facecolor=color, edgecolor=GRID, label=f"{code} = {label}"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=False, fontsize=6.8,
              handlelength=1.2, columnspacing=1.2)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_trackb_status.{ext}", dpi=300, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)


def vs_baseline(a: pd.DataFrame, out: Path):
    # the cells the submitted paper treated as "parseable, genuine" scores (non-zero, gold resolvable)
    s = a[a.reported_acc_pct.notna() & (a.reported_acc_pct > 0) & (a.status != "GOLD_MISSING")].copy()
    s["delta"] = s.reported_acc_pct - s.best_constant_pct
    s["label"] = s.model + " · " + s.dataset
    s = s.sort_values("delta")
    fig, ax = plt.subplots(figsize=(3.5, 3.1))
    colors = ["#2a78d6" if v > 0 else "#eb6834" for v in s.delta]
    ax.barh(range(len(s)), s.delta, color=colors, height=0.62, edgecolor=SURF, linewidth=0.5)
    ax.axvline(0, color=INK2, lw=0.8)
    for i, (d, acc, base, st) in enumerate(zip(s.delta, s.reported_acc_pct, s.best_constant_pct, s.status)):
        tag = {"PROMPT_DEFECT": " (P)"}.get(st, "")
        txt = f"{acc:.1f} vs {base:.1f}{tag}"
        ax.text(d + (0.8 if d > 0 else -0.8), i, txt, va="center", ha="left" if d > 0 else "right", fontsize=6,
                color=INK2)
    ax.set_yticks(range(len(s)), s.label, fontsize=6.3)
    ax.set_xlabel("Accuracy minus best constant-letter\nbaseline (percentage points)")
    ax.set_xlim(min(s.delta) - 22, max(s.delta) + 16)
    ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_trackb_vs_baseline.{ext}", dpi=300, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    a = pd.read_csv(args.audit)
    a["reported_acc_pct"] = 100 * a["correct"] / a["total"]  # avoid double rounding (53.445 -> 53.4)
    status_grid(a, out)
    vs_baseline(a, out)
    print("written to", out)
