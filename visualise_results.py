import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns

# custom colors and plot style setup
PALETTE   = {"fake": "#e05252", "real": "#52a0e0", "uncertain": "#f0c040"}
BG        = "#f8f9fb"
GRID_CLR  = "#e0e4ea"
sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor":   BG,
    "axes.edgecolor":   "#cccccc",
    "grid.color":       GRID_CLR,
    "axes.spines.top":  False,
    "axes.spines.right": False,
})

def parse_log(log_path: str) -> dict:
    # function to read the text log and extract accuracy and metrics
    text = Path(log_path).read_text(encoding="utf-8")

    stats = {}

    # parse labeled dataset distribution
    m = re.search(r"Label distribution — fake:\s*([\d,]+)\s+real:\s*([\d,]+)", text)
    if m:
        stats["labeled_fake"] = int(m.group(1).replace(",", ""))
        stats["labeled_real"] = int(m.group(2).replace(",", ""))

    m = re.search(r"Training on\s*([\d,]+)\s+samples", text)
    if m:
        stats["train_size"] = int(m.group(1).replace(",", ""))

    # parse validation scores
    m = re.search(r"ROC-AUC\s*:\s*([0-9.]+)", text)
    if m:
        stats["val_auc"] = float(m.group(1))

    m = re.search(r"accuracy\s+([\d.]+)", text)
    if m:
        stats["val_accuracy"] = float(m.group(1))

    # grab precision, recall, and f1 for real and fake classes
    for cls in ["real", "fake"]:
        pattern = rf"{cls}\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)"
        m = re.search(pattern, text)
        if m:
            stats[f"{cls}_precision"] = float(m.group(1))
            stats[f"{cls}_recall"]    = float(m.group(2))
            stats[f"{cls}_f1"]        = float(m.group(3))
            stats[f"{cls}_support"]   = int(m.group(4))

    # parse pseudo label statistics
    m = re.search(r"Pseudo-labels accepted:\s*([\d,]+)", text)
    if m:
        stats["pseudo_accepted"] = int(m.group(1).replace(",", ""))

    m = re.search(r"Combined training size:\s*([\d,]+)", text)
    if m:
        stats["combined_train_size"] = int(m.group(1).replace(",", ""))

    # parse predictions from foundational model block
    found_block = re.search(
        r"\[Foundational\] Prediction Distribution.*?Std dev\s*:\s*([0-9.]+)",
        text, re.DOTALL
    )
    if found_block:
        block = found_block.group(0)
        for lbl in ["fake", "uncertain", "real"]:
            m = re.search(rf"{lbl}\s+([\d,]+)\s+([\d.]+)%", block)
            if m:
                stats[f"base_{lbl}_count"] = int(m.group(1).replace(",", ""))
                stats[f"base_{lbl}_pct"]   = float(m.group(2))
        m = re.search(r"Avg fake probability\s*:\s*([0-9.]+)", block)
        if m: stats["base_avg_prob"] = float(m.group(1))
        m = re.search(r"Median fake prob\s*:\s*([0-9.]+)", block)
        if m: stats["base_median_prob"] = float(m.group(1))
        m = re.search(r"Std dev\s*:\s*([0-9.]+)", block)
        if m: stats["base_std"] = float(m.group(1))

    # parse predictions from adapted model block
    adapt_block = re.search(
        r"\[Adapted\] Prediction Distribution.*?Std dev\s*:\s*([0-9.]+)",
        text, re.DOTALL
    )
    if adapt_block:
        block = adapt_block.group(0)
        for lbl in ["fake", "uncertain", "real"]:
            m = re.search(rf"{lbl}\s+([\d,]+)\s+([\d.]+)%", block)
            if m:
                stats[f"adapted_{lbl}_count"] = int(m.group(1).replace(",", ""))
                stats[f"adapted_{lbl}_pct"]   = float(m.group(2))
        m = re.search(r"Avg fake probability\s*:\s*([0-9.]+)", block)
        if m: stats["adapted_avg_prob"] = float(m.group(1))
        m = re.search(r"Median fake prob\s*:\s*([0-9.]+)", block)
        if m: stats["adapted_median_prob"] = float(m.group(1))
        m = re.search(r"Std dev\s*:\s*([0-9.]+)", block)
        if m: stats["adapted_std"] = float(m.group(1))

    # parse agreement metrics between models
    m = re.search(r"Agree\s*:\s*([\d,]+)\s+\(([\d.]+)%\)", text)
    if m:
        stats["agree_count"] = int(m.group(1).replace(",", ""))
        stats["agree_pct"]   = float(m.group(2))

    m = re.search(r"Disagree\s*:\s*([\d,]+)\s+\(([\d.]+)%\)", text)
    if m:
        stats["disagree_count"] = int(m.group(1).replace(",", ""))
        stats["disagree_pct"]   = float(m.group(2))

    m = re.search(r"Adaptation shifted → fake\s*:\s*([\d,]+)", text)
    if m: stats["shifted_to_fake"] = int(m.group(1).replace(",", ""))

    m = re.search(r"Adaptation shifted → real\s*:\s*([\d,]+)", text)
    if m: stats["shifted_to_real"] = int(m.group(1).replace(",", ""))

    m = re.search(r"Avg probability shift \(adapted − base\):\s*([+-]?[0-9.]+)", text)
    if m: stats["avg_prob_shift"] = float(m.group(1))

    # parse score bucket distributions
    bucket_labels = [
        "0.0–0.2 (strong real)", "0.2–0.4 (lean real)",
        "0.4–0.6 (uncertain)",   "0.6–0.8 (lean fake)",
        "0.8–1.0 (strong fake)"
    ]
    bucket_counts = []
    bucket_pcts   = []
    for bl in bucket_labels:
        escaped = re.escape(bl)
        m = re.search(rf"{escaped}\s+([\d,]+)\s+([\d.]+)%", text)
        if m:
            bucket_counts.append(int(m.group(1).replace(",", "")))
            bucket_pcts.append(float(m.group(2)))
        else:
            bucket_counts.append(0)
            bucket_pcts.append(0.0)
    stats["bucket_labels"] = bucket_labels
    stats["bucket_counts"] = bucket_counts
    stats["bucket_pcts"]   = bucket_pcts

    return stats

def _save(fig, path: Path, name: str):
    # helper function to save chart and close figure
    out = path / name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✔  {out}")

def chart_labeled_distribution(s: dict, out: Path):
    # plot the number of real vs fake labeled samples
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor(BG)

    vals   = [s.get("labeled_real", 0), s.get("labeled_fake", 0)]
    labels = ["Real", "Fake"]
    colors = [PALETTE["real"], PALETTE["fake"]]
    bars   = ax.bar(labels, vals, color=colors, width=0.5, edgecolor="white", linewidth=1.2)

    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                f"{v:,}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    total = sum(vals)
    ax.set_title("Stage C — Labeled Training Set Composition", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Number of Reviews")
    ax.set_ylim(0, max(vals) * 1.15)
    ax.text(0.98, 0.97, f"Total: {total:,}", transform=ax.transAxes,
            ha="right", va="top", fontsize=10, color="#666")
    _save(fig, out, "01_labeled_distribution.png")

def chart_validation_metrics(s: dict, out: Path):
    # plot a bar chart comparing scores between classes
    metrics = ["Precision", "Recall", "F1-Score"]
    real_vals = [s.get("real_precision",0), s.get("real_recall",0), s.get("real_f1",0)]
    fake_vals = [s.get("fake_precision",0), s.get("fake_recall",0), s.get("fake_f1",0)]

    x    = np.arange(len(metrics))
    w    = 0.32
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(BG)

    b1 = ax.bar(x - w/2, real_vals, w, label="Real", color=PALETTE["real"],  edgecolor="white")
    b2 = ax.bar(x + w/2, fake_vals, w, label="Fake", color=PALETTE["fake"],  edgecolor="white")

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=9)

    auc = s.get("val_auc", 0)
    acc = s.get("val_accuracy", 0)
    ax.set_title(f"Stage C — Validation Metrics   (AUC {auc:.4f} · Accuracy {acc:.0%})",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.legend(framealpha=0.6)
    _save(fig, out, "02_validation_metrics.png")

def chart_domain_distribution_comparison(s: dict, out: Path):
    # compare base model and adapted model label distributions
    cats   = ["Fake", "Uncertain", "Real"]
    base   = [s.get(f"base_{c.lower()}_pct",   0) for c in cats]
    adapted= [s.get(f"adapted_{c.lower()}_pct", 0) for c in cats]
    colors = [PALETTE["fake"], PALETTE["uncertain"], PALETTE["real"]]

    x = np.arange(len(cats))
    w = 0.32
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(BG)

    b1 = ax.bar(x - w/2, base,    w, label="Foundational (Stage C)", color=colors, alpha=0.60, edgecolor="white")
    b2 = ax.bar(x + w/2, adapted, w, label="Adapted (Stage D)",      color=colors, alpha=1.00, edgecolor="white")

    for bars, vals in [(b1, base), (b2, adapted)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
                    f"{v:.1f}%", ha="center", va="bottom", fontsize=9)

    ax.set_title("Domain Data — Prediction Distribution: Base vs Adapted", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylabel("% of Domain Reviews")
    ax.set_ylim(0, max(max(base), max(adapted)) * 1.18)
    ax.legend(framealpha=0.6)
    _save(fig, out, "03_domain_distribution_comparison.png")

def chart_confidence_buckets(s: dict, out: Path):
    # plot predictions across different confidence buckets
    short_labels = ["Strong\nReal\n(0.0–0.2)", "Lean\nReal\n(0.2–0.4)",
                    "Uncertain\n(0.4–0.6)",     "Lean\nFake\n(0.6–0.8)",
                    "Strong\nFake\n(0.8–1.0)"]
    counts = s.get("bucket_counts", [0]*5)
    pcts   = s.get("bucket_pcts",   [0.0]*5)
    bucket_colors = [PALETTE["real"], "#7ec8e3", PALETTE["uncertain"],
                     "#f0a070", PALETTE["fake"]]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(BG)
    bars = ax.bar(short_labels, counts, color=bucket_colors, edgecolor="white", linewidth=1.2)

    for bar, c, p in zip(bars, counts, pcts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f"{c:,}\n({p:.1f}%)", ha="center", va="bottom", fontsize=9)

    ax.set_title("Adapted Model — Confidence Bucket Distribution", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Number of Reviews")
    ax.set_ylim(0, max(counts) * 1.20)
    _save(fig, out, "04_confidence_buckets.png")

def chart_model_agreement(s: dict, out: Path):
    # plot a pie chart showing model agreement percentage
    agree    = s.get("agree_pct",    90)
    disagree = s.get("disagree_pct", 10)

    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_facecolor(BG)
    wedges, texts, autotexts = ax.pie(
        [agree, disagree],
        labels=["Agree", "Disagree"],
        colors=["#52c07a", "#e07070"],
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops=dict(edgecolor="white", linewidth=2),
    )
    for at in autotexts:
        at.set_fontsize(12)
        at.set_fontweight("bold")

    shifted_fake = s.get("shifted_to_fake", 0)
    shifted_real = s.get("shifted_to_real", 0)
    shift        = s.get("avg_prob_shift",  0)
    ax.set_title("Base vs Adapted — Model Agreement", fontsize=13, fontweight="bold", pad=14)
    fig.text(0.5, 0.03,
             f"Shifted → fake: {shifted_fake}   Shifted → real: {shifted_real}   "
             f"Avg prob shift: {shift:+.4f}",
             ha="center", fontsize=9, color="#555")
    _save(fig, out, "05_model_agreement.png")

def chart_probability_histogram(df: pd.DataFrame, out: Path):
    # plot histograms of probabilities from predictions file
    if "base_fake_probability" not in df.columns or "adapted_fake_probability" not in df.columns:
        print("  ⚠  Skipping histogram — predictions CSV missing probability columns.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    fig.patch.set_facecolor(BG)
    fig.suptitle("Fake Probability Distributions — Domain Reviews", fontsize=13, fontweight="bold", y=1.01)

    for ax, col, title, color in [
        (axes[0], "base_fake_probability",    "Foundational Model (Stage C)", "#7aabda"),
        (axes[1], "adapted_fake_probability", "Adapted Model (Stage D)",      "#da7a7a"),
    ]:
        ax.hist(df[col].dropna(), bins=40, color=color, edgecolor="white",
                linewidth=0.6, alpha=0.85)
        ax.axvline(0.35, color="#555", linestyle="--", linewidth=1, label="Real threshold (0.35)")
        ax.axvline(0.65, color="#222", linestyle="--", linewidth=1, label="Fake threshold (0.65)")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Fake Probability")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8, framealpha=0.6)
        mean_val = df[col].mean()
        ax.axvline(mean_val, color="orange", linestyle="-", linewidth=1.5, label=f"Mean {mean_val:.3f}")
        ax.legend(fontsize=8, framealpha=0.6)

    plt.tight_layout()
    _save(fig, out, "06_probability_histogram.png")

def chart_probability_scatter(df: pd.DataFrame, out: Path):
    # scatter plot for base vs adapted probabilities
    if "base_fake_probability" not in df.columns or "adapted_fake_probability" not in df.columns:
        print("  ⚠  Skipping scatter — predictions CSV missing probability columns.")
        return

    color_map = {"fake": PALETTE["fake"], "real": PALETTE["real"],
                 "uncertain": PALETTE["uncertain"]}
    pred_col = "adapted_prediction" if "adapted_prediction" in df.columns else None

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor(BG)

    if pred_col:
        for lbl, grp in df.groupby(pred_col):
            ax.scatter(grp["base_fake_probability"], grp["adapted_fake_probability"],
                       s=8, alpha=0.4, color=color_map.get(lbl, "#aaa"), label=lbl.capitalize())
        ax.legend(title="Adapted Prediction", framealpha=0.7, markerscale=2)
    else:
        ax.scatter(df["base_fake_probability"], df["adapted_fake_probability"],
                   s=8, alpha=0.4, color="#7aabda")

    lo, hi = 0, 1
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, alpha=0.5, label="y = x (no change)")
    ax.axhline(0.65, color=PALETTE["fake"], linewidth=0.7, linestyle=":", alpha=0.6)
    ax.axhline(0.35, color=PALETTE["real"], linewidth=0.7, linestyle=":", alpha=0.6)
    ax.axvline(0.65, color=PALETTE["fake"], linewidth=0.7, linestyle=":", alpha=0.6)
    ax.axvline(0.35, color=PALETTE["real"], linewidth=0.7, linestyle=":", alpha=0.6)

    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Foundational Model — Fake Probability")
    ax.set_ylabel("Adapted Model — Fake Probability")
    ax.set_title("Probability Shift: Base → Adapted", fontsize=13, fontweight="bold", pad=12)
    _save(fig, out, "07_probability_scatter.png")

def chart_summary_dashboard(s: dict, out: Path):
    # build a large dashboard plot combining multiple charts
    fig = plt.figure(figsize=(14, 8))
    fig.patch.set_facecolor(BG)
    fig.suptitle("Pipeline Summary Dashboard", fontsize=16, fontweight="bold", y=1.01)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.axis("off")
    kpis = [
        ("ROC-AUC",   f"{s.get('val_auc',0):.4f}",      "#52a0e0"),
        ("Accuracy",  f"{s.get('val_accuracy',0):.1%}",  "#52c07a"),
        ("Fake F1",   f"{s.get('fake_f1',0):.2f}",       "#e05252"),
        ("Real F1",   f"{s.get('real_f1',0):.2f}",       "#52a0e0"),
        ("Agreement", f"{s.get('agree_pct',0):.1f}%",    "#888888"),
    ]
    for i, (label, val, col) in enumerate(kpis):
        y = 0.95 - i * 0.19
        ax0.text(0.05, y, label, transform=ax0.transAxes, fontsize=10, color="#555", va="top")
        ax0.text(0.95, y, val,   transform=ax0.transAxes, fontsize=14, color=col,
                 va="top", ha="right", fontweight="bold")
        line_y = y - 0.04
        ax0.plot([0.05, 0.95], [line_y, line_y], color=GRID_CLR, linewidth=0.8,
                 transform=ax0.transAxes, clip_on=False)
    ax0.set_title("Key Metrics", fontsize=11, fontweight="bold")

    ax1 = fig.add_subplot(gs[0, 1])
    metrics = ["Precision", "Recall", "F1"]
    rv = [s.get("real_precision",0), s.get("real_recall",0), s.get("real_f1",0)]
    fv = [s.get("fake_precision",0), s.get("fake_recall",0), s.get("fake_f1",0)]
    x = np.arange(3); w = 0.35
    ax1.bar(x-w/2, rv, w, color=PALETTE["real"], label="Real",  edgecolor="white")
    ax1.bar(x+w/2, fv, w, color=PALETTE["fake"], label="Fake",  edgecolor="white")
    ax1.set_xticks(x); ax1.set_xticklabels(metrics, fontsize=9)
    ax1.set_ylim(0, 1.1); ax1.set_title("Validation Metrics", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=8, framealpha=0.6)

    ax2 = fig.add_subplot(gs[0, 2])
    lr = s.get("labeled_real", 0); lf = s.get("labeled_fake", 0)
    ax2.pie([lr, lf], labels=["Real", "Fake"],
            colors=[PALETTE["real"], PALETTE["fake"]],
            autopct="%1.1f%%", startangle=90,
            wedgeprops=dict(edgecolor="white", linewidth=1.5))
    ax2.set_title("Training Set Balance", fontsize=11, fontweight="bold")

    ax3 = fig.add_subplot(gs[1, 0:2])
    cats    = ["Fake", "Uncertain", "Real"]
    base_v  = [s.get(f"base_{c.lower()}_pct",    0) for c in cats]
    adapt_v = [s.get(f"adapted_{c.lower()}_pct", 0) for c in cats]
    colors  = [PALETTE["fake"], PALETTE["uncertain"], PALETTE["real"]]
    x = np.arange(3); w = 0.32
    ax3.bar(x-w/2, base_v,  w, color=colors, alpha=0.55, edgecolor="white", label="Base")
    ax3.bar(x+w/2, adapt_v, w, color=colors, alpha=1.00, edgecolor="white", label="Adapted")
    ax3.set_xticks(x); ax3.set_xticklabels(cats)
    ax3.set_ylabel("% of domain reviews")
    ax3.set_title("Domain Prediction Distribution", fontsize=11, fontweight="bold")
    ax3.legend(fontsize=8, framealpha=0.6)

    ax4 = fig.add_subplot(gs[1, 2])
    short = ["SR\n0.0–0.2", "LR\n0.2–0.4", "UNC\n0.4–0.6", "LF\n0.6–0.8", "SF\n0.8–1.0"]
    counts = s.get("bucket_counts", [0]*5)
    bcolors = [PALETTE["real"], "#7ec8e3", PALETTE["uncertain"], "#f0a070", PALETTE["fake"]]
    ax4.bar(short, counts, color=bcolors, edgecolor="white")
    ax4.set_title("Confidence Buckets\n(Adapted)", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Count")

    plt.tight_layout()
    _save(fig, out, "00_summary_dashboard.png")

def main():
    parser = argparse.ArgumentParser(description="Visualise pipeline results")
    parser.add_argument("--log",    required=True,
                        help="Path to saved pipeline terminal log  (pipeline_run.log)")
    parser.add_argument("--output", required=True,
                        help="Pipeline output directory containing the predictions CSV")
    parser.add_argument("--charts", default="charts/",
                        help="Where to save chart PNGs  (default: charts/)")
    args = parser.parse_args()

    chart_dir = Path(args.chart_dir if hasattr(args, "chart_dir") else args.charts)
    chart_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nParsing log: {args.log}")
    s = parse_log(args.log)

    if not s:
        print("ERROR: Could not parse any metrics from the log file.")
        sys.exit(1)

    print(f"  Extracted {len(s)} metrics from log\n")

    output_dir = Path(args.output)
    pred_csvs  = list(output_dir.glob("*_predictions.csv"))
    df_pred    = pd.DataFrame()

    if pred_csvs:
        pred_csv = pred_csvs[0]
        print(f"Reading predictions: {pred_csv}")
        df_pred = pd.read_csv(pred_csv)
        if "fake_probability" in df_pred.columns and "base_fake_probability" not in df_pred.columns:
            df_pred["base_fake_probability"]    = df_pred["fake_probability"]
            df_pred["adapted_fake_probability"] = df_pred["fake_probability"]
        if "prediction" in df_pred.columns and "adapted_prediction" not in df_pred.columns:
            df_pred["adapted_prediction"] = df_pred["prediction"]
            df_pred["base_prediction"]    = df_pred["prediction"]
        print(f"  {len(df_pred):,} rows, columns: {df_pred.columns.tolist()}\n")
    else:
        print("No *_predictions.csv found in output dir — skipping CSV-based charts.\n")

    print(f"Generating charts → {chart_dir}/\n")

    chart_summary_dashboard(s, chart_dir)
    chart_labeled_distribution(s, chart_dir)
    chart_validation_metrics(s, chart_dir)
    chart_domain_distribution_comparison(s, chart_dir)
    chart_confidence_buckets(s, chart_dir)
    chart_model_agreement(s, chart_dir)

    if not df_pred.empty:
        chart_probability_histogram(df_pred, chart_dir)
        chart_probability_scatter(df_pred, chart_dir)

    print(f"\nDone — {len(list(chart_dir.glob('*.png')))} charts saved to {chart_dir}/")

if __name__ == "__main__":
    main()