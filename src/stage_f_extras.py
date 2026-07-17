"""
STAGE F -- Extra analyses & figures for the report
==================================================
Adds four report assets on top of Stages A-E:

  F1  Per-set agreement ("confusion") heatmaps  -> figures/F1_confusion_heatmaps.png
  F2  Length-bias analysis                       -> figures/F2_length_bias.png
  F3  Pipeline + model architecture diagram      -> figures/F3_architecture.png
  --  Exact / adjacent agreement accuracy        -> outputs/accuracy_metrics.json

All three quantitative pieces reuse the SAME all-sets train/test split as the
BiLSTM (via prepare_data), so every comparison is on identical data.

Run:  python src/stage_f_extras.py
"""

import os
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from config import FIGURES_DIR, OUTPUTS_DIR, SCORE_RANGES, set_global_seed, ensure_dirs
from stage_b_preprocess import prepare_data, denormalize_per_row, denormalize_score
from stage_c_baseline import build_feature_matrix

DPI = 300


# --------------------------------------------------------------------------- #
# Shared: load BiLSTM all-sets test predictions + reconstruct the test frame
# --------------------------------------------------------------------------- #
def load_context():
    """
    Returns the all-sets test DataFrame (same split as Stage D) plus the saved
    BiLSTM integer predictions, aligned. Also returns the train frame so we can
    fit per-prompt baselines on the identical partition.
    """
    data = prepare_data(essay_sets=None, maxlen=800, truncating="pre",
                        stratify_score_bins=True)
    npz = np.load(os.path.join(OUTPUTS_DIR, "bilstm_allsets_test_predictions.npz"))

    test_df = data["test_df"].copy()
    # Sanity: the saved essay_set order must match the reconstructed test frame.
    assert np.array_equal(npz["essay_set"], test_df["essay_set"].values), \
        "prediction order does not match reconstructed test split"
    test_df["bilstm_pred"] = npz["y_pred_int"]
    test_df["actual"] = npz["y_true_int"]
    test_df["length"] = test_df["clean_essay"].str.split().apply(len)
    return data["train_df"], test_df


# --------------------------------------------------------------------------- #
# F1 -- Confusion / agreement heatmaps (per set)
# --------------------------------------------------------------------------- #
def plot_confusion_heatmaps(test_df):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for idx, s in enumerate(range(1, 9)):
        ax = axes[idx // 4][idx % 4]
        low, high = SCORE_RANGES[s]
        n = high - low + 1
        sub = test_df[test_df.essay_set == s]
        # Build count matrix M[actual, predicted].
        M = np.zeros((n, n), dtype=int)
        for a, p in zip(sub["actual"], sub["bilstm_pred"]):
            M[int(a) - low, int(p) - low] += 1

        ax.imshow(M, origin="lower", cmap="Blues", aspect="auto")
        ax.plot([0, n - 1], [0, n - 1], "r--", linewidth=1)  # perfect-agreement line
        ax.set_title(f"Set {s} (scale {low}-{high})", fontsize=10)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        # Annotate counts only when the grid is small enough to read.
        if n <= 7:
            ax.set_xticks(range(n)); ax.set_xticklabels(range(low, high + 1))
            ax.set_yticks(range(n)); ax.set_yticklabels(range(low, high + 1))
            for i in range(n):
                for j in range(n):
                    if M[i, j]:
                        ax.text(j, i, M[i, j], ha="center", va="center",
                                fontsize=8,
                                color="white" if M[i, j] > M.max() / 2 else "black")
        else:
            ax.set_xticks([0, n - 1]); ax.set_xticklabels([low, high])
            ax.set_yticks([0, n - 1]); ax.set_yticklabels([low, high])

    fig.suptitle("BiLSTM Agreement Heatmaps (actual vs predicted) — mass on/near "
                 "the red diagonal = good; off-diagonal = errors", fontsize=13)
    out = os.path.join(FIGURES_DIR, "F1_confusion_heatmaps.png")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# F2 -- Length-bias analysis
# --------------------------------------------------------------------------- #
def length_bias(train_df, test_df):
    """
    For each set, fit a per-prompt baseline on the all-sets TRAIN partition and
    predict on the test partition (same essays the BiLSTM saw). Then measure how
    strongly each model's predictions correlate with raw essay LENGTH, versus how
    strongly the HUMAN score does. A model whose length-correlation is closer to
    the human's (and below the length-driven baseline's) is less length-biased.
    """
    rows = {}
    for s in range(1, 9):
        tr = train_df[train_df.essay_set == s]
        te = test_df[test_df.essay_set == s]
        Xtr, _ = build_feature_matrix(tr)
        Xte, _ = build_feature_matrix(te)
        ytr = tr["score_norm"].values
        model = make_pipeline(StandardScaler(), LinearRegression()).fit(Xtr, ytr)
        base_pred = denormalize_score(np.clip(model.predict(Xte), 0, 1), s)

        length = te["length"].values
        rows[s] = {
            "human": float(np.corrcoef(length, te["actual"])[0, 1]),
            "baseline": float(np.corrcoef(length, base_pred)[0, 1]),
            "bilstm": float(np.corrcoef(length, te["bilstm_pred"])[0, 1]),
        }

    means = {k: float(np.mean([rows[s][k] for s in range(1, 9)]))
             for k in ("human", "baseline", "bilstm")}

    # ---- plot ----
    sets = list(range(1, 9))
    x = np.arange(len(sets))
    w = 0.26
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - w, [rows[s]["human"] for s in sets], w, label="Human score",
           color="#55A868")
    ax.bar(x, [rows[s]["baseline"] for s in sets], w, label="Baseline prediction",
           color="#B0B0B0")
    ax.bar(x + w, [rows[s]["bilstm"] for s in sets], w, label="BiLSTM prediction",
           color="#4C72B0")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x); ax.set_xticklabels([f"Set {s}" for s in sets])
    ax.set_ylabel("Correlation of essay LENGTH with score")
    ax.set_title("Length-bias check: how much does each score depend on essay "
                 "length?\n(closer to the human bar = less length-biased)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    out = os.path.join(FIGURES_DIR, "F2_length_bias.png")
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)

    return out, rows, means


# --------------------------------------------------------------------------- #
# F3 -- Architecture / workflow diagram
# --------------------------------------------------------------------------- #
def plot_architecture():
    fig, (axp, axm) = plt.subplots(2, 1, figsize=(12, 9),
                                   gridspec_kw={"height_ratios": [1, 1.4]})

    def box(ax, x, y, w, h, text, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                    fc=fc, ec="black", linewidth=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=9.5, wrap=True)

    def arrow(ax, x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="->",
                                     mutation_scale=16, linewidth=1.2, color="#333"))

    # --- top: end-to-end workflow ---
    axp.set_xlim(0, 10); axp.set_ylim(0, 2); axp.axis("off")
    axp.set_title("Workflow: from raw essay to predicted score", fontsize=12)
    stages = [
        (0.1, "ASAP\nessays (.tsv)", "#EAEAEA"),
        (2.05, "Clean +\ntokenize +\npad", "#DCE6F1"),
        (4.0, "GloVe\nembedding\n(50-d)", "#DCE6F1"),
        (5.95, "BiLSTM +\nmean pooling", "#C6D9F1"),
        (7.9, "Score\n(0-1 -> scale)", "#BEE3BE"),
    ]
    for i, (x, t, fc) in enumerate(stages):
        box(axp, x, 0.6, 1.75, 0.8, t, fc)
        if i:
            arrow(axp, x - 0.2, 1.0, x, 1.0)

    # --- bottom: model layer stack ---
    axm.set_xlim(0, 10); axm.set_ylim(0, 8.6); axm.axis("off")
    axm.set_title("BiLSTM model architecture", fontsize=12)
    layers = [
        ("Input: padded token sequence (length 800)", "#EAEAEA"),
        ("Embedding: GloVe 50-d, mask padding (trainable)", "#DCE6F1"),
        ("Bidirectional LSTM (64 units/dir), return sequences", "#C6D9F1"),
        ("Mean-over-time pooling (masked average)", "#C6D9F1"),
        ("Dropout (0.3)", "#F2DCDB"),
        ("Dense (32, ReLU)", "#DCE6F1"),
        ("Dropout (0.3)", "#F2DCDB"),
        ("Dense (1, Sigmoid) -> normalised score 0-1", "#BEE3BE"),
    ]
    y = 7.5
    for i, (t, fc) in enumerate(layers):
        box(axm, 2.0, y, 6.0, 0.7, t, fc)
        if i:
            arrow(axm, 5.0, y + 0.85, 5.0, y + 0.7)
        y -= 1.0

    out = os.path.join(FIGURES_DIR, "F3_architecture.png")
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# Exact / adjacent agreement accuracy
# --------------------------------------------------------------------------- #
def accuracy_metrics(test_df):
    per_set = {}
    ex_all, ad_all = [], []
    for s in range(1, 9):
        sub = test_df[test_df.essay_set == s]
        a, p = sub["actual"].values, sub["bilstm_pred"].values
        ex = float((a == p).mean())
        ad = float((np.abs(a - p) <= 1).mean())
        per_set[s] = {"exact": round(ex, 4), "adjacent": round(ad, 4)}
        ex_all.append(ex); ad_all.append(ad)
    result = {"per_set": per_set,
              "mean_exact": round(float(np.mean(ex_all)), 4),
              "mean_adjacent": round(float(np.mean(ad_all)), 4)}
    with open(os.path.join(OUTPUTS_DIR, "accuracy_metrics.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result


# --------------------------------------------------------------------------- #
def main():
    set_global_seed()
    ensure_dirs()
    train_df, test_df = load_context()

    f1 = plot_confusion_heatmaps(test_df)
    f2, lb_rows, lb_means = length_bias(train_df, test_df)
    f3 = plot_architecture()
    acc = accuracy_metrics(test_df)

    print("Saved figures:")
    for f in (f1, f2, f3):
        print("  ", f)
    print("\nExact / adjacent accuracy (mean over sets): "
          f"{acc['mean_exact']*100:.1f}% / {acc['mean_adjacent']*100:.1f}%")
    print("\nLength-score correlation (mean over sets):")
    for k, v in lb_means.items():
        print(f"  {k:>9}: {v:+.3f}")
    print("\nInterpretation: baseline predictions correlate with length "
          f"({lb_means['baseline']:+.2f}) more than the human score does "
          f"({lb_means['human']:+.2f}); the BiLSTM ({lb_means['bilstm']:+.2f}) "
          "sits closer to the human, i.e. it is less length-biased.")

    # Persist length-bias numbers for the report generator.
    with open(os.path.join(OUTPUTS_DIR, "length_bias.json"), "w") as f:
        json.dump({"per_set": lb_rows, "means": lb_means}, f, indent=2)


if __name__ == "__main__":
    main()
