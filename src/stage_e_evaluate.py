"""
STAGE E -- Evaluation & Report Figures
======================================
Pulls together the baseline (Stage C) and the BiLSTM (Stage D) and produces the
figures + summary tables for the report. Nothing is retrained here except the
per-prompt baselines (fast); the BiLSTM results are loaded from the artefacts
Stage D saved.

The HEADLINE metric is MEAN PER-SET QWK, never the pooled QWK. Pooling 8 prompts
with different score ranges (1-6 vs 0-60) inflates QWK, because cross-set essay
pairs trivially "agree" on being far apart -- the metric ends up partly rewarding
the model for sorting which SET an essay belongs to rather than scoring within a
set. We show that pooled number ONLY as a labelled cautionary annotation.

Figures produced (all 300 dpi, saved to figures/):
  E1  per-set QWK bar chart, baseline vs BiLSTM, with mean lines  (the money chart)
  E2  training/validation loss curves: set-1 run vs all-sets run
  E3  predicted-vs-actual scatter, baseline | BiLSTM, coloured by set
  E4  per-set predicted-vs-actual small multiples (BiLSTM), QWK annotated

Run:  python src/stage_e_evaluate.py
"""

import os
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import FIGURES_DIR, OUTPUTS_DIR, set_global_seed, ensure_dirs
from stage_b_preprocess import denormalize_per_row
from stage_c_baseline import run_baseline
from metrics import quadratic_weighted_kappa

# A colourblind-friendly qualitative palette (one colour per essay set 1-8).
SET_COLORS = plt.get_cmap("tab10")
DPI = 300


# --------------------------------------------------------------------------- #
# DATA GATHERING
# --------------------------------------------------------------------------- #
def load_bilstm_allsets():
    """Load the BiLSTM all-sets metrics + test predictions saved by Stage D."""
    with open(os.path.join(OUTPUTS_DIR, "bilstm_allsets_metrics.json")) as f:
        metrics = json.load(f)
    npz = np.load(os.path.join(OUTPUTS_DIR, "bilstm_allsets_test_predictions.npz"))
    preds = {
        "y_true_int": npz["y_true_int"],
        "y_pred_int": npz["y_pred_int"],
        "essay_set": npz["essay_set"],
    }
    return metrics, preds


def compute_perprompt_baseline():
    """
    Train a SEPARATE baseline per prompt (the strongest, fairest baseline) and
    collect, for each set: its per-set QWK and its denormalised integer
    predictions vs actuals (for the scatter).

    Returns:
        qwk_by_set : {set -> qwk}
        preds      : {"y_true_int", "y_pred_int", "essay_set"} pooled arrays
    """
    qwk_by_set = {}
    true_all, pred_all, set_all = [], [], []

    for s in range(1, 9):
        # quiet run -> just the artefacts we need
        _, art = run_baseline(essay_sets=[s], save_predictions=False, verbose=False)
        es = art["test_df"]["essay_set"].values
        t_int = denormalize_per_row(art["y_test_norm"], es, round_to_int=True)
        p_int = denormalize_per_row(art["y_pred_norm"], es, round_to_int=True)
        qwk_by_set[s] = quadratic_weighted_kappa(t_int, p_int)
        true_all.append(t_int)
        pred_all.append(p_int)
        set_all.append(es)

    preds = {
        "y_true_int": np.concatenate(true_all),
        "y_pred_int": np.concatenate(pred_all),
        "essay_set": np.concatenate(set_all),
    }
    return qwk_by_set, preds


def bilstm_qwk_by_set(bilstm_metrics):
    """Pull the per-set QWK dict out of the saved BiLSTM metrics JSON."""
    return {int(s): m["qwk"] for s, m in bilstm_metrics["per_set"].items()}


# --------------------------------------------------------------------------- #
# E1 -- PER-SET QWK BAR CHART  (the money chart)
# --------------------------------------------------------------------------- #
def plot_qwk_bars(baseline_qwk, bilstm_qwk):
    sets = list(range(1, 9))
    b_vals = [baseline_qwk[s] for s in sets]
    d_vals = [bilstm_qwk[s] for s in sets]
    b_mean = np.mean(b_vals)
    d_mean = np.mean(d_vals)

    x = np.arange(len(sets))
    w = 0.38

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - w / 2, b_vals, w, label="Baseline (per-prompt linear)",
           color="#B0B0B0", edgecolor="black", linewidth=0.5)
    ax.bar(x + w / 2, d_vals, w, label="BiLSTM (single all-sets model)",
           color="#4C72B0", edgecolor="black", linewidth=0.5)

    # Mean lines for each model (the honest headline numbers).
    ax.axhline(b_mean, color="#7A7A7A", linestyle="--", linewidth=1.4,
               label=f"Baseline mean QWK = {b_mean:.3f}")
    ax.axhline(d_mean, color="#2A4D8F", linestyle="--", linewidth=1.4,
               label=f"BiLSTM mean QWK = {d_mean:.3f}")

    # Annotate each bar with its value.
    for xi, v in zip(x - w / 2, b_vals):
        ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    for xi, v in zip(x + w / 2, d_vals):
        ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Set {s}" for s in sets])
    ax.set_ylabel("Quadratic Weighted Kappa (per set)")
    ax.set_ylim(0, 1.0)
    ax.set_title("Per-Set QWK: Baseline vs BiLSTM  (higher = better agreement)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    out = os.path.join(FIGURES_DIR, "E1_qwk_per_set_bars.png")
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# E2 -- LOSS CURVES  (set-1 run vs all-sets run)
# --------------------------------------------------------------------------- #
def plot_loss_curves():
    def load_hist(name):
        p = os.path.join(OUTPUTS_DIR, name)
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f)
        return None

    h_set1 = load_hist("bilstm_set1_history.json")
    h_all = load_hist("bilstm_allsets_history.json")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)

    for ax, hist, title in [
        (axes[0], h_set1, "Set-1 only run (1,247 train essays)"),
        (axes[1], h_all, "All-8-sets run (9,082 train essays)"),
    ]:
        if hist is None:
            ax.set_visible(False)
            continue
        epochs = range(1, len(hist["loss"]) + 1)
        ax.plot(epochs, hist["loss"], color="#4C72B0", label="train loss (MSE)")
        ax.plot(epochs, hist["val_loss"], color="#DD8452", label="val loss (MSE)")
        # Mark the best (lowest) val-loss epoch -- where restore_best_weights kept.
        best = int(np.argmin(hist["val_loss"])) + 1
        ax.axvline(best, color="grey", linestyle=":", linewidth=1,
                   label=f"best epoch = {best}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (MSE, normalised 0-1 target)")
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle("Training vs Validation Loss — more data => healthier convergence",
                 fontsize=13)
    out = os.path.join(FIGURES_DIR, "E2_loss_curves.png")
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# E3 -- PREDICTED vs ACTUAL SCATTER  (pooled, coloured by set)
# --------------------------------------------------------------------------- #
def plot_scatter_pooled(baseline_preds, bilstm_preds):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)

    for ax, preds, title in [
        (axes[0], baseline_preds, "Baseline (per-prompt linear)"),
        (axes[1], bilstm_preds, "BiLSTM (single all-sets model)"),
    ]:
        t = preds["y_true_int"]
        p = preds["y_pred_int"]
        es = preds["essay_set"]
        for s in range(1, 9):
            m = es == s
            ax.scatter(t[m], p[m], s=10, alpha=0.5,
                       color=SET_COLORS((s - 1) % 10), label=f"Set {s}")
        # y = x reference line spanning the full pooled range.
        lo = min(t.min(), p.min())
        hi = max(t.max(), p.max())
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
        ax.set_xlabel("Actual score (original scale)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Predicted score (original scale)")
    axes[1].legend(loc="upper left", fontsize=8, markerscale=1.5, ncol=2)

    fig.suptitle(
        "Predicted vs Actual, pooled & coloured by set — the 8 separated clusters "
        "are exactly why a single pooled QWK is inflated",
        fontsize=12,
    )
    out = os.path.join(FIGURES_DIR, "E3_pred_vs_actual_pooled.png")
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# E4 -- PER-SET SMALL MULTIPLES  (BiLSTM, QWK annotated)
# --------------------------------------------------------------------------- #
def plot_scatter_per_set(bilstm_preds, bilstm_qwk, baseline_qwk):
    t = bilstm_preds["y_true_int"]
    p = bilstm_preds["y_pred_int"]
    es = bilstm_preds["essay_set"]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for idx, s in enumerate(range(1, 9)):
        ax = axes[idx // 4][idx % 4]
        m = es == s
        ax.scatter(t[m], p[m], s=14, alpha=0.5, color=SET_COLORS((s - 1) % 10))
        lo = min(t[m].min(), p[m].min())
        hi = max(t[m].max(), p[m].max())
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
        # Annotate BiLSTM vs baseline QWK; bold the winner.
        d, b = bilstm_qwk[s], baseline_qwk[s]
        winner = "BiLSTM" if d >= b else "Baseline"
        ax.set_title(f"Set {s}  (winner: {winner})", fontsize=10)
        ax.text(0.05, 0.92,
                f"BiLSTM QWK={d:.2f}\nBaseline QWK={b:.2f}",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round", fc="white", alpha=0.7))
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.grid(alpha=0.3)

    fig.suptitle(
        "BiLSTM Predicted vs Actual, per set — content-driven sets (4,6,7) the "
        "BiLSTM wins; length-driven Set 1 the baseline wins",
        fontsize=13,
    )
    out = os.path.join(FIGURES_DIR, "E4_pred_vs_actual_per_set.png")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# SUMMARY TABLE
# --------------------------------------------------------------------------- #
def write_summary(baseline_qwk, bilstm_qwk, bilstm_metrics):
    sets = list(range(1, 9))
    b_mean = float(np.mean([baseline_qwk[s] for s in sets]))
    d_mean = float(np.mean([bilstm_qwk[s] for s in sets]))

    summary = {
        "headline_metric": "mean_per_set_qwk",
        "baseline_perprompt_mean_qwk": round(b_mean, 4),
        "bilstm_allsets_mean_qwk": round(d_mean, 4),
        "pooled_qwk_bilstm_MISLEADING": round(
            bilstm_metrics["metrics"]["qwk"], 4),
        "per_set_qwk": {
            s: {"baseline": round(baseline_qwk[s], 4),
                "bilstm": round(bilstm_qwk[s], 4),
                "winner": "bilstm" if bilstm_qwk[s] >= baseline_qwk[s]
                          else "baseline"}
            for s in sets
        },
    }
    out = os.path.join(OUTPUTS_DIR, "comparison_summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)

    # Pretty console table.
    print("\n================ FINAL COMPARISON (per-set QWK) ================")
    print(f"  {'Set':>4} {'Baseline':>10} {'BiLSTM':>10} {'Winner':>10}")
    for s in sets:
        w = "BiLSTM" if bilstm_qwk[s] >= baseline_qwk[s] else "Baseline"
        print(f"  {s:>4} {baseline_qwk[s]:>10.3f} {bilstm_qwk[s]:>10.3f} {w:>10}")
    print("  " + "-" * 44)
    print(f"  {'MEAN':>4} {b_mean:>10.3f} {d_mean:>10.3f}")
    print("  ---------------------------------------------------------------")
    print(f"  Pooled BiLSTM QWK (MISLEADING, do not headline): "
          f"{bilstm_metrics['metrics']['qwk']:.3f}")
    print("================================================================")
    return out


# --------------------------------------------------------------------------- #
def main():
    set_global_seed()
    ensure_dirs()

    print("Loading BiLSTM all-sets artefacts...")
    bilstm_metrics, bilstm_preds = load_bilstm_allsets()
    bilstm_qwk = bilstm_qwk_by_set(bilstm_metrics)

    print("Training per-prompt baselines for a fair comparison (8 quick fits)...")
    baseline_qwk, baseline_preds = compute_perprompt_baseline()

    print("\nGenerating figures...")
    figs = [
        plot_qwk_bars(baseline_qwk, bilstm_qwk),
        plot_loss_curves(),
        plot_scatter_pooled(baseline_preds, bilstm_preds),
        plot_scatter_per_set(bilstm_preds, bilstm_qwk, baseline_qwk),
    ]
    for f in figs:
        print("  saved:", f)

    write_summary(baseline_qwk, bilstm_qwk, bilstm_metrics)


if __name__ == "__main__":
    main()
