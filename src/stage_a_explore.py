"""
STAGE A -- Data Exploration
===========================
Goal: understand the ASAP essay dataset BEFORE we build anything, exactly as
described in Section 2 (Methodology, step 1 "Dataset Description") of the
Stage-1 report.

What this script does:
  1. Loads the raw TSV with the correct ISO-8859-1 encoding.
  2. Reports the number of essays per essay set.
  3. Reports the score range (min / max / mean) of domain1_score per set.
  4. Reports the essay-length distribution (in words) per set and overall.
  5. Saves three exploration plots to figures/ so they can go in the report.

Run:  python src/stage_a_explore.py
"""

# --- standard library ---
import os

# --- third-party ---
import pandas as pd
import numpy as np
import matplotlib

# Use a non-interactive backend so the script saves PNGs without needing a
# display / popping up windows (important when running from a plain terminal).
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- our project config (paths, seed, score ranges) ---
from config import (
    TSV_PATH,
    TSV_ENCODING,
    FIGURES_DIR,
    SCORE_RANGES,
    set_global_seed,
    ensure_dirs,
)


def load_raw_data() -> pd.DataFrame:
    """
    Read the ASAP TSV into a DataFrame.

    We only keep the three columns this project actually needs:
      - essay_set     : which of the 8 prompts (1-8) the essay answers
      - essay         : the raw essay text
      - domain1_score : the resolved human score we want to predict (the target)
    """
    # sep="\t" because the file is TAB-separated; encoding is Latin-1 (not utf-8).
    df = pd.read_csv(TSV_PATH, sep="\t", encoding=TSV_ENCODING)

    # Keep only the columns we care about and drop any row with a missing score
    # (the target must be present for supervised learning).
    df = df[["essay_id", "essay_set", "essay", "domain1_score"]].copy()
    df = df.dropna(subset=["domain1_score"]).reset_index(drop=True)

    # A quick derived column: essay length measured in words (whitespace split).
    # This is only used for exploration/plots here; Stage B does real tokenising.
    df["word_count"] = df["essay"].astype(str).str.split().apply(len)
    return df


def report_per_set(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build and print a per-essay-set summary table:
    count of essays, observed score min/max/mean, and word-count statistics.
    Returns the summary DataFrame so it can also be saved to disk.
    """
    # Group by essay_set and aggregate several statistics in one pass.
    summary = df.groupby("essay_set").agg(
        n_essays=("essay_id", "count"),
        score_min=("domain1_score", "min"),
        score_max=("domain1_score", "max"),
        score_mean=("domain1_score", "mean"),
        words_min=("word_count", "min"),
        words_max=("word_count", "max"),
        words_mean=("word_count", "mean"),
    )

    # Attach the OFFICIAL score range from config so we can eyeball whether the
    # observed data stays within the documented bounds.
    summary["official_range"] = summary.index.map(
        lambda s: f"{SCORE_RANGES[s][0]}-{SCORE_RANGES[s][1]}"
    )

    # Round the float columns for a clean, readable printout.
    summary["score_mean"] = summary["score_mean"].round(2)
    summary["words_mean"] = summary["words_mean"].round(0).astype(int)

    print("\n================ PER-ESSAY-SET SUMMARY ================")
    print(summary.to_string())
    print("======================================================")
    return summary


def plot_essays_per_set(df: pd.DataFrame) -> str:
    """Bar chart: how many essays are in each of the 8 sets."""
    counts = df["essay_set"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(counts.index.astype(str), counts.values, color="#4C72B0")
    ax.set_xlabel("Essay Set")
    ax.set_ylabel("Number of Essays")
    ax.set_title("Essays per Essay Set (ASAP dataset)")
    # Annotate each bar with its count so the numbers are readable in the report.
    for x, y in zip(counts.index.astype(str), counts.values):
        ax.text(x, y + 5, str(y), ha="center", va="bottom", fontsize=9)

    out = os.path.join(FIGURES_DIR, "A_essays_per_set.png")
    fig.tight_layout()
    fig.savefig(out, dpi=300)  # 300 dpi = print quality for the report
    plt.close(fig)
    return out


def plot_score_ranges(df: pd.DataFrame) -> str:
    """
    Box plot of domain1_score per set.  Because different sets use different
    scales (e.g. set 2 is 1-6 while set 8 is 0-60), this makes the scale
    differences -- and why we must normalise in Stage B -- visually obvious.
    """
    sets = sorted(df["essay_set"].unique())
    # Collect the score arrays, one per set, in order.
    data = [df.loc[df["essay_set"] == s, "domain1_score"].values for s in sets]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot(data, tick_labels=[str(s) for s in sets], showfliers=True)
    ax.set_xlabel("Essay Set")
    ax.set_ylabel("domain1_score (original scale)")
    ax.set_title("Score Distribution per Essay Set — note the differing scales")

    out = os.path.join(FIGURES_DIR, "A_score_ranges_per_set.png")
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def plot_length_distribution(df: pd.DataFrame) -> str:
    """
    Histogram of essay length (in words) across the whole dataset, plus a
    vertical line at the 95th percentile.  This directly informs the fixed
    padding/truncation length we choose in Stage B: pick a length that covers
    most essays without wasting memory on a few very long outliers.
    """
    lengths = df["word_count"].values
    p95 = np.percentile(lengths, 95)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(lengths, bins=60, color="#55A868", edgecolor="white")
    ax.axvline(
        p95,
        color="red",
        linestyle="--",
        label=f"95th percentile = {p95:.0f} words",
    )
    ax.set_xlabel("Essay Length (words)")
    ax.set_ylabel("Number of Essays")
    ax.set_title("Essay-Length Distribution (all sets)")
    ax.legend()

    out = os.path.join(FIGURES_DIR, "A_length_distribution.png")
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out, p95


def main() -> None:
    # Reproducibility + make sure figures/ exists.
    set_global_seed()
    ensure_dirs()

    print("Loading ASAP dataset from:", TSV_PATH)
    df = load_raw_data()
    print(f"Loaded {len(df):,} essays across {df['essay_set'].nunique()} sets.")

    # ---- textual reports ----
    report_per_set(df)

    # ---- plots (saved to figures/) ----
    f1 = plot_essays_per_set(df)
    f2 = plot_score_ranges(df)
    f3, p95 = plot_length_distribution(df)

    print("\nSaved figures:")
    for f in (f1, f2, f3):
        print("  -", f)

    # A concrete recommendation for Stage B's MAX_SEQUENCE_LENGTH, derived from
    # the data rather than guessed.  We round the 95th percentile up to a tidy
    # number so most essays are kept whole and only long outliers get truncated.
    suggested_maxlen = int(np.ceil(p95 / 50.0) * 50)
    print(
        f"\nSuggested MAX_SEQUENCE_LENGTH for Stage B "
        f"(95th pct rounded up): {suggested_maxlen} tokens"
    )


if __name__ == "__main__":
    main()
