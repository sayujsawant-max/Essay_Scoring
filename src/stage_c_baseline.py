"""
STAGE C -- Baseline Model
=========================
A simple, transparent baseline the deep model must beat, exactly as promised in
the Stage-1 report: a LINEAR REGRESSION on HANDCRAFTED features.

Handcrafted features (per essay):
  1. word_count            -- number of words
  2. sentence_count        -- number of sentences (via . ! ? boundaries)
  3. avg_word_length       -- mean characters per word
  4. unique_word_ratio     -- unique words / total words (type-token ratio)
  5. spelling_error_count  -- alphabetic tokens not found in an English dictionary

Why these? They capture surface fluency (length, vocabulary variety, correctness)
WITHOUT understanding meaning -- which is precisely the point of a baseline: if
the BiLSTM cannot beat "long, varied, correctly-spelled essays score higher",
the deep model is not adding value.

The baseline is trained to predict the NORMALISED 0-1 score (same target as the
BiLSTM) and then denormalised, so both models are compared on identical footing
and QWK is computed on rounded integer scores.

Run:  python src/stage_c_baseline.py
"""

import re
import os
import json

import numpy as np
import pandas as pd
from nltk.corpus import words as nltk_words
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from config import OUTPUTS_DIR, set_global_seed, ensure_dirs
from stage_b_preprocess import load_dataframe, split_dataframe
from metrics import evaluate_all

# --------------------------------------------------------------------------- #
# Build the known-words dictionary ONCE at import time (it is reused for every
# essay). We lowercase it so comparison is case-insensitive. This is a proxy
# spell-checker: any purely-alphabetic token absent from this set is counted as
# a spelling error. It is deliberately simple -- a baseline feature, not a real
# spell-checker.
# --------------------------------------------------------------------------- #
ENGLISH_WORDS = frozenset(w.lower() for w in nltk_words.words())

# Regex helpers compiled once.
_SENTENCE_SPLIT = re.compile(r"[.!?]+")   # sentence boundaries
_ALPHA_TOKEN = re.compile(r"^[a-z]+$")    # purely alphabetic (for spell check)


def extract_features(raw_essay: str) -> dict:
    """
    Compute the 5 handcrafted features from ONE raw essay string.

    Note we use the RAW text (not the Stage-B cleaned text) because sentence
    count needs the original punctuation and spelling errors need the original
    (uncleaned) words.
    """
    text = str(raw_essay)

    # --- word tokens: split on whitespace ---
    words_list = text.split()
    n_words = len(words_list)
    # Guard against empty essays to avoid division by zero.
    if n_words == 0:
        return {
            "word_count": 0,
            "sentence_count": 0,
            "avg_word_length": 0.0,
            "unique_word_ratio": 0.0,
            "spelling_error_count": 0,
        }

    # 1. word_count
    word_count = n_words

    # 2. sentence_count: split on . ! ? and count non-empty fragments (min 1).
    sentences = [s for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    sentence_count = max(1, len(sentences))

    # 3. avg_word_length: mean number of characters per word token.
    avg_word_length = float(np.mean([len(w) for w in words_list]))

    # 4. unique_word_ratio: distinct words / total words (lowercased).
    lowered = [w.lower() for w in words_list]
    unique_word_ratio = len(set(lowered)) / n_words

    # 5. spelling_error_count: alphabetic tokens not in the English dictionary.
    #    We strip surrounding punctuation, skip anonymisation tags (@...) and
    #    non-alphabetic tokens (numbers), then look each up.
    spelling_errors = 0
    for w in lowered:
        w_stripped = re.sub(r"[^a-z]", "", w)  # keep letters only
        if w_stripped and _ALPHA_TOKEN.match(w_stripped):
            if w_stripped not in ENGLISH_WORDS:
                spelling_errors += 1

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_word_length": avg_word_length,
        "unique_word_ratio": unique_word_ratio,
        "spelling_error_count": spelling_errors,
    }


def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Apply extract_features to every essay and return an (n, 5) float array."""
    feats = df["essay"].apply(extract_features).apply(pd.Series)
    # Fixed column order so train/val/test matrices line up.
    cols = [
        "word_count",
        "sentence_count",
        "avg_word_length",
        "unique_word_ratio",
        "spelling_error_count",
    ]
    return feats[cols].values.astype("float32"), cols


def run_baseline(essay_sets=None, save_predictions: bool = True, verbose: bool = True):
    """
    Train + evaluate the baseline. Returns (metrics_dict, artefacts_dict).

    essay_sets=[1] mirrors Stage D's first run so the two models are directly
    comparable on the same data.
    """
    set_global_seed()
    ensure_dirs()

    # Load (with normalised scores) and split identically to the deep model.
    df = load_dataframe(essay_sets=essay_sets)
    train_df, val_df, test_df = split_dataframe(df)

    if verbose:
        print(f"Baseline data: train={len(train_df)}, "
              f"val={len(val_df)}, test={len(test_df)}")

    # Build handcrafted-feature matrices for each split.
    X_train, cols = build_feature_matrix(train_df)
    X_val, _ = build_feature_matrix(val_df)
    X_test, _ = build_feature_matrix(test_df)

    # Targets = normalised 0-1 scores (same target the BiLSTM will learn).
    y_train = train_df["score_norm"].values
    y_test = test_df["score_norm"].values

    # StandardScaler + LinearRegression in one pipeline. Scaling matters because
    # the 5 features live on very different scales (word_count ~ hundreds vs
    # unique_word_ratio ~ 0-1); standardising puts them on equal footing.
    model = make_pipeline(StandardScaler(), LinearRegression())
    model.fit(X_train, y_train)

    # Predict on the held-out TEST split.
    y_pred_test = model.predict(X_test)

    # Evaluate with the shared metrics (denormalises internally; QWK on ints).
    metrics = evaluate_all(
        y_true_norm=y_test,
        y_pred_norm=y_pred_test,
        essay_sets=test_df["essay_set"].values,
    )

    # Show which features the linear model leaned on (interpretability point for
    # the report / viva: does it just reward length?).
    lin = model.named_steps["linearregression"]
    # Cast to plain Python float so the dict is JSON-serialisable.
    coefs = {c: float(round(w, 4)) for c, w in zip(cols, lin.coef_)}

    if verbose:
        print("\n---------------- BASELINE RESULTS (test set) ----------------")
        print(f"  RMSE : {metrics['rmse']:.4f}  (original score points)")
        print(f"  MAE  : {metrics['mae']:.4f}  (original score points)")
        print(f"  QWK  : {metrics['qwk']:.4f}  (agreement with human rater)")
        print("  Feature weights (standardised):")
        for k, v in coefs.items():
            print(f"     {k:>22}: {v:+.4f}")
        print("-------------------------------------------------------------")

    artefacts = {
        "test_df": test_df,
        "y_test_norm": y_test,
        "y_pred_norm": y_pred_test,
        "coefficients": coefs,
    }

    if save_predictions:
        # Persist metrics so Stage E can load and compare side-by-side.
        out = os.path.join(OUTPUTS_DIR, "baseline_metrics.json")
        with open(out, "w") as f:
            json.dump({"model": "baseline_linear_regression",
                       "essay_sets": essay_sets or "all",
                       "metrics": metrics,
                       "coefficients": coefs}, f, indent=2)
        print(f"\nSaved metrics -> {out}")

    return metrics, artefacts


if __name__ == "__main__":
    # Match Stage D: run on essay set 1 first for a quick, comparable pipeline.
    run_baseline(essay_sets=[1])
