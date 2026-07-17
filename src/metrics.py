"""
metrics.py
----------
Shared scoring helpers so the baseline (Stage C) and the BiLSTM (Stage E) are
evaluated with the EXACT same code -- otherwise a difference in how a metric is
computed could masquerade as a difference in model quality.

The three metrics match the Stage-1 report:
  * RMSE : Root Mean Squared Error  (penalises large errors more)
  * MAE  : Mean Absolute Error      (average error in score points)
  * QWK  : Quadratic Weighted Kappa (the standard AES metric; agreement with
           the human rater, chance-corrected). QWK is ALWAYS computed on the
           denormalised INTEGER scores, never on the 0-1 values.
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, cohen_kappa_score

from stage_b_preprocess import denormalize_per_row


def rmse(y_true, y_pred) -> float:
    """Root Mean Squared Error. Computed on whatever scale is passed in."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true, y_pred) -> float:
    """Mean Absolute Error."""
    return float(mean_absolute_error(y_true, y_pred))


def quadratic_weighted_kappa(y_true_int, y_pred_int) -> float:
    """
    Quadratic Weighted Kappa between two arrays of INTEGER scores.
    weights='quadratic' makes disagreements cost more the further apart they are,
    which is exactly what AES cares about (predicting 11 when the truth is 12 is
    far better than predicting 2).
    """
    return float(
        cohen_kappa_score(y_true_int, y_pred_int, weights="quadratic")
    )


def evaluate_all(y_true_norm, y_pred_norm, essay_sets):
    """
    Compute all three metrics for a set of predictions.

    Parameters
    ----------
    y_true_norm : normalised (0-1) ground-truth scores
    y_pred_norm : normalised (0-1) model predictions
    essay_sets  : the essay_set of each row (so we denormalise with the right
                  official range per essay set)

    RMSE and MAE are reported on the ORIGINAL score scale (after denormalising
    but WITHOUT rounding) so the numbers are interpretable as "score points".
    QWK is computed on the ROUNDED integer scores, as the report requires.

    Returns a dict: {"rmse":..., "mae":..., "qwk":...}
    """
    essay_sets = np.asarray(essay_sets)

    # Denormalise to the real scale. Continuous (no rounding) for RMSE/MAE ...
    true_cont = denormalize_per_row(y_true_norm, essay_sets, round_to_int=False)
    pred_cont = denormalize_per_row(y_pred_norm, essay_sets, round_to_int=False)

    # ... and rounded integers for QWK (which needs discrete classes).
    true_int = denormalize_per_row(y_true_norm, essay_sets, round_to_int=True)
    pred_int = denormalize_per_row(y_pred_norm, essay_sets, round_to_int=True)

    return {
        "rmse": rmse(true_cont, pred_cont),
        "mae": mae(true_cont, pred_cont),
        "qwk": quadratic_weighted_kappa(true_int, pred_int),
    }


def evaluate_per_set(y_true_norm, y_pred_norm, essay_sets):
    """
    Same as evaluate_all but ALSO broken down PER essay set.

    A single pooled QWK across 8 differently-scaled prompts can look healthy
    while hiding that the model aces long-essay prompts and flops on short ones.
    Reporting QWK/RMSE/MAE per set is the honest picture for the report.

    Returns:
        {
          "overall": {rmse, mae, qwk},           # pooled across all sets
          "per_set": {1: {...}, 2: {...}, ...},  # one dict per essay set
        }
    Note: per-set QWK is computed within a single set, so denormalisation there
    uses that set's own range -- exactly the scale a human rater used.
    """
    y_true_norm = np.asarray(y_true_norm)
    y_pred_norm = np.asarray(y_pred_norm)
    essay_sets = np.asarray(essay_sets)

    overall = evaluate_all(y_true_norm, y_pred_norm, essay_sets)

    per_set = {}
    for s in sorted(np.unique(essay_sets)):
        mask = essay_sets == s
        per_set[int(s)] = evaluate_all(
            y_true_norm[mask], y_pred_norm[mask], essay_sets[mask]
        )
    return {"overall": overall, "per_set": per_set}
