"""
save_demo_artifacts.py
----------------------
Regenerates the two small files the Streamlit demo needs alongside the trained
model (which is produced by `python src/stage_d_bilstm.py`):

  * models/tokenizer_allsets.json  -- the exact tokenizer the all-sets model was
                                      trained with (regenerated deterministically
                                      via the fixed random seed).
  * models/demo_examples.json      -- a few real ASAP essays with their true human
                                      scores, used as one-click examples.

Run once after training the all-sets model:
    python src/save_demo_artifacts.py
"""

import os
import io
import json

import pandas as pd

from config import TSV_PATH, TSV_ENCODING, MODELS_DIR, ensure_dirs
from stage_b_preprocess import prepare_data


def save_tokenizer():
    # Rebuild the identical tokenizer (same seed, same params as Stage D combined).
    data = prepare_data(
        essay_sets=None, maxlen=800, truncating="pre", stratify_score_bins=True
    )
    out = os.path.join(MODELS_DIR, "tokenizer_allsets.json")
    with io.open(out, "w", encoding="utf-8") as f:
        f.write(data["tokenizer"].to_json())
    print(f"Saved tokenizer ({len(data['tokenizer'].word_index)} words) -> {out}")


def save_examples():
    # Pick a few real essays (one low + a couple high) as demo examples.
    df = pd.read_csv(TSV_PATH, sep="\t", encoding=TSV_ENCODING)
    df = df[["essay_set", "essay", "domain1_score"]]
    picks = []
    for s, which in [(3, "low"), (3, "high"), (6, "high")]:
        sub = df[df.essay_set == s]
        row = (sub.nsmallest(1, "domain1_score").iloc[0] if which == "low"
               else sub.nlargest(50, "domain1_score").iloc[10])
        picks.append({"set": int(s),
                      "score": int(row["domain1_score"]),
                      "text": row["essay"][:900]})
    out = os.path.join(MODELS_DIR, "demo_examples.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(picks, f)
    print(f"Saved {len(picks)} demo examples -> {out}")


if __name__ == "__main__":
    ensure_dirs()
    save_tokenizer()
    save_examples()
