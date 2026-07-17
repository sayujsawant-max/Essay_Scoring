"""
demo_infer.py
-------------
Inference helpers shared by the demo app. Loads the trained all-sets BiLSTM and
its tokenizer once, and turns raw essay text into a predicted score on the
chosen prompt's scale.

Nothing here needs the 800 MB GloVe file or the raw dataset: the embedding is
already baked into the saved model, and the tokenizer vocabulary is loaded from
models/tokenizer_allsets.json.
"""

import os
import json

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json

from config import MODELS_DIR, SCORE_RANGES
from stage_b_preprocess import clean_text, texts_to_padded, denormalize_score

# The all-sets model was trained with these settings (Stage D combined run).
MAXLEN = 800
TRUNCATING = "pre"

MODEL_PATH = os.path.join(MODELS_DIR, "bilstm_allsets_best.keras")
TOKENIZER_PATH = os.path.join(MODELS_DIR, "tokenizer_allsets.json")
EXAMPLES_PATH = os.path.join(MODELS_DIR, "demo_examples.json")


def load_artifacts():
    """Load and return (model, tokenizer). Raises a clear error if missing."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. Run "
            "`python src/stage_d_bilstm.py` to train the all-sets model first."
        )
    if not os.path.exists(TOKENIZER_PATH):
        raise FileNotFoundError(
            f"Tokenizer not found at {TOKENIZER_PATH}. Run the tokenizer-save "
            "step (see README) to regenerate it."
        )
    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "r", encoding="utf-8") as f:
        tokenizer = tokenizer_from_json(f.read())
    return model, tokenizer


def load_examples():
    """Optional bundled example essays with their real human scores."""
    if os.path.exists(EXAMPLES_PATH):
        with open(EXAMPLES_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def predict_score(model, tokenizer, essay_text, essay_set):
    """
    Predict the score of one essay for a given prompt (essay_set 1-8).

    Returns a dict:
        predicted_int : integer score on the prompt's official scale
        low, high     : the prompt's official score range
        normalized    : the raw 0-1 model output (before rescaling)
        n_words       : word count of the cleaned essay
    """
    low, high = SCORE_RANGES[essay_set]

    # Same preprocessing the model was trained with.
    cleaned = clean_text(essay_text)
    n_words = len(cleaned.split())
    X = texts_to_padded(tokenizer, [cleaned], maxlen=MAXLEN, truncating=TRUNCATING)

    # Model outputs a 0-1 value; rescale to the prompt's integer range.
    norm = float(model.predict(X, verbose=0).ravel()[0])
    norm = min(max(norm, 0.0), 1.0)  # clamp to [0,1]
    predicted_int = int(denormalize_score(norm, essay_set, round_to_int=True))

    return {
        "predicted_int": predicted_int,
        "low": low,
        "high": high,
        "normalized": norm,
        "n_words": n_words,
    }
