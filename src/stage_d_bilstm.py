"""
STAGE D -- BiLSTM Deep Model
============================
The deep-learning model promised in Section 2 step 3 of the Stage-1 report:

    Embedding (GloVe-initialised)
        -> Bidirectional LSTM
        -> Dropout
        -> Dense (regression head, sigmoid output in 0-1)

Trained with Adam + MSE loss, batch size 32, early stopping on validation loss.
We train on ESSAY SET 1 FIRST so the whole pipeline can be validated quickly.

Design choices made to fight overfitting (only ~1,247 training essays):
  * The GloVe embedding is FROZEN (config.EMBEDDING_TRAINABLE = False).
  * Heavy dropout (0.5).
  * Early stopping with restore_best_weights=True -- we keep the weights from the
    BEST validation epoch, not wherever training happened to stop.

It reuses Stage B's prepare_data() so the train/val/test split is BYTE-IDENTICAL
to Stage C's (same RANDOM_SEED, same function) -- otherwise the baseline vs
BiLSTM comparison would not be on the same data.

Run:  python src/stage_d_bilstm.py
"""

import os
import json

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, Bidirectional, LSTM, Dropout, Dense,
    GlobalAveragePooling1D,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from config import (
    GLOVE_PATH,
    EMBEDDING_DIM,
    MAX_NUM_WORDS,
    MAX_SEQUENCE_LENGTH,
    LSTM_UNITS,
    DENSE_UNITS,
    DROPOUT_RATE,
    BATCH_SIZE,
    EPOCHS,
    EARLY_STOP_PATIENCE,
    EMBEDDING_TRAINABLE,
    MODELS_DIR,
    OUTPUTS_DIR,
    RANDOM_SEED,
    set_global_seed,
    ensure_dirs,
)
from stage_b_preprocess import prepare_data
from metrics import evaluate_per_set


# --------------------------------------------------------------------------- #
# 1. BUILD THE GLOVE EMBEDDING MATRIX
# --------------------------------------------------------------------------- #
def build_embedding_matrix(tokenizer):
    """
    Create the (vocab_size x 50) weight matrix for the Embedding layer.

    We STREAM the GloVe file line-by-line (it is 842 MB / 1.29M words) and only
    keep vectors for words that are actually in OUR vocabulary -- building the
    full dict in memory would be wasteful. Words in our vocab that have no GloVe
    vector keep their small random init (this is the expected fate of the
    anonymisation placeholders and any rare/typo tokens).

    Returns (embedding_matrix, coverage_stats).
    """
    word_index = tokenizer.word_index  # word -> integer index (1-based)

    # Effective vocab size: index 0 is reserved for padding, and we cap at
    # MAX_NUM_WORDS most-frequent words (matching the tokenizer's num_words).
    vocab_size = min(MAX_NUM_WORDS, len(word_index) + 1)

    # Random small init so out-of-GloVe words start from a sensible place rather
    # than zeros. Seeded for reproducibility. Row 0 (padding) is zeroed below.
    rng = np.random.RandomState(RANDOM_SEED)
    embedding_matrix = rng.normal(
        scale=0.6, size=(vocab_size, EMBEDDING_DIM)
    ).astype("float32")
    embedding_matrix[0] = 0.0  # padding index -> all zeros

    # Which of our words we still need a vector for (index within vocab_size).
    wanted = {w: i for w, i in word_index.items() if i < vocab_size}
    found_words = set()

    # Stream GloVe once.
    with open(GLOVE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            # Each line: "word f1 f2 ... f50"
            parts = line.rstrip().split(" ")
            word = parts[0]
            if word in wanted:
                vec = np.asarray(parts[1:1 + EMBEDDING_DIM], dtype="float32")
                embedding_matrix[wanted[word]] = vec
                found_words.add(word)

    # --- coverage statistics (for the report) ---
    n_vocab = len(wanted)
    n_found = len(found_words)
    coverage = {
        "vocab_size_used": vocab_size,
        "words_in_vocab": n_vocab,
        "words_with_glove_vector": n_found,
        "coverage_fraction": round(n_found / max(1, n_vocab), 4),
        "words_random_init": n_vocab - n_found,
    }

    # Specifically check the Stage-B anonymisation placeholders -- the user
    # flagged that they "definitely won't be in GloVe". Because clean_text()
    # collapses @PERSON1 -> "person" (a real English word), most of these DO get
    # a real vector; we report exactly which did and which fell back to random.
    placeholders = ["person", "location", "organization", "caps",
                    "date", "time", "money", "percent", "num"]
    placeholder_cov = {
        p: (p in found_words) for p in placeholders if p in wanted
    }
    coverage["placeholder_glove_hit"] = placeholder_cov

    return embedding_matrix, coverage


# --------------------------------------------------------------------------- #
# 2. BUILD THE MODEL
# --------------------------------------------------------------------------- #
def build_model(embedding_matrix, maxlen=MAX_SEQUENCE_LENGTH):
    """
    Assemble the BiLSTM regressor with the Keras functional API.

    Layer by layer:
      Input           : integer sequence of length MAX_SEQUENCE_LENGTH
      Embedding       : maps each word id to its 50-d GloVe vector. mask_zero=True
                        so padding positions are ignored by the LSTM. Frozen.
      Bidirectional   : an LSTM read forwards AND backwards, returning its output
        (LSTM)          at EVERY timestep (return_sequences=True), so each word
                        position gets a context-aware vector.
      MeanOverTime    : average those per-word vectors across the essay
        (GlobalAvgPool)  (padding positions are masked out, so only real words
                        count). This "mean-over-time" pooling is the AES
                        architecture of Taghipour & Ng (2016) [report ref #7];
                        it gives the dense head a summary of the WHOLE essay
                        instead of just the final LSTM state, which is what a
                        plain last-state BiLSTM discards and why that variant
                        barely beat a mean-predictor.
      Dropout         : randomly zero 50% of activations -> regularisation.
      Dense (relu)    : a small hidden layer to combine the pooled features.
      Dropout         : more regularisation.
      Dense (sigmoid) : single output squashed to 0-1, matching the normalised
                        target. MSE loss then makes this a regression.
    """
    vocab_size, emb_dim = embedding_matrix.shape

    # --- input: one padded essay ---
    inp = Input(shape=(maxlen,), dtype="int32", name="essay_tokens")

    # --- embedding: GloVe weights (trainable per config) ---
    x = Embedding(
        input_dim=vocab_size,
        output_dim=emb_dim,
        weights=[embedding_matrix],
        mask_zero=True,               # ignore padding positions
        trainable=EMBEDDING_TRAINABLE,
        name="glove_embedding",
    )(inp)

    # --- one bidirectional LSTM layer, returning the full sequence ---
    x = Bidirectional(
        LSTM(LSTM_UNITS, return_sequences=True),  # output at every timestep
        name="bilstm",
    )(x)

    # --- mean-over-time pooling (mask-aware: ignores padding steps) ---
    x = GlobalAveragePooling1D(name="mean_over_time")(x)

    # --- regularisation + regression head ---
    x = Dropout(DROPOUT_RATE, name="dropout_1")(x)
    x = Dense(DENSE_UNITS, activation="relu", name="dense_hidden")(x)
    x = Dropout(DROPOUT_RATE, name="dropout_2")(x)
    out = Dense(1, activation="sigmoid", name="score_0_1")(x)

    model = Model(inputs=inp, outputs=out, name="BiLSTM_AES")

    # Adam + MSE (regression), track MAE as a readable secondary metric.
    model.compile(optimizer=Adam(learning_rate=1e-3), loss="mse", metrics=["mae"])
    return model


# --------------------------------------------------------------------------- #
# 3. TRAIN + EVALUATE
# --------------------------------------------------------------------------- #
def train_bilstm(
    essay_sets=None,
    maxlen=MAX_SEQUENCE_LENGTH,
    truncating="post",
    stratify_score_bins=False,
    tag="set1",
):
    """
    Train + evaluate the BiLSTM.

    tag                 : label for the saved artefacts (e.g. "set1", "allsets")
                          so different runs never overwrite each other.
    maxlen/truncating/stratify_score_bins : passed straight to prepare_data so
                          the combined 8-set run can use a longer length, keep
                          conclusions, and stratify by score bin.
    """
    set_global_seed()   # seeds python/numpy/TF for a reproducible run
    ensure_dirs()

    # ---- reuse Stage B pipeline => SAME split as Stage C ----
    data = prepare_data(
        essay_sets=essay_sets,
        maxlen=maxlen,
        truncating=truncating,
        stratify_score_bins=stratify_score_bins,
    )
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]
    tokenizer = data["tokenizer"]

    print(f"Train/val/test sizes: {len(X_train)}/{len(X_val)}/{len(X_test)}")
    print(f"Steps per epoch (batch {BATCH_SIZE}): "
          f"{int(np.ceil(len(X_train) / BATCH_SIZE))}")

    # ---- GloVe embedding matrix + coverage report ----
    print("\nBuilding GloVe embedding matrix (streaming the file)...")
    embedding_matrix, coverage = build_embedding_matrix(tokenizer)
    print(f"  Vocabulary words: {coverage['words_in_vocab']}")
    print(f"  Got a GloVe vector: {coverage['words_with_glove_vector']} "
          f"({coverage['coverage_fraction']*100:.1f}%)")
    print(f"  Random-init (no GloVe): {coverage['words_random_init']}")
    print(f"  Anonymisation placeholders found in GloVe: "
          f"{coverage['placeholder_glove_hit']}")

    # ---- build + summarise the model ----
    print("\nBuilding model...")
    model = build_model(embedding_matrix, maxlen=maxlen)
    model.summary()

    # ---- callbacks ----
    # EarlyStopping with restore_best_weights=True: stop when val_loss stops
    # improving for PATIENCE epochs AND roll the weights back to the best epoch.
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOP_PATIENCE,
        restore_best_weights=True,
        verbose=1,
    )
    # Also checkpoint the best model to disk as a safety net.
    ckpt_path = os.path.join(MODELS_DIR, f"bilstm_{tag}_best.keras")
    checkpoint = ModelCheckpoint(
        ckpt_path, monitor="val_loss", save_best_only=True, verbose=0
    )

    # ---- train ----
    print("\nTraining...\n")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop, checkpoint],
        verbose=2,
    )

    # ---- evaluate on the held-out TEST split ----
    y_pred_test = model.predict(X_test, verbose=0).ravel()
    es = data["test_df"]["essay_set"].values
    # Overall AND per-set metrics (per-set is the honest view across 8 prompts).
    report = evaluate_per_set(
        y_true_norm=y_test, y_pred_norm=y_pred_test, essay_sets=es
    )
    overall = report["overall"]

    print("\n---------------- BiLSTM RESULTS (test set) ----------------")
    print(f"  OVERALL  RMSE={overall['rmse']:.4f}  "
          f"MAE={overall['mae']:.4f}  QWK={overall['qwk']:.4f}")
    if len(report["per_set"]) > 1:
        print("  Per-set QWK (honest breakdown):")
        print(f"    {'set':>4} {'n':>5} {'RMSE':>7} {'MAE':>7} {'QWK':>7}")
        for s, m in report["per_set"].items():
            n = int((es == s).sum())
            print(f"    {s:>4} {n:>5} {m['rmse']:>7.3f} "
                  f"{m['mae']:>7.3f} {m['qwk']:>7.3f}")
    print("-----------------------------------------------------------")

    # ---- persist everything Stage E needs (so it never has to retrain) ----
    # 1. metrics (overall + per-set)
    with open(os.path.join(OUTPUTS_DIR, f"bilstm_{tag}_metrics.json"), "w") as f:
        json.dump({"model": f"bilstm_{tag}",
                   "essay_sets": essay_sets or "all",
                   "maxlen": maxlen,
                   "truncating": truncating,
                   "metrics": overall,
                   "per_set": report["per_set"],
                   "glove_coverage": coverage}, f, indent=2)
    # 2. training history (loss curves)
    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(os.path.join(OUTPUTS_DIR, f"bilstm_{tag}_history.json"), "w") as f:
        json.dump(hist, f, indent=2)
    # 3. test predictions vs actuals (denormalised integers) for the scatter plot
    from stage_b_preprocess import denormalize_per_row
    np.savez(
        os.path.join(OUTPUTS_DIR, f"bilstm_{tag}_test_predictions.npz"),
        y_true_int=denormalize_per_row(y_test, es, round_to_int=True),
        y_pred_int=denormalize_per_row(y_pred_test, es, round_to_int=True),
        essay_set=es,
    )

    print(f"\nSaved model -> {ckpt_path}")
    print(f"Saved metrics/history/predictions ('{tag}') -> {OUTPUTS_DIR}")
    return model, report, history


if __name__ == "__main__":
    import sys

    # Two run modes so the set-1 artefacts are never clobbered:
    #   python stage_d_bilstm.py            -> combined 8-set run (the (B) upgrade)
    #   python stage_d_bilstm.py set1       -> re-run the set-1 baseline-within-baseline
    if len(sys.argv) > 1 and sys.argv[1] == "set1":
        train_bilstm(essay_sets=[1], maxlen=600, truncating="post",
                     stratify_score_bins=False, tag="set1")
    else:
        # Combined run: all 8 sets, longer maxlen with conclusion-preserving
        # truncation, and score-bin stratification. ~9,000 training essays.
        train_bilstm(essay_sets=None, maxlen=800, truncating="pre",
                     stratify_score_bins=True, tag="allsets")
