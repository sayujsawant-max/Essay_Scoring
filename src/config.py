"""
config.py
---------
Central place for all project-wide constants and paths, so that every stage
(A-E) reads the same settings. Keeping these in one file means we only change a
value once and every script picks it up -- this is important for
reproducibility (e.g. the random seed) and for a viva where you can point to a
single source of truth.
"""

import os
import random

import numpy as np

# --------------------------------------------------------------------------- #
# Paths.  We compute them relative to THIS file so the project runs no matter
# what folder you launch Python from.
# --------------------------------------------------------------------------- #
# Directory that contains this config.py  ->  .../aes-microproject/src
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
# Project root is one level up  ->  .../aes-microproject
ROOT_DIR = os.path.dirname(SRC_DIR)

DATA_DIR = os.path.join(ROOT_DIR, "data")
FIGURES_DIR = os.path.join(ROOT_DIR, "figures")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")

# Input data files (as supplied by the user).
TSV_PATH = os.path.join(DATA_DIR, "training_set_rel3.tsv")
GLOVE_PATH = os.path.join(DATA_DIR, "glove.6B.50d.txt")

# The ASAP file is Latin-1 encoded, NOT utf-8 -- reading it as utf-8 crashes on
# special characters, so we always pass this encoding to pandas.
TSV_ENCODING = "ISO-8859-1"

# --------------------------------------------------------------------------- #
# Reproducibility: one global seed used everywhere (numpy, python random and,
# in later stages, TensorFlow).  Setting this makes every run produce the same
# splits, the same shuffles and the same trained weights.
# --------------------------------------------------------------------------- #
RANDOM_SEED = 42

# GloVe embedding dimension (our file is the 50-dimensional version).
EMBEDDING_DIM = 50

# --------------------------------------------------------------------------- #
# Preprocessing hyper-parameters (used by Stage B and consumed by Stage D).
# --------------------------------------------------------------------------- #
# Fixed sequence length every essay is padded / truncated to before it enters
# the embedding layer.  Derived from Stage A: the 95th-percentile essay length
# over ALL sets is ~600 words, so 600 keeps the vast majority of essays whole.
#
#   *** REVISIT-WHEN-SCALING WARNING ***
#   Stage D trains on essay SET 1 first (avg 366 words), where 600 is generous.
#   But essay SET 8 averages ~605 words, so a length of 600 would TRUNCATE
#   roughly half of set-8 essays.  If/when you expand training beyond set 1,
#   reconsider this value per-set (or raise it) so set-8 content is not lost.
MAX_SEQUENCE_LENGTH = 600

# Keep only the most frequent MAX_NUM_WORDS tokens in the vocabulary. Rare words
# add embedding rows that are barely trained; capping the vocab keeps the model
# small and generalisable.  None => keep every word seen in the TRAIN split.
MAX_NUM_WORDS = 20000

# Train / validation / test proportions. They must sum to 1.0.
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

# --------------------------------------------------------------------------- #
# BiLSTM model + training hyper-parameters (Stage D). Values follow the Stage-1
# report: Adam, MSE loss, batch size 32, 15-20 epochs with early stopping.
# --------------------------------------------------------------------------- #
LSTM_UNITS = 64            # hidden units PER DIRECTION in the BiLSTM layer
DENSE_UNITS = 32           # units in the dense layer before the output
DROPOUT_RATE = 0.3         # tuned on set 1: 0.5 under-fit; 0.3 gave the best/most
                           # stable val QWK (0.4 was noisier & stopped too early on
                           # this tiny 268-essay val split)
BATCH_SIZE = 32            # per the report
EPOCHS = 30                # upper bound; early stopping usually ends sooner
EARLY_STOP_PATIENCE = 6    # epochs with no val-loss improvement before stopping

# Fine-tune (unfreeze) the GloVe embedding. Diagnosis on set 1 showed the model
# UNDER-fits (train and val loss stay close), so the extra capacity helps rather
# than hurts: the ~27% of vocab words with no GloVe vector start random and can
# only become useful if the embedding is allowed to learn, and the pretrained
# vectors get adapted to the essay-scoring task (as in Taghipour & Ng 2016).
# Early stopping + restore_best_weights still guards against overfitting.
EMBEDDING_TRAINABLE = True

# --------------------------------------------------------------------------- #
# Official ASAP score range for each of the 8 essay sets, taken from the
# competition description.  domain1_score for every essay in a set lies within
# [low, high].  Stage B uses these bounds for min-max normalisation to 0-1, and
# Stage C/E use them to rescale predictions back to the ORIGINAL integer range
# before computing Quadratic Weighted Kappa (QWK).
#
# Format:  essay_set -> (min_score, max_score)
# --------------------------------------------------------------------------- #
SCORE_RANGES = {
    1: (2, 12),
    2: (1, 6),
    3: (0, 3),
    4: (0, 3),
    5: (0, 4),
    6: (0, 4),
    7: (0, 30),
    8: (0, 60),
}


def set_global_seed(seed: int = RANDOM_SEED) -> None:
    """
    Seed every random-number generator we rely on so results are repeatable.

    Called at the top of each stage.  TensorFlow is seeded lazily (only if it is
    installed / imported) so that Stages A-C, which do not need TF, still run in
    an environment where TF is absent.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)  # stable hashing of Python objects
    random.seed(seed)                         # Python's built-in RNG
    np.random.seed(seed)                      # NumPy RNG (used by pandas/sklearn)
    try:
        import tensorflow as tf               # optional import
        tf.random.set_seed(seed)              # TensorFlow/Keras RNG
    except ImportError:
        pass  # TF not installed yet -- fine for the non-DL stages


def ensure_dirs() -> None:
    """Create the output folders if they do not already exist (idempotent)."""
    for d in (FIGURES_DIR, MODELS_DIR, OUTPUTS_DIR):
        os.makedirs(d, exist_ok=True)
