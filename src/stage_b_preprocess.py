"""
STAGE B -- Preprocessing (reusable module)
==========================================
This module turns raw ASAP essays into model-ready arrays, following Section 2
step 2 ("Data Preprocessing") of the Stage-1 report:

  * Text cleaning        -> lowercase, strip special characters
  * Tokenization         -> Keras Tokenizer, fit on the TRAIN split only
  * Padding / truncation -> every essay becomes a fixed-length integer sequence
  * Score normalization  -> domain1_score mapped to 0-1 PER ESSAY SET using the
                            OFFICIAL ASAP range (not the observed min/max)
  * Train/val/test split -> stratified by essay_set for a representative split

It is written as a library: Stages C, D and E all `import` these functions.
Running it directly (python stage_b_preprocess.py) executes a self-test that
prints the resulting shapes for essay set 1.

WHY OFFICIAL RANGE, NOT OBSERVED MIN/MAX
----------------------------------------
Min-max scaling on the observed data would assume no essay can ever score below
/ above what happened to appear in this file. That breaks if a test essay is a
genuine 0 or a genuine 30 for set 7. Using the documented ASAP range is the
defensible choice and is what published AES work does. All ranges live in
config.SCORE_RANGES.
"""

# --- standard library ---
import re

# --- third-party ---
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- project config ---
from config import (
    TSV_PATH,
    TSV_ENCODING,
    SCORE_RANGES,
    RANDOM_SEED,
    MAX_SEQUENCE_LENGTH,
    MAX_NUM_WORDS,
    TRAIN_FRAC,
    VAL_FRAC,
    TEST_FRAC,
    set_global_seed,
)


# --------------------------------------------------------------------------- #
# 1. TEXT CLEANING
# --------------------------------------------------------------------------- #
def clean_text(text: str) -> str:
    """
    Normalise a single essay string.

    Steps:
      1. Lowercase everything (so "The" and "the" are the same token).
      2. The ASAP corpus anonymises named entities as tokens like @PERSON1,
         @LOCATION2, @CAPS1, @NUM1.  We collapse each family to a single
         placeholder word (e.g. all @PERSON# -> "person") so the model sees a
         meaningful entity signal instead of thousands of unique tokens like
         person1, person2, person3 ...
      3. Remove every remaining character that is not a letter or whitespace
         ("removing special characters" from the report). Digits/punctuation go.
      4. Collapse repeated whitespace into single spaces and trim the ends.

    Returns the cleaned string.
    """
    text = str(text).lower()

    # Map the ASAP @-anonymisation families to plain placeholder words.
    # \d+ matches the trailing number; we drop it so all members collapse to one.
    text = re.sub(r"@person\d+", " person ", text)
    text = re.sub(r"@location\d+", " location ", text)
    text = re.sub(r"@organization\d+", " organization ", text)
    text = re.sub(r"@caps\d+", " caps ", text)          # capitalised word
    text = re.sub(r"@date\d+", " date ", text)
    text = re.sub(r"@time\d+", " time ", text)
    text = re.sub(r"@money\d+", " money ", text)
    text = re.sub(r"@percent\d+", " percent ", text)
    text = re.sub(r"@num\d+", " num ", text)            # anonymised number
    text = re.sub(r"@\w+", " ", text)                   # any other @-tag -> drop

    # Keep letters and spaces only; everything else becomes a space.
    text = re.sub(r"[^a-z\s]", " ", text)

    # Squash runs of whitespace and strip leading/trailing spaces.
    text = re.sub(r"\s+", " ", text).strip()
    return text


# --------------------------------------------------------------------------- #
# 2. SCORE NORMALISATION  (per essay set, OFFICIAL range)
# --------------------------------------------------------------------------- #
def normalize_score(score, essay_set: int):
    """
    Map an original integer score to the 0-1 range using the OFFICIAL ASAP
    range for that essay set:  norm = (score - low) / (high - low).

    Works on a single number or a NumPy array/Series (vectorised).
    """
    low, high = SCORE_RANGES[essay_set]
    return (score - low) / (high - low)


def denormalize_score(norm, essay_set: int, round_to_int: bool = True):
    """
    Inverse of normalize_score: map a 0-1 prediction back to the original score
    scale of the given essay set:  score = norm * (high - low) + low.

    This is exactly how we recover integer scores before computing QWK, so QWK
    is measured on the real scale (per the report), not on the 0-1 values.

    If round_to_int is True we round to the nearest whole number and clip into
    the valid [low, high] range (a model can output slightly <0 or >1).
    """
    low, high = SCORE_RANGES[essay_set]
    scores = np.asarray(norm) * (high - low) + low
    if round_to_int:
        scores = np.clip(np.round(scores), low, high).astype(int)
    return scores


def denormalize_per_row(norm, essay_sets, round_to_int: bool = True):
    """
    Vectorised denormalisation when each prediction belongs to a DIFFERENT essay
    set (needed once training spans multiple sets, because each set has its own
    official range). `norm` and `essay_sets` are equal-length arrays.

    For a single essay set this is equivalent to denormalize_score, but writing
    it per-row now means Stage C/E work unchanged when we scale up.
    """
    norm = np.asarray(norm)
    essay_sets = np.asarray(essay_sets)
    out = np.empty(len(norm), dtype=float)
    # Process one essay set at a time so the correct (low, high) is applied.
    for s in np.unique(essay_sets):
        mask = essay_sets == s
        out[mask] = denormalize_score(norm[mask], int(s), round_to_int=False)
    if round_to_int:
        # Clip/round each row within its own set's range.
        for s in np.unique(essay_sets):
            mask = essay_sets == s
            low, high = SCORE_RANGES[int(s)]
            out[mask] = np.clip(np.round(out[mask]), low, high)
        out = out.astype(int)
    return out


# --------------------------------------------------------------------------- #
# 3. LOAD + FILTER + SPLIT
# --------------------------------------------------------------------------- #
def load_dataframe(essay_sets=None) -> pd.DataFrame:
    """
    Load the ASAP TSV, keep the needed columns, optionally filter to a subset of
    essay sets, clean the text, and add a normalised-score column.

    Parameters
    ----------
    essay_sets : list[int] | None
        Which essay sets to keep (e.g. [1] for Stage D's first run). None = all.

    Returns a DataFrame with columns:
        essay_id, essay_set, essay (raw), clean_essay, domain1_score, score_norm
    """
    df = pd.read_csv(TSV_PATH, sep="\t", encoding=TSV_ENCODING)
    df = df[["essay_id", "essay_set", "essay", "domain1_score"]].copy()
    df = df.dropna(subset=["domain1_score"]).reset_index(drop=True)
    df["domain1_score"] = df["domain1_score"].astype(int)

    # Optionally restrict to specific essay sets.
    if essay_sets is not None:
        df = df[df["essay_set"].isin(essay_sets)].reset_index(drop=True)

    # Clean the raw essay text.
    df["clean_essay"] = df["essay"].apply(clean_text)

    # Normalise each score with its OWN essay-set's official range.
    df["score_norm"] = df.apply(
        lambda r: normalize_score(r["domain1_score"], r["essay_set"]), axis=1
    )
    return df


def _stratify_key(df: pd.DataFrame, stratify_score_bins: bool) -> pd.Series:
    """
    Build the label array that train_test_split stratifies on.

    - stratify_score_bins=False -> stratify by essay_set only (fine for a single
      set; keeps set proportions equal when spanning sets).
    - stratify_score_bins=True  -> stratify by "essay_set x score-tercile", so
      each split gets a representative spread of low/mid/high scorers WITHIN each
      prompt. Terciles are computed per set with qcut(duplicates='drop'); sets
      whose scores have too few distinct values simply get fewer bins. This
      prevents an unlucky cluster of (say) high-scorers landing only in test.
    """
    if not stratify_score_bins:
        return df["essay_set"].astype(str)

    keys = pd.Series(index=df.index, dtype=object)
    for s, grp in df.groupby("essay_set"):
        # Up to 3 bins per set; drop duplicate edges for near-discrete scores.
        try:
            bins = pd.qcut(grp["domain1_score"], q=3, duplicates="drop")
            bin_codes = bins.cat.codes.astype(str)
        except (ValueError, IndexError):
            bin_codes = pd.Series("0", index=grp.index)  # fallback: one bin
        keys.loc[grp.index] = str(s) + "_" + bin_codes
    return keys


def split_dataframe(df: pd.DataFrame, stratify_score_bins: bool = False):
    """
    Split into train / validation / test, STRATIFIED so each split is
    representative. See _stratify_key for how the stratification label is built
    (essay_set alone, or essay_set x score-bin for the combined multi-set run).

    Two-step split:
      1. Carve out the TEST fraction.
      2. Split the remainder into TRAIN and VALIDATION.
    All splits use RANDOM_SEED so they are identical on every run.
    """
    strat = _stratify_key(df, stratify_score_bins)

    # Step 1: separate the test set.
    train_val_df, test_df = train_test_split(
        df,
        test_size=TEST_FRAC,
        random_state=RANDOM_SEED,
        stratify=strat,
    )

    # Step 2: split the rest into train and val. We convert VAL_FRAC (a fraction
    # of the WHOLE dataset) into a fraction of the remaining train_val portion.
    # Rebuild the stratify key on just the remaining rows.
    strat_tv = _stratify_key(train_val_df, stratify_score_bins)
    val_relative = VAL_FRAC / (TRAIN_FRAC + VAL_FRAC)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_relative,
        random_state=RANDOM_SEED,
        stratify=strat_tv,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


# --------------------------------------------------------------------------- #
# 4. TOKENIZE + PAD
# --------------------------------------------------------------------------- #
def build_tokenizer(train_texts) -> Tokenizer:
    """
    Fit a Keras Tokenizer on the TRAINING texts ONLY (fitting on val/test would
    leak information). The tokenizer builds a word -> integer-index vocabulary,
    reserving index 0 for padding and using an <OOV> token for words unseen at
    training time.
    """
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_texts)
    return tokenizer


def texts_to_padded(
    tokenizer: Tokenizer,
    texts,
    maxlen: int = MAX_SEQUENCE_LENGTH,
    truncating: str = "post",
):
    """
    Convert a list of cleaned essays into a fixed-shape integer matrix:
      - words -> integer indices via the fitted tokenizer
      - pad short essays with 0 at the END ('post') up to `maxlen`
      - truncate long essays down to `maxlen`, at the END ('post') or the
        BEGINNING ('pre').
    Returns an int32 array of shape (n_essays, maxlen).

    truncating='post' cuts the END of long essays; 'pre' cuts the BEGINNING and
    therefore KEEPS the conclusion. We use 'post' for the set-1 run (barely any
    truncation there) but switch to 'pre' for the combined 8-set run, where set-8
    essays are long and their conclusions carry real scoring signal.
    """
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(
        sequences,
        maxlen=maxlen,
        padding="post",
        truncating=truncating,
    )
    return padded


# --------------------------------------------------------------------------- #
# 5. ONE-CALL PIPELINE
# --------------------------------------------------------------------------- #
def prepare_data(
    essay_sets=None,
    maxlen: int = MAX_SEQUENCE_LENGTH,
    truncating: str = "post",
    stratify_score_bins: bool = False,
):
    """
    End-to-end convenience wrapper that later stages call.

    Parameters
    ----------
    essay_sets          : which sets to include ([1] for the set-1 run, None=all)
    maxlen              : padding/truncation length (600 for set 1, 800 combined)
    truncating          : 'post' (cut end) or 'pre' (cut start, keep conclusion)
    stratify_score_bins : also stratify the split by per-set score tercile

    Returns a dictionary containing everything the models need:
        X_train / X_val / X_test  : padded integer sequences  (for the BiLSTM)
        y_train / y_val / y_test  : normalised 0-1 target scores
        train_df / val_df / test_df : the underlying DataFrames (Stage C uses
                                      raw text + essay_set + original scores)
        tokenizer                 : the fitted Keras tokenizer (Stage D builds
                                      the GloVe embedding matrix from its vocab)
        maxlen                    : the sequence length actually used
    """
    set_global_seed()  # deterministic cleaning/splitting

    # Load + clean + normalise, then split.
    df = load_dataframe(essay_sets=essay_sets)
    train_df, val_df, test_df = split_dataframe(
        df, stratify_score_bins=stratify_score_bins
    )

    # Fit tokenizer on TRAIN text only, then vectorise all three splits.
    tokenizer = build_tokenizer(train_df["clean_essay"])
    X_train = texts_to_padded(tokenizer, train_df["clean_essay"], maxlen, truncating)
    X_val = texts_to_padded(tokenizer, val_df["clean_essay"], maxlen, truncating)
    X_test = texts_to_padded(tokenizer, test_df["clean_essay"], maxlen, truncating)

    # Targets are the normalised 0-1 scores.
    y_train = train_df["score_norm"].values.astype("float32")
    y_val = val_df["score_norm"].values.astype("float32")
    y_test = test_df["score_norm"].values.astype("float32")

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "train_df": train_df, "val_df": val_df, "test_df": test_df,
        "tokenizer": tokenizer,
        "maxlen": maxlen,
    }


# --------------------------------------------------------------------------- #
# SELF-TEST: run this file directly to sanity-check the pipeline on set 1.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    print("Running Stage B self-test on essay set 1 ...\n")

    data = prepare_data(essay_sets=[1])

    # Show shapes so we can confirm padding / splits worked.
    print("Padded sequence shapes (essays x maxlen):")
    print("  X_train:", data["X_train"].shape)
    print("  X_val  :", data["X_val"].shape)
    print("  X_test :", data["X_test"].shape)

    print("\nTarget (normalised 0-1) ranges:")
    for name in ("y_train", "y_val", "y_test"):
        y = data[name]
        print(f"  {name}: min={y.min():.3f} max={y.max():.3f} mean={y.mean():.3f}")

    print("\nVocabulary size (unique words seen in train):",
          len(data["tokenizer"].word_index))

    # Demonstrate the normalise <-> denormalise round-trip on a few real scores.
    print("\nNormalise / denormalise round-trip check (essay set 1, range 2-12):")
    sample_scores = np.array([2, 6, 8, 12])
    norm = normalize_score(sample_scores, 1)
    back = denormalize_score(norm, 1)
    for orig, n, b in zip(sample_scores, norm, back):
        print(f"  score {orig:>2}  ->  norm {n:.3f}  ->  back {b}")

    # Show one cleaned essay so text cleaning is visible.
    print("\nExample cleaned essay (first 220 chars):")
    print(" ", data["train_df"]["clean_essay"].iloc[0][:220], "...")

    # Confirm the stratified split kept set proportions (trivial for one set).
    print("\nSplit sizes:",
          f"train={len(data['train_df'])}, "
          f"val={len(data['val_df'])}, "
          f"test={len(data['test_df'])}")
