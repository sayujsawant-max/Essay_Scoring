"""
demo_app.py -- Streamlit web demo for Automatic Essay Scoring
=============================================================
An interactive UI: paste an essay, pick which ASAP prompt it answers, and the
trained all-sets BiLSTM predicts a score on that prompt's scale.

Launch (from the project root):
    streamlit run demo_app.py

This starts a LOCAL server on http://localhost:8501 while it is open; it is not
a deployed/always-on service -- close the terminal and it stops.
"""

import os
import sys
import json

import streamlit as st

# Make the src/ modules importable no matter where streamlit is launched from.
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from config import SCORE_RANGES, OUTPUTS_DIR          # noqa: E402
from demo_infer import load_artifacts, load_examples, predict_score  # noqa: E402


# --------------------------------------------------------------------------- #
# One-time cached loads (model is heavy; load once and reuse across reruns).
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading trained BiLSTM model...")
def get_model():
    return load_artifacts()


@st.cache_data
def get_examples():
    return load_examples()


@st.cache_data
def get_per_set_qwk():
    """Per-set QWK from Stage E, to show how reliable the model is per prompt."""
    p = os.path.join(OUTPUTS_DIR, "comparison_summary.json")
    if os.path.exists(p):
        with open(p) as f:
            data = json.load(f)
        return {int(s): v["bilstm"] for s, v in data["per_set_qwk"].items()}
    return {}


def load_example_cb(ex):
    """
    on_click callback for the example buttons. Callbacks run BEFORE the widgets
    are re-instantiated on the rerun, so it is safe to set widget-key state here
    (doing it after instantiation would raise a Streamlit exception).
    """
    st.session_state.essay_text = ex["text"]
    st.session_state.essay_set_select = ex["set"]


# A short human description of what each ASAP prompt asks for (for context).
PROMPT_HINTS = {
    1: "Persuasive: effect of computers on people",
    2: "Persuasive: censorship in libraries",
    3: "Source-based: cyclist in the desert",
    4: "Source-based: 'Winter Hibiscus' story",
    5: "Source-based: author's mood / memoir",
    6: "Source-based: Empire State Building dirigibles",
    7: "Narrative: story about patience",
    8: "Narrative: story about laughter",
}


def main():
    st.set_page_config(page_title="Automatic Essay Scoring — BiLSTM Demo",
                       page_icon="📝", layout="wide")

    # ---------------- Sidebar: about the model ----------------
    with st.sidebar:
        st.header("About this model")
        st.markdown(
            "A **Bidirectional LSTM** with GloVe word embeddings and "
            "mean-over-time pooling, trained on the **ASAP** dataset "
            "(~12,900 essays, 8 prompts)."
        )
        st.markdown(
            "- **One model** scores all 8 prompts\n"
            "- Trained target: score normalised to 0–1 per prompt\n"
            "- **Mean per-set QWK = 0.686** (agreement with human raters)"
        )
        st.info(
            "QWK (Quadratic Weighted Kappa) is the standard AES metric — it "
            "measures chance-corrected agreement with a human rater and weights "
            "bigger disagreements more heavily.",
            icon="ℹ️",
        )
        st.caption("Regression task → metrics are RMSE / MAE / QWK, not accuracy.")

    # ---------------- Header ----------------
    st.title("📝 Automatic Essay Scoring")
    st.markdown(
        "Paste a student essay, choose which prompt it answers, and the trained "
        "BiLSTM predicts a score on that prompt's official scale."
    )

    model, tokenizer = get_model()
    examples = get_examples()
    per_set_qwk = get_per_set_qwk()

    # Initialise session state before the widgets are created.
    st.session_state.setdefault("essay_text", "")

    col_left, col_right = st.columns([3, 2])

    # ---------------- Left: input ----------------
    with col_left:
        # Prompt (essay set) picker with its score range shown. Its key lets the
        # example buttons switch it to the example's prompt via the callback.
        essay_set = st.selectbox(
            "Which ASAP prompt does this essay answer?",
            options=list(range(1, 9)),
            key="essay_set_select",
            format_func=lambda s: (
                f"Set {s}  —  {PROMPT_HINTS[s]}  "
                f"(scale {SCORE_RANGES[s][0]}–{SCORE_RANGES[s][1]})"
            ),
        )

        # Example loader buttons (fill the text box + prompt via on_click).
        if examples:
            st.caption("Or load a real example essay (with its true human score):")
            ex_cols = st.columns(len(examples))
            for i, ex in enumerate(examples):
                label = f"Set {ex['set']} · human score {ex['score']}"
                ex_cols[i].button(label, key=f"ex{i}",
                                  on_click=load_example_cb, args=(ex,))

        essay_text = st.text_area(
            "Essay text",
            key="essay_text",
            height=320,
            placeholder="Paste the student's essay here...",
        )

        score_clicked = st.button("Score essay", type="primary")

    # ---------------- Right: result ----------------
    with col_right:
        st.subheader("Predicted score")

        if score_clicked:
            if len(essay_text.split()) < 20:
                st.warning(
                    "Please enter a longer essay (at least ~20 words) for a "
                    "meaningful prediction."
                )
            else:
                res = predict_score(model, tokenizer, essay_text, essay_set)
                low, high = res["low"], res["high"]

                # Big number + the scale.
                st.metric(
                    label=f"Set {essay_set} score (scale {low}–{high})",
                    value=f"{res['predicted_int']} / {high}",
                )

                # Normalised 0-1 confidence-style bar.
                st.progress(res["normalized"])
                st.caption(
                    f"Model output (normalised): {res['normalized']:.2f} on 0–1  ·  "
                    f"{res['n_words']} words analysed"
                )

                # How reliable is the model on THIS prompt?
                if essay_set in per_set_qwk:
                    q = per_set_qwk[essay_set]
                    st.caption(
                        f"Model's test QWK on Set {essay_set}: **{q:.2f}** "
                        f"(agreement with human raters on this prompt)."
                    )

                st.success("Scored. Try another essay or example.", icon="✅")
        else:
            st.info("Enter an essay and click **Score essay**.", icon="👈")

    st.divider()
    st.caption(
        "Educational demo. Automatic scores should support, not replace, a human "
        "rater — borderline or low-confidence essays need human review."
    )


if __name__ == "__main__":
    main()
