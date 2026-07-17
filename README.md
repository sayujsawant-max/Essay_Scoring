# Automatic Essay Scoring (AES) — Deep Learning Microproject

Predicts a numeric score for a student essay using a BiLSTM (GloVe embeddings),
benchmarked against a handcrafted-feature linear-regression baseline on the
**ASAP** dataset. Follows the Stage-1 report methodology exactly.

## Project layout
```
aes-microproject/
├── data/        <-- PUT YOUR DATA FILES HERE
│   ├── training_set_rel3.tsv    (ASAP dataset, ISO-8859-1 encoded)
│   └── glove.6B.50d.txt         (50-d GloVe embeddings)
├── figures/     generated plots (300 dpi) for the report
├── models/      saved trained models
├── outputs/     metrics tables / CSVs
├── src/         pipeline code, one module per stage
└── requirements.txt
```

## Stages
- **A** Data exploration      → `src/stage_a_explore.py`
- **B** Preprocessing module  → `src/stage_b_preprocess.py`
- **C** Baseline model        → `src/stage_c_baseline.py`
- **D** BiLSTM model          → `src/stage_d_bilstm.py`
- **E** Evaluation & figures  → `src/stage_e_evaluate.py`

## Setup
```bash
pip install -r requirements.txt
```

## Interactive demo (Streamlit web UI)
Paste an essay, pick the prompt, and the trained BiLSTM predicts a score.
```bash
# 1. train the all-sets model (produces models/bilstm_allsets_best.keras)
python src/stage_d_bilstm.py
# 2. regenerate the demo's tokenizer + example essays
python src/save_demo_artifacts.py
# 3. launch the web UI at http://localhost:8501
streamlit run demo_app.py
```
This runs a **local** server on demand (localhost only) while open; it is not a
deployed/always-on service. Nothing else in the project needs a server.

## Action needed
Drop these two files into `data/`:
1. `training_set_rel3.tsv`  (from the Kaggle ASAP competition)
2. `glove.6B.50d.txt`       (from https://nlp.stanford.edu/data/glove.6B.zip)
