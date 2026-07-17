# Results Summary — Automatic Essay Scoring

Drop-in material for **Section 3 (Implementation & Results)** and **Section 4
(Conclusion)** of the microproject report. All figures are in `figures/` at
300 dpi.

## Dataset (Stage A)
- ASAP dataset: **12,976 essays** across **8 prompts (sets)**, no missing scores.
- Score scales differ wildly per set: Set 3 = 0–3, Set 2 = 1–6, Set 8 = 0–60.
  → scores must be normalised **per set** before a shared model can learn them.
- Figures: `A_essays_per_set.png`, `A_score_ranges_per_set.png`,
  `A_length_distribution.png`.

## Method (Stages B–D)
- **Preprocessing:** lowercase, collapse ASAP `@PERSON1`→`person` anonymisation
  tags, strip non-letters; tokenise (fit on **train only**); pad/truncate;
  normalise `domain1_score` to 0–1 using the **official** per-set range.
- **Baseline (Stage C):** linear regression on 5 handcrafted features
  (word count, sentence count, avg word length, unique-word ratio, spelling
  errors). Feature weights show it is dominated by **word count** — essentially
  "longer essays score higher".
- **BiLSTM (Stage D):** GloVe-50d embedding (trainable) → BiLSTM(64) →
  **mean-over-time pooling** → dropout → dense → sigmoid. Adam, MSE, batch 32,
  early stopping (restore best weights). GloVe vocab coverage 77.4%.

### Tuning journey (good methodology-section story)
On set 1 alone the BiLSTM improved as design flaws were fixed:
| Change | Set-1 QWK |
|---|---|
| Frozen embedding + last-state only | 0.30 |
| + mean-over-time pooling | 0.47 |
| + trainable embedding, less dropout | 0.70 |

## Headline comparison (Stage E) — MEAN PER-SET QWK

| Model | Mean per-set QWK |
|---|---|
| **BiLSTM** — one model, all 8 sets | **0.686** |
| Baseline — one global linear model, all 8 sets | 0.361 |
| Baseline — 8 separate per-prompt linear models | 0.672 |

**Thesis:** a *single* BiLSTM generalises across all 8 heterogeneous prompts as
well as **eight separately-tuned baselines combined** (0.686 vs 0.672), and
nearly **2× better than a single feature-based model** (0.361). That is deep
learning earning its complexity.

### Per-set QWK (BiLSTM vs strongest per-prompt baseline) — 4–4 split
| Set | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| BiLSTM | 0.72 | **0.61** | 0.63 | **0.77** | 0.77 | **0.77** | **0.72** | 0.49 |
| Baseline | **0.79** | 0.58 | **0.68** | 0.69 | **0.81** | 0.62 | 0.65 | **0.55** |

- Baseline edges **Set 1** (longest, most length-driven prompt — its trick shines).
- BiLSTM wins where content matters more (Sets 4, 6, 7).
- Figure: `E1_qwk_per_set_bars.png` (the money chart).

### Why not report the pooled QWK?
Pooling all 8 prompts gives an inflated **QWK = 0.98**, which is **misleading**:
essays from sets with different ranges (e.g. a 55 vs a 3) trivially "agree" on
being far apart, so the metric partly rewards sorting *which set* an essay is in
rather than scoring *within* a set. `E3_pred_vs_actual_pooled.png` shows the 8
separated clusters that cause this. Mean **per-set** QWK is the correct metric.

### Set 8 is the weakest (QWK 0.49) — root cause
Smallest training set (505 essays), **widest scale (0–60)**, and fewest test
essays (109) → hardest to learn and noisiest to measure; ~26% of its essays
still exceed the 800-token window (secondary factor).

## Figures index
| File | What it shows |
|---|---|
| `E1_qwk_per_set_bars.png` | Per-set QWK, baseline vs BiLSTM, + mean lines |
| `E2_loss_curves.png` | Train/val loss: set-1 (overfits) vs all-sets (healthier) |
| `E3_pred_vs_actual_pooled.png` | Pooled scatter, coloured by set — the pooling mirage |
| `E4_pred_vs_actual_per_set.png` | Per-set scatter, BiLSTM, QWK annotated |

## Conclusion (for Section 4)
Deep learning **is** suitable for AES: a single BiLSTM reading only essay text
matches an ensemble of eight prompt-specific baselines and far exceeds a single
feature model, while remaining one unified model. Simple length-based features
remain a strong, hard-to-beat baseline on individual well-populated prompts
(notably Set 1). Future work: larger/transformer models (BERT), per-set output
heads, and more data for the small/wide-scale prompts (Set 8).

> Note on metrics: the report's Section 3 table lists Accuracy/Precision/Recall/
> F1, which suit *classification*. AES is framed here as **regression**, so the
> appropriate metrics are **RMSE, MAE, and QWK** (the standard AES measure) —
> reported throughout instead of accuracy/F1.
