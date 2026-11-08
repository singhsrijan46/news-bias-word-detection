# First bias-word models: hard labels against vote shares

The paper calls a word biased when at least 4 of its roughly 10 annotators highlighted it, and trains on that 0/1 label. MBIC's raw annotations also tell us how many annotators highlighted each word. This is the project's first research question (RQ1): if a model learns that vote share instead, is it at least as good at finding biased words, and do its scores say how contested each word is?

## Models

| Model | What it is | Trained on |
| --- | --- | --- |
| M0 | The paper's tuned XGBoost on its word features (`dt_final`), settings as in `src/models/all_models.py` | The paper's label |
| M1 | RoBERTa-base token classifier: one score per word, read at the word's first sub-word token | Hard label: at least 4 votes |
| M2 | The same model, the same settings | Vote share: the fraction of annotators who highlighted the word |

M1 and M2 differ only in their targets. Both use binary cross-entropy:
- **M1:** with 0/1 targets, this is ordinary classification.
- **M2:** with shares as targets, it fits the predicted probability to the share, so the model's score estimates how many readers would flag the word.

Both models are trained for 4 epochs, with a learning rate of 3e-5, batch size 16, and seed 0.

## Evaluation

- **Folds grouped by sentence.** Ten folds, and no sentence appears in both training and test. The repo's earlier split (`get_train_test_split`) shuffles words rather than sentences, so words from one sentence land on both sides, and models can partly learn the sentence instead of the word.
- **Thresholds chosen out of sample.** In each fold, 10% of the training sentences are held out, and each model's decision threshold is the one with the best F1 there.
- **The paper's words and labels.** All three models are scored on the paper's own word list (`dt_final`: 31,794 non-stop words) with its `label4`. Of these, 31,267 (98.3%) line up with our tokens; spaCy splits some words differently (see `align_paper_words`). Scores are out-of-fold: every word is predicted by the model that did not see its sentence.
- **Calibration.** Two kinds:
  - Against the label: Brier score and expected calibration error (ECE).
  - Against the vote share: share ECE (the gap between predicted and actual shares) and Spearman's rho.
- **Accuracy is not reported.** With 9.5% positive words, a model that never flags anything scores 90.5%.

## How to run

```bash
pip install -r requirements-transformers.txt
python scripts/train_token_models.py             # 10 folds; about 30 minutes on a 4 GB laptop GPU
python scripts/train_token_models.py --quick     # 2 folds, 1 epoch: checks that everything runs
```

The script writes these files to `outputs/models/cv/`, which git ignores:
- `metrics.json`: every number.
- `summary.md`: the results table.
- `paper_words_oof.csv` and `tokens_oof.csv`: out-of-fold predictions.

## Results

**Setup:** 10 folds, 4 epochs and seed 0, on an RTX 2050 laptop GPU with 4 GB. Each RoBERTa model took about 80 seconds per fold, and the whole run took 27 minutes. All scores are out-of-fold, on the 31,267 paper words that line up with our tokens.

| Model | F1 | F1 per fold | Precision | Recall | ROC AUC | Avg precision | Brier | ECE | Share ECE | Spearman ρ with share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M0 XGBoost (paper features) | 0.451 | 0.451 ± 0.020 | 0.417 | 0.492 | 0.850 | 0.445 | 0.117 | 0.161 | 0.158 | 0.448 |
| M1 RoBERTa, hard labels | 0.608 | 0.608 ± 0.015 | 0.573 | 0.647 | 0.931 | 0.629 | 0.058 | 0.043 | 0.080 | 0.625 |
| M2 RoBERTa, vote shares | **0.617** | 0.617 ± 0.010 | 0.561 | 0.686 | **0.938** | **0.643** | **0.053** | **0.028** | **0.013** | **0.665** |

For reference, the paper reports F1 0.43 and ROC AUC 0.79 for its XGBoost.

`python scripts/compare_token_models.py` writes the fold-by-fold comparisons, with paired tests, to `paired.md`. With 10 folds, the smallest possible Wilcoxon p-value is 0.002.

1. **Transformers find biased words far better than the paper's feature-based model.** On the same folds, M1 beats M0 by 0.16 in F1 and 0.08 in ROC AUC, and it does so in every fold. M0 itself lands close to the paper's own result: F1 0.45, against the paper's 0.43.
2. **Training on vote shares costs no detection.**
   - **F1:** M2's is 0.009 higher than M1's. It is higher in 6 of 10 folds, which is not significant (p = 0.10).
   - **Ranking:** M2 ranks words better in every fold, with ROC AUC higher by 0.006 (p < 0.001). Its average precision is higher by 0.015, in 8 of 10 folds (p = 0.015).
3. **Training on vote shares makes the scores mean something.**
   - **Matching the vote shares:** M2's predicted probabilities are about six times closer to the actual vote shares. Its share ECE is 0.014 against M1's 0.081, averaged over folds, and lower in all 10 folds. Across all 56,909 tokens it is 0.009 against 0.051.
   - **Ranking by agreement:** M2 also ranks words more closely by how many annotators agreed, with a Spearman ρ of 0.665 against 0.627.
   - **Against the hard label:** M2 is better calibrated there too, with an ECE of 0.030 against 0.044.
4. **M2's decision threshold is stable.** Each fold's best-F1 threshold, chosen on its validation sentences, ranged from 0.21 to 0.31 for M2 but from 0.16 to 0.57 for M1. A threshold that moves less between samples of data is easier to trust on new text.
5. **M2's score reads as a share of readers.** At its own cutoff, a predicted share of at least 0.4 to match "4 of 10 annotators", M2 flags fewer words at higher precision: F1 0.571, precision 0.704, recall 0.480. The best-F1 threshold, about 0.26, trades that precision for recall. Which point a reading aid should use is a design question for RQ4.
6. **M0's probabilities are poorly calibrated (ECE 0.16).** Its 9.5× class weight inflates them on purpose. This reflects how it was trained, not a flaw in its features.

**Labels.** The paper's `label4` agrees with our rebuilt "at least 4 votes" label on 98.3% of words (Cohen's κ = 0.906), so the rebuilt targets are a faithful basis for training. The 527 paper words (1.7%) that did not line up with our tokens are left out of the scores.

## Abstaining on uncertain words

A model that knows when it is unsure can hold back on those words, and be right more often on the rest. This is RQ1's last measure. `scripts/coverage_curves.py` makes each model abstain on the words whose scores are closest to its decision threshold, in each fold. It then scores the model on the words it keeps, against the paper's labels. The students' predictions come from `train_student.py` (`docs/STUDENT_MODEL.md`).

| Model | F1, every word | 95% kept | 90% kept | 80% kept | 70% kept | 50% kept |
| --- | --- | --- | --- | --- | --- | --- |
| M0 XGBoost | 0.451 | 0.473 | 0.496 | 0.489 | 0.319 | 0.000 |
| M1, hard labels | 0.608 | 0.652 | 0.661 | 0.665 | 0.678 | 0.711 |
| M2, vote shares | 0.617 | 0.663 | **0.703** | **0.765** | 0.803 | 0.839 |
| Student | 0.605 | 0.650 | 0.686 | 0.762 | 0.816 | 0.856 |
| Student, INT8 | 0.601 | 0.640 | 0.684 | 0.759 | 0.797 | 0.852 |

1. **M2 gains far more from abstaining than M1.**
   - **With 10% set aside:** M2's F1 rises by 0.09 and M1's by 0.05. Fold by fold, M2 is ahead in 8 of 10 folds (Wilcoxon p = 0.01), and its gain from abstaining is larger (paired t-test p = 0.02).
   - **With 20% set aside:** M2 is ahead in all 10 folds (p = 0.002). M1 hardly improves past 90%: its errors are not where its scores are closest to the threshold.
2. **M2's uncertainty is the readers' disagreement.** Call a word contested when 2 to 5 of 10 annotators highlighted it. Contested words are 17% of all words, but 55% of the 10% M2 sets aside, against 39% for M1. Training on vote shares makes "unsure" mean "readers disagree".
3. **The students abstain as well as M2.** This includes the INT8 model on the board.
4. **Abstaining has a cost.** At 90% coverage M2 sets aside 28% of the biased words, mostly contested ones. So Bias Lens does not hide these words. It marks the ones above the threshold "readers split", which gives no verdict on them. On BABE, experts marked 32% of those words, against 79% of the "most readers" flags (`docs/BABE_TEST.md`).

M0 collapses past 80% because most of its biased words score near its threshold.

## Limits and next steps

- **One run, untuned.** This is one seed and one set of hyperparameters (4 epochs, learning rate 3e-5), fixed before seeing the test folds. More seeds would settle whether M2's small F1 gain is real.
- **Random folds.** The folds are random over sentences. On BABE, an expert-labelled external test set, the models rank words as well as here, including on topics MBIC lacks (`docs/BABE_TEST.md`). A topic-held-out split is the other test in the proposal's plan.
- **Next models.** M3 adds annotator ideology. After that comes the small student distilled for the UNO Q. The on-device benchmark shows that a MiniLM-sized INT8 model handles an article in about 1.3 seconds on the board.
