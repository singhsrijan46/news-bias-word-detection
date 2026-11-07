# The on-device student: distilling the soft-label model for the UNO Q

The soft-label teacher (M2, RoBERTa-base, 124M parameters) is the most accurate model, but it needs about 8 seconds per article on the UNO Q. The on-device benchmark showed that a MiniLM-L6 model (23M parameters) in INT8 handles an article in about 1.3 seconds. This step trains such a student and measures what the smaller size and INT8 cost.

## Method

| Model | What it is |
| --- | --- |
| Teacher | M2: RoBERTa-base trained on vote shares, 4 epochs |
| `s_gold` | MiniLM-L6 (`nreimers/MiniLM-L6-H384-uncased`) trained on the annotators' vote shares only: 10 epochs, learning rate 1e-4, batch 32 |
| `s_distil` | The same student trained on the vote shares plus the teacher's predicted shares for unlabeled sentences: 3 epochs |
| `*_int8` | Each student exported to ONNX, quantized to INT8 and run with ONNX Runtime, as on the board |

- **Unlabeled sentences.** MBIC's 1,700 labeled sentences come from 918 full articles. These hold 31,360 other sentences of 6 to 80 words that nobody annotated (`src/data/articles.py`). The teacher scores them, and the student learns from those scores.
  - In each fold, the teacher sees only that fold's training sentences.
  - The pool leaves out every article that contains a test sentence, and uses 20,000 of the remaining sentences.
  - The gold sentences are repeated three times, so the pool does not swamp them.
- **Folds.** The same 10 sentence-grouped folds as `docs/SOFT_LABEL_MODELS.md`, with thresholds chosen on held-out training sentences, all scored on the paper's 31,267 matched words.
- **A note on "FP32".** In cross-validation, the PyTorch scores were computed under bfloat16 autocast on the GPU. The clean FP32-against-INT8 comparison is the audit's, on the final student's two ONNX models (`docs/AUDIT.md`).

## Run it

```bash
python scripts/train_student.py               # 10 folds, about an hour on a 4 GB GPU
python scripts/train_student.py --final       # train on everything; export the student and the teacher
```

`--final` trains on all 1,700 labeled sentences and the whole pool, and writes:
- `outputs/student/final/`: the student, which the board runs.
- `outputs/student/final_teacher/`: the teacher, used by the audits.

Each folder holds the model in FP32 and INT8 ONNX, its tokenizer, and `model_info.json`. That file includes the student's cross-validated decision threshold, which the app uses.

## Results

**Setup:** 10 folds, on an RTX 2050 laptop GPU with 4 GB, at about 5 minutes per fold. Each fold trained a teacher and two students, scored the 20,000-sentence pool, and exported both students to INT8. Scores are out-of-fold, on the paper's 31,267 matched words.

| Model | Parameters | F1 | Precision | Recall | ROC AUC | Avg precision | ECE | Share ECE | Spearman ρ with share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Teacher (M2, RoBERTa-base) | 124M | 0.617 | 0.561 | 0.686 | 0.938 | 0.643 | 0.028 | 0.013 | 0.665 |
| Student, vote shares only | 23M | 0.581 | 0.527 | 0.647 | 0.923 | 0.611 | 0.021 | 0.017 | 0.642 |
| the same in INT8 | 23M | 0.570 | 0.518 | 0.632 | 0.918 | 0.604 | 0.021 | 0.014 | 0.634 |
| Student, distilled | 23M | 0.605 | 0.554 | 0.667 | 0.934 | 0.642 | 0.024 | 0.015 | 0.660 |
| **the same in INT8: runs on the board** | 23M | **0.601** | 0.557 | 0.653 | 0.932 | 0.639 | 0.026 | 0.015 | 0.655 |

Fold-by-fold comparisons, with paired t-tests and Wilcoxon tests, are written to `outputs/student/cv/summary.md`.

1. **Distillation is what makes the small model good.**
   - **Detection:** against the student trained on vote shares only, the distilled student gains 0.024 in F1, higher in 9 of 10 folds (p = 0.003).
   - **Ranking:** it gains 0.011 in ROC AUC and 0.033 in average precision, higher in all 10 folds.
   - **Calibration:** its scores track the vote shares more closely, with Spearman's rho higher in all 10 folds.

   The teacher's scores on 20,000 unlabeled article sentences teach the small model what one or two thousand labeled sentences cannot.
2. **INT8 costs almost nothing.** The change in F1 is −0.003, which is not significant (p = 0.48). There are small but consistent losses in calibration: ECE is worse by 0.003 and Spearman's rho by 0.004, in nearly every fold.
3. **The board's model keeps 97% of the teacher's F1 at a fifth of its size.**
   - **Detection:** its F1 is 0.601 against 0.617, and its average precision is level with the teacher's.
   - **Speed:** it is about six times faster on the board, at roughly 1.3 seconds against 7.9 seconds per article.
   - **Calibration:** against the hard label it is better calibrated than the teacher (ECE 0.024 against 0.030). Against the vote shares it is slightly worse (0.015 against 0.013).

The board ships the distilled student in INT8. Its decision threshold, 0.267, is the cross-validated mean of each fold's best-F1 threshold, stored in `model_info.json`.
