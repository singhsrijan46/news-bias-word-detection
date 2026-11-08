# External test on BABE: do the models hold up beyond MBIC?

Every result so far comes from MBIC, scored by MBIC's crowd. This test uses BABE, "Bias Annotations By Experts" (Spinde et al., 2021, Findings of EMNLP), from the same research group. It asks two things:

1. Do the models work on news text they have never seen?
2. How do they compare with expert judgements, rather than the crowd's?

## Data

BABE has 4,121 sentences from news articles. Expert annotators labelled each sentence as biased or not, and for most sentences they also listed the words that make it biased.

**1,697 of BABE's sentences are MBIC's own,** relabelled by experts. That splits the test in two:

| Part | Sentences | What is compared |
| --- | --- | --- |
| New sentences | 2,418 (1,971 with expert word labels) | The final models, trained on all of MBIC, against the experts' labels |
| MBIC's own sentences | 1,697 (30,594 of the paper's words) | The models' cross-validated predictions, so no model scores a sentence it trained on, against the experts' labels and the crowd's |

**Rules:**
- **Matching:** sentences are matched on their text, ignoring case, quote marks and spacing. Two sentences that are light edits of MBIC sentences are left out of both parts.
- **Word labels:** the experts' words are matched to the sentence's words the same way as MBIC's highlights. 447 new sentences have no word list, so they count only at sentence level.
- **Thresholds:** each model keeps the threshold chosen in MBIC's cross-validation. Nothing is tuned on BABE.
- **Sentences:** a sentence is flagged when any of its words is flagged, and its score is its highest word score.

The models are the teacher (M2, RoBERTa-base trained on vote shares), the distilled MiniLM-L6 student, and the same student in INT8, which is the model the board runs. The test used BABE's version `749f72c` from Hugging Face.

## Summary

- **The ranking carries over to new text.** On the 2,418 new sentences, word-level ROC AUC is 0.945 to 0.950, as high as in MBIC's cross-validation (0.93 to 0.94).
- **The threshold does not carry over.** Experts mark fewer words than MBIC's crowd: 3.7% of words in the new sentences. At MBIC's threshold the models over-flag. Precision is about 0.40, recall is 0.68 to 0.74, and F1 is 0.52. A threshold chosen on BABE would reach an F1 of 0.55 to 0.57.
- **New topics are no harder.** On the six BABE topics that MBIC lacks, F1 is 0.53 to 0.54, against 0.51 to 0.52 on MBIC's topics.
- **Centre outlets get many false flags.** Experts marked only 0.4% of words from centre outlets. There, nine in ten of the models' flags are words the experts did not mark. Left and right outlets are treated alike.
- **The models match the experts at least as well as the crowd does.** On MBIC's own sentences, the crowd's 4-vote labels agree with the experts only moderately (Cohen's κ 0.42). The models' cross-validated flags agree slightly better (κ 0.43 to 0.44). They also rank words better against the experts than the crowd's own vote shares do (ROC AUC 0.94 against 0.92).
- **The app's two bands mean something to experts too.** Of the board's flags in the "most readers" band, experts marked 79%; in the "readers split" band, only 32%.
- **INT8 costs nothing measurable here.** The INT8 student's F1 is 0.516, against 0.519 for the FP32 student.

## 1. New sentences

| Model | Words | F1 | Precision | Recall | ROC AUC | Average precision | Best threshold on BABE | F1 there |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Teacher | 59,474 | 0.515 | 0.396 | 0.737 | 0.950 | 0.575 | 0.41 | 0.571 |
| Student | 59,474 | 0.519 | 0.413 | 0.698 | 0.948 | 0.578 | 0.39 | 0.559 |
| Student, INT8 | 59,474 | 0.516 | 0.414 | 0.684 | 0.945 | 0.567 | 0.33 | 0.550 |

F1 is lower than MBIC's cross-validated 0.60 to 0.62, but those numbers were measured against the crowd. Against experts on MBIC's own sentences (section 2), F1 is lower still: 0.47 to 0.48. So the drop comes from the experts' stricter labels, not from the new text.

**By part** (the INT8 student; the teacher's and the FP32 student's F1 are within 0.03 of it in every part):

| Part | Words | Marked by experts | F1 | Precision | Recall | ROC AUC |
| --- | --- | --- | --- | --- | --- | --- |
| Topics in MBIC | 44,324 | 3.5% | 0.511 | 0.401 | 0.702 | 0.948 |
| Topics new to the models | 15,150 | 4.3% | 0.530 | 0.452 | 0.640 | 0.939 |
| Outlets in MBIC | 53,704 | 3.9% | 0.522 | 0.424 | 0.681 | 0.945 |
| Outlets new to the models | 5,770 | 2.1% | 0.430 | 0.304 | 0.736 | 0.947 |
| Left outlets | 8,923 | 5.3% | 0.587 | 0.467 | 0.793 | 0.957 |
| Centre outlets | 13,122 | 0.4% | 0.192 | 0.113 | 0.644 | 0.948 |
| Right outlets | 9,050 | 5.9% | 0.587 | 0.516 | 0.680 | 0.941 |
| Leaning not given | 28,379 | 4.0% | 0.498 | 0.407 | 0.642 | 0.936 |

- **Topics:** the new topics are #MeToo, Black Lives Matter, Islam, marriage equality, taxes and universal health care. The models never trained on them, yet F1 there is slightly higher and the ranking almost the same.
- **Outlets:** new outlets, such as the New York Times and The Guardian, have half as many expert-marked words, which lowers precision at a fixed threshold. Their words are ranked as well as the others.
- **Centre outlets:** this is the clearest weakness. The ranking is still good (ROC AUC 0.95), but with so few biased words, the fixed threshold flags about six words for every one the experts marked. MBIC's centre outlets showed the same pattern, less sharply (`docs/AUDIT.md`, section 2).
- **Left against right:** F1 is the same, 0.587 for both. Left-leaning outlets get more recall, right-leaning ones more precision.

**Sentence level.** 51.7% of the new sentences are biased by the experts' labels.

| Model | Sentences | F1 | Precision | Recall | ROC AUC |
| --- | --- | --- | --- | --- | --- |
| Teacher | 2,418 | 0.782 | 0.666 | 0.947 | 0.868 |
| Student | 2,418 | 0.788 | 0.685 | 0.930 | 0.866 |
| Student, INT8 | 2,418 | 0.787 | 0.686 | 0.922 | 0.860 |

The models flag nearly every biased sentence, but also about half of the unbiased ones. A sentence-level verdict is not what they were trained for. The authors' own sentence classifiers are compared on BABE's test split in `docs/AUDIT.md`.

## 2. MBIC's own sentences: crowd against experts

On the same 30,594 words, experts marked 5.0% and the crowd's 4-vote rule marks 9.8%.

| Rater | F1 against experts | Precision | Recall | ROC AUC | Cohen's κ | F1 against the crowd |
| --- | --- | --- | --- | --- | --- | --- |
| Crowd: at least 4 of 10 votes (score: vote share) | 0.460 | 0.347 | 0.682 | 0.924 | 0.422 | — |
| M0, the paper's XGBoost | 0.309 | 0.222 | 0.509 | 0.832 | 0.257 | 0.453 |
| M1, hard labels | 0.478 | 0.353 | 0.739 | 0.936 | 0.440 | 0.627 |
| M2, vote shares (the teacher) | 0.471 | 0.339 | 0.771 | 0.941 | 0.432 | 0.636 |
| Student | 0.475 | 0.344 | 0.766 | 0.940 | 0.436 | 0.625 |
| Student, INT8 | 0.475 | 0.347 | 0.753 | 0.937 | 0.436 | 0.621 |

- **Crowd and experts disagree.** The crowd's labels miss a third of the experts' words, and two thirds of the crowd's words are not the experts'. This matches the group's own finding that crowd labels are much less reliable than expert ones (Spinde et al., 2021, "Towards a reliable ground truth").
- **The models are at least as close to the experts as the crowd is.** They learned from the crowd, but averaging over many sentences smooths out the noise of ten votes per sentence.
- **Soft labels rank better against experts too.** M2 against M1: ROC AUC 0.941 against 0.936, average precision 0.549 against 0.515. M1's F1 is slightly higher.
- **M0 transfers worst.** The paper's feature-based model reaches an F1 of only 0.31 against the experts.

## What this means for Bias Lens

1. **Scores mean "MBIC's readers".** Experts are stricter, so a flag means a crowd of readers would flag the word, not that an expert would. The page already says whose readers they are.
2. **The "readers split" band is the right display.** On BABE's new sentences, the INT8 student put 20% of its flags in the "most readers" band (a share of at least 0.6), and experts marked 79% of those. The other 80% fell in "readers split", where experts marked only 32%. Showing these flags as contested, not as verdicts, is what the experts' labels support.
3. **Be careful with neutral outlets.** On centre outlets, most flags are words experts would not mark. But 91% of those flags fall in the "readers split" band, and experts marked only 7% of them. The page should say that a low-bias article will still show some flags, mostly in that band.
4. **The board's model transfers like the large one.** INT8 and distillation cost no more on BABE than on MBIC.

## Run it

```bash
python scripts/train_student.py               # cross-validated predictions (outputs/student/cv)
python scripts/train_token_models.py          # M0 to M2, cross-validated (outputs/models/cv)
python scripts/train_student.py --final       # the final student and teacher
python scripts/evaluate_babe.py               # downloads BABE (under 1 MB); a few minutes
```

Results go to `outputs/babe/`: `summary.md` and the tables behind it, plus every word's scores.

## Limits

- **Same group, same era.** BABE comes from the same group and the same years as MBIC, and partly the same outlets. It tests new sentences and expert labels, not news from other countries or years.
- **Experts' word lists.** The word labels were made by matching listed words to the sentence. A listed word that occurs twice marks only its first occurrence, as with MBIC's highlights.
- **The sentence verdict is ours.** "Any word flagged" is a simple rule chosen for this comparison. The models were trained on words, not sentences.
