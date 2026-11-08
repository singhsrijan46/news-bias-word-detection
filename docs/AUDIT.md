# Audit: political symmetry, outlet leaning, whose readers, and compression

A tool that flags bias must not be biased itself. This audit asks four things about the models behind Bias Lens:

1. Do they treat the political sides alike?
2. Do they work equally well on left-, centre- and right-leaning outlets?
3. Whose perception of bias do they learn?
4. What does compressing the model for the UNO Q change?

It covers these models:
- **The teacher:** M2, RoBERTa-base trained on vote shares, in FP32.
- **The distilled student:** MiniLM-L6, in FP32.
- **The same student in INT8:** the model the board runs.

The models are described in `docs/SOFT_LABEL_MODELS.md` and `docs/STUDENT_MODEL.md`.

## Summary

- **Nearly symmetric, with one clear exception.** In sentences rewritten to name the other political side, the models flag the swapped terms almost alike: Biden against Trump differs by 0.005, Democrats against Republicans by 0.010 to 0.013. The exception is the words themselves. "Liberal" is flagged much more than "conservative" in identical contexts (0.28 against 0.18 for the teacher), and "liberals" more than "conservatives" (0.18 against 0.10).
- **Compression costs a little symmetry.** INT8 widens the liberal/conservative gap (0.28 against 0.14), triples the pro-choice/pro-life gap, and reverses the direction for left-wing/right-wing. Flags on the other words of a swapped sentence change almost twice as often: 0.31% of words for the teacher, 0.57% in INT8.
- **Weakest on centre outlets.** F1 is about 0.53 on centre outlets against 0.60 to 0.64 elsewhere. Centre outlets simply have fewer biased words. The student loses more against the teacher on right-leaning outlets (−0.019) than on left-leaning ones (−0.007).
- **The models read like liberal annotators, partly.** Even with groups of equal size, predicted shares follow liberal annotators more closely than conservative ones (Spearman's rho 0.48 against 0.40, in all 50 draws). About half of the raw gap is real. The training target pools all annotators, and liberals are the largest group (44% against 27%). M3's conservative output fits conservative annotators better, in all 10 folds.
- **INT8 moves scores very little, but lowers them slightly.** The INT8 and FP32 scores correlate at Spearman's rho 0.993. The board uses the threshold chosen for INT8, which absorbs that shift.

## 1. Counterfactual swaps

**Method.** Sentences that mention only one political side are rewritten to mention the other, and both versions are scored. Capitalisation is kept, and possessives follow ("Trump's" becomes "Biden's").
- **Sentences:** 4,631, drawn from MBIC and from its articles' unlabeled sentences.
- **Pairs:** Democrats/Republicans, Democratic/Republican, liberal(s)/conservative(s), left-wing/right-wing, pro-choice/pro-life, Biden/Trump and Pelosi/McConnell.

A symmetric model scores each swapped term like its counterpart, and leaves every other word's score where it was.

| Model | Swapped terms: left-coded share | Right-coded share | Difference | Other words: mean absolute shift | Other words: flags that flip |
| --- | --- | --- | --- | --- | --- |
| Teacher | 0.067 | 0.062 | −0.005 | 0.002 | 0.31% |
| Student | 0.064 | 0.057 | −0.007 | 0.003 | 0.42% |
| Student, INT8 | 0.063 | 0.056 | −0.007 | 0.005 | 0.57% |

The differences are statistically consistent (Wilcoxon p < 0.01), but small: under one point of share.

**By pair** (the teacher and the INT8 student, using pairs with at least 100 swaps):

| Pair | Swaps | Teacher: left / right | INT8 student: left / right |
| --- | --- | --- | --- |
| Biden / Trump | 3,018 | 0.022 / 0.027 | 0.018 / 0.024 |
| Democrats / Republicans | 725 | 0.059 / 0.049 | 0.056 / 0.042 |
| Democratic / Republican | 697 | 0.048 / 0.045 | 0.041 / 0.042 |
| liberal / conservative | 259 | **0.278 / 0.175** | **0.279 / 0.140** |
| Pelosi / McConnell | 136 | 0.025 / 0.020 | 0.015 / 0.018 |
| pro-choice / pro-life | 132 | 0.507 / 0.528 | 0.476 / 0.540 |
| liberals / conservatives | 112 | **0.176 / 0.099** | **0.172 / 0.090** |
| left-wing / right-wing | 103 | 0.297 / 0.320 | 0.335 / 0.271 |

- **Names and parties are treated almost alike.** Trump is flagged very slightly more than Biden, and Democrats slightly more than Republicans.
- **"Liberal" is the outlier.** MBIC's annotators often met "liberal" as a loaded label in their sentences, and flagged it far more often than "conservative". The earlier perspective analysis (`docs/PERSPECTIVE_ANALYSIS.md`) found that annotators with a negative ideology score flag "liberal" at 0.39, against 0.15 for those with a positive score. The model has learned the word, not just its use. In a neutral sentence, "liberal" is flagged where "conservative" is not.
- **INT8 changes which way some pairs lean.** It widens the liberal/conservative and pro-choice/pro-life gaps, and turns left-wing/right-wing around. These are small samples, from 100 to 260 swaps each, but they show that compression can shift political symmetry, not just accuracy.

**Caveat.** A swap can produce an awkward sentence, such as "a Democratic said" from "a Republican said", and real counterparts are not used identically. The test isolates the model's response to the term, which is what matters for a tool that flags terms.

## 2. Outlet leaning

These are cross-validated predictions on the paper's words, grouped by the outlet's AllSides rating. All three models use their cross-validated thresholds.

| Outlet | Model | Words | F1 | Precision | Recall | ROC AUC | Share ECE | Mean predicted share | Mean actual share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Left | Teacher | 12,795 | 0.637 | 0.584 | 0.700 | 0.937 | 0.014 | 0.105 | 0.117 |
| Left | Student | 12,795 | 0.630 | 0.580 | 0.689 | 0.933 | 0.016 | 0.106 | 0.117 |
| Left | Student, INT8 | 12,795 | 0.622 | 0.579 | 0.670 | 0.931 | 0.017 | 0.102 | 0.117 |
| Centre | Teacher | 5,986 | 0.534 | 0.462 | 0.632 | 0.955 | 0.013 | 0.062 | 0.069 |
| Centre | Student | 5,986 | 0.528 | 0.468 | 0.606 | 0.950 | 0.014 | 0.063 | 0.069 |
| Centre | Student, INT8 | 5,986 | 0.526 | 0.467 | 0.602 | 0.950 | 0.012 | 0.061 | 0.069 |
| Right | Teacher | 12,486 | 0.615 | 0.559 | 0.682 | 0.930 | 0.014 | 0.103 | 0.114 |
| Right | Student | 12,486 | 0.596 | 0.546 | 0.656 | 0.926 | 0.016 | 0.105 | 0.114 |
| Right | Student, INT8 | 12,486 | 0.596 | 0.554 | 0.645 | 0.923 | 0.016 | 0.100 | 0.114 |

- **Centre outlets:** F1 is lowest there, yet ROC AUC is highest. Centre outlets have fewer biased words (mean share 0.069), which lowers precision at a fixed threshold. They are not ranked worse.
- **The student's loss:** against the teacher, the student loses more on right-leaning outlets (−0.019 F1) than on left-leaning ones (−0.007).
- **Under-prediction:** every model slightly under-predicts shares everywhere, by about one point.

## 3. Whose perception do the models learn?

Each model's cross-validated shares were compared with the shares of liberal annotators (ideology below −3) and of conservative ones (above 3), on the 30,551 paper words whose sentences both groups judged.

| Comparison | Spearman's rho with liberal shares | With conservative shares | Gap |
| --- | --- | --- | --- |
| Teacher, groups as they are | 0.579 | 0.421 | 0.158 |
| Teacher, equal-sized groups (50 draws) | 0.484 | 0.402 | 0.082 ± 0.004 |
| Student, equal-sized groups | 0.481 | 0.399 | 0.082 ± 0.004 |
| Student INT8, equal-sized groups | 0.477 | 0.397 | 0.081 ± 0.004 |

A sentence has on average 4.7 liberal annotators but only 2.7 conservative ones, and a model always agrees less with the noisier, smaller group. So `scripts/audit_group_balance.py` gives both groups the same size: in each sentence it draws as many annotators from each group as the smaller one has, and repeats the draw 50 times.

- **About half the gap is group size.** It falls from 0.158 to 0.082.
- **The rest is real.** The liberal side stays higher in all 50 draws. Absolute error tells the same story: 0.112 against liberal shares and 0.122 against conservative ones.
- **Why:** the training target, the share of all annotators, pools everyone, and liberals are the largest group: 44% of annotators, against 27% conservative. So "how many readers would flag this" means how many of MBIC's readers would, and that pool leans liberal.
- **For comparison:** at equal sizes, the two groups' own shares agree with each other at only 0.33. Perception of bias genuinely differs, and small groups are noisy.

**M3, the perspective model,** learns separate outputs per group (`docs/SOFT_LABEL_MODELS.md`, `scripts/train_perspective_model.py`).
- **Conservative annotators:** its conservative output explains their choices better than M2's single share, in all 10 folds (log loss 0.207 against 0.212, p < 0.0001). For liberals there is no gain, because the pooled share already follows them.
- **The predicted gap:** the gap between M3's two outputs follows the observed gap only weakly (Spearman's rho 0.13 on words that someone flagged). It is clearest on words like these:
  - **Liberals flag more:** "fake", "radical", "drastic" and "blasted".
  - **Conservatives flag more:** "fraud", "undocumented", "impeachment" and "asylum".

So M3 is a fair way to show conservative readers' view, but its per-word differences are uncertain. Bias Lens should show them only on request, with that caveat. This is the proposal's RQ4.

## 4. What INT8 changes

These numbers compare the final student's FP32 and INT8 ONNX models on all 1,700 MBIC sentences (56,909 words). This is the clean comparison: in cross-validation, the PyTorch scores were computed under bfloat16.

| Measure | Value |
| --- | --- |
| Mean absolute change in share | 0.007 |
| Largest change | 0.183 |
| Spearman's rho, FP32 against INT8 | 0.993 |
| Words flagged at the threshold of 0.267 | 4,324 in FP32, 3,962 in INT8 |

INT8 lowers scores slightly, so at one fixed threshold it flags 8% fewer words. The board uses the threshold chosen on INT8 scores in cross-validation. There the INT8 student's F1 was 0.601, against 0.605 for the bfloat16 student, a difference that is not significant (`docs/STUDENT_MODEL.md`). Compression's other cost is to symmetry (section 1).

## What this means for Bias Lens

1. **Say whose readers.** The page already says the scores come from US crowd workers rating 2017 to 2020 news. It should also say that this pool leans liberal, and that "readers" means them.
2. **Treat political labels with care.** "Liberal" and "liberals" are flagged far more than their counterparts in identical sentences. A per-term check, such as running this swap test before each release, should guard the shipped model.
3. **Offer the conservative view on request.** M3 fits conservative annotators better. Show its liberal and conservative shares on request, with the weak-gap caveat.
4. **Test symmetry after compression, not only accuracy.** INT8 kept F1, but moved some pairs' balance.

## Run it

```bash
python scripts/train_student.py               # cross-validated predictions (outputs/student/cv)
python scripts/train_student.py --final       # the final student and teacher
python scripts/audit_models.py                # swaps, outlets, groups, INT8 (outputs/audit; about 25 minutes)
python scripts/audit_group_balance.py         # equal-sized groups (about 20 seconds)
python scripts/train_perspective_model.py     # M3 in 10 folds
```

## Limits

- **One dataset.** MBIC only: 1,700 sentences of US political news from 2017 to 2020, judged by US crowd workers, about ten per sentence.
- **A short swap list.** Seven kinds of pair, and they are not perfect counterparts in real use.
- **The paper's grouping.** Self-reported ideology on a −10 to 10 scale, cut at −3 and 3. Groups are small per sentence, which is why section 3 matches their sizes.
- **One run.** One seed for every model, so small differences, such as the per-pair INT8 shifts, should be confirmed with more seeds before drawing conclusions.
