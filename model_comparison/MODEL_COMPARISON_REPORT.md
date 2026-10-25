# Model Comparison Report: Top 10 Models from Research Paper

> **Benchmark Reference**: *Automated identification of bias inducing words in news articles using linguistic and context-oriented features* (Spinde et al., Elsevier 2021)

Evaluated on identical held-out test split of **3,114 words** (302 biased words, 2,812 non-biased words) with fixed `random_state=42`.

## Comprehensive Performance & Confusion Matrix Table

| Rank | Model                              | Accuracy | Correct Words | Failed Words | TP  | TN   | FP  | FN  | F1     | Precision | Recall | ROC AUC |
| -----|------------------------------------|----------|---------------|--------------|-----|------|-----|-----|--------|-----------|--------|-------- |
| 1    | Random Forest (Weighted)           | 91.43%   | 2847 (91.43%) | 267 (8.57%)  | 115 | 2732 | 80  | 187 | 0.4628 | 0.5897    | 0.3808 | 0.8781  |
| 2    | Multilayer Perceptron (MLP)        | 91.36%   | 2845 (91.36%) | 269 (8.64%)  | 75  | 2770 | 42  | 227 | 0.3580 | 0.6410    | 0.2483 | 0.8456  |
| 3    | Linear Discriminant Analysis (LDA) | 90.75%   | 2826 (90.75%) | 288 (9.25%)  | 113 | 2713 | 99  | 189 | 0.4397 | 0.5330    | 0.3742 | 0.8405  |
| 4    | B4: Semi-Automated Bias Lexicon    | 90.75%   | 2826 (90.75%) | 288 (9.25%)  | 36  | 2790 | 22  | 266 | 0.2000 | 0.6207    | 0.1192 | 0.5557  |
| 5    | SVM (RBF Kernel, Weighted)         | 86.83%   | 2704 (86.83%) | 410 (13.17%) | 120 | 2584 | 228 | 182 | 0.3692 | 0.3448    | 0.3974 | 0.7875  |
| 6    | B2: Negative Sentiment Lexicon     | 86.45%   | 2692 (86.45%) | 422 (13.55%) | 142 | 2550 | 262 | 160 | 0.4023 | 0.3515    | 0.4702 | 0.6885  |
| 7    | XGBoost (Fine-Tuned & Weighted)    | 84.3%    | 2625 (84.3%)  | 489 (15.7%)  | 211 | 2414 | 398 | 91  | 0.4632 | 0.3465    | 0.6987 | 0.8766  |
| 8    | Linear SVM (Weighted)              | 78.39%   | 2441 (78.39%) | 673 (21.61%) | 217 | 2224 | 588 | 85  | 0.3921 | 0.2696    | 0.7185 | 0.8491  |
| 9    | Logistic Regression (Weighted)     | 78.1%    | 2432 (78.1%)  | 682 (21.9%)  | 216 | 2216 | 596 | 86  | 0.3878 | 0.2660    | 0.7152 | 0.8473  |
| 10   | Complement Naive Bayes             | 72.96%   | 2272 (72.96%) | 842 (27.04%) | 226 | 2046 | 766 | 76  | 0.3493 | 0.2278    | 0.7483 | 0.8429  |

---

## Deep-Dive: Word Counts & Error Breakdown

| Model | Correct Words Count | Failed Words Count | Sample Correctly Predicted Biased Words | Sample False Alarms (FP Words) | Sample Missed Bias (FN Words) |
| :--- | :---: | :---: | :--- | :--- | :--- |
| **Random Forest (Weighted)** | **2847** (91.43%) | **267** (8.57%) | *rhetoric*, *racist*, *disrupted*, *dangerous*, *corrupt* | *endless*, *propaganda*, *vulnerable*, *criticized*, *lack* | *conspiracy*, *good*, *criticism*, *pomp*, *crisis* |
| **Multilayer Perceptron (MLP)** | **2845** (91.36%) | **269** (8.64%) | *rhetoric*, *racist*, *corrupt*, *predatory*, *drastic* | *endless*, *vulnerable*, *criticized*, *undermine*, *Unfortunately* | *white*, *conspiracy*, *good*, *disrupted*, *dangerous* |
| **Linear Discriminant Analysis (LDA)** | **2826** (90.75%) | **288** (9.25%) | *rhetoric*, *racist*, *disrupted*, *corrupt*, *dramatic* | *endless*, *black*, *ignoring*, *long*, *propaganda* | *white*, *conspiracy*, *good*, *pomp*, *crisis* |
| **B4: Semi-Automated Bias Lexicon** | **2826** (90.75%) | **288** (9.25%) | *rhetoric*, *racist*, *corrupt*, *intolerant*, *ideology* | *ignoring*, *propaganda*, *vulnerable*, *criticized*, *undermine* | *white*, *conspiracy*, *good*, *disrupted*, *dangerous* |
| **SVM (RBF Kernel, Weighted)** | **2704** (86.83%) | **410** (13.17%) | *rhetoric*, *racist*, *corrupt*, *drastic*, *false* | *endless*, *claim*, *emergency*, *voting*, *propaganda* | *good*, *dangerous*, *pomp*, *white*, *predatory* |
| **B2: Negative Sentiment Lexicon** | **2692** (86.45%) | **422** (13.55%) | *rhetoric*, *conspiracy*, *racist*, *disrupted*, *dangerous* | *outbreak*, *trying*, *crisis*, *endless*, *black* | *white*, *good*, *pomp*, *dramatic*, *rights* |
| **XGBoost (Fine-Tuned & Weighted)** | **2625** (84.3%) | **489** (15.7%) | *white*, *rhetoric*, *racist*, *good*, *disrupted* | *white*, *endless*, *federal*, *progressive*, *media* | *pomp*, *longstanding*, *rule*, *domestic*, *claimed* |
| **Linear SVM (Weighted)** | **2441** (78.39%) | **673** (21.61%) | *white*, *rhetoric*, *racist*, *good*, *disrupted* | *abortion*, *white*, *presidential*, *political*, *federal* | *pomp*, *rule*, *domestic*, *prosecutors*, *concocting* |
| **Logistic Regression (Weighted)** | **2432** (78.1%) | **682** (21.9%) | *white*, *rhetoric*, *racist*, *good*, *disrupted* | *abortion*, *white*, *presidential*, *political*, *federal* | *pomp*, *rule*, *domestic*, *prosecutors*, *concocting* |
| **Complement Naive Bayes** | **2272** (72.96%) | **842** (27.04%) | *white*, *rhetoric*, *racist*, *good*, *disrupted* | *presidential*, *federal*, *white*, *abortion*, *political* | *pomp*, *conspiracy*, *rule*, *domestic*, *overstepping* |


---

## Key Empirical Observations

1. **The Accuracy Paradox in Imbalanced Word Data**:
   - Models like *Random Forest* (91.43%) and *Multilayer Perceptron* (91.36%) achieve high accuracy by playing conservative—they predict very few positive words, resulting in low recall (0.24 - 0.38).
   - Conversely, the paper's champion *XGBoost (Fine-Tuned & Weighted)* achieves the highest balanced F1 (0.4632) and high recall (0.6987), but sacrifices raw accuracy (84.30%) due to 398 False Positives.
2. **Lexicon Baselines vs. ML Classifiers**:
   - The *Negative Sentiment Lexicon (B2)* achieves 86.45% accuracy and F1 of 0.4023, confirming that word bias heavily correlates with negative affective words.
   - The *Semi-Automated Bias Lexicon (B4)* has high precision (62.07%) and 90.75% accuracy, but misses 88.08% of biased words (only 36 detected out of 302) because static lexica cannot adapt to dynamic linguistic context.
3. **The Need for Optimization**:
   - To build the ultimate model, we must bridge this divide: achieving the high recall and F1 of XGBoost while eliminating false alarms to attain the **highest accuracy (>91.5%+)** of any model.
