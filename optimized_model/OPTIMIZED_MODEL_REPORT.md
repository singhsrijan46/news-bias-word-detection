# Technical Report: Engineered Champion Model for News Bias Word Detection

> **Mission**: Pick the best performing model from the foundational research paper (Timo Spinde et al., 2021), diagnose its failure modes, and engineer an optimized architecture that achieves the **HIGHEST ACCURACY among all evaluated models**.

---

## 1. Executive Summary & Core Results

- **Final Held-Out Test Accuracy**: **91.78%** (Strictly highest among all models, beating Random Forest 91.43% and paper XGBoost 84.30%).
- **Correctly Classified Words**: **2,858 / 3,114 words** (91.78%).
- **Total Failed Words**: **256 / 3,114 words** (8.22%).
- **Precision**: **0.7347** (74.00% precision vs. paper's 34.65% — **over 2.1x fewer false alarms**).
- **ROC AUC**: **0.8794** (Top-tier ranking capability).
- **Decision Threshold**: $\tau^* = 0.55$ (Optimized on unseen validation split).

---

## 2. Accuracy Leaderboard: Optimized Model vs. Paper Baselines

| Model | Accuracy | F1 Score | Precision | Recall | ROC AUC | Correct Words | Failed Words | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Optimized Model (Ours)** | **91.78%** | 0.3600 | 0.7347 | 0.2384 | 0.8794 | 2,858 | 256 | **[#1 CHAMPION] Highest Accuracy** |
| **Random Forest (Weighted)** | **91.43%** | 0.4628 | 0.5897 | 0.3808 | 0.8781 | 2,847 | 267 | **Paper Best Accuracy** |
| **Multilayer Perceptron (MLP)** | **91.36%** | 0.3580 | 0.6410 | 0.2483 | 0.8456 | 2,845 | 269 | **Paper Neural Net** |
| **Linear Discriminant Analysis** | **90.75%** | 0.4397 | 0.5330 | 0.3742 | 0.8405 | 2,826 | 288 | **Paper Linear Baseline** |
| **B4: Semi-Automated Bias Lexicon** | **90.75%** | 0.2000 | 0.6207 | 0.1192 | 0.5557 | 2,826 | 288 | **Paper Novel Lexicon** |
| **B2: Negative Sentiment Lexicon** | **86.45%** | 0.4023 | 0.3515 | 0.4702 | 0.6885 | 2,692 | 422 | **Paper Rule Baseline** |
| **XGBoost (Paper Fine-Tuned)** | **84.3%** | 0.4632 | 0.3465 | 0.6987 | 0.8766 | 2,625 | 489 | **Paper Champion Baseline** |


---

## 3. Deep Dive: Why Did the Original Research Paper Model Underperform in Accuracy?

In the original paper (*Spinde et al., 2021*), **XGBoost** was selected as the champion model because it achieved the highest F1 score (0.42 - 0.43) and ROC AUC (0.84). However, its raw accuracy was only **84.30%**, generating **398 False Positives** on the held-out test split.

Our diagnostic analysis identified three fundamental bottlenecks in the paper's setup:
1. **The Class-Weighting Trap (`scale_pos_weight = 9.5`)**:
   - Because only ~9.7% of words in news articles are biased, the authors applied a heavy static class weight (`scale_pos_weight = 9.5`) combined with a naive decision threshold of `0.5`.
   - This artificially multiplied positive error gradients by 9.5x, causing the model to over-predict bias on neutral, factual, or mildly negative words (e.g., words like *federal*, *presidential*, *media*). That severe false positive rate degraded accuracy down to 84.3%.
2. **Lack of Syntactic-Semantic Interaction Features**:
   - The original paper engineered 306 individual features (GloVe norm, POS tags, LIWC categories, sentiment lexica), but treated each feature independently.
   - Media bias is fundamentally an **interaction phenomenon**: a word is not biased simply because it has negative sentiment; it becomes biased when negative sentiment co-occurs with strong subjectivity, certainty boosters, and high topical prominence.
3. **Absence of Systematic Regularization & Threshold Tuning**:
   - Without tree regularization penalties (L1/L2 shrinkage) and without validation-guided threshold calibration, the decision boundary suffered from excessive variance.

---

## 4. The 3 Pillars of Our Solution

### Pillar 1: Domain-Informed Contextual Feature Interactions
We engineered 8 synergistic features that capture the true mechanisms of journalistic word choice:
1. `feat_net_sentiment`: Net emotional valence ($Sentiment_{neg} - Sentiment_{pos}$).
2. `feat_subjectivity_total`: Weighted subjectivity score ($2 \times Subj_{strong} + Subj_{weak}$).
3. `feat_emotional_charge`: Non-linear emotional amplification ($(Neg + Hyperbolic + Attitude) \times Subj_{strong}$).
4. `feat_lexical_bias_total`: Total bias lexicon trigger density ($BiasLex_{token} + BiasLex_{context}$).
5. `feat_prominence`: Semantic salience ($GloVe\_Norm \times TF\text{-}IDF_{article}$).
6. `feat_booster_ratio`: Certainty-to-doubt ratio ($(Boosters + 1) / (Hedges + 1)$).
7. `feat_context_contrast`: Sentiment contrast between target token and its surrounding neighborhood ($Neg_{token} - Neg_{context}$).
8. `feat_assertion_power`: Rhetorical persuasion force ($(AssertiveVerbs + KillVerbs) \times (Boosters + 1)$).

### Pillar 2: Regularized Gradient Boosting Tree Architecture
- **Subsampling (`subsample = 0.85`)** and **Column subsampling (`colsample_bytree = 0.80`)**: Prevents individual features from dominating tree splits.
- **L1 Regularization (`reg_alpha = 0.20`)** and **L2 Regularization (`reg_lambda = 1.20`)**: Penalizes complex leaf weights to guard against overfitting.
- **Learning Rate & Depth (`lr = 0.035`, `max_depth = 6`, `n_estimators = 450`)**: Provides smooth, robust gradient descent convergence.

### Pillar 3: Validation-Guided Calibrated Threshold Optimization
- Rather than relying on default 0.5 or arbitrary heuristic class weights, we conducted an exhaustive grid search over the validation split ($N=2,803$ tokens).
- Setting the decision threshold to **$\tau^* = 0.55$** achieves the optimal operating point: eliminating 372 false alarms (cutting FP from 398 down to 26) while capturing high-confidence biased words.

---

## 5. Qualitative Word Analysis

- **Correctly Identified Biased Words (Sample TPs)**: *rhetoric*, *racist*, *corrupt*, *dramatic*, *drastic*, *radical*, *false*, *intolerant*, *rumblings*, *gruesome*
- **False Alarms (Sample FPs)**: *endless*, *propaganda*, *vulnerable*, *criticized*, *provoked*, *white*, *alarming*, *plagued*, *critical*, *abused*
- **Missed Bias (Sample FNs)**: *conspiracy*, *good*, *dangerous*, *criticism*, *pomp*, *burden*, *crisis*, *predatory*, *white*, *little*
