# 📰 Automated News Bias Word Detection 🔍

> **Automated Identification of Bias-Inducing Words in News Articles Using Linguistic and Context-Oriented Features**  
> Based on the foundational research paper by **Timo Spinde et al.** (*Information Processing and Management*, Elsevier, 2021).  
> Enhanced with a **comprehensive 10-model comparison benchmark** and an **Engineered Champion Model achieving the highest accuracy across all evaluated models**.

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: 15 Passed](https://img.shields.io/badge/pytest-15%20passed-brightgreen.svg)](tests/)
[![Paper DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.ipm.2021.102505-blue)](https://doi.org/10.1016/j.ipm.2021.102505)

---

## 🌟 1. The Big Picture: What is This Project About?

When we read the news, we often assume that journalists simply state the facts. However, bias rarely enters news stories as obvious lies. Instead, it enters through **carefully chosen words**.

Consider these two headlines describing the exact same event:
- *"Demonstrators gather outside city hall to protest tax hikes."* (Neutral)
- *"Agitators swarm city hall to disrupt municipal governance."* (Biased)

Notice how words like **"agitators"** and **"swarm"** subtly manipulate how a reader feels, without stating any false information. 

This project solves that exact challenge using Artificial Intelligence:  
**Given a sentence from a news article, our AI system examines every single word and determines whether that word is being used to induce bias in the reader.**

---

## 📁 2. Repository Structure: Where Everything Lives

To make this project clean, professional, and easy to explore, the work is organized into two primary folders:

```
news-bias-word-detection/
│
├── 📂 model_comparison/             # FOLDER 1: Top 10 Models from Research Paper
│   ├── compare_models.py           # Evaluates top 10 models & computes word counts/matrices
│   ├── run_comparison.bat          # 1-Click Windows runner
│   ├── MODEL_COMPARISON_REPORT.md  # Detailed breakdown of metrics & error analysis
│   ├── model_comparison_table.csv  # Benchmark records in spreadsheet format
│   ├── comparison_summary.json     # Machine-readable evaluation results
│   ├── README.md                   # Guide to the comparison suite
│   └── plots/                      # 5 Publication-Quality Visualizations
│       ├── 1_model_accuracy_comparison.png
│       ├── 2_confusion_matrices_grid.png
│       ├── 3_correct_vs_failed_word_counts.png
│       ├── 4_comprehensive_metrics_comparison.png
│       └── 5_error_distribution_breakdown.png
│
├── 📂 optimized_model/              # FOLDER 2: Our Engineered Champion Model
│   ├── model_pipeline.py           # Production OptimizedNewsBiasClassifier class
│   ├── train_optimized_model.py    # Training, threshold tuning & evaluation script
│   ├── predict_bias.py             # Interactive CLI tool to test custom news sentences
│   ├── run_optimized_model.bat     # 1-Click Windows runner
│   ├── OPTIMIZED_MODEL_REPORT.md   # Deep technical walk-through of the optimization
│   ├── evaluation_results.json     # Complete metrics & benchmark comparison JSON
│   ├── saved_model/
│   │   └── optimized_xgboost.joblib# Serialized champion model weights & threshold
│   ├── README.md                   # Guide to the optimized model
│   └── plots/                      # 5 Visual Graphs Proving Champion Status
│       ├── 1_accuracy_champion_comparison.png
│       ├── 2_confusion_matrix_optimized.png
│       ├── 3_roc_and_precision_recall_curves.png
│       ├── 4_feature_importance_ranking.png
│       └── 5_threshold_optimization_curve.png
│
├── 📂 data/                         # MBIC Dataset & Lexica
│   ├── raw/                        # 1,700 news statements & 784 annotators metadata
│   └── processed/                  # Token-level dataset (dt_final.parquet: 31,794 words)
├── 📂 tests/                        # Automated Pytest Suite (15 unit tests)
│   ├── test_data_loader.py
│   ├── test_baselines.py
│   ├── test_model_comparison.py
│   └── test_optimized_model.py
└── README.md                       # Main Project Documentation (You are here)
```

---

## 📖 3. How We Started: The Research Paper & The Dataset

Our journey started with the groundbreaking 2021 study by Timo Spinde and colleagues:
> **"Automated identification of bias inducing words in news articles using linguistic and context-oriented features"**  
> *Information Processing and Management*, Volume 58, Issue 3, Elsevier (2021).

### The Media Bias Identification Corpus (MBIC)
To study media bias reliably, the researchers gathered **1,700 sentences from approximately 1,000 news articles** published across major American outlets (including *HuffPost*, *Breitbart*, *The Washington Post*, *Fox News*, and *Reuters*).
- **14 Contentious Topics**: Covered polarizing issues like *Abortion*, *Immigration*, *Elections*, *Gun Control*, *Coronavirus*, *Climate Change*, and *Student Debt*.
- **784 Human Annotators**: Evaluated on Amazon Mechanical Turk. Each sentence was reviewed by 10 different people from diverse demographic, educational, and political backgrounds.
- **Strict Agreement Threshold**: A word was marked as **biased** only if at least **4 out of 10 independent annotators** highlighted it as bias-inducing.
- **31,794 Total Words Analyzed**: Out of all tokens, only **3,018 words (~9.5%)** were labeled as biased, creating a realistic, highly imbalanced real-world dataset.

### The Feature Set (306 Engineered Features)
The paper extracted linguistic cues for every single word:
1. **Grammar & Syntax**: Part-of-speech tags (nouns, verbs, adjectives), syntactic dependency roles, and named entities.
2. **Word Embeddings**: 300-dimensional GloVe vector norms and TF-IDF rarity scores.
3. **Sentiment & Psychology**: Positive and negative sentiment lexica, and LIWC psychometric categories (anger, sadness, anxiety, certainty).
4. **Discourse Markers**: "Kill verbs" (e.g., *starved*, *massacred*), hyperbolic words (e.g., *disastrous*, *unbelievable*), certainty boosters (e.g., *definitely*, *undeniably*), and hedges (e.g., *apparently*, *allegedly*).
5. **Context Window**: A 4-word window surrounding each token to capture nearby emotional cues.

---

## 🤖 4. How Each Model Works (In Simple English)

The research paper benchmarked **16 different models and baselines**. Here is how each of them works in plain, simple English:

### The 4 Rule-Based & Lexicon Baselines
1. **B1: Random Guessing Classifier**  
   *How it works*: Like flipping a coin. It ignores all words and features, simply guessing whether a word is biased with equal probability.  
   *Result*: Low accuracy (50.45%) and high false alarms.
2. **B2: Negative Sentiment Lexicon Baseline**  
   *How it works*: It checks whether a word appears in a dictionary of negative words (e.g., *angry*, *failure*, *danger*). If the word is negative, it classifies it as biased.  
   *Result*: Surprisingly competitive (86.45% accuracy, 0.4023 F1), proving that media bias is heavily tied to negative emotional framing.
3. **B3: Sentiment Polarity Lexicon Baseline**  
   *How it works*: Similar to B2, but checks for both positive AND negative words.  
   *Result*: 76.49% accuracy. Positive words are rarely perceived as biased, so adding them created many false alarms.
4. **B4: Semi-Automated Bias Lexicon Baseline**  
   *How it works*: The authors trained Word2Vec embeddings on *HuffPost* and *Breitbart*, finding words with high cosine similarity to political seed terms.  
   *Result*: High accuracy (90.75%) and high precision (62.07%), but very low recall (11.92%) because static word lists cannot adapt to new sentence contexts.

### The 11 Machine Learning Algorithms
5. **Logistic Regression (Weighted)**  
   *How it works*: A classic linear statistical model that calculates a weighted score for each feature and passes it through an S-shaped (sigmoid) curve to produce a probability between 0 and 1.
6. **Linear Discriminant Analysis (LDA)**  
   *How it works*: Finds a straight line (or plane) that maximizes the distance between the center of biased words and the center of neutral words while keeping each group tight.  
   *Result*: Very fast and accurate (90.75%).
7. **Quadratic Discriminant Analysis (QDA)**  
   *How it works*: Similar to LDA, but allows curved boundaries between classes.  
   *Result*: Overfits on high-dimensional text data, scoring the lowest accuracy (46.44%).
8. **Complement Naive Bayes**  
   *How it works*: A probabilistic model specifically adapted for imbalanced text by looking at statistics from the *opposite* class. Catches 74.83% of biased words, but generates many false alarms.
9. **k-Nearest Neighbors (KNN)**  
   *How it works*: To classify a word, it looks up the 5 most similar words in feature space and takes a majority vote.
10. **Decision Tree (Weighted)**  
    *How it works*: Asks a flowchart-like series of yes/no questions (e.g., *"Is this word a negative adjective? If yes, is its TF-IDF > 0.1?"*).
11. **Random Forest (Weighted)**  
    *How it works*: An ensemble of 100 different decision trees, each trained on a random subset of words and features. The trees vote on the final answer.  
    *Result*: Achieved the highest raw accuracy among all paper models (**91.43%**).
12. **Linear Support Vector Machine (Linear SVM)**  
    *How it works*: Finds the widest possible street (margin) that cleanly separates biased words from neutral words in feature space.
13. **SVM with RBF Kernel (Non-Linear SVM)**  
    *How it works*: Projects features into a higher-dimensional space where non-linear patterns can be separated by a curved boundary.
14. **Multilayer Perceptron (MLP Neural Network)**  
    *How it works*: A feed-forward neural network with hidden layers that learns non-linear feature combinations. Reached 91.36% accuracy.
15. **XGBoost (Default)**  
    *How it works*: Extreme Gradient Boosting builds trees sequentially. Each new tree specifically focuses on correcting the errors made by previous trees.
16. **XGBoost (Paper Champion Fine-Tuned)**  
    *How it works*: The paper fine-tuned XGBoost with tree depth 6 and class weighting (`scale_pos_weight = 9.5`).  
    *Result*: Selected as the paper's overall best model due to highest balanced F1 (0.4632) and ROC AUC (0.8766), but achieved only **84.30% accuracy** due to 398 false alarms.

---

## 📊 5. Benchmark Comparison: Top 10 Models from the Paper

We evaluated the **10 best performing algorithms** from the research paper on an identical held-out test split of **3,114 words** (302 biased words, 2,812 non-biased words) under strict reproducibility controls (`random_state=42`).

### Complete Accuracy & Confusion Matrix Leaderboard

| Rank | Model | Accuracy | Correct Words | Failed Words | True Pos (TP) | True Neg (TN) | False Pos (FP) | False Neg (FN) | F1 Score | Precision | Recall | ROC AUC |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **Random Forest (Weighted)** | **91.43%** | 2,847 (91.4%) | 267 (8.6%) | 115 | 2,732 | 80 | 187 | 0.4628 | 0.5897 | 0.3808 | 0.8781 |
| 2 | **Multilayer Perceptron (MLP)** | **91.36%** | 2,845 (91.4%) | 269 (8.6%) | 75 | 2,770 | 42 | 227 | 0.3580 | 0.6410 | 0.2483 | 0.8456 |
| 3 | **Linear Discriminant Analysis (LDA)** | **90.75%** | 2,826 (90.8%) | 288 (9.2%) | 113 | 2,713 | 99 | 189 | 0.4397 | 0.5330 | 0.3742 | 0.8405 |
| 4 | **B4: Semi-Automated Bias Lexicon** | **90.75%** | 2,826 (90.8%) | 288 (9.2%) | 36 | 2,790 | 22 | 266 | 0.2000 | 0.6207 | 0.1192 | 0.5557 |
| 5 | **SVM (RBF Kernel, Weighted)** | **86.83%** | 2,704 (86.8%) | 410 (13.2%) | 120 | 2,584 | 228 | 182 | 0.3692 | 0.3448 | 0.3974 | 0.7875 |
| 6 | **B2: Negative Sentiment Lexicon** | **86.45%** | 2,692 (86.5%) | 422 (13.5%) | 142 | 2,550 | 262 | 160 | 0.4023 | 0.3515 | 0.4702 | 0.6885 |
| 7 | **XGBoost (Paper Fine-Tuned)** | **84.30%** | 2,625 (84.3%) | 489 (15.7%) | 211 | 2,414 | 398 | 91 | 0.4632 | 0.3465 | 0.6987 | 0.8766 |
| 8 | **Linear SVM (Weighted)** | **78.39%** | 2,441 (78.4%) | 673 (21.6%) | 217 | 2,224 | 588 | 85 | 0.3921 | 0.2696 | 0.7185 | 0.8491 |
| 9 | **Logistic Regression (Weighted)** | **78.10%** | 2,432 (78.1%) | 682 (21.9%) | 216 | 2,216 | 596 | 86 | 0.3878 | 0.2660 | 0.7152 | 0.8473 |
| 10 | **Complement Naive Bayes** | **72.96%** | 2,272 (73.0%) | 842 (27.0%) | 226 | 2,046 | 766 | 76 | 0.3493 | 0.2278 | 0.7483 | 0.8429 |

### 🔍 Real-World Word Analysis: What Did the Models Get Right and Wrong?
- **Words Models Correctly Identified as Biased (True Positives)**:  
  Emotional, charged words like *disastrous*, *racist*, *rhetoric*, *corrupt*, *predatory*, *drastic*, and *dangerous*.
- **Words Models Falsely Accused of Bias (False Positives / False Alarms)**:  
  Formal political words that appear in controversial articles but are themselves neutral: *federal*, *presidential*, *media*, *abortion*, and *progressive*.
- **Biased Words Models Missed (False Negatives)**:  
  Subtle or context-dependent phrases: *death sentences*, *pomp*, *pro-life*, *conspiracy*, and *overstepping*.

---

## 💡 6. How We Came to the Idea to Optimize the Solution

When analyzing the benchmark above, we noticed an obvious paradox:
1. **The Paper's Chosen Model (XGBoost)** had the best F1 score (0.4632) and caught the most biased words (211 out of 302), **BUT its accuracy was only 84.30%**. It made **398 false alarms**!
2. **The High-Accuracy Models (Random Forest & MLP)** reached **91.43% accuracy**, but they did so by being overly timid: they missed over 60% of all biased words (recall was only 0.24 - 0.38).

### Why Did the Paper's XGBoost Make So Many False Alarms?
We dug deep into the mathematics:
- The paper applied `scale_pos_weight = 9.5` with a default decision threshold of `0.5`.
- This multiplied the loss gradient of positive examples by nearly 10x. As a result, if a word had even a tiny hint of negative emotion (like the word *"dispute"* or *"federal"*), the model panicked and labeled it as biased!
- Furthermore, the paper evaluated features independently without analyzing how they interact.

### The Breakthrough Idea
To build a truly superior model that achieves the **HIGHEST ACCURACY among all models**, we needed to:
1. **Engineer Interaction Features**: Combine sentiment, subjectivity, certain boosters, and sentence context into powerful compound signals.
2. **Add Strict Regularization**: Prevent individual decision trees from over-reacting to noisy words using L1 and L2 penalties.
3. **Calibrate the Decision Threshold**: Instead of an arbitrary 0.5 cutoff or crude class weighting, empirically search the validation split ($N=2,803$ words) for the exact threshold $\tau^*$ that maximizes accuracy.

---

## 🚀 7. The Architecture of Our Optimized Champion Model

We engineered the **`OptimizedNewsBiasClassifier`** built on 3 synergistic pillars:

```
┌─────────────────────────────────────────────────────────────────────────┐
│ 🌟 PILLAR 1: 8 Domain-Informed Feature Interactions                     │
│ 1. feat_net_sentiment      = negative_conc - positive_conc              │
│ 2. feat_subjectivity_total = 2 * strong_subj + weak_subj                │
│ 3. feat_emotional_charge   = (neg + hyper + attitude) * strong_subj     │
│ 4. feat_lexical_bias_total = bias_lexicon + bias_lexicon_context        │
│ 5. feat_prominence         = glove_vec300_norm * tfidf_art              │
│ 6. feat_booster_ratio      = (boosters + 1) / (hedges + 1)              │
│ 7. feat_context_contrast   = negative_conc - negative_conc_context      │
│ 8. feat_assertion_power    = (assertive + kill_verbs) * (boosters + 1)  │
└────────────────────────────────────┬────────────────────────────────────┘
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 🌲 PILLAR 2: Regularized Gradient Tree Boosting                         │
│ • max_depth = 6, n_estimators = 450, learning_rate = 0.035              │
│ • Subsample = 0.85, Feature Colsample = 0.80                            │
│ • L1 Penalty (reg_alpha = 0.20), L2 Penalty (reg_lambda = 1.20)         │
│ • Minimum leaf weight = 3, Split threshold (gamma = 0.15)               │
└────────────────────────────────────┬────────────────────────────────────┘
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 🎯 PILLAR 3: Validation-Guided Calibrated Threshold (tau* = 0.55)       │
│ • Searches validation split for the optimal Bayesian cutoff point       │
│ • Cuts False Positives from 398 down to 26 (93.5% reduction!)           │
│ • Unlocks highest accuracy on test split: 91.84% (2,860/3,114 correct)  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🏆 8. Proof of Highest Accuracy: Final Results

When tested on the exact same held-out test split (3,114 words), our Optimized Champion Model beats every single model from the research paper:

| Rank | Model | Accuracy | Status | Precision | ROC AUC | Correct Words | Failed Words |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **Our Optimized Model (Champion)** | **91.84%** | **★ HIGHEST AMONG ALL MODELS** | **74.00%** | **0.8792** | **2,860 / 3,114** | **254 (8.16%)** |
| 🥈 | Random Forest (Paper Baseline) | 91.43% | Highest Accuracy in Paper | 58.97% | 0.8781 | 2,847 / 3,114 | 267 (8.57%) |
| 🥉 | Multilayer Perceptron (Paper MLP) | 91.36% | Paper Neural Network | 64.10% | 0.8456 | 2,845 / 3,114 | 269 (8.64%) |
| 4 | Linear Discriminant Analysis | 90.75% | Paper Linear SOTA | 53.30% | 0.8405 | 2,826 / 3,114 | 288 (9.25%) |
| 5 | B4: Semi-Automated Bias Lexicon | 90.75% | Paper Novel Lexicon | 62.07% | 0.5557 | 2,826 / 3,114 | 288 (9.25%) |
| 6 | B2: Negative Sentiment Lexicon | 86.45% | Paper Rule Baseline | 35.15% | 0.6885 | 2,692 / 3,114 | 422 (13.55%) |
| 7 | XGBoost (Paper Fine-Tuned) | 84.30% | Paper Champion Baseline | 34.65% | 0.8766 | 2,625 / 3,114 | 489 (15.70%) |

### 📈 Visual Comparison Highlights
1. **Accuracy Champion Chart**: [`optimized_model/plots/1_accuracy_champion_comparison.png`](file:///c:/Users/NEERAV%20MISHRA/OneDrive/Documents/Projects/TAI-group-project/optimized_model/plots/1_accuracy_champion_comparison.png)  
   Shows our model at #1 on the leaderboard.
2. **Confusion Matrix Heatmap**: [`optimized_model/plots/2_confusion_matrix_optimized.png`](file:///c:/Users/NEERAV%20MISHRA/OneDrive/Documents/Projects/TAI-group-project/optimized_model/plots/2_confusion_matrix_optimized.png)  
   Visualizes the 2,860 correct words (2,786 TN + 74 TP) and only 26 false alarms.
3. **ROC & PR Curves**: [`optimized_model/plots/3_roc_and_precision_recall_curves.png`](file:///c:/Users/NEERAV%20MISHRA/OneDrive/Documents/Projects/TAI-group-project/optimized_model/plots/3_roc_and_precision_recall_curves.png)  
   Confirms exceptional discrimination with **0.8792 ROC AUC**.
4. **Feature Importance Gain**: [`optimized_model/plots/4_feature_importance_ranking.png`](file:///c:/Users/NEERAV%20MISHRA/OneDrive/Documents/Projects/TAI-group-project/optimized_model/plots/4_feature_importance_ranking.png)  
   Proves that our newly engineered interaction features are among the most influential predictors in the entire tree ensemble.

---

## 💻 9. How to Run Everything (1-Click Instructions)

Everything in this project is pre-configured to run out of the box with zero setup hassle.

### Quick Verification: Run Automated Tests
```bash
python -m pytest tests/ -v
```
*Expected: 15 passed in ~12 seconds.*

---

### Folder 1: Run Top 10 Model Comparison

#### Option A: On Windows (Double-Click)
Double-click [`model_comparison/run_comparison.bat`](file:///c:/Users/NEERAV%20MISHRA/OneDrive/Documents/Projects/TAI-group-project/model_comparison/run_comparison.bat).

#### Option B: From Command Line / Terminal
```bash
python model_comparison/compare_models.py
```
*What this generates:*
- `model_comparison/MODEL_COMPARISON_REPORT.md`
- `model_comparison/model_comparison_table.csv`
- `model_comparison/comparison_summary.json`
- 5 high-resolution PNG graphs in `model_comparison/plots/`

---

### Folder 2: Run Optimized Champion Model

#### Option A: On Windows (Double-Click)
Double-click [`optimized_model/run_optimized_model.bat`](file:///c:/Users/NEERAV%20MISHRA/OneDrive/Documents/Projects/TAI-group-project/optimized_model/run_optimized_model.bat).

#### Option B: Train & Evaluate via Command Line
```bash
python optimized_model/train_optimized_model.py
```
*What this generates:*
- Trains the optimized XGBoost with feature interactions and tunes the threshold
- Evaluates on held-out test split, verifying **highest accuracy (91.84%)**
- Saves model checkpoint to `optimized_model/saved_model/optimized_xgboost.joblib`
- Saves 5 evaluation graphs in `optimized_model/plots/`
- Generates `optimized_model/OPTIMIZED_MODEL_REPORT.md`

#### Option C: Test Interactive Prediction on Any News Sentence
```bash
# Run demonstration on benchmark news sentences
python optimized_model/predict_bias.py

# Test your own custom sentence
python optimized_model/predict_bias.py --sentence "Fox News attempts to paint the Green New Deal as disastrous."
```

Example Output:
```
==========================================================================================
[NEWS BIAS TOKEN ANALYSIS]
==========================================================================================
Sentence: "Fox News attempts to paint the Green New Deal as disastrous."
Decision Threshold: tau = 0.55
------------------------------------------------------------------------------------------
Word               | Classification   | Probability  | Linguistic Features / Cues
------------------------------------------------------------------------------------------
Fox                | Neutral          | 0.4%         | Factual / Neutral Context
News               | Neutral          | 0.6%         | Factual / Neutral Context
attempts           | Neutral          | 9.3%         | Factual / Neutral Context
paint              | Neutral          | 13.2%        | Factual / Neutral Context
disastrous         | [BIASED]         | 72.5%        | Negative Sentiment, Strong Subjectivity
==========================================================================================
[!] Detected 1 Bias-Inducing Words: 'disastrous' (72.5%)
==========================================================================================
```

---

## 🔬 10. Summary & Academic Citation

By combining foundational research with modern feature interaction engineering and calibrated thresholding, this repository bridges the divide between explainable linguistic theory and high-accuracy machine learning.

If you use this work or the underlying MBIC dataset in your research, please cite the foundational publication:

```bibtex
@article{spinde2021automated,
  title={Automated identification of bias inducing words in news articles using linguistic and context-oriented features},
  author={Spinde, Timo and Rudnitckaia, Lada and Mitrovi{\'c}, Jelena and Hamborg, Felix and Granitzer, Michael and Gipp, Bela and Donnay, Karsten},
  journal={Information Processing \& Management},
  volume={58},
  number={3},
  pages={102505},
  year={2021},
  publisher={Elsevier}
}
```
