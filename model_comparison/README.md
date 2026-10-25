# 📊 Model Comparison Suite: Spinde et al. (2021) Research Paper

> Comprehensive empirical benchmark and comparative analysis of the **top 10 models** from the research paper:  
> **"Automated identification of bias inducing words in news articles using linguistic and context-oriented features"**  
> (Timo Spinde, Lada Rudnitckaia, Jelena Mitrović, Felix Hamborg, Michael Granitzer, Bela Gipp, Karsten Donnay - *Elsevier IPM, 2021*)

---

## 🎯 Purpose of this Folder

This folder provides a self-contained, reproducible evaluation comparing the **10 best performing algorithms** identified in the research paper. It details:
1. **Model Accuracy & Metrics**: Accuracy, F1-Score, Precision, Recall, and ROC AUC.
2. **True Positive (TP), True Negative (TN), False Positive (FP), False Negative (FN)**: Full confusion matrix records.
3. **Word Counts**:
   - Total correctly predicted words ($TP + TN$) and percentage.
   - Total failed / misclassified words ($FP + FN$) and percentage.
   - Specific qualitative word breakdowns (words correctly flagged as biased, false alarms, and missed bias).
4. **Visual Comparison Graphs**: High-resolution publication plots.

---

## 🏆 Top 10 Models Benchmarked

| Model | Category | Paper Reference |
| :--- | :--- | :--- |
| **XGBoost (Fine-Tuned & Weighted)** | Gradient Boosted Decision Trees | Table 7 (Paper Champion) |
| **Random Forest (Weighted)** | Bagged Tree Ensemble | Table 6 |
| **Linear Discriminant Analysis (LDA)** | Linear Probabilistic Discriminant | Table 6 |
| **Multilayer Perceptron (MLP)** | Feed-Forward Neural Network | Table 6 |
| **B4: Semi-Automated Bias Lexicon** | Word2Vec Cosine Semantic Lexicon | Table 7 (Novel Contribution) |
| **B2: Negative Sentiment Lexicon** | Lexicon Rule Matching | Table 7 |
| **SVM (RBF Kernel, Weighted)** | Non-Linear Kernel Support Vector Machine | Table 6 |
| **Linear SVM (Weighted)** | Linear Maximum-Margin Classifier | Table 6 |
| **Logistic Regression (Weighted)** | Generalized Linear Model | Table 6 |
| **Complement Naive Bayes** | Imbalanced Probabilistic Classifier | Table 6 |

---

## 📈 Summary of Results (Held-Out Test Split, N=3,114 Words)

| Rank | Model | Accuracy | Correct Words | Failed Words | TP | TN | FP | FN | F1 | Precision | Recall | ROC AUC |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **Random Forest (Weighted)** | **91.43%** | 2,847 (91.4%) | 267 (8.6%) | 115 | 2,732 | 80 | 187 | 0.4628 | 0.5897 | 0.3808 | 0.8781 |
| 2 | **Multilayer Perceptron (MLP)** | **91.36%** | 2,845 (91.4%) | 269 (8.6%) | 75 | 2,770 | 42 | 227 | 0.3580 | 0.6410 | 0.2483 | 0.8456 |
| 3 | **Linear Discriminant Analysis (LDA)** | **90.75%** | 2,826 (90.8%) | 288 (9.2%) | 113 | 2,713 | 99 | 189 | 0.4397 | 0.5330 | 0.3742 | 0.8405 |
| 4 | **B4: Semi-Automated Bias Lexicon** | **90.75%** | 2,826 (90.8%) | 288 (9.2%) | 36 | 2,790 | 22 | 266 | 0.2000 | 0.6207 | 0.1192 | 0.5557 |
| 5 | **SVM (RBF Kernel, Weighted)** | **86.83%** | 2,704 (86.8%) | 410 (13.2%) | 120 | 2,584 | 228 | 182 | 0.3692 | 0.3448 | 0.3974 | 0.7875 |
| 6 | **B2: Negative Sentiment Lexicon** | **86.45%** | 2,692 (86.5%) | 422 (13.5%) | 142 | 2,550 | 262 | 160 | 0.4023 | 0.3515 | 0.4702 | 0.6885 |
| 7 | **XGBoost (Fine-Tuned & Weighted)** | **84.30%** | 2,625 (84.3%) | 489 (15.7%) | 211 | 2,414 | 398 | 91 | 0.4632 | 0.3465 | 0.6987 | 0.8766 |
| 8 | **Linear SVM (Weighted)** | **78.39%** | 2,441 (78.4%) | 673 (21.6%) | 217 | 2,224 | 588 | 85 | 0.3921 | 0.2696 | 0.7185 | 0.8491 |
| 9 | **Logistic Regression (Weighted)** | **78.10%** | 2,432 (78.1%) | 682 (21.9%) | 216 | 2,216 | 596 | 86 | 0.3878 | 0.2660 | 0.7152 | 0.8473 |
| 10 | **Complement Naive Bayes** | **72.96%** | 2,272 (73.0%) | 842 (27.0%) | 226 | 2,046 | 766 | 76 | 0.3493 | 0.2278 | 0.7483 | 0.8429 |

---

## 🖼️ Generated Comparison Visualizations

The script generates 5 high-resolution figures stored in [`plots/`](file:///c:/Users/NEERAV%20MISHRA/OneDrive/Documents/Projects/TAI-group-project/model_comparison/plots):

1. **[`1_model_accuracy_comparison.png`](file:///c:/Users/NEERAV%20MISHRA/OneDrive/Documents/Projects/TAI-group-project/model_comparison/plots/1_model_accuracy_comparison.png)**:  
   Horizontal bar chart comparing overall accuracy across all 10 models with percentage callouts.
2. **[`2_confusion_matrices_grid.png`](file:///c:/Users/NEERAV%20MISHRA/OneDrive/Documents/Projects/TAI-group-project/model_comparison/plots/2_confusion_matrices_grid.png)**:  
   $2 \times 5$ multi-panel heatmap grid displaying TP, TN, FP, and FN for every model.
3. **[`3_correct_vs_failed_word_counts.png`](file:///c:/Users/NEERAV%20MISHRA/OneDrive/Documents/Projects/TAI-group-project/model_comparison/plots/3_correct_vs_failed_word_counts.png)**:  
   Stacked bar chart illustrating total correctly classified words vs. failed words.
4. **[`4_comprehensive_metrics_comparison.png`](file:///c:/Users/NEERAV%20MISHRA/OneDrive/Documents/Projects/TAI-group-project/model_comparison/plots/4_comprehensive_metrics_comparison.png)**:  
   Grouped comparison of Accuracy, F1-Score, Precision, Recall, and ROC AUC side-by-side.
5. **[`5_error_distribution_breakdown.png`](file:///c:/Users/NEERAV%20MISHRA/OneDrive/Documents/Projects/TAI-group-project/model_comparison/plots/5_error_distribution_breakdown.png)**:  
   Side-by-side analysis of False Positives (False Alarms) vs. False Negatives (Missed Bias).

---

## 🚀 How to Run

### Method 1: On Windows (Double-Click)
Double-click [`run_comparison.bat`](file:///c:/Users/NEERAV%20MISHRA/OneDrive/Documents/Projects/TAI-group-project/model_comparison/run_comparison.bat) in File Explorer.

### Method 2: Terminal / Command Line
```bash
python model_comparison/compare_models.py
```

### Outputs Generated:
- `model_comparison/MODEL_COMPARISON_REPORT.md`
- `model_comparison/model_comparison_table.csv`
- `model_comparison/comparison_summary.json`
- `model_comparison/plots/*.png`
