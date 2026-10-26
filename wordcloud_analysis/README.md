# ☁️ Biased Word Cloud & Model Self-Bias Analysis Suite

> **Multi-Model Word Cloud Comparison & In-Depth Audit of Intrinsic Model Bias**  
> Benchmarks Ground Truth (Human MBIC Annotators), 10 Models from Spinde et al. (2021), and the Optimized Champion Model.

---

## 📁 Folder Structure

```
wordcloud_analysis/
│
├── wordcloud_comparator.py              # Main Python engine for training, word clouds & bias audit
├── run_wordcloud_analysis.bat           # 1-Click Windows execution runner
├── MODEL_BIAS_AND_WORDCLOUD_REPORT.md   # Complete empirical report with metrics and interpretations
├── README.md                            # Documentation and user guide (You are here)
│
├── biased_word_frequency_comparison.csv # Spreadsheet comparing token frequencies across all models
├── model_intrinsic_bias_metrics.json    # Machine-readable metrics (outlet, topic, POS, sentiment)
├── model_outlet_and_topic_bias.csv      # Tabular breakdown of bias rates by outlet and topic
│
└── plots/                               # 5 Publication-grade visualization artifacts
    ├── 1_master_wordcloud_grid.png      # 12-Panel grid showing Ground Truth and all 11 models
    ├── 2_top_biased_words_frequency_comparison.png # Frequency comparison bar chart of top words
    ├── 3_false_positives_vs_negatives_wordclouds.png # False alarms vs. missed bias word clouds
    ├── 4_model_outlet_and_topic_bias_disparity.png # Media outlet & political leaning disparity
    ├── 5_sentiment_conflation_and_pos_bias.png # Sentiment conflation & POS distribution
    │
    └── individual_wordclouds/           # 12 Standalone high-res word clouds for every single model
        ├── 00_ground_truth_annotators.png
        ├── 01_optimized_xgboost_champion.png
        ├── 02_random_forest.png
        ├── 03_multilayer_perceptron.png
        ├── 04_linear_discriminant_analysis.png
        ├── 05_b4_semi_automated_lexicon.png
        ├── 06_svm_rbf_kernel.png
        ├── 07_b2_negative_sentiment_lexicon.png
        ├── 08_xgboost_paper_fine_tuned.png
        ├── 09_linear_svm.png
        ├── 10_logistic_regression.png
        └── 11_complement_naive_bayes.png
```

---

## 🚀 How to Run

### Option A: 1-Click Windows Batch File
Double-click [`wordcloud_analysis/run_wordcloud_analysis.bat`](run_wordcloud_analysis.bat).

### Option B: Command Line
```bash
python wordcloud_analysis/wordcloud_comparator.py
```

### Option C: Run Unit Tests
```bash
python -m pytest tests/test_wordcloud_analysis.py -v
```

---

## 🌟 Key Scientific Insights

1. **Word Clouds Gallery**:
   * Ground truth features a balanced distribution of loaded nouns (*propaganda*, *rhetoric*), emotional adjectives (*racist*, *corrupt*), and aggressive verbs (*infiltrate*, *swarm*).
   * Uncalibrated models (Naive Bayes, Linear SVM, Paper XGBoost) flood their clouds with neutral political words (*federal*, *abortion*, *senate*, *media*).
2. **Frequency Comparison**:
   * **Consensus biased words**: Words flagged by humans and almost all models include *rhetoric*, *racist*, *corrupt*, *intolerant*, *drastic*, and *false*.
   * **False alarms**: Words like *abortion* (flagged 9 times by Linear SVM despite 0 in Ground Truth) reflect topic co-occurrence bias.
3. **Is the Model Biased in Itself?**:
   * **Sentiment Conflation Bias**: Models have $r \approx 0.35 - 0.40$ correlation with negative sentiment, often confusing negative factual reporting (e.g. *crisis*, *death*, *illness*) with editorial bias.
   * **Part-of-Speech Bias**: Models show a massive **Adjective Saliency Bias** (63.3% of flagged words vs. 36.1% in Ground Truth).
   * **Dispositional Bias**: Models split into *Hyper-Alarmist* (Naive Bayes, Linear SVM) vs *Hyper-Conservative* (MLP, B4 Lexicon) vs *Calibrated Champion* (Optimized XGBoost).

For the full detailed breakdown and graphs, see [MODEL_BIAS_AND_WORDCLOUD_REPORT.md](MODEL_BIAS_AND_WORDCLOUD_REPORT.md).
