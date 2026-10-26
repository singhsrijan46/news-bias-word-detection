"""
Unit Tests for Word Cloud & Model Self-Bias Analysis Module.
"""

import os
import json
import numpy as np
import pandas as pd
import pytest

from wordcloud_analysis.wordcloud_comparator import (
    clean_token,
    generate_single_wordcloud,
    MODEL_COLORMAPS,
)


def test_clean_token_utility():
    """Verify that tokens are cleaned and normalized appropriately."""
    assert clean_token("Disastrous!") == "disastrous"
    assert clean_token("  'corrupt'  ") == "corrupt"
    assert clean_token("a") == ""  # Single characters dropped
    assert clean_token("---") == ""  # Punctuation-only dropped
    assert clean_token(None) == ""
    assert clean_token(123) == ""


def test_model_colormaps_exist_for_all_models():
    """Verify all 12 model keys have corresponding visual colormaps."""
    expected_models = [
        "Ground Truth (Human Annotators)",
        "Optimized XGBoost (Champion)",
        "Random Forest (Weighted)",
        "Multilayer Perceptron (MLP)",
        "Linear Discriminant Analysis (LDA)",
        "B4: Semi-Automated Bias Lexicon",
        "SVM (RBF Kernel, Weighted)",
        "B2: Negative Sentiment Lexicon",
        "XGBoost (Paper Fine-Tuned)",
        "Linear SVM (Weighted)",
        "Logistic Regression (Weighted)",
        "Complement Naive Bayes",
    ]
    for m in expected_models:
        assert m in MODEL_COLORMAPS, f"Colormap missing for model: {m}"


def test_wordcloud_artifact_files_exist():
    """Verify that generated CSVs, JSON metrics, and plot images exist."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    wc_dir = os.path.join(base_dir, "wordcloud_analysis")
    plots_dir = os.path.join(wc_dir, "plots")
    indiv_dir = os.path.join(plots_dir, "individual_wordclouds")

    # CSV and JSON files
    csv_freq = os.path.join(wc_dir, "biased_word_frequency_comparison.csv")
    csv_bias = os.path.join(wc_dir, "model_outlet_and_topic_bias.csv")
    json_metrics = os.path.join(wc_dir, "model_intrinsic_bias_metrics.json")

    assert os.path.exists(csv_freq), f"Missing {csv_freq}"
    assert os.path.exists(csv_bias), f"Missing {csv_bias}"
    assert os.path.exists(json_metrics), f"Missing {json_metrics}"

    # Plots
    plot_files = [
        "1_master_wordcloud_grid.png",
        "2_top_biased_words_frequency_comparison.png",
        "3_false_positives_vs_negatives_wordclouds.png",
        "4_model_outlet_and_topic_bias_disparity.png",
        "5_sentiment_conflation_and_pos_bias.png",
    ]
    for p in plot_files:
        assert os.path.exists(os.path.join(plots_dir, p)), f"Missing plot: {p}"

    # Individual wordclouds
    indiv_clouds = os.listdir(indiv_dir)
    assert len(indiv_clouds) >= 12, f"Expected 12 individual word clouds, found {len(indiv_clouds)}"


def test_intrinsic_bias_metrics_integrity():
    """Verify JSON structure and metrics sanity."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_metrics = os.path.join(base_dir, "wordcloud_analysis", "model_intrinsic_bias_metrics.json")

    with open(json_metrics, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "dataset_summary" in data
    assert "models" in data
    assert len(data["models"]) == 12

    for model_name, m_data in data["models"].items():
        assert "total_biased_words_flagged" in m_data
        assert "disposition_classification" in m_data
        assert "sentiment_conflation" in m_data
        assert "political_leaning_bias" in m_data
        assert "outlets" in m_data
        assert "topics" in m_data
        assert "pos_distribution_pct" in m_data


def test_generate_single_wordcloud_executes(tmp_path):
    """Verify word cloud generation runs end-to-end on synthetic frequencies."""
    freqs = {"racist": 10, "disastrous": 8, "corrupt": 6, "drastic": 4}
    out_file = os.path.join(str(tmp_path), "test_cloud.png")
    generate_single_wordcloud(freqs, title="Test Cloud", output_path=out_file)
    assert os.path.exists(out_file)
    assert os.path.getsize(out_file) > 1000  # Non-trivial image file
