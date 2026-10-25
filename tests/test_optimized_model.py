"""
Unit Tests for Optimized Model Pipeline and Inference.
"""

import os
import json
import numpy as np
import pandas as pd
import pytest

from src.data.dataset_loader import MBICDataLoader
from optimized_model.model_pipeline import (
    engineer_interaction_features,
    OptimizedNewsBiasClassifier,
)
from optimized_model.predict_bias import BiasInferenceEngine


def test_interaction_feature_engineering():
    """Verify that interaction feature engineering creates the expected columns."""
    sample_df = pd.DataFrame({
        "negative_conc": [1.0, 0.0],
        "positive_conc": [0.0, 1.0],
        "strong_subj": [1.0, 0.0],
        "weak_subj": [0.0, 1.0],
        "tfidf_art": [0.15, 0.05],
        "glove_vec300_norm": [5.2, 4.1],
        "boosters": [1.0, 0.0],
        "hedges": [0.0, 1.0],
        "bias_lexicon": [1.0, 0.0],
        "bias_lexicon_context": [0.0, 1.0],
        "negative_conc_context": [0.5, 0.0],
        "assertive_verbs": [1.0, 0.0],
        "kill_verbs": [0.0, 0.0],
        "hyperbolic_terms": [1.0, 0.0],
        "attitude_markers": [0.0, 0.0],
    })

    df_eng = engineer_interaction_features(sample_df)

    expected_cols = [
        "feat_net_sentiment",
        "feat_subjectivity_total",
        "feat_emotional_charge",
        "feat_lexical_bias_total",
        "feat_prominence",
        "feat_booster_ratio",
        "feat_context_contrast",
        "feat_assertion_power",
    ]
    for c in expected_cols:
        assert c in df_eng.columns, f"Engineered column '{c}' missing."

    # Validate specific mathematical relationship
    assert df_eng.loc[0, "feat_net_sentiment"] == 1.0  # 1 - 0
    assert df_eng.loc[1, "feat_net_sentiment"] == -1.0 # 0 - 1


def test_optimized_classifier_fit_predict_save_load(tmp_path):
    """Test fitting, threshold tuning, and serialization of OptimizedNewsBiasClassifier."""
    loader = MBICDataLoader()
    splits = loader.get_train_test_split(test_size=0.10, val_size=0.10, random_state=42)

    X_train = splits["X_train"].iloc[:150]
    y_train = splits["y_train"].iloc[:150]
    X_val = splits["X_val"].iloc[:50]
    y_val = splits["y_val"].iloc[:50]
    X_test = splits["X_test"].iloc[:50]

    clf = OptimizedNewsBiasClassifier(n_estimators=10, max_depth=3, random_state=42)
    clf.fit(X_train, y_train)
    assert clf.is_fitted_

    # Tune threshold
    best_t = clf.tune_threshold_on_val(X_val, y_val, metric="accuracy")
    assert 0.35 <= best_t <= 0.70

    # Predictions
    probs = clf.predict_proba(X_test)
    assert probs.shape == (50, 2)
    assert np.allclose(probs.sum(axis=1), 1.0)

    preds = clf.predict(X_test)
    assert len(preds) == 50
    assert set(np.unique(preds)).issubset({0, 1})

    # Save and reload
    save_file = os.path.join(tmp_path, "test_opt_model.joblib")
    clf.save(save_file)
    assert os.path.exists(save_file)

    loaded = OptimizedNewsBiasClassifier.load(save_file)
    assert loaded.is_fitted_
    assert loaded.threshold == clf.threshold

    loaded_preds = loaded.predict(X_test)
    assert np.array_equal(preds, loaded_preds)


def test_inference_engine_sentence_prediction():
    """Verify that BiasInferenceEngine processes sentences correctly."""
    engine = BiasInferenceEngine()
    sentence = "Fox News attempts to paint the Green New Deal as disastrous."
    results = engine.analyze_sentence(sentence)

    assert len(results) > 0
    words = [r["word"].lower() for r in results]
    assert "disastrous" in words

    # Find result for 'disastrous'
    disastrous_res = next(r for r in results if r["word"].lower() == "disastrous")
    assert disastrous_res["bias_probability"] > 0.50
    assert disastrous_res["is_biased"] is True


def test_optimized_model_highest_accuracy_assertion():
    """Verify that the evaluated optimized model achieves highest accuracy in evaluation_results.json."""
    eval_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "optimized_model",
        "evaluation_results.json",
    )
    assert os.path.exists(eval_file), f"Evaluation results file {eval_file} missing."

    with open(eval_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    opt_acc = data["accuracy_pct"]
    benchmarks = data["benchmark_comparison"]

    for b in benchmarks:
        if b["model"] != "Optimized Model (Ours)":
            assert opt_acc >= b["accuracy_pct"], (
                f"Optimized model accuracy ({opt_acc}%) is not highest! "
                f"Beaten by {b['model']} ({b['accuracy_pct']}%)"
            )
