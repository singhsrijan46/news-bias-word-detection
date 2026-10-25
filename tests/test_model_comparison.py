"""
Unit Tests for Model Comparison Module.
"""

import os
import json
import numpy as np
import pandas as pd
import pytest

from src.data.dataset_loader import MBICDataLoader
from src.models.all_models import build_all_models_suite
from src.evaluation.metrics import evaluate_model
from model_comparison.compare_models import TOP_10_MODEL_NAMES


def test_top_10_models_exist():
    """Verify all top 10 models are present in the model suite."""
    suite = build_all_models_suite(random_state=42)
    assert len(TOP_10_MODEL_NAMES) == 10
    for name in TOP_10_MODEL_NAMES:
        assert name in suite, f"Model '{name}' missing from model suite."


def test_model_comparison_metrics_integrity():
    """Verify metrics calculation and word counts logic on a fast subset."""
    loader = MBICDataLoader()
    splits = loader.get_train_test_split(test_size=0.10, val_size=0.10, random_state=42)
    X_train, y_train = splits["X_train"].iloc[:100], splits["y_train"].iloc[:100]
    X_test, y_test = splits["X_test"].iloc[:50], splits["y_test"].iloc[:50]

    suite = build_all_models_suite(random_state=42)
    # Test fast baseline
    model = suite["B2: Negative Sentiment Lexicon"]
    model.fit(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test, model_name="B2 Test")

    assert "accuracy" in metrics
    assert "tp" in metrics and "tn" in metrics and "fp" in metrics and "fn" in metrics
    total_words = metrics["tp"] + metrics["tn"] + metrics["fp"] + metrics["fn"]
    assert total_words == len(X_test)


def test_comparison_summary_files_exist():
    """Check that comparison summary JSON and CSV files exist and have valid structure."""
    comp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model_comparison")
    json_path = os.path.join(comp_dir, "comparison_summary.json")
    csv_path = os.path.join(comp_dir, "model_comparison_table.csv")
    report_path = os.path.join(comp_dir, "MODEL_COMPARISON_REPORT.md")

    assert os.path.exists(json_path), f"Missing {json_path}"
    assert os.path.exists(csv_path), f"Missing {csv_path}"
    assert os.path.exists(report_path), f"Missing {report_path}"

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert len(data) == 10
    assert all("accuracy" in item and "correct_words_count" in item for item in data)
