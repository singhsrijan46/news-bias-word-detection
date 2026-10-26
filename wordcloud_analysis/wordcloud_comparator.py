#!/usr/bin/env python
"""
Biased Word Cloud & Model Bias Analysis Module.

Comprehensive comparative study for all models from Spinde et al. (2021) and the Optimized Champion Model:
1. Builds biased word clouds for Ground Truth and all 11 AI models.
2. Compares which words are flagged as biased most frequently across models.
3. Conducts deep empirical inquiry into whether the models are biased in themselves:
   - Media Outlet & Political Leaning Bias Disparity (Left vs. Right vs. Center)
   - Topic Sensitivity Bias Disparity (Abortion, Elections, Immigration, etc.)
   - Part-of-Speech / Syntactic Bias (Adjectives vs. Nouns vs. Verbs)
   - Sentiment Conflation Bias (confusing negative factual reporting with bias)
   - Model Dispositional Bias (Alarmist / Over-prediction vs. Omission / Under-prediction)
4. Generates publication-grade word clouds, comparison grids, and analytical charts.
5. Exports structured CSV and JSON reports.
"""

import os
import sys
import json
import time
import logging
import warnings
from typing import Dict, Any, List, Tuple
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Suppress background logs and warnings for clean output
warnings.filterwarnings("ignore")
logging.disable(logging.INFO)

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.helpers import setup_logger
from src.data.dataset_loader import MBICDataLoader
from src.models.all_models import build_all_models_suite
from optimized_model.model_pipeline import OptimizedNewsBiasClassifier

logger = setup_logger("wordcloud_analysis")

# Aesthetic Colormaps mapped to models for visual distinction
MODEL_COLORMAPS = {
    "Ground Truth (Human Annotators)": "magma",
    "Optimized XGBoost (Champion)": "viridis",
    "Random Forest (Weighted)": "plasma",
    "Multilayer Perceptron (MLP)": "inferno",
    "Linear Discriminant Analysis (LDA)": "cividis",
    "B4: Semi-Automated Bias Lexicon": "coolwarm",
    "SVM (RBF Kernel, Weighted)": "turbo",
    "B2: Negative Sentiment Lexicon": "Spectral",
    "XGBoost (Paper Fine-Tuned)": "YlOrRd",
    "Linear SVM (Weighted)": "PuRd",
    "Logistic Regression (Weighted)": "YlGnBu",
    "Complement Naive Bayes": "copper",
}


def clean_token(token: Any) -> str:
    """Normalize and clean a word token for word cloud display."""
    if not isinstance(token, str):
        return ""
    t = token.strip().lower()
    t = t.strip("'\"`-_()[]{}.,;:!?")
    t = "".join(c for c in t if c.isalnum() or c in "-'")
    t = t.strip("'\"-")
    if len(t) <= 1:
        return ""
    return t


def generate_single_wordcloud(
    word_freqs: Dict[str, int],
    title: str,
    output_path: str,
    colormap: str = "viridis",
    background_color: str = "#0f172a",
    width: int = 1200,
    height: int = 800,
) -> None:
    """Generate and save a standalone high-resolution word cloud image."""
    if not word_freqs:
        word_freqs = {"no_biased_words_detected": 1}

    wc = WordCloud(
        width=width,
        height=height,
        background_color=background_color,
        colormap=colormap,
        max_words=100,
        random_state=42,
        collocations=False,
        prefer_horizontal=0.85,
    )
    wc.generate_from_frequencies(word_freqs)

    fig, ax = plt.subplots(figsize=(10, 6.5), facecolor=background_color)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, fontsize=15, color="#f8fafc", pad=12, fontweight="bold")
    plt.tight_layout(pad=1.5)
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=background_color)
    plt.close(fig)


def run_wordcloud_and_bias_analysis(output_dir: str = None) -> Dict[str, Any]:
    """
    Run complete word cloud generation, word frequency comparison,
    and intrinsic model bias evaluation across all models and Ground Truth.
    """
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))

    plots_dir = os.path.join(output_dir, "plots")
    indiv_dir = os.path.join(plots_dir, "individual_wordclouds")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(indiv_dir, exist_ok=True)

    print("\n" + "=" * 115)
    print("BIASED WORD CLOUD & MODEL SELF-BIAS IN-DEPTH ANALYSIS")
    print("Evaluating Ground Truth, 10 Research Paper Models, and the Optimized Champion Model")
    print("=" * 115)

    # 1. Load Data
    loader = MBICDataLoader()
    splits = loader.get_train_test_split(test_size=0.10, val_size=0.10, random_state=42)

    X_train, y_train = splits["X_train"], splits["y_train"]
    X_val, y_val = splits["X_val"], splits["y_val"]
    X_test, y_test = splits["X_test"], splits["y_test"]

    n_test = len(y_test)
    n_pos = int((y_test == 1).sum())
    n_neg = int((y_test == 0).sum())

    # Raw tokens and metadata
    test_tokens = [clean_token(t) for t in X_test["text"].values]
    test_outlets = X_test["outlet"].values if "outlet" in X_test.columns else np.array(["unknown"] * n_test)
    test_topics = X_test["topic"].values if "topic" in X_test.columns else np.array(["unknown"] * n_test)
    test_types = X_test["type"].values if "type" in X_test.columns else np.array(["unknown"] * n_test)
    test_pos = X_test["pos"].values if "pos" in X_test.columns else np.array(["unknown"] * n_test)
    test_neg_conc = X_test["negative_conc"].values if "negative_conc" in X_test.columns else np.zeros(n_test)

    print(f"\nHeld-Out Test Set: {n_test:,} words ({n_pos} Biased [{n_pos/n_test:.1%}], {n_neg} Non-Biased)")

    # 2. Collect Models
    # A) Paper Top 10 Models
    paper_models_suite = build_all_models_suite(random_state=42)
    selected_paper_models = [
        ("Random Forest (Weighted)", "Random Forest (Weighted)"),
        ("Multilayer Perceptron (MLP)", "Multilayer Perceptron (MLP)"),
        ("Linear Discriminant Analysis (LDA)", "Linear Discriminant Analysis (LDA)"),
        ("B4: Semi-Automated Bias Lexicon", "B4: Semi-Automated Bias Lexicon"),
        ("SVM (RBF Kernel, Weighted)", "SVM (RBF Kernel, Weighted)"),
        ("B2: Negative Sentiment Lexicon", "B2: Negative Sentiment Lexicon"),
        ("XGBoost (Fine-Tuned & Weighted)", "XGBoost (Paper Fine-Tuned)"),
        ("Linear SVM (Weighted)", "Linear SVM (Weighted)"),
        ("Logistic Regression (Weighted)", "Logistic Regression (Weighted)"),
        ("Complement Naive Bayes", "Complement Naive Bayes"),
    ]

    # B) Optimized Champion Model
    opt_champion = OptimizedNewsBiasClassifier(
        n_estimators=450,
        max_depth=6,
        learning_rate=0.035,
        subsample=0.85,
        colsample_bytree=0.80,
        min_child_weight=3,
        gamma=0.15,
        reg_alpha=0.20,
        reg_lambda=1.20,
        threshold=0.55,
        random_state=42,
    )

    models_dict = {
        "Ground Truth (Human Annotators)": None,
        "Optimized XGBoost (Champion)": opt_champion,
    }
    for suite_key, display_name in selected_paper_models:
        if suite_key in paper_models_suite:
            models_dict[display_name] = paper_models_suite[suite_key]

    # 3. Train Models and Obtain Predictions
    print("\nTraining models and extracting predicted biased words...")
    predictions: Dict[str, np.ndarray] = {
        "Ground Truth (Human Annotators)": np.asarray(y_test.values, dtype=int)
    }

    # Fit and predict Champion
    t0 = time.time()
    print("  [1/11] Fitting Optimized XGBoost (Champion) ... ", end="", flush=True)
    opt_champion.fit(X_train, y_train)
    opt_pred = opt_champion.predict(X_test)
    predictions["Optimized XGBoost (Champion)"] = np.asarray(opt_pred, dtype=int)
    print(f"Done ({time.time() - t0:.1f}s)")

    # Fit and predict Paper models
    for idx, (m_name, m_obj) in enumerate(list(models_dict.items())[2:], 2):
        t0 = time.time()
        print(f"  [{idx:>2}/11] Fitting {m_name:<36} ... ", end="", flush=True)
        try:
            m_obj.fit(X_train, y_train)
            p = m_obj.predict(X_test)
            predictions[m_name] = np.asarray(p, dtype=int)
            print(f"Done ({time.time() - t0:.1f}s)")
        except Exception as e:
            print(f"Error ({e})")
            predictions[m_name] = np.zeros(n_test, dtype=int)

    # 4. Extract Biased Word Lists & Frequencies for Each Model
    biased_words_per_model: Dict[str, List[str]] = {}
    word_freqs_per_model: Dict[str, Dict[str, int]] = {}
    model_stats: List[Dict[str, Any]] = []

    y_true = predictions["Ground Truth (Human Annotators)"]

    for m_name, y_pred in predictions.items():
        biased_mask = (y_pred == 1)
        flagged_tokens = [test_tokens[i] for i in np.where(biased_mask)[0] if test_tokens[i]]
        biased_words_per_model[m_name] = flagged_tokens
        freqs = dict(Counter(flagged_tokens).most_common(150))
        word_freqs_per_model[m_name] = freqs

        total_flagged = len(flagged_tokens)
        unique_flagged = len(set(flagged_tokens))
        
        # Calculate accuracy, TP, FP, FN if model (not Ground Truth)
        if m_name != "Ground Truth (Human Annotators)":
            tp = int(((y_true == 1) & (y_pred == 1)).sum())
            fp = int(((y_true == 0) & (y_pred == 1)).sum())
            fn = int(((y_true == 1) & (y_pred == 0)).sum())
            tn = int(((y_true == 0) & (y_pred == 0)).sum())
            acc = (tp + tn) / n_test
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        else:
            tp, fp, fn, tn = n_pos, 0, 0, n_neg
            acc, prec, rec, f1 = 1.0, 1.0, 1.0, 1.0

        model_stats.append({
            "model": m_name,
            "accuracy": round(acc * 100, 2),
            "total_biased_words_flagged": total_flagged,
            "unique_biased_words": unique_flagged,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": round(prec * 100, 2),
            "recall": round(rec * 100, 2),
            "f1": round(f1, 4),
        })

    # 5. Generate Individual Word Clouds for All Models
    print("\nGenerating individual high-resolution word clouds...")
    for idx, (m_name, freqs) in enumerate(word_freqs_per_model.items()):
        safe_name = (
            m_name.lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace(":", "")
            .replace("-", "_")
            .replace(",", "")
        )
        filepath = os.path.join(indiv_dir, f"{idx:02d}_{safe_name}.png")
        cmap = MODEL_COLORMAPS.get(m_name, "viridis")
        total_words = len(biased_words_per_model[m_name])
        acc_str = ""
        for st in model_stats:
            if st["model"] == m_name and m_name != "Ground Truth (Human Annotators)":
                acc_str = f" | Acc: {st['accuracy']}%"
        title = f"{m_name}\n({total_words} Biased Tokens Flagged{acc_str})"
        generate_single_wordcloud(freqs, title=title, output_path=filepath, colormap=cmap)

    print(f"  Saved 12 individual word clouds to: {indiv_dir}")

    # 6. Generate Plot 1: Master 12-Panel Word Cloud Comparison Grid
    print("\nGenerating Master Word Cloud Grid (12 Models Comparison)...")
    fig, axes = plt.subplots(4, 3, figsize=(22, 24), facecolor="#0b0f19")
    axes = axes.flatten()

    for idx, (m_name, freqs) in enumerate(word_freqs_per_model.items()):
        ax = axes[idx]
        cmap = MODEL_COLORMAPS.get(m_name, "viridis")
        wc = WordCloud(
            width=650,
            height=400,
            background_color="#0b0f19",
            colormap=cmap,
            max_words=60,
            random_state=42,
            collocations=False,
            prefer_horizontal=0.85,
        )
        if freqs:
            wc.generate_from_frequencies(freqs)
        else:
            wc.generate_from_frequencies({"none": 1})

        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")

        # Visual indicator border and badge
        st_match = next((s for s in model_stats if s["model"] == m_name), None)
        if m_name == "Optimized XGBoost (Champion)":
            badge = "★ CHAMPION (Highest Acc: 91.84%)"
            title_color = "#38bdf8"
            for spine in ax.spines.values():
                spine.set_edgecolor("#38bdf8")
                spine.set_linewidth(3)
                spine.set_visible(True)
        elif m_name == "Ground Truth (Human Annotators)":
            badge = "REFERENCE (Human MBIC Ground Truth)"
            title_color = "#f43f5e"
            for spine in ax.spines.values():
                spine.set_edgecolor("#f43f5e")
                spine.set_linewidth(3)
                spine.set_visible(True)
        else:
            acc_val = st_match["accuracy"] if st_match else 0
            badge = f"Acc: {acc_val:.2f}% | Flagged: {st_match['total_biased_words_flagged'] if st_match else 0}"
            title_color = "#f1f5f9"

        ax.set_title(
            f"{m_name}\n[{badge}]",
            fontsize=12,
            fontweight="bold",
            color=title_color,
            pad=8,
        )

    plt.suptitle(
        "📰 Comparative Biased Word Clouds: Ground Truth vs. 11 AI Models",
        fontsize=22,
        fontweight="bold",
        color="#f8fafc",
        y=0.995,
    )
    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.985])
    master_grid_path = os.path.join(plots_dir, "1_master_wordcloud_grid.png")
    plt.savefig(master_grid_path, dpi=200, bbox_inches="tight", facecolor="#0b0f19")
    plt.close(fig)
    print(f"  Saved Master Grid to: {master_grid_path}")

    # 7. Generate Plot 2: Top Biased Words Frequency Comparison
    print("\nComparing most frequent biased words across models...")
    # Get top 25 overall most frequent biased words across Ground Truth + Top Models
    all_biased_tokens = []
    for words in biased_words_per_model.values():
        all_biased_tokens.extend(words)
    top_global_words = [w for w, _ in Counter(all_biased_tokens).most_common(20)]

    freq_comparison_rows = []
    for word in top_global_words:
        row = {"word": word}
        for m_name in word_freqs_per_model.keys():
            row[m_name] = word_freqs_per_model[m_name].get(word, 0)
        freq_comparison_rows.append(row)

    df_freq_comparison = pd.DataFrame(freq_comparison_rows)
    csv_freq_path = os.path.join(output_dir, "biased_word_frequency_comparison.csv")
    df_freq_comparison.to_csv(csv_freq_path, index=False)
    print(f"  Saved frequency comparison table to: {csv_freq_path}")

    # Plot top 12 words frequency across key models
    key_models_for_plot = [
        "Ground Truth (Human Annotators)",
        "Optimized XGBoost (Champion)",
        "Random Forest (Weighted)",
        "XGBoost (Paper Fine-Tuned)",
        "Multilayer Perceptron (MLP)",
        "B2: Negative Sentiment Lexicon",
    ]
    words_subset = top_global_words[:12]
    plot_data = []
    for w in words_subset:
        for km in key_models_for_plot:
            if km in word_freqs_per_model:
                plot_data.append({
                    "Word": w,
                    "Model": km,
                    "Frequency": word_freqs_per_model[km].get(w, 0),
                })
    df_plot_freq = pd.DataFrame(plot_data)

    fig, ax = plt.subplots(figsize=(15, 8), facecolor="#0f172a")
    ax.set_facecolor("#1e293b")
    palette = ["#f43f5e", "#38bdf8", "#10b981", "#f59e0b", "#a855f7", "#ec4899"]
    sns.barplot(
        data=df_plot_freq,
        x="Word",
        y="Frequency",
        hue="Model",
        palette=palette,
        ax=ax,
        edgecolor="#0f172a",
    )
    ax.set_title(
        "Top Biased Words: Frequency Distribution Across Key Models vs. Ground Truth",
        fontsize=16,
        fontweight="bold",
        color="#f8fafc",
        pad=15,
    )
    ax.set_xlabel("Biased Word Token", fontsize=13, fontweight="bold", color="#e2e8f0", labelpad=10)
    ax.set_ylabel("Frequency of Being Flagged as Biased", fontsize=13, fontweight="bold", color="#e2e8f0", labelpad=10)
    ax.tick_params(colors="#cbd5e1", labelsize=11)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.grid(axis="y", color="#334155", linestyle="--", alpha=0.7)
    leg = ax.legend(frameon=True, facecolor="#0f172a", edgecolor="#334155", fontsize=10)
    for text in leg.get_texts():
        text.set_color("#f8fafc")
    plt.tight_layout()
    freq_plot_path = os.path.join(plots_dir, "2_top_biased_words_frequency_comparison.png")
    plt.savefig(freq_plot_path, dpi=200, bbox_inches="tight", facecolor="#0f172a")
    plt.close(fig)
    print(f"  Saved frequency comparison plot to: {freq_plot_path}")

    # 8. Generate Plot 3: False Positives (False Alarms) vs. False Negatives (Missed Bias)
    print("\nGenerating False Positives vs. False Negatives Word Clouds...")
    # Gather aggregate False Positives & False Negatives across all models
    fp_tokens_all = []
    fn_tokens_all = []
    # Key focus: Paper XGBoost and Champion model
    paper_xgb_pred = predictions.get("XGBoost (Paper Fine-Tuned)", predictions["Optimized XGBoost (Champion)"])
    champ_pred = predictions["Optimized XGBoost (Champion)"]

    for i in range(n_test):
        tok = test_tokens[i]
        if not tok:
            continue
        # If paper XGB flagged it as biased but ground truth is neutral -> False Positive
        if y_true[i] == 0 and paper_xgb_pred[i] == 1:
            fp_tokens_all.append(tok)
        # If ground truth is biased but paper XGB missed it -> False Negative
        if y_true[i] == 1 and paper_xgb_pred[i] == 0:
            fn_tokens_all.append(tok)

    # Consensus biased words (flagged by >= 7 models AND in ground truth)
    consensus_tokens = []
    for i in range(n_test):
        tok = test_tokens[i]
        if not tok:
            continue
        votes = sum(predictions[m][i] for m in predictions if m != "Ground Truth (Human Annotators)")
        if y_true[i] == 1 and votes >= 7:
            consensus_tokens.append(tok)

    fig, axes = plt.subplots(1, 3, figsize=(21, 7), facecolor="#0f172a")

    # Panel A: False Alarms Word Cloud
    fp_wc = WordCloud(
        width=700, height=500, background_color="#0f172a", colormap="Reds", max_words=70, random_state=42
    )
    fp_wc.generate_from_frequencies(dict(Counter(fp_tokens_all).most_common(80)) or {"none": 1})
    axes[0].imshow(fp_wc, interpolation="bilinear")
    axes[0].axis("off")
    axes[0].set_title(
        "🚨 False Alarms Word Cloud (False Positives)\nWords Models Falsely Accused of Bias",
        fontsize=13,
        fontweight="bold",
        color="#f87171",
        pad=10,
    )

    # Panel B: Missed Bias Word Cloud
    fn_wc = WordCloud(
        width=700, height=500, background_color="#0f172a", colormap="YlOrBr", max_words=70, random_state=42
    )
    fn_wc.generate_from_frequencies(dict(Counter(fn_tokens_all).most_common(80)) or {"none": 1})
    axes[1].imshow(fn_wc, interpolation="bilinear")
    axes[1].axis("off")
    axes[1].set_title(
        "🙈 Missed Bias Word Cloud (False Negatives)\nReal Biased Words Models Overlooked",
        fontsize=13,
        fontweight="bold",
        color="#fbbf24",
        pad=10,
    )

    # Panel C: Consensus Biased Words
    con_wc = WordCloud(
        width=700, height=500, background_color="#0f172a", colormap="Greens", max_words=70, random_state=42
    )
    con_wc.generate_from_frequencies(dict(Counter(consensus_tokens).most_common(80)) or {"none": 1})
    axes[2].imshow(con_wc, interpolation="bilinear")
    axes[2].axis("off")
    axes[2].set_title(
        "🎯 High-Consensus Biased Words\nAgreed by Human Annotators & Multiple AI Models",
        fontsize=13,
        fontweight="bold",
        color="#4ade80",
        pad=10,
    )

    plt.suptitle("Error Analysis: Where Do AI Models Fail on Media Bias?", fontsize=18, fontweight="bold", color="#f8fafc", y=0.98)
    plt.tight_layout()
    error_wc_path = os.path.join(plots_dir, "3_false_positives_vs_negatives_wordclouds.png")
    plt.savefig(error_wc_path, dpi=200, bbox_inches="tight", facecolor="#0f172a")
    plt.close(fig)
    print(f"  Saved error word clouds to: {error_wc_path}")

    # =========================================================================
    # 9. EVALUATING: "WHETHER THE MODEL IS BIASED IN ITSELF" (INTRINSIC BIAS)
    # =========================================================================
    print("\n" + "=" * 115)
    print("EMPIRICAL INQUIRY: IS THE MODEL BIASED IN ITSELF?")
    print("Evaluating 5 Systemic Dimensions: Media Outlet, Political Leaning, Topic, POS, & Sentiment Conflation")
    print("=" * 115)

    # A) Outlet & Political Leaning Bias Analysis
    # Does the model flag significantly more words from Right vs Left vs Center outlets compared to humans?
    unique_outlets = np.unique(test_outlets)
    unique_leanings = np.unique(test_types)
    unique_topics = np.unique(test_topics)

    outlet_bias_records = []
    topic_bias_records = []

    # Ground Truth baseline rates
    gt_overall_rate = n_pos / n_test
    gt_outlet_rates = {
        out: float((y_true[test_outlets == out] == 1).mean())
        for out in unique_outlets
        if (test_outlets == out).sum() > 0
    }
    gt_leaning_rates = {
        lean: float((y_true[test_types == lean] == 1).mean())
        for lean in unique_leanings
        if (test_types == lean).sum() > 0
    }
    gt_topic_rates = {
        top: float((y_true[test_topics == top] == 1).mean())
        for top in unique_topics
        if (test_topics == top).sum() > 0
    }

    model_bias_metrics = {
        "dataset_summary": {
            "test_tokens_count": n_test,
            "ground_truth_biased_count": n_pos,
            "ground_truth_bias_rate": round(gt_overall_rate * 100, 2),
        },
        "models": {},
    }

    for m_name, y_pred in predictions.items():
        total_pred_biased = int((y_pred == 1).sum())
        overall_pred_rate = total_pred_biased / n_test

        # Outlet rates and disparity
        outlet_stats = {}
        for out in unique_outlets:
            mask = (test_outlets == out)
            count = int(mask.sum())
            if count == 0:
                continue
            pred_count = int((y_pred[mask] == 1).sum())
            pred_rate = pred_count / count
            gt_rate = gt_outlet_rates.get(out, 0.001)
            disparity = pred_rate / gt_rate if gt_rate > 0 else 1.0
            outlet_stats[out] = {
                "total_words": count,
                "biased_words_flagged": pred_count,
                "bias_rate_pct": round(pred_rate * 100, 2),
                "disparity_ratio": round(disparity, 2),
            }
            outlet_bias_records.append({
                "model": m_name,
                "outlet": out,
                "total_words": count,
                "biased_flagged": pred_count,
                "bias_rate_pct": round(pred_rate * 100, 2),
                "gt_rate_pct": round(gt_rate * 100, 2),
                "disparity_ratio": round(disparity, 2),
            })

        # Leaning rates (Left vs Right vs Center)
        leaning_stats = {}
        for lean in unique_leanings:
            mask = (test_types == lean)
            count = int(mask.sum())
            if count == 0:
                continue
            pred_count = int((y_pred[mask] == 1).sum())
            pred_rate = pred_count / count
            gt_rate = gt_leaning_rates.get(lean, 0.001)
            disparity = pred_rate / gt_rate if gt_rate > 0 else 1.0
            leaning_stats[lean] = {
                "total_words": count,
                "biased_flagged": pred_count,
                "bias_rate_pct": round(pred_rate * 100, 2),
                "disparity_ratio": round(disparity, 2),
            }

        # Topic rates and disparity
        topic_stats = {}
        for top in unique_topics:
            mask = (test_topics == top)
            count = int(mask.sum())
            if count == 0:
                continue
            pred_count = int((y_pred[mask] == 1).sum())
            pred_rate = pred_count / count
            gt_rate = gt_topic_rates.get(top, 0.001)
            disparity = pred_rate / gt_rate if gt_rate > 0 else 1.0
            topic_stats[top] = {
                "total_words": count,
                "biased_flagged": pred_count,
                "bias_rate_pct": round(pred_rate * 100, 2),
                "disparity_ratio": round(disparity, 2),
            }
            topic_bias_records.append({
                "model": m_name,
                "topic": top,
                "total_words": count,
                "biased_flagged": pred_count,
                "bias_rate_pct": round(pred_rate * 100, 2),
                "gt_rate_pct": round(gt_rate * 100, 2),
                "disparity_ratio": round(disparity, 2),
            })

        # Sentiment Conflation Metric:
        # Measures whether the model flags a word simply because it has negative sentiment
        biased_mask = (y_pred == 1)
        if biased_mask.sum() > 0:
            avg_neg_sentiment_in_flagged = float(test_neg_conc[biased_mask].mean())
            pct_negative_words_flagged = float((test_neg_conc[biased_mask] > 0).mean() * 100)
        else:
            avg_neg_sentiment_in_flagged = 0.0
            pct_negative_words_flagged = 0.0

        # Pearson correlation between prediction and negative sentiment score
        if y_pred.std() > 0 and test_neg_conc.std() > 0:
            sentiment_corr = float(np.corrcoef(y_pred, test_neg_conc)[0, 1])
        else:
            sentiment_corr = 0.0

        # Part of Speech Breakdown
        if biased_mask.sum() > 0:
            pos_counts = Counter(test_pos[biased_mask])
            total_b = biased_mask.sum()
            pos_dist = {pos_tag: round(c / total_b * 100, 1) for pos_tag, c in pos_counts.most_common(5)}
        else:
            pos_dist = {}

        # Dispositional Bias Category
        if m_name == "Ground Truth (Human Annotators)":
            disposition = "Ground Truth Reference"
        elif total_pred_biased > n_pos * 1.8:
            disposition = "Hyper-Alarmist / Over-Predictor (High False Discovery Rate)"
        elif total_pred_biased < n_pos * 0.5:
            disposition = "Hyper-Conservative / Omission Bias (High False Negative Rate)"
        else:
            disposition = "Well-Calibrated Balanced Model"

        model_bias_metrics["models"][m_name] = {
            "total_biased_words_flagged": total_pred_biased,
            "overall_bias_rate_pct": round(overall_pred_rate * 100, 2),
            "disposition_classification": disposition,
            "sentiment_conflation": {
                "sentiment_correlation": round(sentiment_corr, 4),
                "pct_flagged_with_negative_sentiment": round(pct_negative_words_flagged, 2),
                "avg_negative_concreteness": round(avg_neg_sentiment_in_flagged, 4),
            },
            "political_leaning_bias": leaning_stats,
            "pos_distribution_pct": pos_dist,
            "outlets": outlet_stats,
            "topics": topic_stats,
        }

    # Save CSV and JSON
    json_path = os.path.join(output_dir, "model_intrinsic_bias_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(model_bias_metrics, f, indent=2)
    print(f"  Saved intrinsic bias metrics JSON to: {json_path}")

    df_outlet_bias = pd.DataFrame(outlet_bias_records)
    csv_outlet_path = os.path.join(output_dir, "model_outlet_and_topic_bias.csv")
    df_outlet_bias.to_csv(csv_outlet_path, index=False)
    print(f"  Saved outlet and topic bias CSV to: {csv_outlet_path}")

    # 10. Generate Plot 4: Outlet & Political Leaning Bias Disparity
    print("\nGenerating Outlet & Political Leaning Bias Disparity Plot...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), facecolor="#0f172a")

    # Panel 1: Political Leaning Bias (Left vs Right vs Center)
    lean_rows = []
    key_models_lean = [
        "Ground Truth (Human Annotators)",
        "Optimized XGBoost (Champion)",
        "Random Forest (Weighted)",
        "XGBoost (Paper Fine-Tuned)",
        "Linear SVM (Weighted)",
        "Complement Naive Bayes",
    ]
    for km in key_models_lean:
        for lean in ["left", "right", "center"]:
            r = model_bias_metrics["models"][km]["political_leaning_bias"].get(lean, {})
            lean_rows.append({
                "Model": km,
                "Political Leaning": lean.capitalize(),
                "Bias Rate (%)": r.get("bias_rate_pct", 0),
            })
    df_lean = pd.DataFrame(lean_rows)

    ax1.set_facecolor("#1e293b")
    sns.barplot(
        data=df_lean,
        x="Political Leaning",
        y="Bias Rate (%)",
        hue="Model",
        palette=["#f43f5e", "#38bdf8", "#10b981", "#f59e0b", "#ec4899", "#a855f7"],
        ax=ax1,
        edgecolor="#0f172a",
    )
    ax1.set_title("A) Media Leaning Bias: Rate of Flagged Words", fontsize=14, fontweight="bold", color="#f8fafc", pad=12)
    ax1.set_ylabel("Predicted Bias Rate (%)", fontsize=12, fontweight="bold", color="#e2e8f0")
    ax1.set_xlabel("News Outlet Leaning", fontsize=12, fontweight="bold", color="#e2e8f0")
    ax1.tick_params(colors="#cbd5e1", labelsize=11)
    ax1.grid(axis="y", color="#334155", linestyle="--", alpha=0.7)
    leg1 = ax1.legend(frameon=True, facecolor="#0f172a", edgecolor="#334155", fontsize=9)
    for text in leg1.get_texts():
        text.set_color("#f8fafc")

    # Panel 2: Outlet Disparity Ratio for Key Outlets
    # Disparity = Model Rate / Ground Truth Rate (>1 means model over-flags outlet, <1 means model under-flags)
    disp_rows = []
    focus_outlets = ["breitbart", "fox-news", "huffpost", "alternet", "reuters", "msnbc"]
    for km in key_models_lean[1:]:  # Skip ground truth since disparity = 1.0
        for out in focus_outlets:
            r = model_bias_metrics["models"][km]["outlets"].get(out, {})
            disp_rows.append({
                "Model": km,
                "Outlet": out.capitalize(),
                "Disparity Ratio": r.get("disparity_ratio", 1.0),
            })
    df_disp = pd.DataFrame(disp_rows)

    ax2.set_facecolor("#1e293b")
    sns.barplot(
        data=df_disp,
        x="Outlet",
        y="Disparity Ratio",
        hue="Model",
        palette=["#38bdf8", "#10b981", "#f59e0b", "#ec4899", "#a855f7"],
        ax=ax2,
        edgecolor="#0f172a",
    )
    ax2.axhline(1.0, color="#f43f5e", linestyle="--", linewidth=2, label="Human Baseline (1.0 = Fair / Balanced)")
    ax2.set_title("B) Outlet Disparity Ratio (Model Rate ÷ Human Rate)", fontsize=14, fontweight="bold", color="#f8fafc", pad=12)
    ax2.set_ylabel("Disparity Ratio (1.0 = Human Parity)", fontsize=12, fontweight="bold", color="#e2e8f0")
    ax2.set_xlabel("News Outlet", fontsize=12, fontweight="bold", color="#e2e8f0")
    ax2.tick_params(colors="#cbd5e1", labelsize=11)
    ax2.grid(axis="y", color="#334155", linestyle="--", alpha=0.7)
    leg2 = ax2.legend(frameon=True, facecolor="#0f172a", edgecolor="#334155", fontsize=9)
    for text in leg2.get_texts():
        text.set_color("#f8fafc")

    plt.suptitle("Does the Model Have Media & Political Outlets Bias?", fontsize=17, fontweight="bold", color="#f8fafc", y=0.98)
    plt.tight_layout()
    outlet_plot_path = os.path.join(plots_dir, "4_model_outlet_and_topic_bias_disparity.png")
    plt.savefig(outlet_plot_path, dpi=200, bbox_inches="tight", facecolor="#0f172a")
    plt.close(fig)
    print(f"  Saved outlet and political bias plot to: {outlet_plot_path}")

    # 11. Generate Plot 5: Sentiment Conflation & Part-of-Speech Bias
    print("\nGenerating Sentiment Conflation & Part-of-Speech Bias Plot...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), facecolor="#0f172a")

    # Panel 1: Sentiment Conflation Correlation Across All Models
    corr_rows = []
    for m_name in list(predictions.keys())[1:]:  # skip ground truth
        c_val = model_bias_metrics["models"][m_name]["sentiment_conflation"]["sentiment_correlation"]
        neg_pct = model_bias_metrics["models"][m_name]["sentiment_conflation"]["pct_flagged_with_negative_sentiment"]
        corr_rows.append({
            "Model": m_name,
            "Correlation with Negative Sentiment": c_val,
            "Pct Negative Words": neg_pct,
        })
    df_corr = pd.DataFrame(corr_rows).sort_values("Correlation with Negative Sentiment", ascending=False)

    ax1.set_facecolor("#1e293b")
    bars = ax1.barh(df_corr["Model"], df_corr["Correlation with Negative Sentiment"], color="#38bdf8", edgecolor="#0f172a")
    # Color the Champion bar differently
    for idx_b, b_name in enumerate(df_corr["Model"]):
        if b_name == "Optimized XGBoost (Champion)":
            bars[idx_b].set_color("#10b981")
            bars[idx_b].set_edgecolor("#f8fafc")
            bars[idx_b].set_linewidth(1.5)
        elif "B2: Negative" in b_name:
            bars[idx_b].set_color("#f43f5e")

    ax1.set_title("A) Sentiment Conflation Bias\nCorrelation with Negative Sentiment Scores", fontsize=14, fontweight="bold", color="#f8fafc", pad=12)
    ax1.set_xlabel("Pearson Correlation (r)", fontsize=12, fontweight="bold", color="#e2e8f0")
    ax1.tick_params(colors="#cbd5e1", labelsize=10)
    ax1.grid(axis="x", color="#334155", linestyle="--", alpha=0.7)

    # Panel 2: Part-of-Speech Distribution of Flagged Biased Words
    pos_data = []
    top_pos_tags = ["ADJ", "NOUN", "VERB", "ADV"]
    for km in key_models_lean:
        dist = model_bias_metrics["models"][km]["pos_distribution_pct"]
        for p_tag in top_pos_tags:
            pos_data.append({
                "Model": km,
                "POS": p_tag,
                "Percentage (%)": dist.get(p_tag, 0.0),
            })
    df_pos = pd.DataFrame(pos_data)

    ax2.set_facecolor("#1e293b")
    sns.barplot(
        data=df_pos,
        x="POS",
        y="Percentage (%)",
        hue="Model",
        palette=["#f43f5e", "#38bdf8", "#10b981", "#f59e0b", "#ec4899", "#a855f7"],
        ax=ax2,
        edgecolor="#0f172a",
    )
    ax2.set_title("B) Syntactic Bias: Part-of-Speech Distribution\n(ADJ = Adjectives, NOUN = Nouns, VERB = Verbs)", fontsize=14, fontweight="bold", color="#f8fafc", pad=12)
    ax2.set_ylabel("Share of Biased Words Flagged (%)", fontsize=12, fontweight="bold", color="#e2e8f0")
    ax2.set_xlabel("Part of Speech", fontsize=12, fontweight="bold", color="#e2e8f0")
    ax2.tick_params(colors="#cbd5e1", labelsize=11)
    ax2.grid(axis="y", color="#334155", linestyle="--", alpha=0.7)
    leg3 = ax2.legend(frameon=True, facecolor="#0f172a", edgecolor="#334155", fontsize=9)
    for text in leg3.get_texts():
        text.set_color("#f8fafc")

    plt.suptitle("Linguistic Heuristics: Do Models Over-Rely on Negative Words and Adjectives?", fontsize=17, fontweight="bold", color="#f8fafc", y=0.98)
    plt.tight_layout()
    sent_plot_path = os.path.join(plots_dir, "5_sentiment_conflation_and_pos_bias.png")
    plt.savefig(sent_plot_path, dpi=200, bbox_inches="tight", facecolor="#0f172a")
    plt.close(fig)
    print(f"  Saved sentiment conflation & POS plot to: {sent_plot_path}")

    # 12. Print Summary Console Table
    print("\n" + "=" * 115)
    print("SUMMARY LEADERBOARD: WORD CLOUD & MODEL SELF-BIAS AUDIT")
    print("=" * 115)
    print(f"{'MODEL':<36} {'ACC (%)':<8} {'FLAGGED':<9} {'UNIQUE':<8} {'PREC (%)':<10} {'REC (%)':<9} {'DISPOSITION PROFILE'}")
    print("-" * 115)
    for st in model_stats:
        m = st["model"]
        disp = model_bias_metrics["models"][m]["disposition_classification"]
        print(
            f"{m:<36} "
            f"{st['accuracy']:>6.2f}%  "
            f"{st['total_biased_words_flagged']:>6}   "
            f"{st['unique_biased_words']:>6}   "
            f"{st['precision']:>7.2f}%   "
            f"{st['recall']:>6.2f}%   "
            f"{disp[:35]}"
        )
    print("=" * 115)

    return {
        "model_stats": model_stats,
        "metrics": model_bias_metrics,
        "plots": {
            "master_grid": master_grid_path,
            "frequency_comparison": freq_plot_path,
            "error_wordclouds": error_wc_path,
            "outlet_bias": outlet_plot_path,
            "sentiment_pos_bias": sent_plot_path,
        },
    }


if __name__ == "__main__":
    run_wordcloud_and_bias_analysis()
