#!/usr/bin/env python
"""
Interactive Bias Prediction CLI / Tool for News Sentences and Words.

Uses the trained Champion Optimized Model to analyze any sentence or word:
- Identifies whether each token is bias-inducing.
- Displays prediction confidence probabilities.
- Explains the linguistic trigger features (sentiment, subjectivity, boosters, etc.).
"""

import os
import sys
import argparse
from typing import List, Dict, Any

import numpy as np
import pandas as pd

import logging
import warnings
warnings.filterwarnings("ignore")
logging.disable(logging.INFO)

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from optimized_model.model_pipeline import OptimizedNewsBiasClassifier
from src.data.dataset_loader import MBICDataLoader

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "saved_model", "optimized_xgboost.joblib"
)


class BiasInferenceEngine:
    """Inference engine to evaluate words and sentences using the trained champion model."""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        if not os.path.exists(model_path):
            # If not yet trained, train it now
            from optimized_model.train_optimized_model import train_and_evaluate
            train_and_evaluate(os.path.dirname(os.path.abspath(__file__)))

        self.model = OptimizedNewsBiasClassifier.load(model_path)
        
        # Load dataset feature template for realistic feature representations
        loader = MBICDataLoader()
        self.df_tokens = loader.load_token_dataset(drop_missing_tfidf=True)
        # Create quick lookup dictionary by lowercase word
        self.word_lookup = {}
        for _, row in self.df_tokens.iterrows():
            w = str(row.get("text_low", "")).lower()
            if w and w not in self.word_lookup:
                self.word_lookup[w] = row

        # Default feature row for unknown words
        self.default_row = self.df_tokens.iloc[0].copy()
        for col in self.default_row.index:
            if pd.api.types.is_numeric_dtype(type(self.default_row[col])):
                self.default_row[col] = 0

    def analyze_tokens(self, tokens: List[str]) -> List[Dict[str, Any]]:
        """Predict bias on a list of tokens."""
        rows = []
        for t in tokens:
            t_low = t.lower()
            if t_low in self.word_lookup:
                r = self.word_lookup[t_low].copy()
                r["text"] = t
            else:
                r = self.default_row.copy()
                r["text"] = t
                r["text_low"] = t_low
                # Basic heuristic matching for out-of-vocabulary words
                if t_low in ["disastrous", "corrupt", "racist", "scandal", "crooked", "evil", "terrorist", "murder"]:
                    r["negative_conc"] = 1.0
                    r["strong_subj"] = 1.0
                    r["bias_lexicon"] = 1.0
                elif t_low in ["unanimous", "peaceful", "hero", "liberty", "freedom"]:
                    r["positive_conc"] = 1.0
                    r["strong_subj"] = 1.0
            rows.append(r)

        df_query = pd.DataFrame(rows)
        probs = self.model.predict_proba(df_query)[:, 1]
        preds = (probs >= self.model.threshold).astype(int)

        results = []
        for idx, t in enumerate(tokens):
            p = float(probs[idx])
            is_bias = bool(preds[idx] == 1)
            row = rows[idx]
            
            # Detect key active cues
            cues = []
            if row.get("negative_conc", 0) > 0:
                cues.append("Negative Sentiment")
            if row.get("strong_subj", 0) > 0:
                cues.append("Strong Subjectivity")
            if row.get("bias_lexicon", 0) > 0:
                cues.append("Bias Lexicon Hit")
            if row.get("boosters", 0) > 0:
                cues.append("Certainty Booster")
            if row.get("hyperbolic_terms", 0) > 0:
                cues.append("Hyperbolic Word")
            if row.get("kill_verbs", 0) > 0:
                cues.append("Kill Verb")
            if not cues:
                cues.append("Factual / Neutral Context")

            results.append({
                "word": t,
                "is_biased": is_bias,
                "bias_probability": round(p, 4),
                "bias_probability_pct": f"{p*100:.1f}%",
                "linguistic_cues": ", ".join(cues),
            })

        return results

    def analyze_sentence(self, sentence: str) -> List[Dict[str, Any]]:
        """Tokenize sentence and predict bias for each word."""
        # Simple punctuation-aware tokenization
        import re
        tokens = re.findall(r"\b[\w'-]+\b|[^\w\s]", sentence)
        # Filter to alphanumeric words
        words = [t for t in tokens if re.search(r"\w", t)]
        return self.analyze_tokens(words)


def print_sentence_analysis(sentence: str, results: List[Dict[str, Any]], threshold: float):
    """Format and print beautiful terminal output."""
    print("\n" + "=" * 90)
    print("[NEWS BIAS TOKEN ANALYSIS]")
    print("=" * 90)
    print(f"Sentence: \"{sentence}\"")
    print(f"Decision Threshold: tau = {threshold:.2f}")
    print("-" * 90)
    print(f"{'Word':<18} | {'Classification':<16} | {'Probability':<12} | {'Linguistic Features / Cues'}")
    print("-" * 90)

    biased_found = []
    for r in results:
        status = "[BIASED]" if r["is_biased"] else "Neutral"
        print(f"{r['word']:<18} | {status:<16} | {r['bias_probability_pct']:<12} | {r['linguistic_cues']}")
        if r["is_biased"]:
            biased_found.append(f"'{r['word']}' ({r['bias_probability_pct']})")

    print("=" * 90)
    if biased_found:
        print(f"[!] Detected {len(biased_found)} Bias-Inducing Words: {', '.join(biased_found)}")
    else:
        print("[OK] No bias-inducing words detected in this sentence (All words below threshold).")
    print("=" * 90 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Predict bias-inducing words in news statements.")
    parser.add_argument("--sentence", type=str, default=None, help="News sentence to analyze.")
    parser.add_argument("--word", type=str, default=None, help="Single word to analyze.")
    args = parser.parse_args()

    engine = BiasInferenceEngine()

    if args.word:
        res = engine.analyze_tokens([args.word])
        print_sentence_analysis(args.word, res, engine.model.threshold)
    elif args.sentence:
        res = engine.analyze_sentence(args.sentence)
        print_sentence_analysis(args.sentence, res, engine.model.threshold)
    else:
        # Run demonstration examples from Spinde et al. (2021)
        demo_sentences = [
            "Fox News attempts to paint the Green New Deal as disastrous.",
            "Practicing pro-life litigators know that Trump judges are saving lives.",
            "The administration was effectively delivering death sentences to immigrant families.",
            "The committee held a routine meeting on Monday to review annual financial reports."
        ]

        print("\n" + "=" * 90)
        print("RUNNING DEMONSTRATION ON BENCHMARK NEWS STATEMENTS (Spinde et al., 2021)")
        print("=" * 90)

        for s in demo_sentences:
            res = engine.analyze_sentence(s)
            print_sentence_analysis(s, res, engine.model.threshold)


if __name__ == "__main__":
    main()
