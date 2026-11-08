#!/usr/bin/env python
"""
Give the authors' released sentence classifiers the audit our word models had (docs/AUDIT.md).

The Media Bias Group released sentence-level bias classifiers fine-tuned on BABE, its expert-labelled
dataset. Each gives one probability per sentence that the sentence is biased:
  - DA-RoBERTa, from the BABE paper (Spinde et al., 2021): mediabiasgroup/da-roberta-babe-ft
  - MAGPIE (Horych et al., 2024): mediabiasgroup/magpie-babe-ft

Two tests, with our teacher and INT8 student alongside; a word model's sentence score is its highest
word score, and it flags a sentence when it flags any word:
  1. Counterfactual swaps: the sentences of audit_models.py, rewritten to name the other political
     side. A symmetric model gives both versions the same score. The classifiers may have trained on
     an original sentence (MBIC's sentences are in BABE) but never on its swapped version, so the
     swaps are also summarised for originals that are not in BABE, and by the side each original named.
  2. Outlet leaning: detection by outlet rating, on BABE's test split without MBIC's sentences. The
     model cards do not say which BABE sentences the classifiers were trained on, so if they saw the
     test split, their scores here are optimistic. Our models never saw BABE's labels.

The classifiers' training data includes MBIC's sentences with expert labels, so the annotator-group
test of docs/AUDIT.md, which needs unseen MBIC sentences, does not apply to them.

Usage:
    python scripts/audit_sentence_classifiers.py
    python scripts/audit_sentence_classifiers.py --max-sentences 300    # a quick run
    python scripts/audit_sentence_classifiers.py --reuse-scores         # remake the tables from saved scores
"""

import argparse
import importlib.util
import json
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.data.articles import article_pool  # noqa: E402
from src.data.dataset_loader import MBICDataLoader  # noqa: E402
from src.evaluation.word_metrics import detection_metrics  # noqa: E402
from src.models.onnx_token_model import OnnxTokenModel  # noqa: E402


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = _script("audit_models")
babe_test = _script("evaluate_babe")

CLASSIFIERS = {  # name: (Hugging Face repository, the version tested)
    "da_roberta": ("mediabiasgroup/da-roberta-babe-ft", "a1e1b9e2ac97f87387fb0b78e5a1914b5e25532f"),
    "magpie": ("mediabiasgroup/magpie-babe-ft", "49023cca9abfa0ae2db65f17deae5d061a874b04"),
}


def biased_index(id2label: Dict) -> int:
    """The output that means 'biased': a label naming bias (not 'non-biased'), else the second one, as in BABE."""
    for index, name in (id2label or {}).items():
        name = str(name).lower()
        if "bias" in name and not name.startswith(("non", "un", "not")):
            return int(index)
    return 1


class SentenceClassifier:
    """P(biased) for each sentence, from a Hugging Face sequence classifier, in batches."""

    def __init__(self, repo: str, revision: str, batch_size: int = 32, max_length: int = 128):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)
        self.model = AutoModelForSequenceClassification.from_pretrained(repo, revision=revision, use_safetensors=True)
        self.model.to(self.device).eval()
        self.biased = biased_index(self.model.config.id2label)
        self.batch_size, self.max_length = batch_size, max_length

    def probabilities(self, texts: List[str]) -> np.ndarray:
        out = []
        with self.torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch = self.tokenizer(texts[i:i + self.batch_size], padding=True, truncation=True,
                                       max_length=self.max_length, return_tensors="pt").to(self.device)
                logits = self.model(**batch).logits.float()
                out.append(self.torch.softmax(logits, dim=-1)[:, self.biased].cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)


def word_model_scores(model: OnnxTokenModel, texts: List[str]) -> np.ndarray:
    return np.array([babe_test.sentence_score(model.score(t)[1]) for t in texts])


def pair_name(text: str) -> str:
    """Which left/right pair a sentence mentions, or 'several' if more than one."""
    found = {w.lower() for w in audit.PAIR_WORD.findall(text)}
    pairs = {f"{left.lower()} / {right.lower()}" for left, right in audit.PAIRS
             if left.lower() in found or right.lower() in found}
    return pairs.pop() if len(pairs) == 1 else "several"


def swap_pairs(texts: List[str]) -> pd.DataFrame:
    """Each sentence that mentions one side, as a left-coded and a right-coded version."""
    rows = []
    for text in texts:
        made = audit.counterfactual(text)
        if made is None:
            continue
        swapped, side = made
        left, right = (text, swapped) if side == "left" else (swapped, text)
        rows.append({"original_side": side, "pair": pair_name(text), "left_text": left, "right_text": right})
    return pd.DataFrame(rows)


def swap_summary(swaps: pd.DataFrame, models: List[str], thresholds: Dict[str, float]) -> pd.DataFrame:
    rows = []
    for m in models:
        left, right = swaps[f"{m}_left"], swaps[f"{m}_right"]
        diff = right - left
        t = thresholds[m]
        rows.append({"model": m, "sentence_pairs": int(len(swaps)), "score_left": float(left.mean()),
                     "score_right": float(right.mean()), "right_minus_left": float(diff.mean()),
                     "p": float(wilcoxon(diff).pvalue) if (diff != 0).sum() > 10 else None,
                     "mean_abs_change": float(diff.abs().mean()),
                     "decision_flips": float(((left >= t) != (right >= t)).mean()),
                     "flagged_left_only": float(((left >= t) & (right < t)).mean()),
                     "flagged_right_only": float(((right >= t) & (left < t)).mean())})
    return pd.DataFrame(rows)


def mark_seen_originals(swaps: pd.DataFrame, babe_texts: set) -> pd.DataFrame:
    """Whether each pair's original sentence is in BABE, where the classifiers may have trained on it."""
    original = swaps.left_text.where(swaps.original_side == "left", swaps.right_text)
    return swaps.assign(original_in_babe=original.map(babe_test.normalize_text).isin(babe_texts))


def swap_controls(swaps: pd.DataFrame, models: List[str]) -> pd.DataFrame:
    """Right minus left by whether the original is in BABE and which side it named."""
    rows = []
    for (in_babe, side), part in swaps.groupby(["original_in_babe", "original_side"]):
        rows.append({"original_in_babe": bool(in_babe), "original_side": side, "n": int(len(part)),
                     **{m: float((part[f"{m}_right"] - part[f"{m}_left"]).mean()) for m in models}})
    return pd.DataFrame(rows)


def swap_by_pair(swaps: pd.DataFrame, models: List[str]) -> pd.DataFrame:
    rows = []
    for pair, part in swaps.groupby("pair", sort=False):
        row = {"pair": pair, "n": int(len(part))}
        for m in models:
            row[f"{m}_left"] = float(part[f"{m}_left"].mean())
            row[f"{m}_right"] = float(part[f"{m}_right"].mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("n", ascending=False)


def outlet_table(test: pd.DataFrame, models: List[str], thresholds: Dict[str, float]) -> pd.DataFrame:
    rows = []
    for leaning, part in [("all", test)] + list(test.groupby("leaning", sort=True)):
        for m in models:
            y, s = part.label.to_numpy(), part[m].to_numpy()
            d = detection_metrics(y, s, (s >= thresholds[m]).astype(int))
            rows.append({"leaning": leaning, "model": m, "sentences": int(len(part)), "biased_share": float(y.mean()),
                         "f1": d["f1"], "precision": d["precision"], "recall": d["recall"], "roc_auc": d["roc_auc"],
                         "mean_score": float(s.mean()), "flagged_share": d["flagged_share"]})
    return pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--student", default=os.path.join(ROOT, "outputs", "student", "final"))
    parser.add_argument("--teacher", default=os.path.join(ROOT, "outputs", "student", "final_teacher"))
    parser.add_argument("--student-cv", default=os.path.join(ROOT, "outputs", "student", "cv"))
    parser.add_argument("--max-sentences", type=int, help="limit the counterfactual sentences")
    parser.add_argument("--reuse-scores", action="store_true", help="read the scores saved by an earlier run")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--out", default=os.path.join(ROOT, "outputs", "audit", "classifiers"))
    args = parser.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    loader = MBICDataLoader()
    labeled = loader.load_raw_sentences()
    texts = list(dict.fromkeys(labeled.sentence.astype(str).tolist() + article_pool(labeled).sentence.tolist()))
    swaps = swap_pairs(texts)
    if args.max_sentences:
        swaps = swaps.head(args.max_sentences)
    babe = babe_test.load_babe()
    _, new, _ = babe_test.split_by_mbic(babe, labeled)
    test = new[new.split == "test"].reset_index(drop=True)
    print(f"{len(swaps)} counterfactual pairs; {len(test)} BABE test sentences not in MBIC")

    with open(os.path.join(args.student_cv, "metrics.json"), encoding="utf-8") as f:
        cv = json.load(f)["results"]
    thresholds = {"teacher": cv["teacher"]["threshold_mean"], "s_distil_int8": cv["s_distil_int8"]["threshold_mean"],
                  **{name: 0.5 for name in CLASSIFIERS}}
    models = list(CLASSIFIERS) + ["teacher", "s_distil_int8"]
    saved_swaps = os.path.join(args.out, "swap_sentence_scores.csv")
    saved_test = os.path.join(args.out, "babe_test_scores.csv")
    if args.reuse_scores:
        swaps = pd.read_csv(saved_swaps)
        test = test.merge(pd.read_csv(saved_test)[["babe_index"] + models], on="babe_index")
    else:
        for name, (repo, revision) in CLASSIFIERS.items():
            classifier = SentenceClassifier(repo, revision)
            print(f"{name}: scoring on {classifier.device}")
            swaps[f"{name}_left"] = classifier.probabilities(swaps.left_text.tolist())
            swaps[f"{name}_right"] = classifier.probabilities(swaps.right_text.tolist())
            test[name] = classifier.probabilities(test.text.astype(str).tolist())
            del classifier
        for name, model in (("teacher", OnnxTokenModel(args.teacher, "fp32", args.threads)),
                            ("s_distil_int8", OnnxTokenModel(args.student, "int8", args.threads))):
            print(f"{name}: scoring on the CPU")
            swaps[f"{name}_left"] = word_model_scores(model, swaps.left_text.tolist())
            swaps[f"{name}_right"] = word_model_scores(model, swaps.right_text.tolist())
            test[name] = word_model_scores(model, test.text.astype(str).tolist())
        swaps.to_csv(saved_swaps, index=False)
        test[["babe_index", "leaning", "label"] + models].to_csv(saved_test, index=False)

    swaps = mark_seen_originals(swaps, set(babe.text.map(babe_test.normalize_text)))
    summary = swap_summary(swaps, models, thresholds)
    unseen = swap_summary(swaps[~swaps.original_in_babe], models, thresholds)
    controls = swap_controls(swaps, models)
    by_pair = swap_by_pair(swaps, models)
    outlets = outlet_table(test, models, thresholds)
    summary.to_csv(os.path.join(args.out, "swap_summary.csv"), index=False)
    unseen.to_csv(os.path.join(args.out, "swap_summary_not_in_babe.csv"), index=False)
    controls.to_csv(os.path.join(args.out, "swap_controls.csv"), index=False)
    by_pair.to_csv(os.path.join(args.out, "swap_by_pair.csv"), index=False)
    outlets.to_csv(os.path.join(args.out, "outlet_breakdown.csv"), index=False)
    info = {"classifiers": {k: {"repo": r, "revision": v} for k, (r, v) in CLASSIFIERS.items()},
            "thresholds": thresholds, "swap_pairs": int(len(swaps)),
            "swap_pairs_not_in_babe": int((~swaps.original_in_babe).sum()), "babe_test_sentences": int(len(test))}
    with open(os.path.join(args.out, "info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    report = "\n".join([
        "## Counterfactual swaps: sentence scores", "", summary.round(4).to_markdown(index=False), "",
        "## The same, for originals not in BABE", "", unseen.round(4).to_markdown(index=False), "",
        "## Right minus left, by the original's side and whether it is in BABE", "",
        controls.round(4).to_markdown(index=False), "",
        "## By pair", "", by_pair.round(3).to_markdown(index=False), "",
        "## Outlet leaning: BABE's test split, without MBIC's sentences", "", outlets.round(3).to_markdown(index=False), "",
    ])
    with open(os.path.join(args.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
