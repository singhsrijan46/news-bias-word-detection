#!/usr/bin/env python
"""
Distil the soft-label teacher (M2) into a small student for the UNO Q, and measure what that costs.

  teacher         M2: RoBERTa-base trained on vote shares, as in train_token_models.py
  s_gold          MiniLM-L6 trained on the annotators' vote shares only
  s_distil        MiniLM-L6 trained on the vote shares plus the teacher's scores for
                  unlabeled sentences from MBIC's full articles
  s_*_int8        each student exported to ONNX and quantized to INT8, run the way it runs on the board

The folds are the same 10 sentence-grouped folds as train_token_models.py (seed 0). In each
fold the teacher sees only the training sentences, and the unlabeled pool leaves out every
article that contains a test sentence. Thresholds are chosen on held-out training sentences.

Usage:
    python scripts/train_student.py                # 10 folds, about an hour on a 4 GB GPU
    python scripts/train_student.py --quick        # 2 folds, a small pool, 1 epoch
    python scripts/train_student.py --final        # train on everything and export the board's model
"""

import argparse
import json
import os
import shutil
import sys
import time
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.data.articles import article_pool, normalize, sentence_articles  # noqa: E402
from src.data.dataset_loader import MBICDataLoader  # noqa: E402
from src.data.word_targets import SentenceTargets, align_paper_words, build_sentence_targets  # noqa: E402
from src.evaluation.word_metrics import (  # noqa: E402
    best_f1_threshold, calibration_metrics, detection_metrics, paired_comparison, per_fold_metrics,
)
from src.models.onnx_token_model import OnnxTokenModel, export, word_spans  # noqa: E402
from src.utils.helpers import setup_logger  # noqa: E402
from train_token_models import NATURAL_SHARE, make_folds, split_validation, word_scores  # noqa: E402

logger = setup_logger("student")

TEACHER = "FacebookAI/roberta-base"
STUDENT = "nreimers/MiniLM-L6-H384-uncased"
MODELS = ["teacher", "s_gold", "s_gold_int8", "s_distil", "s_distil_int8"]
NAMES = {"teacher": "Teacher M2 (RoBERTa-base, 124M)", "s_gold": "Student, vote shares only (MiniLM-L6, 23M)",
         "s_gold_int8": "  the same in INT8", "s_distil": "Student, distilled (MiniLM-L6, 23M)",
         "s_distil_int8": "  the same in INT8 (runs on the board)"}


def unlabeled(sentences: List[str]) -> List[SentenceTargets]:
    """Pool sentences in the shape TokenModel expects; their votes are unused."""
    out = []
    for text in sentences:
        spans = word_spans(text)
        out.append(SentenceTargets(-1, text, "", spans, np.zeros(len(spans), dtype=np.int64), 1))
    return out


def as_targets(predictions: List[np.ndarray]) -> List[np.ndarray]:
    """Teacher scores as training targets; words the teacher could not score (NaN) are ignored (-1)."""
    return [np.where(np.isnan(p), -1.0, p).astype(np.float32) for p in predictions]


def train_teacher(train: List[SentenceTargets], epochs: int, seed: int):
    from src.models.token_transformer import TokenModel, TrainConfig

    return TokenModel(TrainConfig(model_name=TEACHER, epochs=epochs, seed=seed)).fit(
        train, [t.shares.astype(np.float32) for t in train])


def train_student(sentences: List[SentenceTargets], targets: List[np.ndarray], epochs: int, seed: int):
    from src.models.token_transformer import TokenModel, TrainConfig

    config = TrainConfig(model_name=STUDENT, epochs=epochs, batch_size=32, learning_rate=1e-4, seed=seed)
    return TokenModel(config).fit(sentences, targets)


def student_data(train: List[SentenceTargets], pool: List[SentenceTargets], pseudo: List[np.ndarray],
                 gold_repeats: int):
    """Gold sentences (repeated, so they are not swamped by the pool) plus teacher-scored pool sentences."""
    gold = [t.shares.astype(np.float32) for t in train]
    return train * gold_repeats + pool, gold * gold_repeats + pseudo


def run_cv(args, targets, paper, found, fold_of, pool_df, article_of):
    oof = {m: np.full(len(paper), np.nan) for m in MODELS}
    flags = {m: np.full(len(paper), -1) for m in MODELS}
    thresholds: Dict[str, List[float]] = {m: [] for m in MODELS}
    n_run = args.max_folds or args.folds
    for k in range(n_run):
        started = time.time()
        test_ids = np.where(fold_of == k)[0]
        train_ids, val_ids = split_validation(np.where(fold_of != k)[0], 0.1, args.seed + k)
        in_set = lambda ids: np.where(np.isin(paper.sentence_id, ids) & found)[0]
        val_rows, test_rows = in_set(val_ids), in_set(test_ids)
        train = [targets[i] for i in train_ids]
        val, test = [targets[i] for i in val_ids], [targets[i] for i in test_ids]

        test_articles = {article_of.get(normalize(t.text), -1) for t in test} - {-1}
        fold_pool = pool_df[~pool_df.article_id.isin(test_articles)]
        fold_pool = fold_pool.sample(n=min(args.pool_size, len(fold_pool)), random_state=args.seed + k)
        pool = unlabeled(fold_pool.sentence.tolist())

        teacher = train_teacher(train, args.teacher_epochs, args.seed)
        preds = {"teacher": (teacher.predict(val), teacher.predict(test))}
        pseudo = as_targets(teacher.predict(pool))
        teacher.release()
        del teacher

        for variant, (sents, ys, epochs) in {
            "s_gold": (train, [t.shares.astype(np.float32) for t in train], args.gold_epochs),
            "s_distil": (*student_data(train, pool, pseudo, args.gold_repeats), args.distil_epochs),
        }.items():
            student = train_student(sents, ys, epochs, args.seed)
            preds[variant] = (student.predict(val), student.predict(test))
            fold_dir = export(student, os.path.join(args.out, "tmp", f"fold{k}_{variant}"))
            student.release()
            del student
            onnx = OnnxTokenModel(fold_dir, "int8", threads=4)
            preds[f"{variant}_int8"] = (onnx.predict(val), onnx.predict(test))
            del onnx
            shutil.rmtree(fold_dir, ignore_errors=True)

        for m in MODELS:
            val_scores = word_scores(paper, val_rows, dict(zip(val_ids, preds[m][0])))
            test_scores = word_scores(paper, test_rows, dict(zip(test_ids, preds[m][1])))
            threshold = best_f1_threshold(paper.label4.values[val_rows], val_scores)
            thresholds[m].append(threshold)
            oof[m][test_rows] = test_scores
            flags[m][test_rows] = (test_scores >= threshold).astype(int)
        f1 = {m: detection_metrics(paper.label4.values[test_rows], oof[m][test_rows], flags[m][test_rows])["f1"]
              for m in MODELS}
        logger.info(f"fold {k + 1}/{n_run} ({len(pool)} pool sentences, {time.time() - started:.0f} s): "
                    + ", ".join(f"{m} F1 {v:.3f}" for m, v in f1.items()))
    return oof, flags, thresholds, n_run


def report_cv(args, paper, found, oof, flags, thresholds, n_run, pool_size):
    scored = np.where(found & (paper.fold >= 0) & (paper.fold < n_run))[0]
    words = paper.iloc[scored][["sentence_id", "word_index", "fold", "label4", "share"]].copy()
    for m in MODELS:
        words[m], words[f"{m}_flag"] = oof[m][scored], flags[m][scored]
    words.to_csv(os.path.join(args.out, "paper_words_oof.csv"), index=False)

    rows, results = [], {}
    for m in MODELS:
        y, s, share = words.label4.values, words[m].values, words.share.values
        metrics = {**detection_metrics(y, s, words[f"{m}_flag"].values), **calibration_metrics(y, share, s)}
        results[m] = {**metrics, "threshold_mean": float(np.mean(thresholds[m])), "thresholds": thresholds[m]}
        rows.append({"model": NAMES[m], "F1": metrics["f1"], "precision": metrics["precision"],
                     "recall": metrics["recall"], "ROC AUC": metrics["roc_auc"],
                     "avg precision": metrics["average_precision"], "ECE": metrics["ece"],
                     "share ECE": metrics["share_ece"], "Spearman vs share": metrics["spearman_vs_share"],
                     "threshold (mean)": float(np.mean(thresholds[m]))})
    table = pd.DataFrame(rows).round(3).to_markdown(index=False)
    parts = [table, ""]
    if n_run > 1:
        per_fold = per_fold_metrics(words, MODELS)
        per_fold.to_csv(os.path.join(args.out, "per_fold_metrics.csv"), index=False)
        for a, b in (("s_gold", "s_distil"), ("s_distil", "s_distil_int8"), ("teacher", "s_distil")):
            pair = paired_comparison(per_fold, a, b)
            pair.to_csv(os.path.join(args.out, f"paired_{a}_{b}.csv"), index=False)
            parts += [f"## {b} against {a}, paired by fold", "", pair.round(4).to_markdown(index=False), ""]
    text = "\n".join(parts)
    with open(os.path.join(args.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(text)
    with open(os.path.join(args.out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"settings": {**vars(args), "pool_size_used": pool_size, "paper_words_scored": int(len(scored))},
                   "results": results}, f, indent=2)
    print("\n" + text)


def run_final(args, targets, pool_df):
    """Train on all 1,700 sentences and the whole pool, and export the model the board runs."""
    started = time.time()
    teacher = train_teacher(targets, args.teacher_epochs, args.seed)
    pool = unlabeled(pool_df.sentence.tolist())
    pseudo = as_targets(teacher.predict(pool))
    export(teacher, args.out_final_teacher, {"role": "teacher", "training_sentences": len(targets)})
    teacher.release()
    del teacher
    sents, ys = student_data(list(targets), pool, pseudo, args.gold_repeats)
    student = train_student(sents, ys, args.distil_epochs, args.seed)

    info = {"teacher": TEACHER, "training_sentences": len(targets), "pool_sentences": len(pool),
            "natural_share": NATURAL_SHARE}
    cv_metrics = os.path.join(args.out_cv, "metrics.json")
    if os.path.exists(cv_metrics):  # carry the cross-validated threshold to the board
        with open(cv_metrics, encoding="utf-8") as f:
            cv = json.load(f)["results"].get("s_distil_int8", {})
        info.update({"flag_threshold": cv.get("threshold_mean"), "cv_f1": cv.get("f1"),
                     "cv_roc_auc": cv.get("roc_auc"), "cv_share_ece": cv.get("share_ece")})
    out = export(student, args.out_final, info)
    logger.info(f"final student exported to {os.path.normpath(out)} ({time.time() - started:.0f} s)")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--max-folds", type=int, help="run only the first N folds")
    parser.add_argument("--pool-size", type=int, default=20000, help="unlabeled sentences per fold")
    parser.add_argument("--teacher-epochs", type=int, default=4)
    parser.add_argument("--gold-epochs", type=int, default=10)
    parser.add_argument("--distil-epochs", type=int, default=3)
    parser.add_argument("--gold-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true", help="2 folds, 2,000 pool sentences, 1 epoch each")
    parser.add_argument("--final", action="store_true", help="train on everything and export for the board")
    parser.add_argument("--out", default=os.path.join(ROOT, "outputs", "student", "cv"))
    parser.add_argument("--out-final", default=os.path.join(ROOT, "outputs", "student", "final"))
    parser.add_argument("--out-final-teacher", default=os.path.join(ROOT, "outputs", "student", "final_teacher"),
                        help="--final also exports the teacher here, for the audits")
    args = parser.parse_args(argv)
    args.out_cv = args.out
    if args.quick:
        args.max_folds, args.pool_size = 2, 2000
        args.teacher_epochs = args.gold_epochs = args.distil_epochs = 1
    os.makedirs(args.out, exist_ok=True)

    loader = MBICDataLoader()
    labeled = loader.load_raw_sentences()
    targets = build_sentence_targets(loader.load_annotations())
    pool_df = article_pool(labeled)
    article_of = sentence_articles(labeled)
    logger.info(f"{len(targets)} labeled sentences; {len(pool_df)} unlabeled article sentences")
    if args.final:
        run_final(args, targets, pool_df)
        return

    dt = loader.load_token_dataset(drop_missing_tfidf=False).reset_index(drop=True)
    paper = align_paper_words(targets, dt).set_index("dt_index").loc[dt.index].reset_index()
    found = (paper.sentence_id >= 0) & (paper.word_index >= 0)
    paper["share"] = [targets[s].shares[w] if ok else np.nan
                      for s, w, ok in zip(paper.sentence_id, paper.word_index, found)]
    fold_of = make_folds(len(targets), args.folds, args.seed)
    paper["fold"] = np.where(paper.sentence_id >= 0, fold_of[paper.sentence_id.clip(lower=0)], -1)
    oof, flags, thresholds, n_run = run_cv(args, targets, paper, found.values, fold_of, pool_df, article_of)
    report_cv(args, paper, found.values, oof, flags, thresholds, n_run, args.pool_size)


if __name__ == "__main__":
    main()
