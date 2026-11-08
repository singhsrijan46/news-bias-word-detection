#!/usr/bin/env python
"""
Test the bias-word models on BABE, the expert-labelled dataset from the same research group.

BABE (Spinde et al., 2021, Findings of EMNLP) has 4,121 sentences. Each has an expert label for the
whole sentence (biased or not), and most also list the words the experts marked as biased. 1,699 of
them are MBIC's own sentences, so the test has two parts:

  1. New sentences: BABE sentences that are not in MBIC, scored by the final models, which were
     trained on all of MBIC. Word level on those with expert word labels; sentence level on all.
     Broken down by whether the topic and the outlet occur in MBIC, and by outlet leaning.
  2. MBIC's own sentences, labelled by experts instead of the crowd. The models' cross-validated
     predictions are compared with the experts' labels, next to the crowd's own 4-vote labels.

A sentence counts as flagged when any of its words is flagged, and its score is its highest word score.

Needs the final models from "train_student.py --final" and the cross-validated predictions of
train_student.py and train_token_models.py. Downloads BABE (under 1 MB) from Hugging Face.

Usage:
    python scripts/evaluate_babe.py
"""

import argparse
import ast
import json
import os
import re
import sys
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.data.annotations import match_highlights  # noqa: E402
from src.data.dataset_loader import MBICDataLoader  # noqa: E402
from src.data.word_targets import build_sentence_targets  # noqa: E402
from src.evaluation.word_metrics import best_f1_threshold, detection_metrics  # noqa: E402
from src.models.onnx_token_model import OnnxTokenModel, word_spans  # noqa: E402

BABE_REPO = "mediabiasgroup/BABE"
BABE_REVISION = "749f72c1b90f9ed6a9fb9dc7169db18f82dd8754"  # the version tested
# BABE spells some topics two ways; these map them onto one name, and onto MBIC's names.
TOPIC_ALIASES = {"black lives matter": "blm", "gun control": "gun-control", "vaccine": "vaccines"}
FINAL_MODELS = ["teacher", "s_distil", "s_distil_int8"]
# In cross-validation the teacher's predictions are M2's, so the teacher is listed as m2 there.
CV_MODELS = ["m0", "m1", "m2", "s_distil", "s_distil_int8"]


def normalize_text(text) -> str:
    """For matching sentences across datasets: lower case, one kind of quote mark, single spaces."""
    text = re.sub(r"[“”\"‘’`']", "'", str(text))
    return re.sub(r"\s+", " ", text).strip().lower()


def normalize_topic(topic) -> str:
    topic = str(topic).strip().lower()
    return TOPIC_ALIASES.get(topic, topic)


def normalize_outlet(outlet) -> str:
    """'The Federalist' and 'Federalist', 'HuffPost' and 'Huffpost' are the same outlet."""
    return re.sub(r"^the ", "", str(outlet).strip().lower())


def parse_word_list(value) -> Optional[List[str]]:
    """BABE's biased_words, a list written as text such as "['radical', 'far-left']". None if missing."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, str):
        value = ast.literal_eval(value)
    return [str(w) for w in value]


def expert_labels(text: str, words: Iterable[str]) -> np.ndarray:
    """1 for each word of `text` (as word_spans splits it) that the experts marked, matched like MBIC's highlights."""
    tokens = [text[a:b].lower() for a, b in word_spans(text)]
    labels = np.zeros(len(tokens), dtype=np.int64)
    for position in match_highlights(tokens, ", ".join(words)):
        labels[position] = 1
    return labels


def load_babe(revision: str = BABE_REVISION) -> pd.DataFrame:
    from huggingface_hub import hf_hub_download

    parts = []
    for split in ("train", "test"):
        path = hf_hub_download(BABE_REPO, f"data/{split}-00000-of-00001.parquet", repo_type="dataset",
                               revision=revision)
        parts.append(pd.read_parquet(path).assign(split=split))
    babe = pd.concat(parts, ignore_index=True)
    babe["babe_index"] = np.arange(len(babe))
    babe["words"] = babe.biased_words.map(parse_word_list)
    babe["topic_norm"] = babe.topic.map(normalize_topic)
    babe["outlet_norm"] = babe.outlet.map(normalize_outlet)
    babe["leaning"] = babe["type"].fillna("unknown")
    return babe


def split_by_mbic(babe: pd.DataFrame, mbic: pd.DataFrame):
    """BABE's sentences that are MBIC's own, and the rest.

    A sentence whose first 60 characters match an MBIC sentence's is a near duplicate (a light edit
    of an MBIC sentence) and is left out of both parts.
    """
    texts = babe.text.map(normalize_text)
    mbic_texts = set(mbic.sentence.map(normalize_text))
    prefixes = {t[:60] for t in mbic_texts}
    exact = texts.isin(mbic_texts)
    near = ~exact & texts.str[:60].isin(prefixes)
    first = ~texts.duplicated()
    return babe[exact & first].copy(), babe[~exact & ~near & first].copy(), int(near.sum())


def sentence_score(scores: np.ndarray) -> float:
    """A sentence's score: its highest word score (NaN words, cut by truncation, are ignored)."""
    finite = scores[np.isfinite(scores)]
    return float(finite.max()) if len(finite) else float("nan")


def score_new(models: Dict[str, OnnxTokenModel], new: pd.DataFrame):
    """One row per word of the sentences with expert word labels, and one row per sentence."""
    word_rows, sentence_rows = [], []
    context = ["babe_index", "topic_in_mbic", "outlet_in_mbic", "leaning"]
    for n, row in enumerate(new.itertuples(index=False)):
        text = str(row.text)
        scores = {name: model.score(text)[1] for name, model in models.items()}
        base = {c: getattr(row, c) for c in context}
        sentence_rows.append({**base, "label": int(row.label), "has_word_labels": row.words is not None,
                              **{name: sentence_score(s) for name, s in scores.items()}})
        if row.words is not None:
            labels = expert_labels(text, row.words)
            for w, label in enumerate(labels):
                word_rows.append({**base, "word_index": w, "label": int(label),
                                  **{name: float(s[w]) for name, s in scores.items()}})
        if (n + 1) % 500 == 0:
            print(f"  scored {n + 1} of {len(new)} sentences")
    words = pd.DataFrame(word_rows).dropna(subset=list(models))
    return words, pd.DataFrame(sentence_rows)


def metrics_table(table: pd.DataFrame, models: List[str], thresholds: Dict[str, float], by: Optional[str] = None,
                  label: str = "label") -> pd.DataFrame:
    """Detection metrics of each model's scores against `label`, at its cross-validated threshold.

    best_threshold and best_f1 show what a threshold chosen on this data would reach, which separates
    how well the scores rank from how well the threshold carries over.
    """
    groups = [("all", table)] if by is None else list(table.groupby(by, sort=True))
    rows = []
    for key, part in groups:
        for m in models:
            ok = part[m].notna()
            y, s = part.loc[ok, label].to_numpy(), part.loc[ok, m].to_numpy()
            if y.min() == y.max():
                continue
            d = detection_metrics(y, s, (s >= thresholds[m]).astype(int))
            best = best_f1_threshold(y, s)
            rows.append({(by or "part"): key, "model": m, "n": int(len(y)), "positive_share": float(y.mean()),
                         "threshold": thresholds[m], "f1": d["f1"], "precision": d["precision"],
                         "recall": d["recall"], "roc_auc": d["roc_auc"], "average_precision": d["average_precision"],
                         "best_threshold": best, "best_f1": detection_metrics(y, s, (s >= best).astype(int))["f1"]})
    return pd.DataFrame(rows)


def overlap_words(overlap: pd.DataFrame, targets, student_oof: pd.DataFrame, models_oof: pd.DataFrame) -> pd.DataFrame:
    """Expert labels for MBIC's paper words, next to the crowd's labels and the cross-validated scores."""
    by_text = {normalize_text(t.text): t for t in targets}
    rows = []
    for row in overlap.itertuples(index=False):
        target = by_text.get(normalize_text(row.text))
        if row.words is None or target is None:
            continue
        for w, label in enumerate(expert_labels(target.text, row.words)):
            rows.append({"sentence_id": target.sentence_id, "word_index": w, "expert": int(label)})
    keys = ["sentence_id", "word_index"]
    expert = pd.DataFrame(rows).drop_duplicates(keys)
    student = student_oof[keys + ["label4", "share"] + [c for m in FINAL_MODELS for c in (m, f"{m}_flag")]]
    baselines = models_oof[keys + [c for m in ("m0", "m1", "m2") for c in (m, f"{m}_flag")]]
    return (expert.merge(student.drop_duplicates(keys), on=keys)
            .merge(baselines.drop_duplicates(keys), on=keys))


def rater_table(words: pd.DataFrame, raters: Dict[str, tuple], target: str) -> pd.DataFrame:
    """Each rater's 0/1 flags and scores against `target` labels, with Cohen's kappa of the flags."""
    rows = []
    for name, (flag, score) in raters.items():
        part = words.dropna(subset=[score])
        y, flags = part[target].astype(int), part[flag].astype(int)
        d = detection_metrics(y, part[score], flags)
        rows.append({"rater": name, "n": int(len(part)), "f1": d["f1"], "precision": d["precision"],
                     "recall": d["recall"], "roc_auc": d["roc_auc"], "average_precision": d["average_precision"],
                     "kappa": float(cohen_kappa_score(y, flags))})
    return pd.DataFrame(rows)


def overlap_tables(words: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    raters = {"crowd (4 of 10 votes)": ("label4", "share"), **{m: (f"{m}_flag", m) for m in CV_MODELS}}
    vs_expert = rater_table(words, raters, "expert")
    vs_crowd = rater_table(words, {m: (f"{m}_flag", m) for m in CV_MODELS}, "label4")
    # Sentence level: flagged if any word is flagged; scored by the highest word score.
    agg = {"expert": "max", "label4": "max", "share": "max", **{c: "max" for m in CV_MODELS for c in (m, f"{m}_flag")}}
    sentences = words.groupby("sentence_id").agg(agg)
    return {"vs_expert": vs_expert, "vs_crowd": vs_crowd,
            "sentences_vs_expert": rater_table(sentences, raters, "expert")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--student", default=os.path.join(ROOT, "outputs", "student", "final"))
    parser.add_argument("--teacher", default=os.path.join(ROOT, "outputs", "student", "final_teacher"))
    parser.add_argument("--student-cv", default=os.path.join(ROOT, "outputs", "student", "cv"))
    parser.add_argument("--models-cv", default=os.path.join(ROOT, "outputs", "models", "cv"))
    parser.add_argument("--revision", default=BABE_REVISION, help="BABE's version on Hugging Face")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--limit", type=int, help="score only this many new sentences (a quick run)")
    parser.add_argument("--out", default=os.path.join(ROOT, "outputs", "babe"))
    args = parser.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    loader = MBICDataLoader()
    mbic = loader.load_raw_sentences()
    babe = load_babe(args.revision)
    overlap, new, near = split_by_mbic(babe, mbic)
    if args.limit:
        new = new.head(args.limit)
    new["topic_in_mbic"] = new.topic_norm.isin(set(mbic.topic.map(normalize_topic)))
    new["outlet_in_mbic"] = new.outlet_norm.isin(set(mbic.outlet.map(normalize_outlet)))
    with open(os.path.join(args.student_cv, "metrics.json"), encoding="utf-8") as f:
        thresholds = {m: r["threshold_mean"] for m, r in json.load(f)["results"].items()}

    # 1. New sentences, with the final models.
    models = {"teacher": OnnxTokenModel(args.teacher, "fp32", args.threads),
              "s_distil": OnnxTokenModel(args.student, "fp32", args.threads),
              "s_distil_int8": OnnxTokenModel(args.student, "int8", args.threads)}
    print(f"BABE: {len(babe)} sentences; {len(overlap)} are MBIC's, {len(new)} are new, {near} near duplicates left out")
    words, sentences = score_new(models, new)
    word_all = metrics_table(words, FINAL_MODELS, thresholds)
    word_parts = pd.concat([metrics_table(words, FINAL_MODELS, thresholds, by=c)
                            for c in ("topic_in_mbic", "outlet_in_mbic", "leaning")], ignore_index=True)
    sentence_all = metrics_table(sentences, FINAL_MODELS, thresholds)
    sentence_parts = metrics_table(sentences, FINAL_MODELS, thresholds, by="has_word_labels")

    # 2. MBIC's own sentences: crowd and cross-validated models against the experts.
    student_oof = pd.read_csv(os.path.join(args.student_cv, "paper_words_oof.csv"))
    models_oof = pd.read_csv(os.path.join(args.models_cv, "paper_words_oof.csv"))
    shared = overlap_words(overlap, build_sentence_targets(loader.load_annotations()), student_oof, models_oof)
    tables = overlap_tables(shared)

    info = {"babe_repo": BABE_REPO, "babe_revision": args.revision, "babe_sentences": int(len(babe)),
            "mbic_sentences_in_babe": int(len(overlap)), "new_sentences": int(len(new)), "near_duplicates": near,
            "new_with_word_labels": int(new.words.notna().sum()), "new_words": int(len(words)),
            "new_word_positive_share": float(words.label.mean()), "shared_words": int(len(shared)),
            "shared_expert_positive_share": float(shared.expert.mean()),
            "shared_crowd_positive_share": float(shared.label4.mean()), "thresholds": thresholds}
    words.to_csv(os.path.join(args.out, "new_words.csv"), index=False)
    sentences.to_csv(os.path.join(args.out, "new_sentences.csv"), index=False)
    shared.to_csv(os.path.join(args.out, "shared_words.csv"), index=False)
    for name, table in {"new_word_metrics": pd.concat([word_all, word_parts], ignore_index=True),
                        "new_sentence_metrics": pd.concat([sentence_all, sentence_parts], ignore_index=True),
                        **{f"shared_{k}": v for k, v in tables.items()}}.items():
        table.to_csv(os.path.join(args.out, f"{name}.csv"), index=False)
    with open(os.path.join(args.out, "info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    report = "\n".join([
        "## New sentences: words", "", word_all.round(3).to_markdown(index=False), "",
        "## New sentences: words, by part", "", word_parts.round(3).to_markdown(index=False), "",
        "## New sentences: sentences", "", pd.concat([sentence_all, sentence_parts]).round(3).to_markdown(index=False), "",
        "## MBIC's sentences: against the experts' word labels", "", tables["vs_expert"].round(3).to_markdown(index=False), "",
        "## MBIC's sentences: the same words against the crowd's labels", "",
        tables["vs_crowd"].round(3).to_markdown(index=False), "",
        "## MBIC's sentences: sentence level, against the experts", "",
        tables["sentences_vs_expert"].round(3).to_markdown(index=False), "",
        "```", json.dumps(info, indent=2), "```", "",
    ])
    with open(os.path.join(args.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
