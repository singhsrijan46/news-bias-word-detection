#!/usr/bin/env python
"""
Audit the bias-word models for political symmetry, outlet leaning, annotator groups and quantization.

  1. Counterfactual swaps. Sentences that mention one side (Democrats, Biden, liberal, pro-choice...)
     are rewritten to mention the other (Republicans, Trump, conservative, pro-life...), and back.
     A symmetric model flags the swapped words alike, and its flags on the other words do not move.
  2. Outlet leaning. Detection and calibration on sentences from left, centre and right outlets
     (AllSides ratings), from the cross-validated predictions of train_student.py.
  3. Annotator groups. Whether predicted shares track liberal or conservative annotators' own
     shares more closely (the paper's grouping: liberal below -3, conservative above 3).
  4. Quantization. What INT8 changes in the final student's scores and flags, against its FP32 model.

Needs the models from "train_student.py --final" (student and teacher) and the predictions from
its cross-validation.

Usage:
    python scripts/audit_models.py
    python scripts/audit_models.py --max-sentences 300    # a quick run
"""

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_1samp, wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.data.annotations import annotator_ideology, ideology_groups, rebuild_sentence_annotations  # noqa: E402
from src.data.articles import article_pool  # noqa: E402
from src.data.dataset_loader import MBICDataLoader  # noqa: E402
from src.evaluation.word_metrics import calibration_metrics, detection_metrics  # noqa: E402
from src.models.onnx_token_model import OnnxTokenModel  # noqa: E402

# Left-coded and right-coded terms that can replace each other in a sentence.
PAIRS = [("Democrats", "Republicans"), ("Democratic", "Republican"), ("liberals", "conservatives"),
         ("liberal", "conservative"), ("left-wing", "right-wing"), ("pro-choice", "pro-life"),
         ("Biden", "Trump"), ("Pelosi", "McConnell")]
LEFT = {left.lower(): right for left, right in PAIRS}
RIGHT = {right.lower(): left for left, right in PAIRS}
PAIR_WORD = re.compile(r"\b(" + "|".join(re.escape(w) for pair in PAIRS for w in pair) + r")\b", re.IGNORECASE)


def match_case(template: str, word: str) -> str:
    if template.isupper():
        return word.upper()
    if template[:1].isupper():
        return word[:1].upper() + word[1:]
    return word.lower()


def counterfactual(text: str):
    """The sentence with every pair word moved to the other side, and which side it started on.

    Returns (swapped text, side) where side is 'left' or 'right', or None when the sentence
    mentions both sides or neither (then the direction of a swap is not defined).
    """
    found = PAIR_WORD.findall(text)
    sides = {"left" if w.lower() in LEFT else "right" for w in found}
    if len(sides) != 1:
        return None
    swap = {**LEFT, **RIGHT}
    return PAIR_WORD.sub(lambda m: match_case(m.group(0), swap[m.group(0).lower()]), text), sides.pop()


def swap_audit(models: Dict[str, OnnxTokenModel], sentences: List[str], thresholds: Dict[str, float]):
    """Score each sentence and its counterfactual; compare the pair words and every other word."""
    pair_rows, sentence_rows = [], []
    for text in sentences:
        made = counterfactual(text)
        if made is None:
            continue
        swapped, side = made
        left_text, right_text = (text, swapped) if side == "left" else (swapped, text)
        for name, model in models.items():
            spans_l, shares_l = model.score(left_text)
            spans_r, shares_r = model.score(right_text)
            if len(spans_l) != len(spans_r):  # the swap changed the word count; skip
                continue
            words_l = [left_text[a:b] for a, b in spans_l]
            words_r = [right_text[a:b] for a, b in spans_r]
            is_pair = np.array([wl != wr for wl, wr in zip(words_l, words_r)])  # includes "Trump's" / "Biden's"
            ok = ~np.isnan(shares_l) & ~np.isnan(shares_r)
            for i in np.where(is_pair & ok)[0]:
                pair_rows.append({"model": name, "left_word": words_l[i], "right_word": words_r[i],
                                  "share_left": float(shares_l[i]), "share_right": float(shares_r[i])})
            other = ~is_pair & ok
            if other.any():
                t = thresholds[name]
                sentence_rows.append({
                    "model": name, "side": side, "words": int(other.sum()),
                    "shift_right_minus_left": float(np.mean(shares_r[other] - shares_l[other])),
                    "abs_shift": float(np.mean(np.abs(shares_r[other] - shares_l[other]))),
                    "flag_flips": int(np.sum((shares_r[other] >= t) != (shares_l[other] >= t))),
                })
    return pd.DataFrame(pair_rows), pd.DataFrame(sentence_rows)


def summarize_swaps(pairs: pd.DataFrame, sentences: pd.DataFrame, thresholds: Dict[str, float]) -> pd.DataFrame:
    rows = []
    for name, group in sentences.groupby("model", sort=False):
        p = pairs[pairs.model == name]
        diff = p.share_right - p.share_left
        t = thresholds[name]
        rows.append({
            "model": name, "sentence_pairs": int(len(group)), "pair_words": int(len(p)),
            "pair_word_share_left": float(p.share_left.mean()), "pair_word_share_right": float(p.share_right.mean()),
            "pair_word_right_minus_left": float(diff.mean()),
            "pair_word_p": float(wilcoxon(diff).pvalue) if len(p) > 10 else None,
            "pair_word_flag_differs": float(((p.share_left >= t) != (p.share_right >= t)).mean()),
            "other_words_shift_right_minus_left": float(group.shift_right_minus_left.mean()),
            "other_words_shift_p": float(ttest_1samp(group.shift_right_minus_left, 0).pvalue),
            "other_words_mean_abs_shift": float(np.average(group.abs_shift, weights=group.words)),
            "other_words_flag_flip_rate": float(group.flag_flips.sum() / group.words.sum()),
        })
    return pd.DataFrame(rows)


def per_pair(pairs: pd.DataFrame) -> pd.DataFrame:
    p = pairs.assign(pair=pairs.left_word.str.lower() + " / " + pairs.right_word.str.lower())
    return (p.groupby(["model", "pair"], sort=False)
            .agg(n=("share_left", "size"), share_left=("share_left", "mean"), share_right=("share_right", "mean"))
            .assign(right_minus_left=lambda d: d.share_right - d.share_left).reset_index())


def outlet_breakdown(oof: pd.DataFrame, outlet: Dict[int, str], models: List[str]) -> pd.DataFrame:
    rows = []
    for leaning in ("left", "center", "right"):
        part = oof[oof.sentence_id.map(outlet) == leaning]
        for m in models:
            d = detection_metrics(part.label4, part[m], part[f"{m}_flag"])
            c = calibration_metrics(part.label4, part.share, part[m])
            rows.append({"outlet": leaning, "model": m, "words": int(len(part)), "f1": d["f1"],
                         "precision": d["precision"], "recall": d["recall"], "roc_auc": d["roc_auc"],
                         "share_ece": c["share_ece"], "mean_predicted": float(part[m].mean()),
                         "mean_actual_share": float(part.share.mean())})
    return pd.DataFrame(rows)


def group_shares(annotations: pd.DataFrame) -> Dict[Tuple[int, int], Tuple[float, float]]:
    """(sentence_id, word_index) -> (liberal share, conservative share), for words seen by both groups."""
    groups = ideology_groups(annotator_ideology(annotations), centre_band=3)
    out = {}
    for sid, sentence in enumerate(rebuild_sentence_annotations(annotations)):
        by_group = {"liberal": [], "conservative": []}
        for j in sentence.judgements:
            if groups.get(j.annotator_id) in by_group:
                by_group[groups[j.annotator_id]].append(j.positions)
        if not by_group["liberal"] or not by_group["conservative"]:
            continue
        for w in range(len(sentence.tokens)):
            lib = np.mean([w in p for p in by_group["liberal"]])
            con = np.mean([w in p for p in by_group["conservative"]])
            out[(sid, w)] = (float(lib), float(con))
    return out


def group_alignment(oof: pd.DataFrame, shares: Dict, models: List[str]) -> pd.DataFrame:
    keys = list(zip(oof.sentence_id, oof.word_index))
    lib = np.array([shares.get(k, (np.nan, np.nan))[0] for k in keys])
    con = np.array([shares.get(k, (np.nan, np.nan))[1] for k in keys])
    ok = ~np.isnan(lib) & ~np.isnan(con)
    rows = [{"model": "annotators: liberal vs conservative shares", "words": int(ok.sum()),
             "spearman_liberal": float(spearmanr(lib[ok], con[ok]).correlation), "spearman_conservative": None,
             "mean_error_vs_liberal": float(np.mean(con[ok] - lib[ok])), "mean_error_vs_conservative": None,
             "abs_error_vs_liberal": float(np.mean(np.abs(con[ok] - lib[ok]))), "abs_error_vs_conservative": None}]
    for m in models:
        p = oof[m].values[ok]
        rows.append({"model": m, "words": int(ok.sum()),
                     "spearman_liberal": float(spearmanr(p, lib[ok]).correlation),
                     "spearman_conservative": float(spearmanr(p, con[ok]).correlation),
                     "mean_error_vs_liberal": float(np.mean(p - lib[ok])),
                     "mean_error_vs_conservative": float(np.mean(p - con[ok])),
                     "abs_error_vs_liberal": float(np.mean(np.abs(p - lib[ok]))),
                     "abs_error_vs_conservative": float(np.mean(np.abs(p - con[ok])))})
    return pd.DataFrame(rows)


def quantization(fp32: OnnxTokenModel, int8: OnnxTokenModel, texts: List[str], threshold: float) -> Dict:
    """The final student's FP32 and INT8 ONNX models on the same sentences: what quantization changes.

    (In cross-validation the PyTorch "FP32" scores were computed under bfloat16 autocast on the
    GPU, so this is the clean FP32-against-INT8 comparison.)
    """
    a, b = [], []
    for text in texts:
        a.append(fp32.score(text)[1])
        b.append(int8.score(text)[1])
    a, b = np.concatenate(a), np.concatenate(b)
    ok = ~np.isnan(a) & ~np.isnan(b)
    a, b = a[ok], b[ok]
    flagged_fp32, flagged_int8 = a >= threshold, b >= threshold
    return {"sentences": len(texts), "words": int(ok.sum()), "threshold": threshold,
            "mean_abs_share_change": float(np.mean(np.abs(a - b))), "max_abs_share_change": float(np.max(np.abs(a - b))),
            "flagged_fp32": int(flagged_fp32.sum()), "flagged_int8": int(flagged_int8.sum()),
            "flag_changes": int((flagged_fp32 != flagged_int8).sum()),
            "flag_change_rate_among_flagged": float((flagged_fp32 != flagged_int8).sum() / max(1, flagged_fp32.sum())),
            "spearman_fp32_int8": float(spearmanr(a, b).correlation)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--student", default=os.path.join(ROOT, "outputs", "student", "final"))
    parser.add_argument("--teacher", default=os.path.join(ROOT, "outputs", "student", "final_teacher"))
    parser.add_argument("--cv-dir", default=os.path.join(ROOT, "outputs", "student", "cv"))
    parser.add_argument("--max-sentences", type=int, help="limit the counterfactual sentences")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--out", default=os.path.join(ROOT, "outputs", "audit"))
    args = parser.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    loader = MBICDataLoader()
    labeled = loader.load_raw_sentences()
    annotations = loader.load_annotations()
    oof = pd.read_csv(os.path.join(args.cv_dir, "paper_words_oof.csv"))
    with open(os.path.join(args.cv_dir, "metrics.json"), encoding="utf-8") as f:
        cv = json.load(f)["results"]

    # 1. Counterfactual swaps, on MBIC sentences and unlabeled article sentences.
    texts = list(dict.fromkeys(labeled.sentence.astype(str).tolist() + article_pool(labeled).sentence.tolist()))
    texts = [t for t in texts if counterfactual(t) is not None]
    if args.max_sentences:
        texts = texts[:args.max_sentences]
    models = {"teacher": OnnxTokenModel(args.teacher, "fp32", args.threads),
              "s_distil": OnnxTokenModel(args.student, "fp32", args.threads),
              "s_distil_int8": OnnxTokenModel(args.student, "int8", args.threads)}
    thresholds = {m: cv[m]["threshold_mean"] for m in models}
    print(f"Counterfactual swaps on {len(texts)} sentences that mention one side")
    pairs, sentence_shifts = swap_audit(models, texts, thresholds)
    swaps = summarize_swaps(pairs, sentence_shifts, thresholds)
    by_pair = per_pair(pairs)

    # 2-4. From the cross-validated predictions.
    from src.data.word_targets import build_sentence_targets

    outlet = {t.sentence_id: t.outlet_type for t in build_sentence_targets(annotations)}
    cv_models = ["teacher", "s_distil", "s_distil_int8"]
    outlets = outlet_breakdown(oof, outlet, cv_models)
    groups = group_alignment(oof, group_shares(annotations), cv_models)
    quant = quantization(models["s_distil"], models["s_distil_int8"], labeled.sentence.astype(str).tolist(),
                         thresholds["s_distil_int8"])

    pairs.to_csv(os.path.join(args.out, "swap_pair_words.csv"), index=False)
    sentence_shifts.to_csv(os.path.join(args.out, "swap_sentences.csv"), index=False)
    swaps.to_csv(os.path.join(args.out, "swap_summary.csv"), index=False)
    by_pair.to_csv(os.path.join(args.out, "swap_by_pair.csv"), index=False)
    outlets.to_csv(os.path.join(args.out, "outlet_breakdown.csv"), index=False)
    groups.to_csv(os.path.join(args.out, "annotator_groups.csv"), index=False)
    with open(os.path.join(args.out, "quantization.json"), "w", encoding="utf-8") as f:
        json.dump(quant, f, indent=2)
    report = "\n".join([
        "## Counterfactual swaps", "", swaps.round(4).to_markdown(index=False), "",
        "## By pair", "", by_pair.round(3).to_markdown(index=False), "",
        "## Outlet leaning (cross-validated)", "", outlets.round(3).to_markdown(index=False), "",
        "## Annotator groups (cross-validated)", "", groups.round(4).to_markdown(index=False), "",
        "## INT8 against FP32: the final student's ONNX models on MBIC's 1,700 sentences", "",
        json.dumps(quant, indent=2), "",
    ])
    with open(os.path.join(args.out, "audit.md"), "w", encoding="utf-8") as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
