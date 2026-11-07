#!/usr/bin/env python
"""
Summarize the on-board benchmark into tables.

    python scripts/edge/summarize_benchmark.py                  # outputs/edge/results/
    python scripts/edge/summarize_benchmark.py --results DIR

Writes summary.csv and summary.md next to results.jsonl and prints the tables.
"""

import argparse
import json
import os
from typing import List

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def read_jsonl(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def flatten(rows: List[dict]) -> pd.DataFrame:
    """One row per run, the latest run winning when a configuration was repeated."""
    flat = {}
    for r in rows:
        if "error" in r:
            continue
        flat[(r["model"], r["precision"], r["threads"])] = {
            "model": r["model"], "params_m": r["params_m"], "precision": r["precision"], "threads": r["threads"],
            "file_mb": r["file_mb"], "load_s": r["load_s"], "peak_rss_mb": r["peak_rss_mb"],
            "sentence_tokens": r["sentence_tokens_median"],
            "sentence_ms": r["sentence"]["median_ms"], "sentence_p95_ms": r["sentence"]["p95_ms"],
            "article_s": r["article_sequential"]["median_ms"] / 1000.0,
            "article_batched_s": r["article_batched"]["median_ms"] / 1000.0,
            "words_per_s": r["words_per_s_sequential"], "words_per_s_batched": r["words_per_s_batched"],
            "temp_max_c": r["temp_max_c"], "freq_mean_mhz": r["freq_mean_mhz"], "freq_min_mhz": r["freq_min_mhz"],
        }
    return pd.DataFrame(list(flat.values()))


def int8_speedups(df: pd.DataFrame) -> pd.DataFrame:
    """FP32 time divided by INT8 time, per model and thread count (above 1 means INT8 is faster)."""
    keys = ["model", "threads"]
    fp32 = df[df.precision == "fp32"].set_index(keys)
    int8 = df[df.precision == "int8"].set_index(keys)
    both = fp32.index.intersection(int8.index)
    out = pd.DataFrame({
        "sentence": fp32.loc[both, "sentence_ms"] / int8.loc[both, "sentence_ms"],
        "article": fp32.loc[both, "article_s"] / int8.loc[both, "article_s"],
        "memory_saved_mb": fp32.loc[both, "peak_rss_mb"] - int8.loc[both, "peak_rss_mb"],
    }).reset_index()
    order = {m: i for i, m in enumerate(df["model"].drop_duplicates())}
    return out.sort_values(["model", "threads"], key=lambda s: s.map(order) if s.name == "model" else s)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default=os.path.join(ROOT, "outputs", "edge", "results"))
    args = parser.parse_args(argv)

    rows = read_jsonl(os.path.join(args.results, "results.jsonl"))
    errors = [r for r in rows if "error" in r]
    df = flatten(rows)
    main_cols = ["model", "params_m", "precision", "threads", "file_mb", "peak_rss_mb", "load_s",
                 "sentence_ms", "sentence_p95_ms", "article_s", "article_batched_s", "words_per_s", "temp_max_c",
                 "freq_min_mhz"]
    table = df[main_cols].round(2)
    speedups = int8_speedups(df).round(2)
    df.to_csv(os.path.join(args.results, "summary.csv"), index=False)

    parts = ["## All runs", "", table.to_markdown(index=False), "", "## INT8 speed-up over FP32", "",
             speedups.to_markdown(index=False), ""]
    sustained_path = os.path.join(args.results, "sustained_summary.jsonl")
    if os.path.exists(sustained_path):
        parts += ["## Sustained runs", "", pd.DataFrame(read_jsonl(sustained_path)).round(2).to_markdown(index=False), ""]
    if errors:
        parts += [f"{len(errors)} runs failed: " + ", ".join(f"{e['model']} {e['precision']} x{e['threads']}" for e in errors)]
    text = "\n".join(parts)
    with open(os.path.join(args.results, "summary.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
