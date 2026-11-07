import importlib.util
import os

import numpy as np
import pandas as pd
import pytest

EDGE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "edge")


def load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(EDGE, f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


export = load("export_encoders")
board = load("board_benchmark")
summary = load("summarize_benchmark")


def test_split_sentences_handles_lines_and_closing_quotes():
    text = 'First line here.\nHe said "stop." Then he left! Did it work?\n\n'
    assert export.split_sentences(text) == ["First line here.", 'He said "stop."', "Then he left!", "Did it work?"]


def test_pack_and_unpack_round_trip():
    sequences = [[101, 7, 102], [101, 102], [101, 5, 6, 9, 102]]
    flat, offsets = export.pack(sequences)
    assert list(offsets) == [0, 3, 5, 10]
    assert [list(s) for s in board.unpack(flat, offsets)] == sequences


def test_choose_workload_is_seeded_and_respects_article_length():
    df = pd.DataFrame({
        "sentence": [f"Sentence {i}." for i in range(30)],
        "article": ["word " * 500] * 10 + ["word " * 50] * 10 + [None] * 10,
    })
    df.loc[:9, "article"] = [("word " * 500) + str(i) for i in range(10)]
    first = export.choose_workload(df, n_sentences=5, n_articles=3, word_range=(400, 1500), seed=1)
    again = export.choose_workload(df, n_sentences=5, n_articles=3, word_range=(400, 1500), seed=1)
    assert first == again
    sentences, articles = first
    assert len(sentences) == 5 and len(articles) == 3
    assert all(400 <= len(a.split()) <= 1500 for a in articles)


def test_summarize_reports_milliseconds():
    result = board.summarize([0.010, 0.020, 0.030, 0.040])
    assert result["n"] == 4
    assert result["median_ms"] == pytest.approx(25.0)
    assert result["max_ms"] == pytest.approx(40.0)
    assert board.summarize([]) is None


def test_cpu_temp_uses_only_cpu_zones(tmp_path):
    for i, (kind, millideg) in enumerate([("cpuss0-thermal", 41000), ("gpu-thermal", 70000), ("cpuss1-thermal", 43500)]):
        zone = tmp_path / f"thermal_zone{i}"
        zone.mkdir()
        (zone / "type").write_text(kind + "\n")
        (zone / "temp").write_text(f"{millideg}\n")
    zones = board.find_zones(str(tmp_path))
    assert sorted(zones) == ["cpuss0-thermal", "cpuss1-thermal"]
    assert board.cpu_temp(zones) == pytest.approx(43.5)


def test_int8_speedups_divide_fp32_by_int8_time():
    rows = []
    for precision, ms, seconds, mb in (("fp32", 10.0, 0.4, 120.0), ("int8", 5.0, 0.2, 100.0)):
        rows.append({"model": "m", "params_m": 1.0, "precision": precision, "threads": 1, "file_mb": 1.0,
                     "load_s": 0.1, "peak_rss_mb": mb, "sentence_tokens_median": 40.0,
                     "sentence": {"median_ms": ms, "p95_ms": ms}, "article_sequential": {"median_ms": seconds * 1000},
                     "article_batched": {"median_ms": seconds * 1000}, "words_per_s_sequential": 1.0,
                     "words_per_s_batched": 1.0, "temp_max_c": 50.0, "freq_mean_mhz": 2016.0, "freq_min_mhz": 2016.0})
    table = summary.int8_speedups(summary.flatten(rows))
    assert table.loc[0, "sentence"] == pytest.approx(2.0)
    assert table.loc[0, "memory_saved_mb"] == pytest.approx(20.0)
