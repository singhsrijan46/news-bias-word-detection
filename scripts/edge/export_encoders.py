#!/usr/bin/env python
"""
Export candidate encoders to ONNX for the on-device benchmark.

Each candidate gets a token-level head with one output per token: the shape the
soft-label model (M2) and its distilled student will have. The heads are
untrained, because speed and memory do not depend on the weights. Each model is
saved in FP32 and in INT8 (ONNX Runtime dynamic quantization), together with the
benchmark's MBIC sentences and articles already tokenized for it, so the board
needs only numpy and ONNX Runtime.

Usage:
    python scripts/edge/export_encoders.py                        # all candidates
    python scripts/edge/export_encoders.py --models bert-tiny     # a subset
"""

import argparse
import json
import os
import re
import sys
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.data.dataset_loader import MBICDataLoader  # noqa: E402

# Benchmark order: smallest first. Names are used as folder names on the board.
CANDIDATES: Dict[str, str] = {
    "bert-tiny": "google/bert_uncased_L-2_H-128_A-2",
    "bert-mini": "google/bert_uncased_L-4_H-256_A-4",
    "electra-small": "google/electra-small-discriminator",
    "minilm-l6": "nreimers/MiniLM-L6-H384-uncased",
    "bert-small": "google/bert_uncased_L-4_H-512_A-8",
    "distilroberta": "distilbert/distilroberta-base",
    "roberta-base": "FacebookAI/roberta-base",
}

N_SENTENCES = 200
N_ARTICLES = 10
ARTICLE_WORDS = (400, 1500)  # middle of MBIC's range: its median article has 608 words
MAX_TOKENS = 128
SEED = 0

# A sentence ends at . ! or ?, optionally followed by a closing quote or bracket.
SENTENCE_END = re.compile(r"(?:(?<=[.!?])|(?<=[.!?][\"'”’)]))\s+")


def split_sentences(text: str) -> List[str]:
    """Split article text into sentences: first by line, then at sentence-ending punctuation."""
    sentences: List[str] = []
    for paragraph in text.splitlines():
        paragraph = paragraph.strip()
        if paragraph:
            sentences.extend(s.strip() for s in SENTENCE_END.split(paragraph) if s.strip())
    return sentences


def pack(sequences: Sequence[Sequence[int]]) -> Tuple[np.ndarray, np.ndarray]:
    """Store variable-length token sequences as one flat array plus start offsets."""
    lengths = np.array([len(s) for s in sequences], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    flat = (np.concatenate([np.asarray(s, dtype=np.int64) for s in sequences])
            if len(sequences) else np.zeros(0, dtype=np.int64))
    return flat, offsets


def choose_workload(df, n_sentences: int = N_SENTENCES, n_articles: int = N_ARTICLES,
                    word_range: Tuple[int, int] = ARTICLE_WORDS, seed: int = SEED) -> Tuple[List[str], List[str]]:
    """A fixed random sample of MBIC sentences and of whole MBIC articles."""
    rng = np.random.default_rng(seed)
    sentences = df["sentence"].astype(str).tolist()
    picked = rng.choice(len(sentences), size=min(n_sentences, len(sentences)), replace=False)
    articles = df["article"].dropna().astype(str).drop_duplicates()
    words = articles.str.split().str.len()
    pool = articles[(words >= word_range[0]) & (words <= word_range[1])].tolist()
    chosen = rng.choice(len(pool), size=min(n_articles, len(pool)), replace=False)
    return [sentences[i] for i in sorted(picked)], [pool[i] for i in sorted(chosen)]


def tokenize_workload(tokenizer, sentence_texts: List[str], article_texts: List[str]) -> Dict[str, np.ndarray]:
    """Token ids for the sentence sample and for every sentence of every article."""
    def encode(texts: List[str]) -> List[List[int]]:
        return tokenizer(texts, truncation=True, max_length=MAX_TOKENS)["input_ids"]

    article_sentences = [split_sentences(a) for a in article_texts]
    sentence_flat, sentence_offsets = pack(encode(sentence_texts))
    article_flat, article_sentence_offsets = pack(encode([s for sents in article_sentences for s in sents]))
    sizes = np.array([len(s) for s in article_sentences], dtype=np.int64)
    return {
        "sentence_ids": sentence_flat,
        "sentence_offsets": sentence_offsets,
        "article_sentence_ids": article_flat,
        "article_sentence_offsets": article_sentence_offsets,
        "article_offsets": np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64),
        "article_words": np.array([len(a.split()) for a in article_texts], dtype=np.int64),
    }


def export_fp32(model, tokenizer, path: str) -> str:
    """Export to ONNX with dynamic batch and sequence length. Returns the exporter used."""
    import torch

    class TokenScores(torch.nn.Module):
        """Return only the per-token logits, so the graph has one output."""

        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, input_ids, attention_mask):
            return self.inner(input_ids=input_ids, attention_mask=attention_mask).logits

    wrapped = TokenScores(model).eval()
    sample = tokenizer(["A sample sentence for the export."], return_tensors="pt")
    args = (sample["input_ids"], sample["attention_mask"])
    names = dict(input_names=["input_ids", "attention_mask"], output_names=["logits"])
    try:
        torch.onnx.export(
            wrapped, args, path, opset_version=17, dynamo=False,
            dynamic_axes={n: {0: "batch", 1: "sequence"} for n in ("input_ids", "attention_mask", "logits")},
            **names,
        )
        return "torchscript"
    except Exception as legacy_error:  # the legacy exporter may be gone in newer PyTorch
        print(f"  legacy exporter failed ({type(legacy_error).__name__}: {legacy_error}); trying dynamo")
        batch, sequence = torch.export.Dim("batch", max=256), torch.export.Dim("sequence", max=512)
        shapes = {"input_ids": {0: batch, 1: sequence}, "attention_mask": {0: batch, 1: sequence}}
        torch.onnx.export(wrapped, args, path, dynamo=True, dynamic_shapes=shapes, external_data=False, **names)
        return "dynamo"


def max_abs_diff(onnx_path: str, model, tokenizer, reference: np.ndarray = None) -> Tuple[float, np.ndarray]:
    """Largest gap between ONNX Runtime's output and a reference (PyTorch's, unless given)."""
    import onnxruntime as ort
    import torch

    texts = ["The senator's radical plan was slammed by critics.", "Rates rose."]
    enc = tokenizer(texts, padding=True, return_tensors="pt")
    if reference is None:
        with torch.no_grad():
            reference = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits.numpy()
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    got = session.run(None, {"input_ids": enc["input_ids"].numpy(), "attention_mask": enc["attention_mask"].numpy()})[0]
    mask = enc["attention_mask"].numpy().astype(bool)
    return float(np.max(np.abs(reference[mask] - got[mask]))), got


def export_candidate(name: str, hf_id: str, out_dir: str, sentence_texts: List[str], article_texts: List[str]) -> dict:
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    started = time.time()
    model_dir = os.path.join(out_dir, name)
    os.makedirs(model_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    torch.manual_seed(SEED)
    model = AutoModelForTokenClassification.from_pretrained(hf_id, num_labels=1).eval()

    fp32_path, int8_path = os.path.join(model_dir, "fp32.onnx"), os.path.join(model_dir, "int8.onnx")
    exporter = export_fp32(model, tokenizer, fp32_path)
    fp32_diff, fp32_out = max_abs_diff(fp32_path, model, tokenizer)
    quantize_dynamic(fp32_path, int8_path, weight_type=QuantType.QInt8)
    int8_diff, _ = max_abs_diff(int8_path, model, tokenizer, reference=fp32_out)

    workload = tokenize_workload(tokenizer, sentence_texts, article_texts)
    np.savez(os.path.join(model_dir, "inputs.npz"), **workload)

    config = model.config
    sentence_lengths = np.diff(workload["sentence_offsets"])
    manifest = {
        "name": name,
        "hf_id": hf_id,
        "params": int(sum(p.numel() for p in model.parameters())),
        "layers": getattr(config, "num_hidden_layers", None),
        "hidden_size": getattr(config, "hidden_size", None),
        "heads": getattr(config, "num_attention_heads", None),
        "vocab_size": getattr(config, "vocab_size", None),
        "pad_token_id": int(tokenizer.pad_token_id),
        "exporter": exporter,
        "onnx_opset": 17 if exporter == "torchscript" else None,
        "fp32_mb": os.path.getsize(fp32_path) / 2**20,
        "int8_mb": os.path.getsize(int8_path) / 2**20,
        "fp32_vs_pytorch_max_abs_diff": fp32_diff,
        "int8_vs_fp32_max_abs_diff": int8_diff,
        "sentences": int(len(sentence_lengths)),
        "sentence_tokens_median": float(np.median(sentence_lengths)),
        "articles": int(len(workload["article_words"])),
        "article_sentences_median": float(np.median(np.diff(workload["article_offsets"]))),
        "article_words_median": float(np.median(workload["article_words"])),
        "max_tokens": MAX_TOKENS,
        "seed": SEED,
    }
    with open(os.path.join(model_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"  {name}: {manifest['params'] / 1e6:.1f}M params, FP32 {manifest['fp32_mb']:.0f} MB, "
          f"INT8 {manifest['int8_mb']:.0f} MB, ONNX vs PyTorch {fp32_diff:.1e}, "
          f"INT8 vs FP32 {int8_diff:.1e} ({time.time() - started:.0f} s, {exporter})")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", choices=list(CANDIDATES), default=list(CANDIDATES))
    parser.add_argument("--out", default=os.path.join(ROOT, "outputs", "edge", "models"))
    args = parser.parse_args(argv)

    sentence_texts, article_texts = choose_workload(MBICDataLoader().load_raw_sentences())
    print(f"Workload: {len(sentence_texts)} sentences and {len(article_texts)} articles from MBIC")
    os.makedirs(args.out, exist_ok=True)
    for name in args.models:
        export_candidate(name, CANDIDATES[name], args.out, sentence_texts, article_texts)

    exported = [n for n in CANDIDATES if os.path.exists(os.path.join(args.out, n, "manifest.json"))]
    with open(os.path.join(args.out, "index.json"), "w", encoding="utf-8") as f:
        json.dump({"models": exported}, f, indent=2)
    print(f"Saved to {os.path.normpath(args.out)}")


if __name__ == "__main__":
    main()
