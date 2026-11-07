"""
Export a trained token model to ONNX, and run it with ONNX Runtime.

OnnxTokenModel gives the same per-word scores as src.models.token_transformer,
but needs only numpy, onnxruntime and tokenizers. This is the code that runs on
the UNO Q, so it does not import anything else from this repository.

An exported model folder holds:
    model_fp32.onnx, model_int8.onnx   the network, in full precision and in INT8
    tokenizer.json                     the fast tokenizer
    model_info.json                    settings, and the decision thresholds found in cross-validation
"""

import json
import os
import re
from typing import List, Sequence, Tuple

import numpy as np

# The same word pattern as src.data.annotations: words keep inner apostrophes and hyphens.
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:['’\-][A-Za-z0-9]+)*")


def word_spans(text: str) -> List[Tuple[int, int]]:
    return [m.span() for m in TOKEN_PATTERN.finditer(text)]


def first_subword_positions(spans: Sequence[Tuple[int, int]], offsets: Sequence[Tuple[int, int]]) -> np.ndarray:
    """Index of the first sub-word token overlapping each word, or -1 if truncation removed it.

    Same as src.data.word_targets.first_subword_positions, repeated so this file stands alone.
    """
    positions = np.full(len(spans), -1, dtype=np.int64)
    t, n = 0, len(offsets)
    for w, (start, end) in enumerate(spans):
        while t < n and (offsets[t][1] <= start or offsets[t][0] == offsets[t][1]):
            t += 1
        if t < n and offsets[t][0] < end:
            positions[w] = t
    return positions


class OnnxTokenModel:
    """Per-word probabilities from an exported model, one sentence at a time."""

    def __init__(self, model_dir: str, precision: str = "int8", threads: int = 4):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        with open(os.path.join(model_dir, "model_info.json"), encoding="utf-8") as f:
            self.info = json.load(f)
        self.tokenizer = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        self.tokenizer.no_padding()
        self.tokenizer.enable_truncation(self.info["max_length"])
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(os.path.join(model_dir, f"model_{precision}.onnx"), options,
                                            providers=["CPUExecutionProvider"])
        self.precision = precision

    def score(self, text: str) -> Tuple[List[Tuple[int, int]], np.ndarray]:
        """Character spans of the sentence's words, and each word's probability (NaN if truncated)."""
        encoding = self.tokenizer.encode(text)
        ids = np.array([encoding.ids], dtype=np.int64)
        logits = self.session.run(None, {"input_ids": ids, "attention_mask": np.ones_like(ids)})[0][0, :, 0]
        probs = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
        spans = word_spans(text)
        positions = first_subword_positions(spans, encoding.offsets)
        scores = np.full(len(spans), np.nan, dtype=np.float32)
        keep = positions >= 0
        scores[keep] = probs[positions[keep]]
        return spans, scores

    def predict(self, sentences) -> List[np.ndarray]:
        """Scores for objects with a `.text` (such as SentenceTargets), like TokenModel.predict."""
        return [self.score(s.text)[1] for s in sentences]


def export(token_model, out_dir: str, extra_info: dict = None) -> str:
    """Save a trained TokenModel as ONNX (FP32 and INT8) with its tokenizer, for OnnxTokenModel."""
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic

    os.makedirs(out_dir, exist_ok=True)
    network = token_model.model.float().cpu().eval()

    class Logits(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, input_ids, attention_mask):
            return self.inner(input_ids=input_ids, attention_mask=attention_mask).logits

    sample = token_model.tokenizer(["A sample sentence for the export."], return_tensors="pt")
    fp32, int8 = os.path.join(out_dir, "model_fp32.onnx"), os.path.join(out_dir, "model_int8.onnx")
    torch.onnx.export(Logits(network), (sample["input_ids"], sample["attention_mask"]), fp32, opset_version=17,
                      dynamo=False, input_names=["input_ids", "attention_mask"], output_names=["logits"],
                      dynamic_axes={n: {0: "batch", 1: "sequence"} for n in ("input_ids", "attention_mask", "logits")})
    quantize_dynamic(fp32, int8, weight_type=QuantType.QInt8)
    token_model.tokenizer.backend_tokenizer.save(os.path.join(out_dir, "tokenizer.json"))
    info = {"model_name": token_model.config.model_name, "max_length": token_model.config.max_length,
            "pad_token_id": int(token_model.tokenizer.pad_token_id), **(extra_info or {})}
    with open(os.path.join(out_dir, "model_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    if torch.cuda.is_available() and token_model.device == "cuda":
        token_model.model.to("cuda")
    return out_dir
