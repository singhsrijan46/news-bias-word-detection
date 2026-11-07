#!/usr/bin/env python
"""
Convert the candidate encoders to ncnn, to run them on the UNO Q's Adreno 702 GPU.

ONNX Runtime has no GPU support on the board. ncnn runs on the GPU through
Vulkan, and the board's Mesa driver (Turnip) offers Vulkan 1.0. The converter is
pnnx, which traces the PyTorch model.

These models take token ids only, with no attention mask: the benchmark runs
one sentence at a time, so there is no padding to mask. That also avoids a
fault on the GPU, where ncnn computes in FP16: the mask adds a very large
negative number, which overflows to minus infinity in FP16, and 0 * -inf gives NaN.

The heads use the same seed as export_encoders.py, so ncnn and ONNX Runtime
outputs can be compared directly.

Usage:
    python scripts/edge/export_ncnn.py --models bert-tiny minilm-l6
"""

import argparse
import json
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from export_encoders import CANDIDATES, ROOT, SEED  # noqa: E402

TRACE_TOKENS = (42, 96)  # two lengths, so pnnx keeps the sequence length dynamic


def ncnn_logits(param: str, weights: str, ids: np.ndarray) -> np.ndarray:
    """Run an ncnn model on the CPU: token ids in, one score per token out."""
    import ncnn

    net = ncnn.Net()
    net.opt.use_vulkan_compute = False
    net.load_param(param)
    net.load_model(weights)
    extractor = net.create_extractor()
    extractor.input("in0", ncnn.Mat(ids.astype(np.int32)).clone())
    _, out = extractor.extract("out0")
    return np.array(out).reshape(len(ids), -1)


def unmasked_encoder(model):
    """The token classifier rebuilt without any attention-mask code, for one unpadded sentence.

    Hugging Face models build an all-ones mask even when none is passed, and that
    code does not convert to ncnn. Without padding the mask changes nothing, so
    this runs the embeddings, the encoder layers and the head directly.
    """
    import torch

    base_name = model.base_model_prefix  # "bert", "electra" or "roberta"
    base = getattr(model, base_name)
    emb = base.embeddings
    # RoBERTa numbers positions from padding_idx + 1; BERT and ELECTRA from 0.
    offset = emb.padding_idx + 1 if base_name == "roberta" else 0
    project = getattr(base, "embeddings_project", None)  # ELECTRA's embedding-to-hidden projection

    class Unmasked(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb, self.layers, self.project, self.classifier = emb, base.encoder.layer, project, model.classifier

        def forward(self, input_ids):
            positions = self.emb.position_ids[:, offset:offset + input_ids.shape[1]]
            hidden = (self.emb.word_embeddings(input_ids) + self.emb.position_embeddings(positions)
                      + self.emb.token_type_embeddings.weight[0])
            hidden = self.emb.LayerNorm(hidden)
            if self.project is not None:
                hidden = self.project(hidden)
            for layer in self.layers:
                hidden = layer(hidden)[0]
            return self.classifier(hidden)

    return Unmasked().eval()


def export_candidate(name: str, out_dir: str, onnx_dir: str) -> dict:
    import pnnx
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    started = time.time()
    hf_id = CANDIDATES[name]
    model_dir = os.path.join(out_dir, name)
    os.makedirs(model_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    torch.manual_seed(SEED)
    model = AutoModelForTokenClassification.from_pretrained(hf_id, num_labels=1, attn_implementation="eager").eval()
    wrapped = unmasked_encoder(model)

    ids = np.array(tokenizer("The senator's radical plan was slammed by critics.")["input_ids"], dtype=np.int64)
    with torch.no_grad():
        reference = model(input_ids=torch.from_numpy(ids).unsqueeze(0)).logits.numpy()[0]
        rebuilt = wrapped(torch.from_numpy(ids).unsqueeze(0)).numpy()[0]
    rebuild_diff = float(np.max(np.abs(reference - rebuilt)))
    if rebuild_diff > 1e-4:
        raise RuntimeError(f"{name}: unmasked rebuild differs from the model by {rebuild_diff:.1e}")

    examples = [torch.randint(1000, 2000, (1, n), dtype=torch.long) for n in TRACE_TOKENS]
    cwd = os.getcwd()
    os.chdir(model_dir)  # pnnx writes its files next to the traced model
    try:
        pnnx.export(wrapped, "model.pt", (examples[0],), (examples[1],))
    except SyntaxError as error:  # pnnx's own Python re-export can fail after the ncnn files are written
        print(f"  pnnx could not re-import its Python version ({error.msg}); using the ncnn files")
    finally:
        os.chdir(cwd)

    param, weights = os.path.join(model_dir, "model.ncnn.param"), os.path.join(model_dir, "model.ncnn.bin")
    with open(param, encoding="utf-8") as f:
        layer_types = {line.split()[0] for line in f.read().splitlines()[2:] if line.strip()}
    unconverted = sorted(t for t in layer_types if "." in t or "::" in t)
    if unconverted:
        raise RuntimeError(f"{name}: pnnx left layers ncnn cannot run: {unconverted}")
    got = ncnn_logits(param, weights, ids)
    diff = float(np.max(np.abs(reference - got)))

    for leftover in os.listdir(model_dir):  # keep only what the board needs
        if leftover not in ("model.ncnn.param", "model.ncnn.bin"):
            path = os.path.join(model_dir, leftover)
            shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)
    shutil.copy(os.path.join(onnx_dir, name, "inputs.npz"), os.path.join(model_dir, "inputs.npz"))
    with open(os.path.join(onnx_dir, name, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    manifest.update({"format": "ncnn", "ncnn_vs_pytorch_max_abs_diff": diff,
                     "ncnn_mb": (os.path.getsize(param) + os.path.getsize(weights)) / 2**20})
    with open(os.path.join(model_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"  {name}: ncnn {manifest['ncnn_mb']:.0f} MB, ncnn vs PyTorch {diff:.1e} ({time.time() - started:.0f} s)")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", choices=list(CANDIDATES), default=list(CANDIDATES))
    parser.add_argument("--out", default=os.path.join(ROOT, "outputs", "edge", "ncnn"))
    parser.add_argument("--onnx-dir", default=os.path.join(ROOT, "outputs", "edge", "models"),
                        help="export_encoders.py output, for the tokenized workload")
    args = parser.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    for name in args.models:
        export_candidate(name, args.out, args.onnx_dir)
    exported = [n for n in CANDIDATES if os.path.exists(os.path.join(args.out, n, "manifest.json"))]
    with open(os.path.join(args.out, "index.json"), "w", encoding="utf-8") as f:
        json.dump({"models": exported}, f, indent=2)
    print(f"Saved to {os.path.normpath(args.out)}")


if __name__ == "__main__":
    main()
