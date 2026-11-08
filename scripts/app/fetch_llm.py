#!/usr/bin/env python
"""
Download the small language model that explains single words on the board.

    python scripts/app/fetch_llm.py        # outputs/llm/qwen2.5-0.5b-instruct (790 MB)

It is Qwen2.5-0.5B-Instruct (Apache 2.0), converted to ONNX with 4-bit weights by the
onnx-community project, at a pinned version. The 4-bit version ran faster on the board than
the INT8 one (3.1 against 1.8 tokens a second), and in our tests the INT8 versions wrote worse
text. Bigger models explain better, but a 1.5B model needs more memory than the board can give
it under ONNX Runtime (docs/EXPLANATIONS.md).

deploy_app.py install copies the folder to the board.
"""

import argparse
import json
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO = "onnx-community/Qwen2.5-0.5B-Instruct"
REVISION = "22942cb7d7ba4cc81bb4673549ca4d4614469b5e"
FILES = {"onnx/model_q4.onnx": "model.onnx", "tokenizer.json": "tokenizer.json"}
INFO = {"name": "Qwen2.5-0.5B-Instruct (4-bit)", "repo": REPO, "revision": REVISION,
        "base_model": "Qwen/Qwen2.5-0.5B-Instruct", "licence": "Apache-2.0",
        "eos_token_ids": [151645, 151643],  # <|im_end|> and <|endoftext|>
        # In testing, this model's replacement words often changed the meaning ("extreme" -> "moderate"),
        # and it could not judge meaning itself, so the board does not offer them (docs/EXPLANATIONS.md).
        "suggest_alternatives": False,
        # How often its reasons held up, in docs/EXPLANATIONS.md; the page shows this next to each answer.
        "sensible_reasons": ("the board showed a reason for 13 of 40 flagged words, and about a third of "
                             "those reasons were sensible")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=os.path.join(ROOT, "outputs", "llm", "qwen2.5-0.5b-instruct"))
    args = parser.parse_args(argv)
    from huggingface_hub import hf_hub_download

    os.makedirs(args.out, exist_ok=True)
    for remote, local in FILES.items():
        cached = os.path.realpath(hf_hub_download(REPO, remote, revision=REVISION))  # the cache may hold symlinks
        target = os.path.join(args.out, local)
        if os.path.lexists(target):  # also a broken link left by an earlier run
            os.remove(target)
        try:
            os.link(cached, target)  # no second copy on disk
        except OSError:
            shutil.copy(cached, target)
    with open(os.path.join(args.out, "llm_info.json"), "w", encoding="utf-8") as f:
        json.dump(INFO, f, indent=2)
    size = sum(os.path.getsize(os.path.join(args.out, n)) for n in os.listdir(args.out))
    print(f"{INFO['name']} in {args.out}: {size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
