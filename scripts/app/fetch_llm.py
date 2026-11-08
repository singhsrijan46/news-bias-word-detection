#!/usr/bin/env python
"""
Download the small language model that explains single words on the board.

    python scripts/app/fetch_llm.py                                 # outputs/llm/qwen2.5-1.5b-instruct (1.1 GB)
    python scripts/app/fetch_llm.py --model qwen2.5-0.5b-instruct   # the smaller model (790 MB)

The board's model is Qwen2.5-1.5B-Instruct (Apache 2.0) with 4-bit weights, in Qwen's own GGUF
version (Q4_K_M), at a pinned version. It runs with llama.cpp's server: the project's release build
for 64-bit ARM Linux, at a pinned release and checked against its published SHA-256. Only the server
and the libraries it loads are kept. The RPC backend, which can hand work to other machines, is
left out.

Under ONNX Runtime the 1.5B model would not fit in the board's memory. The 0.5B model, which does,
runs with the app's own ONNX Runtime and needs no llama.cpp, but gave sensible reasons far less
often (docs/EXPLANATIONS.md).

deploy_app.py install copies the folder to the board.
"""

import argparse
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LLAMA_CPP = {"release": "b12207", "commit": "998200491327347eed85f007e4de7a1a7c871453",
             "asset": "llama-b12207-bin-ubuntu-arm64.tar.gz",
             "sha256": "cb415cd1ad180f05545d488577005a42d706886bcf4eb284ff0043b67c8559c0", "licence": "MIT"}
LLAMA_URL = "https://github.com/ggml-org/llama.cpp/releases/download/{release}/{asset}"
# The server and the libraries it links. ggml's CPU backends come in variants for different ARM
# CPUs; the server loads the best one for the CPU it finds (on the UNO Q, armv8.0).
LLAMA_FILES = ["llama-server", "libllama-server-impl.so", "libllama-common.so.0", "libllama.so.0", "libmtmd.so.0",
               "libggml.so.0", "libggml-base.so.0", "LICENSE"]
LLAMA_BACKENDS = "libggml-cpu-"

MODELS = {
    "qwen2.5-1.5b-instruct": {
        "repo": "Qwen/Qwen2.5-1.5B-Instruct-GGUF", "revision": "91cad51170dc346986eccefdc2dd33a9da36ead9",
        "files": {"qwen2.5-1.5b-instruct-q4_k_m.gguf": "model.gguf"},
        "sha256": {"model.gguf": "6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e"},
        "llama_cpp": True,
        "info": {"name": "Qwen2.5-1.5B-Instruct (4-bit)", "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
                 "licence": "Apache-2.0", "runtime": "llama.cpp", "model_file": "model.gguf",
                 "repeat_penalty": 1.1,  # better reasons than none, in the board's test
                 # A quarter of its replacement words changed the meaning ("kill" -> "target"), so the board
                 # does not offer them (docs/EXPLANATIONS.md).
                 "suggest_alternatives": False,
                 # From the 40-word test on the board; the page shows it next to each answer.
                 "sensible_reasons": ("it gave a reason for 26 of 40 flagged words: 7 were sensible, 10 partly "
                                      "right and 9 wrong"),
                 "typical_seconds": 20}},
    "qwen2.5-0.5b-instruct": {
        "repo": "onnx-community/Qwen2.5-0.5B-Instruct", "revision": "22942cb7d7ba4cc81bb4673549ca4d4614469b5e",
        "files": {"onnx/model_q4.onnx": "model.onnx", "tokenizer.json": "tokenizer.json"},
        "sha256": {"model.onnx": "09235a3b1c135cd04ef570e5053b5c079e028078a2cc5f76ba6a251e91bf3296",
                   "tokenizer.json": "a8506e7111b80c6d8635951a02eab0f4e1a8e4e5772da83846579e97b16f61bf"},
        "llama_cpp": False,
        "info": {"name": "Qwen2.5-0.5B-Instruct (4-bit)", "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
                 "licence": "Apache-2.0", "runtime": "onnxruntime",
                 "eos_token_ids": [151645, 151643],  # <|im_end|> and <|endoftext|>
                 # Its replacement words often changed the meaning ("extreme" -> "moderate"), and it could
                 # not judge meaning itself, so the board does not offer them (docs/EXPLANATIONS.md).
                 "suggest_alternatives": False,
                 "sensible_reasons": ("it gave a reason for 13 of 40 flagged words: 3 were sensible, 3 partly "
                                      "right and 7 wrong"),
                 "typical_seconds": 15}},
}


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def link_or_copy(source: str, target: str) -> None:
    if os.path.lexists(target):  # also a broken link left by an earlier run
        os.remove(target)
    try:
        os.link(source, target)  # no second copy on disk
    except OSError:
        shutil.copy(source, target)


def fetch_llama_cpp(out: str) -> list:
    """The server and its libraries from llama.cpp's release archive, into out/llama: their paths in out."""
    target = os.path.join(out, "llama")
    shutil.rmtree(target, ignore_errors=True)
    os.makedirs(target)
    with tempfile.TemporaryDirectory() as tmp:
        archive = os.path.join(tmp, LLAMA_CPP["asset"])
        with urllib.request.urlopen(LLAMA_URL.format(**LLAMA_CPP), timeout=60) as response, \
                open(archive, "wb") as f:
            shutil.copyfileobj(response, f)
        if sha256(archive) != LLAMA_CPP["sha256"]:
            raise SystemExit(f"{LLAMA_CPP['asset']} does not match its published SHA-256")
        folder = f"llama-{LLAMA_CPP['release']}/"
        with tarfile.open(archive) as tar:
            names = [m.name[len(folder):] for m in tar.getmembers() if m.name.startswith(folder)]
            wanted = LLAMA_FILES + sorted(n for n in names if n.startswith(LLAMA_BACKENDS) and n.endswith(".so"))
            for name in wanted:
                source = tar.extractfile(folder + name)  # follows the archive's own links, such as libllama.so.0
                if source is None:
                    raise SystemExit(f"{LLAMA_CPP['asset']} lacks {name}")
                with open(os.path.join(target, name), "wb") as f:
                    shutil.copyfileobj(source, f)
    return [f"llama/{name}" for name in wanted]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=sorted(MODELS), default="qwen2.5-1.5b-instruct")
    parser.add_argument("--out", help="default: outputs/llm/<model>")
    args = parser.parse_args(argv)
    from huggingface_hub import hf_hub_download

    spec = MODELS[args.model]
    out = args.out or os.path.join(ROOT, "outputs", "llm", args.model)
    os.makedirs(out, exist_ok=True)
    for remote, local in spec["files"].items():
        # realpath: the Hugging Face cache may hold symlinks
        cached = os.path.realpath(hf_hub_download(spec["repo"], remote, revision=spec["revision"]))
        if sha256(cached) != spec["sha256"][local]:
            raise SystemExit(f"{spec['repo']}/{remote} does not match its SHA-256")
        link_or_copy(cached, os.path.join(out, local))
    files = list(spec["files"].values()) + (fetch_llama_cpp(out) if spec["llama_cpp"] else [])
    info = {**spec["info"], "repo": spec["repo"], "revision": spec["revision"], "sha256": spec["sha256"],
            **({"llama_cpp": LLAMA_CPP} if spec["llama_cpp"] else {}), "files": files}
    with open(os.path.join(out, "llm_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    size = sum(os.path.getsize(os.path.join(out, n)) for n in files)
    print(f"{info['name']} in {out}: {size / 1e6:.0f} MB in {len(files)} files")


if __name__ == "__main__":
    main()
