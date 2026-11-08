import hashlib
import importlib.util
import io
import json
import os
import tarfile
import types

import pytest

APP_SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "app")


def load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(APP_SCRIPTS, f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fetch_llm = load("fetch_llm")
deploy_app = load("deploy_app")


def release_archive(path):
    """A small stand-in for llama.cpp's release archive, with its versioned libraries and their links."""
    files = {"llama-server": b"server", "libllama-server-impl.so": b"impl", "LICENSE": b"MIT",
             "libllama-common.so.0.9.0": b"common", "libllama.so.0.9.0": b"llama", "libmtmd.so.0.9.0": b"mtmd",
             "libggml.so.0.29.0": b"ggml", "libggml-base.so.0.29.0": b"base", "libggml-cpu-armv8.0_1.so": b"a53",
             "libggml-cpu-armv9.2_1.so": b"v9", "libggml-rpc.so": b"rpc", "llama-cli": b"cli"}
    links = {"libllama-common.so.0": "libllama-common.so.0.9.0", "libllama.so.0": "libllama.so.0.9.0",
             "libmtmd.so.0": "libmtmd.so.0.9.0", "libggml.so.0": "libggml.so.0.29.0",
             "libggml-base.so.0": "libggml-base.so.0.29.0"}
    with tarfile.open(path, "w:gz") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(f"llama-b12207/{name}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        for name, target in links.items():
            info = tarfile.TarInfo(f"llama-b12207/{name}")
            info.type, info.linkname = tarfile.SYMTYPE, target
            tar.addfile(info)
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def test_only_the_server_its_libraries_and_the_cpu_backends_are_kept(tmp_path, monkeypatch):
    archive = tmp_path / "release.tar.gz"
    digest = release_archive(archive)
    monkeypatch.setattr(fetch_llm, "LLAMA_CPP", {**fetch_llm.LLAMA_CPP, "sha256": digest})
    monkeypatch.setattr(fetch_llm.urllib.request, "urlopen", lambda url, timeout=None: open(archive, "rb"))
    out = tmp_path / "model"
    files = fetch_llm.fetch_llama_cpp(str(out))
    assert files == [f"llama/{n}" for n in fetch_llm.LLAMA_FILES] + \
        ["llama/libggml-cpu-armv8.0_1.so", "llama/libggml-cpu-armv9.2_1.so"]
    assert sorted(os.listdir(out / "llama")) == sorted(f.split("/")[1] for f in files)  # no RPC backend, no CLI
    assert (out / "llama" / "libllama.so.0").read_bytes() == b"llama"  # the link's target, under the linked name

    monkeypatch.setattr(fetch_llm, "LLAMA_CPP", {**fetch_llm.LLAMA_CPP, "sha256": "0" * 64})
    with pytest.raises(SystemExit, match="SHA-256"):
        fetch_llm.fetch_llama_cpp(str(out))


class FakeBoard:
    def __init__(self, listing):
        self.listing, self.commands, self.pushed = listing, [], []

    def shell(self, command, check=True, capture=False):
        self.commands.append(command)
        return types.SimpleNamespace(stdout=self.listing if "find . -type f -printf" in command else "", returncode=0)

    def run(self, *args, check=True, capture=False):
        self.pushed.append(args[2])


def test_pushing_a_model_skips_unchanged_files_and_removes_the_previous_models(tmp_path):
    (tmp_path / "llama").mkdir()
    (tmp_path / "model.gguf").write_bytes(b"x" * 1000)
    (tmp_path / "llama" / "llama-server").write_bytes(b"server")
    (tmp_path / "llm_info.json").write_text(json.dumps({"files": ["model.gguf", "llama/llama-server"]}))
    board = FakeBoard("1000 model.gguf\n5 llm_info.json\n786156820 model.onnx\n7031673 tokenizer.json\n")
    deploy_app.push_llm(board, str(tmp_path))
    remote = f"{deploy_app.REMOTE_CACHE}/llm"
    assert board.pushed == [f"{remote}/llama/llama-server", f"{remote}/llm_info.json"]  # model.gguf is unchanged
    removals = [c for c in board.commands if "rm -f" in c]
    assert removals == [f"cd {remote} && rm -f 'model.onnx' 'tokenizer.json' "
                        f"&& find . -mindepth 1 -type d -empty -delete"]
    assert any(c == f"mkdir -p {remote}/llama" for c in board.commands)
    assert "chmod 755 llama/llama-server llama/*.so*" in board.commands[-1]


def test_a_model_folder_without_its_files_is_refused(tmp_path):
    (tmp_path / "llm_info.json").write_text(json.dumps({"files": ["model.gguf"]}))
    with pytest.raises(SystemExit, match="lacks"):
        deploy_app.push_llm(FakeBoard(""), str(tmp_path))
