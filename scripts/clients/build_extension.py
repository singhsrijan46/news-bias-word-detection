#!/usr/bin/env python
"""
Assemble the Bias Lens browser extension into a folder Chrome or Edge can load.

    python scripts/clients/build_extension.py
    ->  outputs/clients/bias-lens-extension/       load with "Load unpacked" (developer mode)
    ->  outputs/clients/bias-lens-extension.zip    the same, for sharing

The extension's own files are in clients/extension; the pairing, encryption and overlay code
it shares with the board's web page is in clients/shared.
"""

import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SHARED = ["biaslens-client.js", "noble-bundle.js", "overlay-core.js", "NOTICE.md"]


def build(out_parent: str = os.path.join(ROOT, "outputs", "clients")) -> str:
    target = os.path.join(out_parent, "bias-lens-extension")
    shutil.rmtree(target, ignore_errors=True)
    shutil.copytree(os.path.join(ROOT, "clients", "extension"), target)
    for name in SHARED:
        shutil.copy(os.path.join(ROOT, "clients", "shared", name), target)
    archive = shutil.make_archive(target, "zip", target)
    print(f"Extension folder: {os.path.normpath(target)}\nZip: {os.path.normpath(archive)}")
    return target


if __name__ == "__main__":
    build()
