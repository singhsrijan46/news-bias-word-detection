"""
Bias Lens on the Arduino UNO Q.

Serves a web page and an API on port 7000 to devices on the home network. Devices pair
once (a 6-digit code on the LED matrix), then send encrypted requests:
    POST /api/pair/start, /api/pair/exchange, /api/pair/confirm   pairing (see security.py)
    POST /api/secure     encrypted {op: "analyze" | "perspective" | "info" | "devices" | "forget"}
    GET  /api/info       the model and its settings (nothing private)
    POST /api/analyze    plain-text analysis, refused unless data/settings.json allows unpaired use
The model runs on the board's CPU and the LED matrix shows what is happening. No text is
stored, and none leaves the board.
"""

import json
import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "vendor"))  # onnxruntime, tokenizers, rbcl: installed by deploy_app.py
sys.path.insert(0, HERE)

from arduino.app_bricks.web_ui import WebUI  # noqa: E402
from arduino.app_utils import App, Bridge  # noqa: E402
from fastapi import Body, HTTPException  # noqa: E402

from analyzer import Analyzer, Perspective  # noqa: E402
from security import Pairing, SecurityError  # noqa: E402

DATA = os.path.join(os.path.dirname(HERE), "data")  # kept across redeploys
analyzer = Analyzer.from_dir(os.path.join(HERE, "model"))
perspective = Perspective(os.path.join(HERE, "perspective"), analyzer.threshold)  # M3, loaded on first use
ui = WebUI()


def settings() -> dict:
    path = os.path.join(DATA, "settings.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {"allow_unpaired": False}


def to_matrix(method: str, *params) -> None:
    """Tell the sketch what to show, without making the web request wait for it."""
    def send():
        try:
            Bridge.call(method, *params, timeout=3)
        except Exception as error:  # the page still works if the microcontroller does not answer
            print(f"LED matrix: {method} failed: {error}")
    threading.Thread(target=send, daemon=True).start()


def show_code(code: str) -> None:
    to_matrix("show_code", code)
    # For testing without looking at the matrix: data/settings.json {"debug_write_code": true} also writes
    # the code to data/pairing_code.txt, readable only with access to the board itself (adb or SSH).
    if settings().get("debug_write_code"):
        path = os.path.join(DATA, "pairing_code.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code + "\n")
        os.chmod(path, 0o600)


pairing = Pairing(os.path.join(DATA, "devices.json"),
                  show_code=show_code,
                  on_paired=lambda: to_matrix("show_paired"),
                  on_failed=lambda: to_matrix("show_error"))


def run_analysis(text: str) -> dict:
    text = str(text).strip()
    if not text:
        raise HTTPException(status_code=400, detail="No text given.")
    to_matrix("show_busy")
    try:
        result = analyzer.analyze(text)
    except Exception:
        to_matrix("show_error")
        raise
    to_matrix("show_levels", "".join(str(level) for level in result["summary"]["strip"]))
    s = result["summary"]
    print(f"Analysed {s['words']} words in {s['seconds']:.2f} s: {s['flagged']} flagged")  # counts only, never text
    return result


def run_perspective(text: str) -> dict:
    text = str(text).strip()
    if not text:
        raise HTTPException(status_code=400, detail="No text given.")
    if not perspective.available:
        raise HTTPException(status_code=404, detail="The perspective model is not installed on this board.")
    to_matrix("show_busy")
    try:
        result = perspective.analyze(text)
    except Exception:
        to_matrix("show_error")
        raise
    to_matrix("show_levels", "".join(str(level) for level in result["strip"]))
    print(f"Perspective view of {len(result['words'])} words in {result['seconds']:.2f} s")  # counts only
    return result


def guarded(action):
    try:
        return action()
    except SecurityError as error:
        if error.status in (401, 403):
            time.sleep(0.5)  # slow down guessing
        raise HTTPException(status_code=error.status, detail=str(error))
    except KeyError as error:
        raise HTTPException(status_code=400, detail=f"Missing field {error}.")


def pair_start(payload: dict = Body(...)):
    return guarded(lambda: pairing.start(payload.get("name", "device")))


def pair_exchange(payload: dict = Body(...)):
    return guarded(lambda: pairing.exchange(payload["sid"], payload["ya"]))


def pair_confirm(payload: dict = Body(...)):
    return guarded(lambda: pairing.confirm(payload["sid"], payload["device"], payload["mac"]))


def secure(payload: dict = Body(...)):
    device, nonce, request = guarded(lambda: pairing.open_request(payload))
    op = request.get("op")
    if op == "analyze":
        result = run_analysis(request.get("text", ""))
    elif op == "perspective":
        result = run_perspective(request.get("text", ""))
    elif op == "info":
        result = {**analyzer.describe(), "perspective": perspective.available}
    elif op == "devices":
        result = {"devices": pairing.list_devices(), "you": device}
    elif op == "forget":
        result = {"removed": pairing.forget(request.get("device") or device)}
    else:
        raise HTTPException(status_code=400, detail="Unknown operation.")
    return pairing.seal_response(device, nonce, result)


def analyze_plain(payload: dict = Body(...)):
    if not settings().get("allow_unpaired"):
        raise HTTPException(status_code=403, detail="Pair this device first.")
    return run_analysis(payload.get("text", ""))


def info():
    return {**analyzer.describe(), "paired_devices": len(pairing.devices), "perspective": perspective.available,
            "allow_unpaired": bool(settings().get("allow_unpaired"))}


def greet_matrix():
    time.sleep(5)  # give the microcontroller's Bridge time to start
    to_matrix("show_idle")


ui.expose_api("POST", "/api/pair/start", pair_start)
ui.expose_api("POST", "/api/pair/exchange", pair_exchange)
ui.expose_api("POST", "/api/pair/confirm", pair_confirm)
ui.expose_api("POST", "/api/secure", secure)
ui.expose_api("POST", "/api/analyze", analyze_plain)
ui.expose_api("GET", "/api/info", info)
threading.Thread(target=greet_matrix, daemon=True).start()
print(f"Bias Lens ready: flag threshold {analyzer.threshold:.2f}, {len(analyzer.lexicon)} lexicon words, "
      f"{len(pairing.devices)} paired devices")
App.run()
