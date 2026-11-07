"""
Bias Lens on the Arduino UNO Q.

Serves a web page and a REST API on port 7000 to every device on the home network:
    POST /api/analyze   {"text": "..."}  ->  every sentence's word scores, and a summary
    GET  /api/info                        ->  the model and its settings
The model runs on the board's CPU, and the LED matrix shows the last article's heat
strip. No text leaves the board.
"""

import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "vendor"))  # onnxruntime and tokenizers, installed by deploy_app.py
sys.path.insert(0, HERE)

from arduino.app_bricks.web_ui import WebUI  # noqa: E402
from arduino.app_utils import App, Bridge  # noqa: E402
from fastapi import Body, HTTPException  # noqa: E402

from analyzer import Analyzer  # noqa: E402

analyzer = Analyzer.from_dir(os.path.join(HERE, "model"))
ui = WebUI()


def to_matrix(method: str, *params) -> None:
    """Tell the sketch what to show, without making the web request wait for it."""
    def send():
        try:
            Bridge.call(method, *params, timeout=3)
        except Exception as error:  # the page still works if the microcontroller does not answer
            print(f"LED matrix: {method} failed: {error}")
    threading.Thread(target=send, daemon=True).start()


def analyze(payload: dict = Body(...)):
    text = str(payload.get("text", "")).strip()
    if not text:
        raise HTTPException(status_code=400, detail="No text given.")
    to_matrix("show_busy")
    try:
        result = analyzer.analyze(text)
    except Exception:
        to_matrix("show_idle")
        raise
    to_matrix("show_levels", "".join(str(level) for level in result["summary"]["strip"]))
    s = result["summary"]
    print(f"Analysed {s['words']} words in {s['seconds']:.2f} s: {s['flagged']} flagged")
    return result


def info():
    return analyzer.describe()


def greet_matrix():
    time.sleep(5)  # give the microcontroller's Bridge time to start
    to_matrix("show_idle")


ui.expose_api("POST", "/api/analyze", analyze)
ui.expose_api("GET", "/api/info", info)
threading.Thread(target=greet_matrix, daemon=True).start()
print(f"Bias Lens ready: flag threshold {analyzer.threshold:.2f}, {len(analyzer.lexicon)} lexicon words")
App.run()
