import hashlib
import json
import os
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "app", "bias-lens", "python"))

rbcl = pytest.importorskip("rbcl")
from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # noqa: E402

from security import (  # noqa: E402
    Pairing, SecurityError, b64, derive_keys, generator, request_aad, response_aad, unb64,
)


class Clock:
    def __init__(self):
        self.now = 1_000_000.0

    def __call__(self):
        return self.now


@pytest.fixture
def board(tmp_path):
    clock = Clock()
    pairing = Pairing(str(tmp_path / "devices.json"), clock=clock)
    pairing.shown, pairing.test_clock = {}, clock
    pairing.show_code = lambda code: pairing.shown.update(code=code)  # what the LED matrix would show
    return pairing


def device_side(board, typed=lambda code: code, name="laptop"):
    """The device's half of pairing, in Python; `typed` turns the shown code into what the user types."""
    sid_b64 = board.start(name)["sid"]
    sid = unb64(sid_b64)
    scalar = rbcl.crypto_core_ristretto255_scalar_random()
    ya = rbcl.crypto_scalarmult_ristretto255(scalar, generator(sid, typed(board.shown["code"])))
    reply = board.exchange(sid_b64, b64(ya))
    yb = unb64(reply["yb"])
    keys = derive_keys(sid, ya, yb, rbcl.crypto_scalarmult_ristretto255(scalar, yb))
    return sid_b64, ya, reply, keys


def paired(board):
    sid_b64, _, reply, keys = device_side(board)
    board.confirm(sid_b64, reply["device"], b64(keys["client_mac"]))
    return reply["device"], keys["device_key"]


def seal(device, key, payload, ts, nonce=b"\x01" * 12):
    ct = AESGCM(key).encrypt(nonce, json.dumps(payload).encode(), request_aad(device, ts))
    return {"device": device, "ts": ts, "nonce": b64(nonce), "ct": b64(ct)}


def test_pairing_with_the_right_code_stores_the_device(board):
    device, key = paired(board)
    assert device in board.devices and len(key) == 32
    assert json.load(open(board.store_path))[device]["name"] == "laptop"


def test_the_board_confirms_with_a_tag_the_device_can_check(board):
    _, _, reply, keys = device_side(board, name="phone")
    assert unb64(reply["mac"]) == keys["board_mac"]


def test_a_wrong_code_fails_and_burns_the_session(board):
    wrong = lambda code: f"{(int(code) + 1) % 10**6:06d}"
    sid_b64, ya, reply, keys = device_side(board, typed=wrong, name="intruder")
    assert unb64(reply["mac"]) != keys["board_mac"]  # the device notices the wrong code
    with pytest.raises(SecurityError) as error:
        board.confirm(sid_b64, reply["device"], b64(keys["client_mac"]))
    assert error.value.status == 403
    with pytest.raises(SecurityError):  # the session cannot be retried
        board.exchange(sid_b64, b64(ya))
    assert board.devices == {}


def test_sessions_expire(board):
    sid_b64 = board.start("slow")["sid"]
    board.test_clock.now += 121
    with pytest.raises(SecurityError) as error:
        board.exchange(sid_b64, b64(rbcl.crypto_core_ristretto255_random()))
    assert error.value.status == 410


def test_encrypted_request_round_trip_and_replay_protection(board):
    device, key = paired(board)
    ts = int(board.test_clock.now)
    envelope = seal(device, key, {"op": "analyze", "text": "hi"}, ts)
    got_device, nonce, payload = board.open_request(envelope)
    assert (got_device, payload) == (device, {"op": "analyze", "text": "hi"})
    sealed = board.seal_response(device, nonce, {"ok": 1})
    plain = AESGCM(key).decrypt(unb64(sealed["nonce"]), unb64(sealed["ct"]), response_aad(device, nonce))
    assert json.loads(plain) == {"ok": 1}
    with pytest.raises(SecurityError) as error:
        board.open_request(envelope)
    assert error.value.status == 409


def test_requests_are_rejected_when_stale_tampered_or_unknown(board):
    device, key = paired(board)
    now = int(board.test_clock.now)
    with pytest.raises(SecurityError):
        board.open_request(seal(device, key, {"op": "info"}, now - 1000, nonce=b"\x02" * 12))
    tampered = seal(device, key, {"op": "info"}, now, nonce=b"\x03" * 12)
    tampered["ts"] = now + 1  # changes the associated data
    with pytest.raises(SecurityError):
        board.open_request(tampered)
    with pytest.raises(SecurityError) as error:
        board.open_request(seal("0000", key, {"op": "info"}, now, nonce=b"\x04" * 12))
    assert error.value.status == 401


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_javascript_client_computes_the_same_keys(tmp_path):
    sid = hashlib.sha256(b"sid").digest()[:16]
    code = "042917"
    ya_scalar = rbcl.crypto_core_ristretto255_scalar_reduce(hashlib.sha512(b"ya").digest())
    yb_scalar = rbcl.crypto_core_ristretto255_scalar_reduce(hashlib.sha512(b"yb").digest())
    g = generator(sid, code)
    ya = rbcl.crypto_scalarmult_ristretto255(ya_scalar, g)
    yb = rbcl.crypto_scalarmult_ristretto255(yb_scalar, g)
    k = rbcl.crypto_scalarmult_ristretto255(yb_scalar, ya)
    keys = derive_keys(sid, ya, yb, k)
    aad = "BiasLens-v1|req|test|1"
    nonce = b"\x09" * 12
    vector = {"sid": sid.hex(), "code": code, "ya_scalar": ya_scalar.hex(), "yb_scalar": yb_scalar.hex(),
              "aad": aad, "py_nonce": b64(nonce),
              "py_ct": b64(AESGCM(keys["device_key"]).encrypt(nonce, b"hello from Python", aad.encode()))}
    path = tmp_path / "vector.json"
    path.write_text(json.dumps(vector))
    out = subprocess.run(["node", os.path.join(ROOT, "tests", "js", "interop.mjs"), str(path)],
                         capture_output=True, text=True, check=True, cwd=ROOT)
    js = json.loads(out.stdout.strip().splitlines()[-1])
    assert js["g"] == g.hex()
    assert (js["Ya"], js["Yb"], js["K"]) == (ya.hex(), yb.hex(), k.hex())
    assert js["board_mac"] == keys["board_mac"].hex() and js["client_mac"] == keys["client_mac"].hex()
    assert js["device_key"] == keys["device_key"].hex()
    assert js["from_python"] == "hello from Python"
    from_js = AESGCM(keys["device_key"]).decrypt(unb64(js["js_nonce"]), unb64(js["js_ct"]), aad.encode())
    assert from_js == b"hello from JavaScript"
