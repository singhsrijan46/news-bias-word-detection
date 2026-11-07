"""
Pairing and encryption between Bias Lens and the reader's devices.

Pairing (CPace-style password-authenticated key exchange over ristretto255):
  1. A device asks to pair. The board opens a 2-minute session and shows a 6-digit
     code on its LED matrix, so only someone who can see the board learns it.
  2. Both sides turn the session id and the code into the same group element g, and
     exchange Y = y * g for fresh secret scalars y. Both then compute K = y_mine * Y_other.
     Without the code, an attacker can test only one guess per session, and someone
     listening in learns nothing that lets them test guesses offline.
  3. Both prove they derived the same K with an HMAC, which also yields a 256-bit key
     for this device. Each session allows one attempt, so a wrong code burns it.

Requests after pairing are AES-256-GCM encrypted under the device's key, with the
device id and a timestamp bound in as associated data. Timestamps must be within five
minutes, and every nonce is accepted once.

The protocol uses standard parts (ristretto255 as in RFC 9496, SHA-512, HMAC-SHA256,
AES-256-GCM) but its composition has not been reviewed by a third party.
"""

import base64
import hashlib
import hmac
import json
import os
import secrets
import threading
import time
from typing import Callable, Dict, Optional, Tuple

DSI = b"BiasLens-CPace-v1"
ISK_LABEL = b"BiasLens-ISK-v1"
SESSION_SECONDS = 120
CLOCK_SKEW_SECONDS = 300
MAX_STARTS_PER_HOUR = 30


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def unb64(text: str) -> bytes:
    return base64.b64decode(text.encode("ascii"), validate=True)


def generator(sid: bytes, code: str) -> bytes:
    """The session's generator: the pairing code and session id hashed onto ristretto255."""
    import rbcl

    digest = hashlib.sha512(DSI + b"\x00" + sid + b"\x00" + code.encode("utf-8")).digest()
    return rbcl.crypto_core_ristretto255_from_hash(digest)


def derive_keys(sid: bytes, ya: bytes, yb: bytes, k: bytes) -> Dict[str, bytes]:
    """Confirmation tags and the device key, from the shared point K and the transcript."""
    transcript = sid + ya + yb
    isk = hashlib.sha512(ISK_LABEL + transcript + k).digest()
    tag = lambda label: hmac.new(isk, label, hashlib.sha256).digest()
    return {
        "board_mac": hmac.new(tag(b"board confirm"), transcript, hashlib.sha256).digest(),
        "client_mac": hmac.new(tag(b"client confirm"), transcript, hashlib.sha256).digest(),
        "device_key": tag(b"device key"),
    }


def request_aad(device: str, ts: int) -> bytes:
    return f"BiasLens-v1|req|{device}|{ts}".encode("utf-8")


def response_aad(device: str, request_nonce: str) -> bytes:
    return f"BiasLens-v1|res|{device}|{request_nonce}".encode("utf-8")


class SecurityError(Exception):
    """A request failed authentication; `status` is the HTTP status to answer with."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


class Pairing:
    """Pairing sessions, paired devices (kept in a JSON file) and encrypted requests."""

    def __init__(self, store_path: str, show_code: Callable[[str], None] = lambda code: None,
                 on_paired: Callable[[], None] = lambda: None, on_failed: Callable[[], None] = lambda: None,
                 clock: Callable[[], float] = time.time):
        self.store_path = store_path
        self.show_code, self.on_paired, self.on_failed = show_code, on_paired, on_failed
        self.clock = clock
        self._lock = threading.Lock()
        self._session: Optional[Dict] = None
        self._starts: list = []
        self._seen_nonces: Dict[str, float] = {}
        self.devices: Dict[str, Dict] = self._load()

    # ------------------------------------------------------------ storage

    def _load(self) -> Dict[str, Dict]:
        if os.path.exists(self.store_path):
            with open(self.store_path, encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.store_path) or ".", exist_ok=True)
        tmp = self.store_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.devices, f, indent=2)
        os.replace(tmp, self.store_path)

    # ------------------------------------------------------------ pairing

    def start(self, name: str) -> Dict:
        now = self.clock()
        with self._lock:
            self._starts = [t for t in self._starts if now - t < 3600]
            if len(self._starts) >= MAX_STARTS_PER_HOUR:
                raise SecurityError(429, "Too many pairing attempts; try again later.")
            self._starts.append(now)
            code = f"{secrets.randbelow(10**6):06d}"
            sid = secrets.token_bytes(16)
            self._session = {"sid": sid, "code": code, "name": str(name)[:60] or "device",
                             "expires": now + SESSION_SECONDS, "stage": "started"}
        self.show_code(code)
        return {"sid": b64(sid), "expires_in": SESSION_SECONDS}

    def _active(self, sid_b64: str, stage: str) -> Dict:
        session = self._session
        if (session is None or b64(session["sid"]) != sid_b64 or session["stage"] != stage
                or self.clock() > session["expires"]):
            raise SecurityError(410, "This pairing session is over; start again.")
        return session

    def exchange(self, sid_b64: str, ya_b64: str) -> Dict:
        import rbcl

        with self._lock:
            session = self._active(sid_b64, "started")
            session["stage"] = "exchanged"  # one attempt per session
            ya = unb64(ya_b64)
            if len(ya) != 32 or not rbcl.crypto_core_ristretto255_is_valid_point(ya):
                raise SecurityError(400, "Invalid key share.")
            yb_scalar = rbcl.crypto_core_ristretto255_scalar_random()
            g = generator(session["sid"], session["code"])
            yb = rbcl.crypto_scalarmult_ristretto255(yb_scalar, g)
            k = rbcl.crypto_scalarmult_ristretto255(yb_scalar, ya)
            keys = derive_keys(session["sid"], ya, yb, k)
            session.update(device=secrets.token_hex(8), keys=keys)
            return {"yb": b64(yb), "mac": b64(keys["board_mac"]), "device": session["device"]}

    def confirm(self, sid_b64: str, device: str, mac_b64: str) -> Dict:
        with self._lock:
            session = self._active(sid_b64, "exchanged")
            self._session = None
            if device != session["device"] or not hmac.compare_digest(unb64(mac_b64), session["keys"]["client_mac"]):
                self.on_failed()
                raise SecurityError(403, "The code did not match.")
            self.devices[device] = {"name": session["name"], "key": b64(session["keys"]["device_key"]),
                                    "paired_at": int(self.clock())}
            self._save()
        self.on_paired()
        return {"ok": True, "device": device}

    def forget(self, device: str) -> bool:
        with self._lock:
            removed = self.devices.pop(device, None) is not None
            if removed:
                self._save()
        return removed

    def list_devices(self):
        return [{"device": d, "name": v["name"], "paired_at": v["paired_at"]} for d, v in self.devices.items()]

    # ------------------------------------------------------------ encrypted requests

    def open_request(self, envelope: Dict) -> Tuple[str, str, Dict]:
        """Check and decrypt a request; returns (device, request nonce, payload)."""
        from cryptography.exceptions import InvalidTag
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        try:
            device, ts, nonce_b64, ct = envelope["device"], int(envelope["ts"]), envelope["nonce"], envelope["ct"]
            nonce, ciphertext = unb64(nonce_b64), unb64(ct)
        except (KeyError, TypeError, ValueError):
            raise SecurityError(400, "Malformed request.")
        record = self.devices.get(device)
        if record is None:
            raise SecurityError(401, "Unknown device; pair it first.")
        now = self.clock()
        if abs(now - ts) > CLOCK_SKEW_SECONDS:
            raise SecurityError(401, "Request too old or clock out of sync.")
        replay_key = device + "|" + nonce_b64
        with self._lock:
            self._seen_nonces = {k: t for k, t in self._seen_nonces.items() if now - t < 2 * CLOCK_SKEW_SECONDS}
            if replay_key in self._seen_nonces:
                raise SecurityError(409, "Repeated request.")
            self._seen_nonces[replay_key] = now
        try:
            plain = AESGCM(unb64(record["key"])).decrypt(nonce, ciphertext, request_aad(device, ts))
        except InvalidTag:
            raise SecurityError(401, "Request failed authentication.")
        return device, nonce_b64, json.loads(plain.decode("utf-8"))

    def seal_response(self, device: str, request_nonce: str, payload: Dict) -> Dict:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        nonce = secrets.token_bytes(12)
        ct = AESGCM(unb64(self.devices[device]["key"])).encrypt(
            nonce, json.dumps(payload).encode("utf-8"), response_aad(device, request_nonce))
        return {"nonce": b64(nonce), "ct": b64(ct)}
