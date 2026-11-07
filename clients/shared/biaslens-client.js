// Bias Lens client: pair with the board and send encrypted requests.
// Shared by the browser extension and the board's own web page. The protocol is described in
// app/bias-lens/python/security.py; this is the device's side of it.
//
// Uses the noble libraries (audited, pure JavaScript) rather than WebCrypto, because the
// board's page is served over plain HTTP on the home network, where browsers hide WebCrypto.

import { RistrettoPoint, ed25519, sha512, sha256, hmac, gcm, randomBytes } from "./noble-bundle.js";

const encoder = new TextEncoder();
const decoder = new TextDecoder();
const ORDER = ed25519.CURVE.n;

export function b64(bytes) {
  let s = "";
  for (const b of bytes) s += String.fromCharCode(b);
  return btoa(s);
}

export function unb64(text) {
  return Uint8Array.from(atob(text), (c) => c.charCodeAt(0));
}

function concat(...parts) {
  const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
  let offset = 0;
  for (const p of parts) {
    out.set(p, offset);
    offset += p.length;
  }
  return out;
}

function equal(a, b) {
  if (a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a[i] ^ b[i];
  return diff === 0;
}

// Little-endian bytes to a scalar mod the group order.
export function scalarFromBytes(bytes) {
  let x = 0n;
  for (let i = bytes.length - 1; i >= 0; i--) x = (x << 8n) + BigInt(bytes[i]);
  return x % ORDER;
}

function randomScalar() {
  for (;;) {
    const s = scalarFromBytes(randomBytes(64));
    if (s !== 0n) return s;
  }
}

export function generator(sid, code) {
  const input = concat(encoder.encode("BiasLens-CPace-v1"), [0], sid, [0], encoder.encode(code));
  return RistrettoPoint.hashToCurve(sha512(input));
}

export function deriveKeys(sid, ya, yb, k) {
  const transcript = concat(sid, ya, yb);
  const isk = sha512(concat(encoder.encode("BiasLens-ISK-v1"), transcript, k));
  const tag = (label) => hmac(sha256, isk, encoder.encode(label));
  return {
    boardMac: hmac(sha256, tag("board confirm"), transcript),
    clientMac: hmac(sha256, tag("client confirm"), transcript),
    deviceKey: tag("device key"),
  };
}

async function postJson(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || `${response.status} ${response.statusText}`);
  return data;
}

// Step 1: ask the board for a pairing session; it shows a 6-digit code on its LED matrix.
export async function startPairing(baseUrl, name) {
  return postJson(`${baseUrl}/api/pair/start`, { name });
}

// Steps 2 and 3: run the key exchange with the code the user read off the board.
// Resolves to the pairing record to keep; rejects if the code was wrong.
export async function finishPairing(baseUrl, sidB64, code, name, scalar = randomScalar()) {
  const sid = unb64(sidB64);
  const ya = generator(sid, code).multiply(scalar).toBytes();
  const reply = await postJson(`${baseUrl}/api/pair/exchange`, { sid: sidB64, ya: b64(ya) });
  const yb = unb64(reply.yb);
  const shared = RistrettoPoint.fromBytes(yb).multiply(scalar);
  if (shared.is0()) throw new Error("Pairing failed.");
  const keys = deriveKeys(sid, ya, yb, shared.toBytes());
  if (!equal(keys.boardMac, unb64(reply.mac))) {
    throw new Error("The code did not match. Start again for a new code.");
  }
  await postJson(`${baseUrl}/api/pair/confirm`, { sid: sidB64, device: reply.device, mac: b64(keys.clientMac) });
  return { baseUrl, device: reply.device, key: b64(keys.deviceKey), name, pairedAt: Date.now() };
}

// An encrypted call to the board: {op: "analyze", text} and friends.
export async function secureCall(pairing, payload) {
  const key = unb64(pairing.key);
  const nonce = randomBytes(12);
  const ts = Math.floor(Date.now() / 1000);
  const aad = encoder.encode(`BiasLens-v1|req|${pairing.device}|${ts}`);
  const ct = gcm(key, nonce, aad).encrypt(encoder.encode(JSON.stringify(payload)));
  const reply = await postJson(`${pairing.baseUrl}/api/secure`, {
    device: pairing.device, ts, nonce: b64(nonce), ct: b64(ct),
  });
  const replyAad = encoder.encode(`BiasLens-v1|res|${pairing.device}|${b64(nonce)}`);
  const plain = gcm(key, unb64(reply.nonce), replyAad).decrypt(unb64(reply.ct));
  return JSON.parse(decoder.decode(plain));
}
