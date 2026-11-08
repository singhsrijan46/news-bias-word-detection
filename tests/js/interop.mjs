// Interop check for tests/test_security.py: computes the pairing values and an encrypted
// message with the JavaScript client, from the fixed inputs that Python wrote to a file.
//   node tests/js/interop.mjs vector.json   ->   prints JSON
import { readFileSync } from "node:fs";
import { b64, unb64, generator, deriveKeys, scalarFromBytes } from "../../clients/shared/biaslens-client.js";
import { RistrettoPoint, gcm } from "../../clients/shared/noble-bundle.js";

const v = JSON.parse(readFileSync(process.argv[2], "utf8"));
const hex = (bytes) => Buffer.from(bytes).toString("hex");
const fromHex = (h) => Uint8Array.from(Buffer.from(h, "hex"));

const sid = fromHex(v.sid);
const ya = scalarFromBytes(fromHex(v.ya_scalar));
const yb = scalarFromBytes(fromHex(v.yb_scalar));
const g = generator(sid, v.code);
const Ya = g.multiply(ya).toBytes();
const Yb = g.multiply(yb).toBytes();
const K = RistrettoPoint.fromBytes(Yb).multiply(ya).toBytes();
const keys = deriveKeys(sid, Ya, Yb, K);

// Decrypt Python's message, and encrypt one for Python to decrypt.
const aad = new TextEncoder().encode(v.aad);
const fromPython = new TextDecoder().decode(gcm(keys.deviceKey, unb64(v.py_nonce), aad).decrypt(unb64(v.py_ct)));
const nonce = fromHex("000102030405060708090a0b");
const toPython = gcm(keys.deviceKey, nonce, aad).encrypt(new TextEncoder().encode("hello from JavaScript"));

console.log(JSON.stringify({
  g: hex(g.toBytes()), Ya: hex(Ya), Yb: hex(Yb), K: hex(K),
  board_mac: hex(keys.boardMac), client_mac: hex(keys.clientMac), device_key: hex(keys.deviceKey),
  from_python: fromPython, js_nonce: b64(nonce), js_ct: b64(toPython),
}));
