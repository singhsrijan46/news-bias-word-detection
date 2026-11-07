# Third-party code

`noble-bundle.js` bundles, minified with esbuild, three audited pure-JavaScript libraries by Paul Miller, all under the MIT licence:
- `@noble/curves` 1.9.7: ristretto255
- `@noble/hashes` 1.8.0: SHA-256, SHA-512, HMAC
- `@noble/ciphers` 1.3.0: AES-GCM

To rebuild it:

```bash
npm install @noble/curves@1.9.7 @noble/hashes@1.8.0 @noble/ciphers@1.3.0 esbuild
cat > entry.mjs <<'EOF'
export { RistrettoPoint, ed25519 } from "@noble/curves/ed25519";
export { sha512, sha256 } from "@noble/hashes/sha2";
export { hmac } from "@noble/hashes/hmac";
export { gcm } from "@noble/ciphers/aes";
export { randomBytes } from "@noble/hashes/utils";
EOF
npx esbuild entry.mjs --bundle --format=esm --minify --outfile=noble-bundle.js
```
